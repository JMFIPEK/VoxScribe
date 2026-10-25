"""Remote transcription through a configured Provider (any OpenAI-compatible endpoint)."""

import time

import numpy as np
import soundfile as sf

from voxscribe.transcription.audio import SAMPLE_RATE
from voxscribe.transcription.backend import FallbackRequested, TranscribeRequest
from voxscribe.transcription.devices import describe_whisper_device
from voxscribe.transcription.language import _guess_language_from_text, _normalize_language_code
from voxscribe.transcription.model_ids import REMOTE_MODEL_PREFIX


class _PayloadTooLarge(Exception):
    """Internal marker: the server rejected the upload with 413."""


REMOTE_CHUNK_SECONDS = 180.0


REMOTE_MIN_CHUNK_SECONDS = 5.0


# Chunk length for providers that ignore response_format=verbose_json and
# only return plain text (observed with the KIT Toolbox, 2026-10): every
# chunk then becomes one single segment, and all timestamps/speaker turns
# depend on wav2vec2 alignment inside it - which degrades badly over
# minutes-long segments. Short chunks keep that error bounded. Also used as
# the size of the very first request, which doubles as the capability probe
# (see transcribe_remote()).
REMOTE_TEXT_ONLY_CHUNK_SECONDS = 30.0


# Chunk boundaries are moved back to the quietest moment within this window
# before the nominal cut, so no word gets cut in half.
_REMOTE_SPLIT_SEARCH_SECONDS = 4.0


def _find_quiet_split(audio: np.ndarray, start: int, nominal_end: int) -> int:
    """Returns a cut position <= nominal_end at the quietest 100ms frame within
    the last _REMOTE_SPLIT_SEARCH_SECONDS before nominal_end."""
    if nominal_end >= len(audio):
        return len(audio)
    hop = SAMPLE_RATE // 10
    search_start = max(start + hop, nominal_end - int(_REMOTE_SPLIT_SEARCH_SECONDS * SAMPLE_RATE))
    n = (nominal_end - search_start) // hop
    if n <= 0:
        return nominal_end
    window = audio[search_start:search_start + n * hop].reshape(n, hop)
    quietest = int(np.argmin(np.mean(window ** 2, axis=1)))
    return search_start + quietest * hop + hop // 2


def _post_audio_chunk(chunk: np.ndarray, language: str | None, model_name: str,
                       api_key: str, base_url: str, timeout: float = 1800.0
                       ) -> tuple[list[dict], str | None, bool]:
    """Sends a single audio chunk to the server.

    Returns (segments, raw_language_from_response, server_returned_segments).
    The last flag is False for text-only responses, where the chunk's whole
    text was turned into one single segment here.

    `timeout` is deliberately a parameter rather than a fixed value: normal
    file transcription can happily wait a long time for long recordings
    (default 1800s), but other callers transcribing short, rolling snippets
    repeatedly may want a much shorter timeout so a stuck/very slow server
    surfaces as a visible error quickly instead of hanging silently.
    """
    import io
    import requests

    # FLAC instead of WAV: lossless but noticeably smaller uploads (~50-60%),
    # so even long recordings don't run into server upload-size limits.
    buf = io.BytesIO()
    sf.write(buf, chunk, SAMPLE_RATE, format="FLAC")
    buf.seek(0)

    url = base_url.rstrip("/") + "/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model_name, "response_format": "verbose_json"}
    if language:
        data["language"] = language
    files = {"file": ("audio.flac", buf, "audio/flac")}

    resp = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)
    if resp.status_code == 413:
        raise _PayloadTooLarge()
    resp.raise_for_status()
    payload = resp.json()

    segments = []
    for seg in payload.get("segments") or []:
        segments.append({
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", ""),
        })
    has_segments = bool(segments)
    if not segments and (payload.get("text") or "").strip():
        duration = len(chunk) / SAMPLE_RATE
        segments = [{"start": 0.0, "end": duration, "text": payload["text"]}]
    return segments, payload.get("language"), has_segments


# Transient server errors (timeouts, dropped connections, 5xx, 429) are
# retried with these waits (seconds) before a chunk is given up on. Without
# this, a single hiccup in a 100-chunk meeting threw the whole recording
# onto the local fallback model.
_REMOTE_RETRY_WAITS = (5.0, 15.0, 45.0)


def _post_chunk_with_retry(chunk: np.ndarray, language: str | None, model_name: str,
                           api_key: str, base_url: str, timeout: float = 1800.0
                           ) -> tuple[list[dict], str | None, bool]:
    import requests

    for attempt, wait in enumerate((*_REMOTE_RETRY_WAITS, None)):
        try:
            return _post_audio_chunk(chunk, language, model_name, api_key, base_url, timeout=timeout)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if wait is None or not (status == 429 or status >= 500):
                raise
            reason = f"HTTP {status}"
        except (requests.ConnectionError, requests.Timeout) as e:
            if wait is None:
                raise
            reason = type(e).__name__
        print(f"     [WARNING] Chunk request failed ({reason}), retry {attempt + 1}/"
              f"{len(_REMOTE_RETRY_WAITS)} in {wait:.0f}s")
        time.sleep(wait)


def _transcribe_chunk_with_backoff(chunk: np.ndarray, language: str | None, model_name: str,
                                    api_key: str, base_url: str, timeout: float = 1800.0
                                    ) -> tuple[list[dict], str | None, bool]:
    """Sends a chunk; if rejected with 413, recursively halves it and retries."""
    try:
        return _post_chunk_with_retry(chunk, language, model_name, api_key, base_url, timeout=timeout)
    except _PayloadTooLarge:
        duration = len(chunk) / SAMPLE_RATE
        if duration <= REMOTE_MIN_CHUNK_SECONDS:
            raise ValueError(
                "Server rejects uploads with 413 (Request Entity Too Large), "
                "even for very short audio chunks. Check the server configuration."
            )
        mid = _find_quiet_split(chunk, 0, len(chunk) // 2)
        left, lang_left, segs_left = _transcribe_chunk_with_backoff(
            chunk[:mid], language, model_name, api_key, base_url, timeout=timeout)
        right, lang_right, segs_right = _transcribe_chunk_with_backoff(
            chunk[mid:], language, model_name, api_key, base_url, timeout=timeout)
        offset = mid / SAMPLE_RATE
        for seg in right:
            seg["start"] += offset
            seg["end"] += offset
        return left + right, (lang_left or lang_right), (segs_left or segs_right)


def transcribe_remote(
    audio: np.ndarray,
    language: str | None,
    model_name: str,
    api_key: str,
    base_url: str,
    chunk_seconds: float = REMOTE_CHUNK_SECONDS,
    on_progress=None,
    timeout: float = 1800.0,
    on_segments=None,
) -> dict:
    """Sends audio in chunks to any OpenAI-compatible transcription endpoint.

    Long recordings are split into `chunk_seconds` pieces (and further halved
    automatically on a server 413 error), since server endpoints typically
    have an upload size limit. Cuts are placed at the quietest moment just
    before each nominal boundary (_find_quiet_split()).

    The first request is only REMOTE_TEXT_ONLY_CHUNK_SECONDS long and doubles
    as a capability probe: if the server returns real segments, the rest is
    sent in `chunk_seconds` pieces; if it only returns plain text, all
    remaining chunks stay short too (see REMOTE_TEXT_ONLY_CHUNK_SECONDS).

    If `language` is None/empty, no language is sent and the server detects
    it per chunk, so meetings that switch languages work. (Forcing the first
    chunk's language on the rest made Whisper *translate* every later German
    passage of an English-first meeting into English.) Chunks then also stay
    REMOTE_TEXT_ONLY_CHUNK_SECONDS long, because Whisper picks one language
    per uploaded file. Every segment is tagged with its chunk's language
    ("language" key): the server's report, else a guess from the returned
    text (_guess_language_from_text()), else the previous chunk's language -
    short/quiet chunks don't carry enough text to decide on their own. The
    two chunks at every language switch are then transcribed again in ~10s
    pieces (_refine_language_switches()), so the switch lands within ~10s
    instead of ~30s. The alignment step uses the tags to pick the right
    wav2vec2 model per segment (see _align_by_language()).

    If `language` is given, it's sent with every chunk, as before.

    `on_segments(segments)` gets each chunk's segments as soon as they arrive
    (before the language refinement, which may still replace some of them).

    Returns a dict in WhisperX format ({"segments": [...], "language": ...})
    that can be fed into alignment and diarization afterward. "language" is
    the language covering most of the speech.
    """
    if not api_key:
        raise ValueError(
            "No API key set for this provider - configure it in Settings."
        )

    total_samples = len(audio)
    probe_samples = int(min(chunk_seconds, REMOTE_TEXT_ONLY_CHUNK_SECONDS) * SAMPLE_RATE)
    chunk_samples = probe_samples

    # One record per chunk: sample range, segments (absolute times) and the
    # language the chunk decided on by itself (None = not enough evidence).
    chunks: list[dict] = []
    offset = 0
    while offset < total_samples:
        end = _find_quiet_split(audio, offset, offset + chunk_samples)
        segments, own_language, has_segments = _transcribe_span(
            audio, offset, end, language, model_name, api_key, base_url, timeout)
        if not chunks:
            if has_segments and language:
                chunk_samples = max(1, int(chunk_seconds * SAMPLE_RATE))
            elif not has_segments:
                print(f"     Provider returned text only (no segments/timestamps) - "
                      f"using {REMOTE_TEXT_ONLY_CHUNK_SECONDS:.0f}s chunks")
        chunks.append({"start": offset, "end": end, "segments": segments,
                       "language": own_language})
        if on_segments and segments:
            on_segments([dict(seg) for seg in segments])
        if on_progress:
            on_progress(min(end / max(total_samples, 1), 1.0))
        offset = end

    fallback = language or "en"
    _fill_chunk_languages(chunks, fallback)
    if not language:
        chunks = _refine_language_switches(
            audio, chunks, model_name, api_key, base_url, timeout)
        previous = None
        for chunk in chunks:
            if chunk["language"] != previous:
                print(f"     {chunk['start'] / SAMPLE_RATE / 60:5.1f} min: language '{chunk['language']}'")
                previous = chunk["language"]

    all_segments = []
    for chunk in chunks:
        for seg in chunk["segments"]:
            seg["language"] = chunk["language"]
            all_segments.append(seg)
    speech_seconds: dict[str, float] = {}
    for seg in all_segments:
        speech_seconds[seg["language"]] = speech_seconds.get(seg["language"], 0.0)             + max(0.0, seg["end"] - seg["start"])
    main_language = max(speech_seconds, key=speech_seconds.get) if speech_seconds else fallback
    return {"segments": all_segments, "language": main_language}


def _transcribe_span(audio: np.ndarray, start: int, end: int, language: str | None,
                     model_name: str, api_key: str, base_url: str, timeout: float
                     ) -> tuple[list[dict], str | None, bool]:
    """Transcribes audio[start:end]. Returns (segments with absolute times,
    the span's own language or None, server_returned_segments)."""
    segments, raw_language, has_segments = _transcribe_chunk_with_backoff(
        audio[start:end], language or None, model_name, api_key, base_url, timeout=timeout)
    offset = start / SAMPLE_RATE
    for seg in segments:
        seg["start"] += offset
        seg["end"] += offset
    own_language = language or _normalize_language_code(raw_language)         or _guess_language_from_text(" ".join(s["text"] for s in segments))
    return segments, own_language, has_segments


def _fill_chunk_languages(chunks: list[dict], fallback: str) -> None:
    """Undecided chunks (short/quiet, too little text) take the previous
    chunk's language; leading ones the first decided language."""
    first_known = next((c["language"] for c in chunks if c["language"]), fallback)
    current = first_known
    for chunk in chunks:
        current = chunk["language"] or current
        chunk["language"] = current


# Around a language switch, the chunks on both sides are transcribed again in
# pieces of about this length, so the switch lands within ~10s instead of
# ~30s. Not used for whole recordings: Whisper's language ID gets unreliable
# on short audio and loses context, so only the switch windows pay that price.
REMOTE_REFINE_PIECE_SECONDS = 10.0


def _refine_language_switches(audio: np.ndarray, chunks: list[dict], model_name: str,
                              api_key: str, base_url: str, timeout: float) -> list[dict]:
    """Re-transcribes the two chunks at every language switch in
    REMOTE_REFINE_PIECE_SECONDS pieces, each with its own language detection,
    and replaces them with the pieces. Without this, the end of an English
    talk that shares a 30s chunk with a mostly German discussion came out
    translated into German."""
    refine = set()
    for i in range(1, len(chunks)):
        if chunks[i]["language"] != chunks[i - 1]["language"]:
            refine.update((i - 1, i))
    if not refine:
        return chunks

    piece_samples = int(REMOTE_REFINE_PIECE_SECONDS * SAMPLE_RATE)
    min_samples = int(REMOTE_MIN_CHUNK_SECONDS * SAMPLE_RATE)
    refined: list[dict] = []
    i = 0
    while i < len(chunks):
        if i not in refine:
            refined.append(chunks[i])
            i += 1
            continue
        j = i
        while j + 1 < len(chunks) and j + 1 in refine:
            j += 1
        start, end = chunks[i]["start"], chunks[j]["end"]
        print(f"     Refining language switch {start / SAMPLE_RATE / 60:.1f}-"
              f"{end / SAMPLE_RATE / 60:.1f} min in {REMOTE_REFINE_PIECE_SECONDS:.0f}s pieces")
        previous = chunks[i]["language"]
        pos = start
        while pos < end:
            cut = _find_quiet_split(audio[:end], pos, pos + piece_samples)
            if end - cut < min_samples:
                cut = end
            segments, own_language, _ = _transcribe_span(
                audio, pos, cut, None, model_name, api_key, base_url, timeout)
            previous = own_language or previous
            refined.append({"start": pos, "end": cut, "segments": segments,
                            "language": previous})
            pos = cut
        i = j + 1
    return refined


class RemoteBackend:
    name = "Provider"
    skips_alignment = False

    def describe(self, model_id: str, device: str) -> str:
        return describe_whisper_device(device)

    def transcribe(self, request: TranscribeRequest) -> dict:
        from voxscribe.providers import get_provider

        provider_id = request.model_id[len(REMOTE_MODEL_PREFIX):]
        provider = get_provider(provider_id)
        api_key, api_base_url = request.api_key, request.api_base_url
        if provider is None and not (api_key and api_base_url):
            raise ValueError(
                f"Unknown provider '{provider_id}' - configure it in Settings, "
                "or pass api_key/api_base_url explicitly."
            )
        model_name = provider["model"] if provider else (request.model_id or "whisper-1")
        resolved_api_key = api_key or (provider["api_key"] if provider else None)
        resolved_base_url = api_base_url or (provider["base_url"] if provider else None)
        provider_label = provider["name"] if provider else provider_id

        print(f"1/3  Transcribing via provider ({provider_label} / {model_name})...")
        request.progress(0.0, "Loading audio...")
        audio = request.audio.load()

        def _progress(frac):
            request.progress(0.10 + frac * 0.30, f"Sending audio to server... {frac * 100:.0f}%")

        request.progress(0.10, f"Sending audio to server ({provider_label})...")
        try:
            return transcribe_remote(
                audio, request.language, model_name,
                api_key=resolved_api_key, base_url=resolved_base_url,
                on_progress=_progress, on_segments=request.on_segments,
            )
        except Exception as e:
            raise FallbackRequested(str(e)) from e
