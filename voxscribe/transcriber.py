"""WhisperX-based transcription with alignment and speaker diarization."""

import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import whisperx
from whisperx.utils import LANGUAGES as _WHISPER_LANGUAGE_NAMES

from voxscribe.paths import get_base_dir, get_bundled_models_dir

# whisperx.diarize (pyannote.audio + lightning) is imported lazily inside
# transcribe()'s diarization step and via prewarm_diarization(): it's ~80%
# of this module's import time (~11s warm, far more on a cold start), and
# the GUI imports this module long before any diarization happens.

# Must be set before the first MPS op: pyannote/wav2vec2 use a few ops that
# aren't (yet) implemented for Apple's MPS backend - without the fallback
# flag those crash with NotImplementedError instead of transparently running
# on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SAMPLE_RATE = 16000

# Map full language names (e.g. "german", as returned by OpenAI-compatible
# APIs) to ISO-639-1 codes (e.g. "de"), which is what whisperx expects for
# alignment.
_LANGUAGE_NAME_TO_CODE = {name.lower(): code for code, name in _WHISPER_LANGUAGE_NAMES.items()}


def _normalize_language_code(raw: str | None, fallback: str | None = None) -> str | None:
    """Normalizes a language identifier (name or code) returned by a server API."""
    if not raw:
        return fallback
    raw = raw.strip().lower()
    if raw in _WHISPER_LANGUAGE_NAMES:
        return raw
    return _LANGUAGE_NAME_TO_CODE.get(raw, fallback)


# Very common function words per language - enough to tell the GUI's
# languages (see qt_app/constants.py LANGUAGES) apart reliably from a few
# sentences of transcribed text. Used where the transcription backend
# returns no language of its own (text-only providers, Parakeet), since the
# alignment step needs the correct language-specific wav2vec2 model.
_STOPWORDS = {
    "de": {"der", "die", "das", "und", "ist", "nicht", "ich", "wir", "es", "ein", "eine",
           "zu", "mit", "auf", "für", "auch", "dass", "sie", "den", "dem", "von", "haben",
           "aber", "noch", "jetzt", "also", "oder", "wenn", "schon", "mal", "kann", "sind"},
    "en": {"the", "and", "is", "not", "we", "it", "a", "to", "of", "that", "you", "this",
           "for", "with", "on", "have", "are", "be", "but", "so", "what", "if", "can",
           "there", "they", "do", "was", "would", "just", "or", "will", "at"},
    "fr": {"le", "la", "les", "et", "est", "pas", "je", "nous", "un", "une", "des", "que",
           "pour", "avec", "sur", "dans", "ce", "qui", "mais", "vous", "il", "elle", "du"},
    "es": {"el", "la", "los", "las", "y", "es", "no", "yo", "un", "una", "que", "de",
           "para", "con", "en", "por", "pero", "lo", "se", "del", "como", "está", "muy"},
    "it": {"il", "la", "le", "e", "è", "non", "io", "noi", "un", "una", "che", "di",
           "per", "con", "su", "ma", "sono", "gli", "della", "anche", "questo", "molto"},
}
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _guess_language_from_text(text: str) -> str | None:
    """Best-guess ISO-639-1 code from transcribed text, or None if there's
    too little evidence (short/empty text, or no clear winner)."""
    words = _WORD_RE.findall((text or "").lower())
    if len(words) < 8:
        return None
    scores = {lang: sum(1 for w in words if w in sw) for lang, sw in _STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_score), (_, second_score) = ranked[0], ranked[1]
    if best_score < 3 or best_score < 1.5 * second_score:
        return None
    return best


_get_base_dir = get_base_dir
_get_bundled_models_dir = get_bundled_models_dir


def load_audio_without_ffmpeg(audio_path: str) -> np.ndarray:
    """Loads an audio file and converts it to 16kHz mono float32 (no ffmpeg)."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SAMPLE_RATE, sr)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return data


def _load_audio_via_container(audio_path: str) -> np.ndarray:
    """Extracts the audio track from an arbitrary container (MKV, MP4, MOV, ...) via PyAV.

    Used as a fallback when soundfile can't read the format directly (e.g.
    video containers like MKV, where only the audio track is needed).
    """
    import av

    container = av.open(audio_path)
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        container.close()
        raise ValueError(f"No audio track found in: {audio_path}")

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray())
    for resampled in resampler.resample(None):
        chunks.append(resampled.to_ndarray())
    container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    data = np.concatenate(chunks, axis=1).reshape(-1)
    return (data.astype(np.float32) / 32768.0)


def load_audio_universal(audio_path: str) -> np.ndarray:
    """Loads an audio or video file as a 16kHz mono float32 array.

    Tries soundfile first (fast path for WAV/FLAC/OGG etc.). If that fails
    (e.g. video containers like MKV/MP4, or compressed formats libsndfile
    doesn't know), falls back to decoding just the audio track via PyAV.
    """
    try:
        return load_audio_without_ffmpeg(audio_path)
    except Exception:
        return _load_audio_via_container(audio_path)


REMOTE_MODEL_PREFIX = "server:"


def is_remote_model(model_size: str) -> bool:
    """Checks whether this refers to a server-hosted (provider) model."""
    return model_size.startswith(REMOTE_MODEL_PREFIX)


def remote_model_id(provider_id: str) -> str:
    """Builds the model_size string for a configured provider, e.g. 'server:my-provider'."""
    return f"{REMOTE_MODEL_PREFIX}{provider_id}"


OPENVINO_MODEL_PREFIX = "openvino:"


def is_openvino_model(model_size: str) -> bool:
    """Checks whether transcription should run via the Intel OpenVINO backend
    (Arc GPU or NPU) instead of CTranslate2 (WhisperX' default backend, which
    has no Intel GPU/NPU support at all - see CLAUDE.md)."""
    return model_size.startswith(OPENVINO_MODEL_PREFIX)


def openvino_model_id(ov_device: str, whisper_size: str) -> str:
    """Builds the model_size string for transcribe(), e.g. 'openvino:GPU:medium'."""
    return f"{OPENVINO_MODEL_PREFIX}{ov_device}:{whisper_size}"


def _parse_openvino_model(model_size: str) -> tuple[str, str]:
    """'openvino:GPU:medium' -> ('GPU', 'medium')."""
    rest = model_size[len(OPENVINO_MODEL_PREFIX):]
    ov_device, whisper_size = rest.split(":", 1)
    return ov_device, whisper_size


_OPENVINO_LANG_TOKEN_RE = re.compile(r"<\|([a-z]{2})\|>")


def _detect_language_from_token_ids(processor, generated_ids) -> str | None:
    """Extracts the language token Whisper predicted (e.g. '<|de|>') from the
    first generated tokens, when generate() ran without a forced language."""
    tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids[0].tolist()[:3])
    for tok in tokens:
        m = _OPENVINO_LANG_TOKEN_RE.match(tok)
        if m:
            return m.group(1)
    return None


_OPENVINO_CHUNK_SECONDS = 30


def transcribe_openvino(
    audio_path: str,
    language: str | None,
    ov_device: str,
    whisper_size: str,
    on_progress=None,
) -> dict:
    """Transcribes via Intel's OpenVINO backend (Arc GPU or NPU) instead of CTranslate2.

    CTranslate2 (WhisperX' default inference backend, see transcribe()) has no
    Intel GPU/NPU support at all - this path instead loads a Whisper model
    exported to OpenVINO IR via `optimum-intel` (export step: see
    download_models.py::export_openvino_model()) and runs it on the requested
    OpenVINO device ("GPU" = Arc, "NPU" = Intel AI Boost, "CPU").

    Chunking is implemented manually in 30s windows instead of using
    HuggingFace's own ASR pipeline (`transformers.pipeline(
    "automatic-speech-recognition", chunk_length_s=...)`), because its
    internal chunking iterators unconditionally import `torchcodec`
    (ffmpeg-based) - deliberately not installed in this project, see
    load_audio_universal()/load_audio_without_ffmpeg() elsewhere in this file
    for the same ffmpeg-avoidance reason.

    Returns the same {"segments": [...], "language": ...} shape the
    CTranslate2 path returns, so the subsequent alignment step (which adds
    the word-level timestamps this path doesn't produce on its own) and
    diarization can run unmodified afterward.
    """
    from transformers import AutoProcessor
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq

    bundled = _get_bundled_models_dir()
    if not bundled:
        raise FileNotFoundError(
            "No bundled_models folder found - run 'python download_models.py' "
            "to export the OpenVINO model."
        )
    model_dir = os.path.join(bundled, "openvino", f"whisper-{whisper_size}")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"OpenVINO model not found: {model_dir}. "
            "Run 'python download_models.py' first."
        )
    cache_dir = os.path.join(bundled, "openvino", "_ov_cache")

    processor = AutoProcessor.from_pretrained(model_dir)
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_dir, device=ov_device, ov_config={"CACHE_DIR": cache_dir}
    )

    audio = load_audio_universal(audio_path)
    sr = SAMPLE_RATE
    chunk_samples = _OPENVINO_CHUNK_SECONDS * sr
    n_chunks = max(1, -(-len(audio) // chunk_samples))

    detected_language = language
    segments = []
    for i, start in enumerate(range(0, len(audio), chunk_samples)):
        chunk = audio[start:start + chunk_samples]
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        gen_kwargs = {"task": "transcribe"}
        if detected_language:
            gen_kwargs["language"] = detected_language
        generated_ids = model.generate(inputs["input_features"], **gen_kwargs)
        if detected_language is None:
            # Whisper detects the language itself on the first (unforced)
            # call - force it consistently for every subsequent chunk from
            # here on, mirroring the CTranslate2 path which also only uses
            # the first 30s for detection.
            detected_language = _detect_language_from_token_ids(processor, generated_ids) or "en"
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if text:
            segments.append({
                "start": start / sr,
                "end": min(start + chunk_samples, len(audio)) / sr,
                "text": text,
            })
        if on_progress:
            on_progress((i + 1) / n_chunks * 100)

    del model
    return {"segments": segments, "language": detected_language or "en"}


PARAKEET_MODEL_PREFIX = "parakeet:"
PARAKEET_MODEL = "parakeet:tdt-0.6b-v3"
_PARAKEET_ONNX_ASR_NAMES = {"tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3"}


def is_parakeet_model(model_size: str) -> bool:
    """Checks whether transcription should run via NVIDIA Parakeet (onnx-asr)."""
    return model_size.startswith(PARAKEET_MODEL_PREFIX)


def _parakeet_local_dirs(variant: str) -> tuple[str | None, str | None]:
    """Bundled model/VAD dirs (see download_models.py), or (None, None) to use
    the HuggingFace cache instead."""
    bundled = _get_bundled_models_dir()
    if not bundled:
        return None, None
    model_dir = os.path.join(bundled, "parakeet", variant)
    vad_dir = os.path.join(bundled, "parakeet", "silero-vad")
    return (model_dir if os.path.isdir(model_dir) else None,
            vad_dir if os.path.isdir(vad_dir) else None)


def transcribe_parakeet(audio: np.ndarray, language: str | None, model_size: str,
                        on_progress=None) -> dict:
    """Transcribes via NVIDIA Parakeet TDT 0.6B v3 (25 European languages,
    including German/English) through `onnx-asr` on ONNX Runtime (CPU).

    Roughly 8-10x realtime on a laptop CPU (measured on a Core Ultra 7 258V),
    i.e. several times faster than Whisper int8 on the same CPU, at
    comparable accuracy. Parakeet only accepts ~20-30s of audio at once, so
    long recordings are cut into speech segments by Silero VAD first (also
    via onnx-asr); those segments become the WhisperX-style segments that the
    regular alignment/diarization steps then process unchanged.

    Parakeet detects the language internally but doesn't report it, so it's
    guessed from the text (_guess_language_from_text()) unless given.
    """
    import onnx_asr

    variant = model_size[len(PARAKEET_MODEL_PREFIX):]
    name = _PARAKEET_ONNX_ASR_NAMES.get(variant)
    if name is None:
        raise ValueError(f"Unknown Parakeet model: {model_size}")
    model_dir, vad_dir = _parakeet_local_dirs(variant)

    model = onnx_asr.load_model(name, model_dir, quantization="int8")
    vad = onnx_asr.load_vad("silero", vad_dir)
    # min_silence_duration_ms: merge speech separated by short pauses into
    # longer segments - Parakeet is noticeably more accurate with a full
    # sentence of context than with the VAD's default ~100ms fragments.
    asr = model.with_vad(vad, batch_size=4, max_speech_duration_s=25,
                         min_silence_duration_ms=800)

    duration = len(audio) / SAMPLE_RATE
    segments = []
    for seg in asr.recognize(audio, sample_rate=SAMPLE_RATE):
        text = (seg.text or "").strip()
        if text:
            segments.append({"start": float(seg.start), "end": float(seg.end), "text": text})
        if on_progress and duration > 0:
            on_progress(min(seg.end / duration, 1.0))

    detected = language or _guess_language_from_text(" ".join(s["text"] for s in segments))
    del model, vad, asr
    return {"segments": segments, "language": detected or "en"}


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


def _transcribe_chunk_with_backoff(chunk: np.ndarray, language: str | None, model_name: str,
                                    api_key: str, base_url: str, timeout: float = 1800.0
                                    ) -> tuple[list[dict], str | None, bool]:
    """Sends a chunk; if rejected with 413, recursively halves it and retries."""
    try:
        return _post_audio_chunk(chunk, language, model_name, api_key, base_url, timeout=timeout)
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

    If `language` is None/empty, no language is sent (the server detects it
    itself). The language detected from the first chunk is then used for all
    subsequent chunks, so detection doesn't flip-flop chunk to chunk (short/
    quiet chunks can otherwise easily detect the wrong language). Servers
    that don't report a language get it guessed from the returned text
    (_guess_language_from_text()) - otherwise alignment would silently run
    the English model on e.g. German text.

    Returns a dict in WhisperX format ({"segments": [...], "language": ...})
    that can be fed into alignment and diarization afterward.
    """
    if not api_key:
        raise ValueError(
            "No API key set for this provider - configure it in Settings."
        )

    total_samples = len(audio)
    probe_samples = int(min(chunk_seconds, REMOTE_TEXT_ONLY_CHUNK_SECONDS) * SAMPLE_RATE)
    chunk_samples = probe_samples

    all_segments = []
    detected_language_code = None
    pending_text = ""  # text-only servers: accumulated until the language is clear
    offset = 0
    first = True
    request_language = language
    while offset < total_samples:
        end = _find_quiet_split(audio, offset, offset + chunk_samples)
        chunk = audio[offset:end]
        chunk_start_time = offset / SAMPLE_RATE

        chunk_segments, raw_language, has_segments = _transcribe_chunk_with_backoff(
            chunk, request_language, model_name, api_key, base_url, timeout=timeout)
        for seg in chunk_segments:
            seg["start"] += chunk_start_time
            seg["end"] += chunk_start_time
            all_segments.append(seg)

        if first:
            first = False
            if has_segments:
                chunk_samples = max(1, int(chunk_seconds * SAMPLE_RATE))
            else:
                print(f"     Provider returned text only (no segments/timestamps) - "
                      f"using {REMOTE_TEXT_ONLY_CHUNK_SECONDS:.0f}s chunks")

        if detected_language_code is None:
            detected_language_code = _normalize_language_code(raw_language, fallback=language)
            if detected_language_code is None:
                pending_text += " " + " ".join(s["text"] for s in chunk_segments)
                detected_language_code = _guess_language_from_text(pending_text)
                if detected_language_code:
                    print(f"     Provider reported no language - detected '{detected_language_code}' from text")
            if not request_language and detected_language_code:
                # Lock in the language for the remaining chunks, see docstring.
                request_language = detected_language_code

        if on_progress:
            on_progress(min(end / max(total_samples, 1), 1.0))
        offset = end

    return {"segments": all_segments, "language": detected_language_code or language or "en"}


APPLE_MODEL_PREFIX = "apple:"
APPLE_SPEECHANALYZER_MODEL = "apple:speechanalyzer"


def is_apple_model(model_size: str) -> bool:
    """Checks whether the Apple SpeechAnalyzer engine (macOS only) should be used."""
    return model_size.startswith(APPLE_MODEL_PREFIX)


# ISO-639-1 -> full BCP-47 locale, as expected by SpeechTranscriber. Unlike
# WhisperX/provider models, SpeechAnalyzer (as of macOS 26) has no automatic
# language detection of its own - see detect_apple_language() for how
# transcribe_apple() works around that instead of just guessing "en".
_APPLE_LOCALE_BY_LANGUAGE = {
    "de": "de-DE", "en": "en-US", "fr": "fr-FR", "es": "es-ES",
    "it": "it-IT", "pt": "pt-PT", "ja": "ja-JP", "ko": "ko-KR", "zh": "zh-CN",
}

# Languages detect_apple_language() actually probes for - this app's users
# only ever record German or English meetings, not the full
# _APPLE_LOCALE_BY_LANGUAGE list above (that list stays for explicit
# language= callers, e.g. a future non-GUI use). Kept short deliberately:
# each additional candidate this Mac has never used before costs ~20s for a
# one-time on-device model download (measured directly - de-DE/en-US came
# back in ~0.4s once cached, fr-FR/es-ES/it-IT took 18-24s on first use).
_APPLE_AUTODETECT_CANDIDATES = ("de", "en")


def _apple_locale_identifier(language: str | None) -> str:
    return _APPLE_LOCALE_BY_LANGUAGE.get((language or "en").lower(), "en-US")


def _apple_speechanalyzer_binary_path() -> str:
    """Path to the compiled SpeechAnalyzer helper (see macos/README.md)."""
    return os.path.join(_get_base_dir(), "macos", "SpeechAnalyzerTranscribe")


def _apple_transcribe_raw(audio_path: str, locale_id: str) -> list[dict]:
    """Runs the SpeechAnalyzer helper once, blocking, and returns its raw
    parsed segments (word list included) - the shared building block behind
    both transcribe_apple() (full run, streamed) and
    detect_apple_language() (short probe, only needs the confidence scores).
    Returns [] on failure instead of raising, since detect_apple_language()
    treats a failed/empty candidate as simply losing the confidence
    comparison rather than aborting transcription over it."""
    binary = _apple_speechanalyzer_binary_path()
    try:
        proc = subprocess.run(
            [binary, audio_path, locale_id],
            capture_output=True, text=True, timeout=120,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if proc.returncode != 0:
        return []
    segments = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "segment":
            segments.append(obj)
    return segments


def _apple_probe_segment_path(audio_path: str, probe_seconds: float = 12.0) -> str:
    """Writes a short snippet to a temp WAV for detect_apple_language() to
    probe. Starts 10% into the file (floor 3s) rather than at 0 to skip
    likely dead air/silence at the very start of a recording, which would
    otherwise make every candidate language score equally low."""
    info = sf.info(audio_path)
    total_seconds = info.frames / info.samplerate
    offset_seconds = min(max(total_seconds * 0.1, 3.0), max(total_seconds - probe_seconds, 0.0))
    data, sr = sf.read(
        audio_path,
        start=int(offset_seconds * info.samplerate),
        frames=int(probe_seconds * info.samplerate),
    )
    fd, probe_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(probe_path, data, sr)
    return probe_path


def detect_apple_language(audio_path: str) -> str:
    """Guesses German vs. English for the Apple SpeechAnalyzer path: runs a
    short probe snippet through both locales and keeps whichever one
    SpeechTranscriber was more confident about (mean per-word "score").
    This is the same signal that first exposed the bug this replaces - German
    audio forced through en-US scored ~0.24 average confidence vs. ~0.80 for
    the correct de-DE - so the two candidates are normally far enough apart
    for this to be a reliable, model-free way to tell them apart, without
    adding a real audio-language-ID model or a Whisper dependency on macOS
    (see CLAUDE.md's "macOS transcription runs on Apple's SpeechAnalyzer, not
    WhisperX" for why the latter is avoided deliberately). Falls back to "de"
    if the probe segment has no recognizable words in either language (e.g.
    it landed on silence).
    """
    probe_path = _apple_probe_segment_path(audio_path)
    try:
        best_lang, best_score = _APPLE_AUTODETECT_CANDIDATES[0], -1.0
        for lang in _APPLE_AUTODETECT_CANDIDATES:
            segments = _apple_transcribe_raw(probe_path, _apple_locale_identifier(lang))
            scores = [w["score"] for seg in segments for w in seg.get("words", [])]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if avg_score > best_score:
                best_lang, best_score = lang, avg_score
        return best_lang
    finally:
        os.remove(probe_path)


def default_model_size() -> str:
    """Platform-wide default for model_size when the caller has no explicit
    preference: the user's configured default provider (see providers.py),
    if one is set up - otherwise falls back to the best local model for this
    platform/hardware (default_local_model_size()). transcribe() also falls
    back to a local model automatically if a chosen provider call fails
    (network, server down, ...), see there.
    """
    from voxscribe.providers import get_default_provider
    provider = get_default_provider()
    if provider:
        return remote_model_id(provider["id"])
    return default_local_model_size()


def default_local_model_size() -> str:
    """Recommended *local* model - used as a fallback whenever a remote
    provider call fails or isn't configured, and anywhere a local model is
    explicitly wanted. Platform-/hardware-dependent:
    - macOS: always Apple's SpeechAnalyzer (Neural Engine) - the only local
      transcription engine there, no Whisper model download/inference at
      all, see CLAUDE.md ("Apple Silicon (MPS) acceleration is partial by
      design").
    - Windows/Linux: whatever hardware_detect.recommend_model() picks - a
      dedicated NVIDIA GPU (CUDA) recommends large-v3/medium/base depending
      on VRAM, an Intel Arc GPU (no CUDA present) recommends the OpenVINO
      model, otherwise it falls back to a RAM-sized CPU model.
    """
    if sys.platform == "darwin":
        return APPLE_SPEECHANALYZER_MODEL
    from voxscribe.hardware_detect import recommend_model
    model_size, _device, _reason = recommend_model()
    return model_size


def transcribe_apple(
    audio_path: str,
    language: str | None = None,
    on_progress=None,
) -> dict:
    """Transcribes via Apple's SpeechAnalyzer/SpeechTranscriber (macOS 26+, Neural Engine).

    Runs the compiled Swift helper macos/SpeechAnalyzerTranscribe as a
    one-shot subprocess (no continuous stream like the system-audio
    recording path, so no watcher thread is needed either - the process
    exits on its own) and reads its NDJSON output line by line for progress.

    SpeechTranscriber already provides a per-word timestamp
    (attributeOptions: [.audioTimeRange] in the Swift helper) - verified
    word-accurate on real hardware against test audio. The returned dict is
    therefore already in the "aligned" format that WhisperX's wav2vec2
    alignment step would otherwise produce ({"segments": [...{"words":
    [...]}...], "word_segments": [...]}) - the alignment step is skipped
    entirely for this path (see the branch in transcribe()).

    language=None (the GUI's default) runs detect_apple_language() first -
    unlike WhisperX/remote providers, SpeechTranscriber has no auto-detect of
    its own, and silently assuming English previously produced garbage on
    German audio (~0.24 avg word confidence vs. ~0.80 for the correct
    locale - this is exactly the bug that motivated adding
    detect_apple_language()).
    """
    binary = _apple_speechanalyzer_binary_path()
    if not os.path.isfile(binary):
        raise RuntimeError(
            f"SpeechAnalyzer helper not found ({binary}). "
            "Run 'cd macos && ./build.sh' first."
        )

    if language is None:
        if on_progress:
            on_progress(0.0)
        language = detect_apple_language(audio_path)

    locale_id = _apple_locale_identifier(language)
    try:
        duration = sf.info(audio_path).duration or 1.0
    except Exception:
        duration = 1.0

    proc = subprocess.Popen(
        [binary, audio_path, locale_id],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )

    segments = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "segment":
            continue
        words = [
            {"word": w["word"], "start": w["start"], "end": w["end"], "score": w["score"]}
            for w in obj.get("words", [])
        ]
        segments.append({
            "start": obj["start"], "end": obj["end"], "text": obj["text"], "words": words,
        })
        if on_progress:
            on_progress(min(obj["end"] / duration, 1.0))

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().strip() if proc.stderr else ""
        raise RuntimeError(f"SpeechAnalyzer transcription failed: {stderr}")

    word_segments = [w for seg in segments for w in seg["words"]]
    return {
        "segments": segments,
        "word_segments": word_segments,
        "language": (language or "en").lower(),
    }


def prewarm_diarization() -> None:
    """Imports the diarization stack (pyannote.audio) ahead of time - for
    the GUI to call in an idle background thread after startup, so the first
    transcription doesn't pay the import at its diarization step. Import
    only, no model is loaded (no VRAM/RAM for weights)."""
    import whisperx.diarize  # noqa: F401


def transcribe(
    audio_path: str,
    language: str | None = "en",
    model_size: str = "large-v2",
    diarize: bool = True,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    batch_size: int = 16,
    device: str | None = None,
    on_progress=None,
    api_key: str | None = None,
    api_base_url: str | None = None,
) -> dict:
    """Runs the full WhisperX pipeline.

    1. Transcription (batched inference)
    2. Forced alignment (word-level timestamps)
    3. Speaker diarization (optional)

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        language: Language of the audio (e.g. "en", "de"). None = automatic
            detection - WhisperX/remote providers detect from the audio itself;
            Apple's SpeechAnalyzer has no such capability, so
            transcribe_apple() runs its own German-vs-English probe instead
            (see detect_apple_language())
        model_size: Whisper model size ("large-v2", "large-v3", "medium",
            "base"), a provider model (prefix "server:", see is_remote_model()),
            an OpenVINO model (prefix "openvino:", see is_openvino_model()),
            "parakeet:tdt-0.6b-v3" (NVIDIA Parakeet via onnx-asr, see
            transcribe_parakeet()), or
            "apple:speechanalyzer" for Apple's SpeechAnalyzer (macOS 26+ only,
            see is_apple_model()/transcribe_apple())
        diarize: Enable speaker diarization
        hf_token: HuggingFace token for pyannote (only needed for the first download)
        min_speakers: Minimum number of speakers (optional)
        max_speakers: Maximum number of speakers (optional)
        batch_size: Inference batch size (smaller = less VRAM)
        device: "cuda", "xpu", "mps" or "cpu" (auto-detected if None). Only
            affects alignment/diarization (plain PyTorch) - Whisper
            transcription itself runs via CTranslate2, which has no MPS/XPU
            support and therefore always falls back to "cpu" for that step.
        api_key: Override the configured provider's API key (optional, for
            ad-hoc use without a saved provider)
        api_base_url: Override the configured provider's base URL (optional,
            same purpose as api_key)

    Returns:
        Dict with "segments", "language", and speaker labels if applicable
    """
    if device is None:
        # Priority dGPU > iGPU > CPU: CUDA (dedicated NVIDIA GPU) first, then
        # Intel XPU (Arc GPU - only available if torch was installed with an
        # XPU-capable build, see gpu_setup.py; the standard CUDA install has
        # torch.xpu.is_available() == False and falls through here
        # transparently), then Apple MPS.
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            device = "xpu"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # CTranslate2 (WhisperX' inference backend) only knows "cuda"/"cpu" -
    # neither MPS nor XPU. Alignment (wav2vec2) and diarization (pyannote)
    # are plain PyTorch and can use MPS/XPU instead, hence two separate
    # device variables.
    whisper_device = device if device == "cuda" else "cpu"
    torch_device = device

    compute_type = "float16" if whisper_device == "cuda" else "int8"
    remote = is_remote_model(model_size)
    apple = is_apple_model(model_size)
    openvino = is_openvino_model(model_size)
    parakeet = is_parakeet_model(model_size)

    if parakeet:
        print(f"Device: transcription=Parakeet (ONNX Runtime, CPU), alignment/diarization={torch_device}")
    elif apple:
        print(f"Device: transcription=Apple Neural Engine (SpeechAnalyzer), diarization={torch_device}")
    elif openvino:
        ov_device, _ = _parse_openvino_model(model_size)
        print(f"Device: transcription=Intel OpenVINO ({ov_device}), alignment/diarization={torch_device}")
    elif device == "mps":
        print(f"Device: Whisper={whisper_device} ({compute_type}), alignment/diarization={torch_device} (Apple MPS)")
    elif device == "xpu":
        print(f"Device: Whisper={whisper_device} ({compute_type}), alignment/diarization={torch_device} (Intel Arc GPU)")
    else:
        print(f"Device: {device} ({compute_type})")
    print(f"Model: {model_size}")
    print(f"Language: {language or 'auto-detect'}")
    print(f"Audio: {audio_path}")
    print()

    # Progress ranges: transcription 0-40%, alignment 40-65%, diarization 65-100%
    def _prog(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    def _transcribe_progress(pct_within):
        """pct_within: 0-100 from the whisperx callback -> mapped to 0.15-0.40"""
        mapped = 0.15 + (pct_within / 100) * 0.25
        _prog(mapped, f"Transcribing... {pct_within:.0f}%")

    def _align_progress(pct_within):
        """pct_within: 0-100 from the whisperx callback -> mapped to 0.50-0.63"""
        mapped = 0.50 + (pct_within / 100) * 0.13
        _prog(mapped, f"Aligning... {pct_within:.0f}%")

    # --- 1. Transcription ---
    t0 = time.time()

    # Use the bundled model path if present
    bundled = _get_bundled_models_dir()

    if apple:
        print("1/3  Transcribing via Apple SpeechAnalyzer (Neural Engine)...")
        _prog(0.0, "SpeechAnalyzer transcribing...")

        def _apple_progress(frac):
            mapped = frac * 0.40
            _prog(mapped, f"Transcribing... {frac * 100:.0f}%")

        result = transcribe_apple(audio_path, language, on_progress=_apple_progress)
        # The diarization step below needs the audio as an array (the Swift
        # helper loads/decodes the file again internally).
        audio = load_audio_universal(audio_path)

        t1 = time.time()
        n_segs = len(result["segments"])
        print(f"     Transcription complete ({t1 - t0:.1f}s)")
        print(f"     {n_segs} segments detected")
        _prog(0.40, f"Transcription done - {n_segs} segments ({t1 - t0:.0f}s)")
    elif openvino:
        ov_device, whisper_size = _parse_openvino_model(model_size)
        print(f"1/3  Transcribing via Intel OpenVINO ({ov_device})...")
        _prog(0.0, "Loading OpenVINO model...")

        def _openvino_progress(frac):
            mapped = frac / 100 * 0.40
            _prog(mapped, f"Transcribing... {frac:.0f}%")

        result = transcribe_openvino(
            audio_path, language, ov_device, whisper_size, on_progress=_openvino_progress)
        audio = load_audio_universal(audio_path)

        t1 = time.time()
        n_segs = len(result["segments"])
        print(f"     Transcription complete ({t1 - t0:.1f}s)")
        print(f"     {n_segs} segments detected")
        _prog(0.40, f"Transcription done - {n_segs} segments ({t1 - t0:.0f}s)")
    elif parakeet:
        print("1/3  Transcribing via Parakeet (ONNX Runtime)...")
        _prog(0.0, "Loading audio...")
        audio = load_audio_universal(audio_path)
        _prog(0.05, "Loading Parakeet model...")

        def _parakeet_progress(frac):
            mapped = 0.05 + frac * 0.35
            _prog(mapped, f"Transcribing... {frac * 100:.0f}%")

        result = transcribe_parakeet(audio, language, model_size, on_progress=_parakeet_progress)

        t1 = time.time()
        n_segs = len(result["segments"])
        print(f"     Transcription complete ({t1 - t0:.1f}s)")
        print(f"     {n_segs} segments detected, language: {result['language']}")
        _prog(0.40, f"Transcription done - {n_segs} segments ({t1 - t0:.0f}s)")
    elif remote:
        from voxscribe.providers import get_provider

        provider_id = model_size[len(REMOTE_MODEL_PREFIX):]
        provider = get_provider(provider_id)
        if provider is None and not (api_key and api_base_url):
            raise ValueError(
                f"Unknown provider '{provider_id}' - configure it in Settings, "
                "or pass api_key/api_base_url explicitly."
            )
        remote_model_name = provider["model"] if provider else (model_size or "whisper-1")
        resolved_api_key = api_key or (provider["api_key"] if provider else None)
        resolved_base_url = api_base_url or (provider["base_url"] if provider else None)
        provider_label = provider["name"] if provider else provider_id

        print(f"1/3  Transcribing via provider ({provider_label} / {remote_model_name})...")
        _prog(0.0, "Loading audio...")
        audio = load_audio_universal(audio_path)

        def _remote_progress(frac):
            mapped = 0.10 + frac * 0.30
            _prog(mapped, f"Sending audio to server... {frac * 100:.0f}%")

        _prog(0.10, f"Sending audio to server ({provider_label})...")
        try:
            result = transcribe_remote(
                audio, language, remote_model_name,
                api_key=resolved_api_key, base_url=resolved_base_url,
                on_progress=_remote_progress,
            )
        except Exception as e:
            fallback_model = default_local_model_size()
            print(f"     [WARNING] Provider request failed ({e}) "
                  f"- falling back to local model ({fallback_model})")
            _prog(0.0, f"Provider request failed - falling back to {fallback_model}...")
            return transcribe(
                audio_path, language=language, model_size=fallback_model, diarize=diarize,
                hf_token=hf_token, min_speakers=min_speakers, max_speakers=max_speakers,
                batch_size=batch_size, device=device, on_progress=on_progress,
                api_key=api_key, api_base_url=api_base_url,
            )

        t1 = time.time()
        n_segs = len(result["segments"])
        print(f"     Transcription complete ({t1 - t0:.1f}s)")
        print(f"     {n_segs} segments detected")
        _prog(0.40, f"Transcription done - {n_segs} segments ({t1 - t0:.0f}s)")
    else:
        print("1/3  Transcribing...")
        _prog(0.0, "Loading Whisper model...")

        whisper_arch = model_size
        whisper_local_only = False
        if bundled:
            whisper_path = os.path.join(bundled, "whisper", model_size)
            if os.path.isdir(whisper_path):
                # faster_whisper's WhisperModel takes a directory as-is (no HF
                # lookup at all) if given a path directly - passing it via
                # download_root instead would be treated as a cache_dir
                # expecting the models--org--name/snapshots/<rev>/... HF cache
                # layout, which download_models.py's flat output_dir=...
                # download doesn't produce.
                whisper_arch = whisper_path
                whisper_local_only = True

        model = whisperx.load_model(
            whisper_arch, whisper_device, compute_type=compute_type, language=language,
            download_root=None, local_files_only=whisper_local_only,
        )

        _prog(0.10, "Loading audio...")
        audio = load_audio_universal(audio_path)

        _prog(0.15, "Transcribing...")
        result = model.transcribe(audio, batch_size=batch_size,
                                  progress_callback=_transcribe_progress)

        t1 = time.time()
        n_segs = len(result['segments'])
        print(f"     Transcription complete ({t1 - t0:.1f}s)")
        print(f"     {n_segs} segments detected")
        _prog(0.40, f"Transcription done - {n_segs} segments ({t1 - t0:.0f}s)")

        # Unload the model
        del model
        gc.collect()
        if whisper_device == "cuda":
            torch.cuda.empty_cache()

    # --- 2. Alignment ---
    # SpeechAnalyzer already provides word-accurate timestamps (see
    # transcribe_apple()) - the wav2vec2 alignment step is redundant for that
    # result and would additionally load another PyTorch model on macOS that
    # isn't needed at all.
    if apple:
        print("2/3  Alignment skipped (SpeechAnalyzer already provides word timestamps)")
        _prog(0.65, "Alignment not needed (Apple SpeechAnalyzer)")
    else:
        print("2/3  Aligning...")
        _prog(0.45, "Loading alignment model...")
        t2 = time.time()

        # Bundled alignment model path
        align_model_dir = None
        align_cache_only = False
        if bundled:
            align_path = os.path.join(bundled, "align")
            if os.path.isdir(align_path):
                align_model_dir = align_path
                align_cache_only = True

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=torch_device,
            model_dir=align_model_dir, model_cache_only=align_cache_only,
        )

        _prog(0.50, "Aligning words...")
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, torch_device,
            return_char_alignments=False,
            progress_callback=_align_progress,
        )

        t3 = time.time()
        print(f"     Alignment complete ({t3 - t2:.1f}s)")
        _prog(0.65, f"Alignment done ({t3 - t2:.0f}s)")

        # Unload the model
        del model_a
        gc.collect()
        if torch_device == "cuda":
            torch.cuda.empty_cache()
        elif torch_device == "xpu":
            torch.xpu.empty_cache()
        elif torch_device == "mps":
            torch.mps.empty_cache()

    # --- 3. Speaker diarization ---
    if diarize:
        if not hf_token:
            print("3/3  Diarization skipped (no HF_TOKEN set)")
            print("     Set HF_TOKEN in .env for speaker recognition")
            _prog(1.0, "Done (diarization skipped - no token)")
        else:
            print("3/3  Running speaker diarization...")
            _prog(0.70, "Loading diarization model...")
            t4 = time.time()

            # Bundled diarization model path
            diarize_cache = None
            if bundled:
                diarize_path = os.path.join(bundled, "diarize")
                if os.path.isdir(diarize_path):
                    diarize_cache = diarize_path

            from whisperx.diarize import DiarizationPipeline

            diarize_model = DiarizationPipeline(
                token=hf_token, device=torch_device, cache_dir=diarize_cache
            )

            def _diarize_progress(pct_within):
                """pct_within: 0-100 from whisperx's DiarizationPipeline
                (which itself maps pyannote's two internal steps -
                segmentation, embeddings - into one 0-100 range) -> mapped to
                0.80-0.92."""
                mapped = 0.80 + (pct_within / 100) * 0.12
                _prog(mapped, f"Running speaker diarization... {pct_within:.0f}%")

            _prog(0.80, "Running speaker diarization...")
            diarize_segments, speaker_embeddings = diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True,
                progress_callback=_diarize_progress,
            )

            _prog(0.92, "Assigning speakers...")
            result = whisperx.assign_word_speakers(
                diarize_segments, result, speaker_embeddings=speaker_embeddings)

            # "Microphone + system" recordings: use the separate tracks (see
            # tracks.py) to split speakers that pyannote merged across the
            # local/remote boundary. Runs on raw labels, before voice-print
            # matching - split-off speakers have no embedding of their own
            # and so are simply left unrecognized.
            split_speakers = {}
            try:
                from voxscribe.tracks import load_tracks, split_speakers_by_track
                tracks = load_tracks(audio_path)
                if tracks is not None:
                    split_speakers = split_speakers_by_track(result, *tracks)
                    if split_speakers:
                        print("     Split by microphone/system track: " + ", ".join(
                            f"{old} -> {old} + {new}" for new, old in split_speakers.items()))
            except Exception as e:
                print(f"     [WARNING] Track-based speaker split failed: {e}")

            # Automatically recognize known speakers (voice prints) and label
            # their segments/words directly with the recognized name.
            # speaker_id_map remembers raw diarization ID -> currently
            # displayed label, so the GUI can find the right embedding when
            # the user confirms/corrects a name via the "remember" checkbox.
            speaker_id_map = {spk: spk for spk in (speaker_embeddings or {})}
            for new_spk in split_speakers:
                speaker_id_map[new_spk] = new_spk
            if speaker_embeddings:
                try:
                    from voxscribe.speaker_profiles import match_speakers
                    suggestions = match_speakers(speaker_embeddings)
                except Exception:
                    suggestions = {}
                for spk_id, name in suggestions.items():
                    speaker_id_map[spk_id] = name
                    for seg in result.get("segments", []):
                        if seg.get("speaker") == spk_id:
                            seg["speaker"] = name
                    for word in result.get("word_segments", []) or []:
                        if word.get("speaker") == spk_id:
                            word["speaker"] = name
            result["speaker_id_map"] = speaker_id_map

            t5 = time.time()
            print(f"     Diarization complete ({t5 - t4:.1f}s)")

            # Count speakers
            speakers = set()
            for seg in result.get("segments", []):
                if "speaker" in seg:
                    speakers.add(seg["speaker"])
            print(f"     {len(speakers)} speakers detected: {', '.join(sorted(speakers))}")
            _prog(0.98, f"Diarization done - {len(speakers)} speakers ({t5 - t4:.0f}s)")

            del diarize_model
            gc.collect()
            if torch_device == "cuda":
                torch.cuda.empty_cache()
            elif torch_device == "xpu":
                torch.xpu.empty_cache()
            elif torch_device == "mps":
                torch.mps.empty_cache()
    else:
        print("3/3  Diarization disabled")

    total = time.time() - t0
    _prog(1.0, f"Done! (total: {total:.0f}s)")
    print(f"\nTotal: {total:.1f}s")

    # For transcribe_multi(): the file's duration, to correctly shift
    # subsequent files in the combined transcript without decoding the audio
    # file a second time.
    result["duration"] = len(audio) / SAMPLE_RATE

    return result


def transcribe_multi(
    audio_paths: list[str],
    language: str | None = "en",
    model_size: str = "large-v2",
    diarize: bool = True,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    batch_size: int = 16,
    device: str | None = None,
    on_progress=None,
    api_key: str | None = None,
    api_base_url: str | None = None,
) -> dict:
    """Transcribes multiple audio/video files in sequence (via `transcribe()`,
    unchanged) and merges the results into one continuous transcript - for the
    multi-file selection in the Transcription tab (qt_app/pages/transcribe_page.py).

    Each file's timestamps are shifted by the cumulative duration of the
    preceding files, so the combined transcript has one continuous timeline
    (file 2 starts where file 1 ended, and so on).

    Speaker labels are NOT blindly unified across files: diarization runs
    per-file independently, so SPEAKER_00 in file 1 isn't necessarily the same
    person as SPEAKER_00 in file 2. Speakers not yet recognized via voice
    print (see speaker_profiles.py) therefore get a file prefix ("File 2:
    SPEAKER_00") so they can be told apart clearly in the rename UI - already
    recognized/named speakers (whose label is already a real name rather than
    the raw SPEAKER_NN label) are left unchanged and merge naturally across
    files.
    """
    if not audio_paths:
        raise ValueError("No files given to transcribe.")

    n_files = len(audio_paths)
    combined_segments = []
    combined_word_segments = []
    combined_speaker_id_map = {}
    combined_speaker_embeddings = {}
    detected_language = None
    cumulative_offset = 0.0

    for idx, path in enumerate(audio_paths):
        file_label = f"File {idx + 1}/{n_files} ({os.path.basename(path)})"

        def _file_progress(pct, msg, idx=idx, file_label=file_label):
            if on_progress:
                overall = (idx + pct) / n_files
                on_progress(overall, f"{file_label}: {msg}")

        # After the first file, lock in the language it detected for the rest
        # (mirroring the "detect once, then lock in" pattern in
        # transcribe_remote()), so detection doesn't flip-flop per file.
        file_language = language if language else detected_language

        result = transcribe(
            audio_path=path, language=file_language, model_size=model_size,
            diarize=diarize, hf_token=hf_token, min_speakers=min_speakers,
            max_speakers=max_speakers, batch_size=batch_size, device=device,
            on_progress=_file_progress, api_key=api_key, api_base_url=api_base_url,
        )

        if detected_language is None:
            detected_language = result.get("language")

        # Rename speakers not yet recognized (label == raw diarization ID) in
        # this file, to avoid collisions with other files; leave already
        # voice-print-recognized names unchanged so the same person is merged
        # across files.
        rename_map = {}
        file_speaker_id_map = result.get("speaker_id_map") or {}
        file_embeddings = result.get("speaker_embeddings") or {}
        for raw_id, current_label in file_speaker_id_map.items():
            if n_files > 1 and raw_id == current_label:
                new_label = f"File {idx + 1}: {current_label}"
                new_key = f"file{idx}:{raw_id}"
            else:
                new_label = current_label
                new_key = current_label
            rename_map[current_label] = new_label
            combined_speaker_id_map[new_key] = new_label
            if raw_id in file_embeddings:
                combined_speaker_embeddings[new_key] = file_embeddings[raw_id]

        for seg in result.get("segments", []):
            if rename_map and seg.get("speaker") in rename_map:
                seg["speaker"] = rename_map[seg["speaker"]]
            seg["start"] = seg.get("start", 0.0) + cumulative_offset
            seg["end"] = seg.get("end", 0.0) + cumulative_offset
            for w in seg.get("words", []) or []:
                if "start" in w:
                    w["start"] += cumulative_offset
                if "end" in w:
                    w["end"] += cumulative_offset
            combined_segments.append(seg)

        for w in result.get("word_segments", []) or []:
            if rename_map and w.get("speaker") in rename_map:
                w["speaker"] = rename_map[w["speaker"]]
            if "start" in w:
                w["start"] += cumulative_offset
            if "end" in w:
                w["end"] += cumulative_offset
            combined_word_segments.append(w)

        cumulative_offset += result.get("duration", 0.0)

    combined = {
        "segments": combined_segments,
        "word_segments": combined_word_segments,
        "language": detected_language or "en",
        "duration": cumulative_offset,
    }
    if diarize:
        combined["speaker_id_map"] = combined_speaker_id_map
        combined["speaker_embeddings"] = combined_speaker_embeddings
    return combined


def format_transcript(result: dict, include_speakers: bool = True) -> str:
    """Formats the transcript as readable text with timestamps."""
    lines = []
    for seg in result.get("segments", []):
        start = _format_time(seg.get("start", 0))
        end = _format_time(seg.get("end", 0))
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")

        if include_speakers and speaker:
            lines.append(f"[{start} - {end}] {speaker}: {text}")
        else:
            lines.append(f"[{start} - {end}] {text}")

    return "\n".join(lines)


def save_transcript(result: dict, output_path: str,
                    formats: list[str] | None = None):
    """Saves the transcript in various formats.

    Args:
        result: WhisperX result dict
        output_path: Base path without extension (e.g. "recordings/meeting")
        formats: List of formats ("txt", "srt", "json"). Default: ["txt"]
    """
    if formats is None:
        formats = ["txt"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    saved = []

    for fmt in formats:
        path = f"{output_path}.{fmt}"
        if fmt == "txt":
            content = format_transcript(result)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        elif fmt == "srt":
            content = _to_srt(result)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        else:
            print(f"  Unknown format: {fmt}")
            continue
        saved.append(path)
        print(f"  Saved: {path}")

    return saved


def _format_time(seconds: float) -> str:
    """Formats seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_time_srt(seconds: float) -> str:
    """Formats seconds in SRT format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _to_srt(result: dict) -> str:
    """Converts a WhisperX result to SRT format."""
    lines = []
    for i, seg in enumerate(result.get("segments", []), 1):
        start = _format_time_srt(seg.get("start", 0))
        end = _format_time_srt(seg.get("end", 0))
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)
