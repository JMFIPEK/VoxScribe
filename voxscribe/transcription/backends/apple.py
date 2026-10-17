"""Apple SpeechAnalyzer (macOS 26+, Neural Engine) via a Swift helper subprocess."""

import json
import os
import subprocess
import tempfile

import soundfile as sf

from voxscribe.paths import get_base_dir as _get_base_dir
from voxscribe.transcription.backend import TranscribeRequest


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


class AppleBackend:
    name = "Apple SpeechAnalyzer"
    # SpeechAnalyzer already provides word-accurate timestamps (see transcribe_apple()).
    skips_alignment = True

    def describe(self, model_id: str, device: str) -> str:
        return f"Device: transcription=Apple Neural Engine (SpeechAnalyzer), diarization={device}"

    def transcribe(self, request: TranscribeRequest) -> dict:
        print("1/3  Transcribing via Apple SpeechAnalyzer (Neural Engine)...")
        request.progress(0.0, "SpeechAnalyzer transcribing...")

        def _progress(frac):
            request.progress(frac * 0.40, f"Transcribing... {frac * 100:.0f}%")

        return transcribe_apple(request.audio_path, request.language, on_progress=_progress)
