"""Candidate (#57/#58): NVIDIA Canary-1B-v2 via onnx-asr on ONNX Runtime (CPU).

Reachable by Model ID (`canary:1b-v2`) from the CLI and the benchmark, not
offered in the GUI. Same approach as Parakeet (onnx_asr_common): Silero VAD
cuts the audio into speech segments of at most 25s, which become the
WhisperX-style segments the regular alignment/diarization steps process.
"""

import numpy as np

from voxscribe.transcription.audio import SAMPLE_RATE
from voxscribe.transcription.backend import TranscribeRequest
from voxscribe.transcription.backends.onnx_asr_common import (
    CANARY_VARIANTS, bundled_model_dir, load_vad_asr, recognize_segments,
)
from voxscribe.transcription.model_ids import CANARY_MODEL_PREFIX


# Canary has no language detection: onnx-asr's prompt names the speech
# language and the output language, and defaults both to English - German
# speech then comes out *translated* into English. For auto-detect, the
# first CANARY_PROBE_SECONDS are transcribed once per language here, and the
# one the model is most confident in (mean token log-probability) is used
# for the whole recording. Only the study's languages are probed.
CANARY_PROBE_LANGUAGES = ("de", "en")
CANARY_PROBE_SECONDS = 30.0


def _mean_logprob(results) -> float | None:
    logprobs = [lp for r in results for lp in (getattr(r, "logprobs", None) or [])]
    return float(np.mean(logprobs)) if logprobs else None


def detect_canary_language(asr, audio: np.ndarray) -> str:
    """The CANARY_PROBE_LANGUAGES entry Canary is most confident in on the
    first CANARY_PROBE_SECONDS of `audio`; English if none produced text."""
    probe = audio[:int(CANARY_PROBE_SECONDS * SAMPLE_RATE)]
    timestamped = asr.with_timestamps()
    scores = {}
    for lang in CANARY_PROBE_LANGUAGES:
        results = list(timestamped.recognize(probe, sample_rate=SAMPLE_RATE, language=lang))
        score = _mean_logprob(results)
        if score is not None:
            scores[lang] = score
    if not scores:
        print("     [WARNING] Canary: no speech in the first "
              f"{CANARY_PROBE_SECONDS:.0f}s to detect the language - assuming 'en'")
        return "en"
    best = max(scores, key=scores.get)
    print("     Canary language probe: " + ", ".join(
        f"{lang} {score:.3f}" for lang, score in scores.items()) + f" -> '{best}'")
    return best


def transcribe_canary(audio: np.ndarray, language: str | None, model_size: str,
                      on_progress=None, on_segments=None) -> dict:
    """Transcribes via NVIDIA Canary-1B-v2 (25 European languages) through
    onnx-asr. Speech and output language are both set to `language` (so it
    transcribes, never translates); auto-detect picks one first, see
    detect_canary_language()."""
    variant = model_size[len(CANARY_MODEL_PREFIX):]
    name = CANARY_VARIANTS.get(variant)
    if name is None:
        raise ValueError(f"Unknown Canary model: {model_size}")

    asr = load_vad_asr(name, bundled_model_dir("canary", variant))
    if not language:
        language = detect_canary_language(asr, audio)
    try:
        segments = recognize_segments(asr, audio, on_progress, on_segments,
                                      language=language, target_language=language, pnc=True)
    except KeyError as e:  # onnx-asr looks the language token up in Canary's vocabulary
        raise ValueError(f"Canary does not support the language '{language}'") from e
    del asr
    return {"segments": segments, "language": language}


class CanaryBackend:
    name = "Canary"
    skips_alignment = False

    def describe(self, model_id: str, device: str) -> str:
        return f"Device: transcription=Canary (ONNX Runtime, CPU), alignment/diarization={device}"

    def transcribe(self, request: TranscribeRequest) -> dict:
        print("1/3  Transcribing via Canary (ONNX Runtime)...")
        request.progress(0.0, "Loading audio...")
        audio = request.audio.load()
        request.progress(0.05, "Loading Canary model... (first use downloads it)")

        def _progress(frac):
            request.progress(0.05 + frac * 0.35, f"Transcribing... {frac * 100:.0f}%")

        return transcribe_canary(audio, request.language, request.model_id,
                                 on_progress=_progress, on_segments=request.on_segments)
