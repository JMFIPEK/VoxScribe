"""Shared by the onnx-asr Backends (Parakeet, Canary): model lookup and the
Silero-VAD loop that turns a recording into WhisperX-style segments.

Imports onnx-asr only inside the functions, so the cheap callers (the
benchmark's model_presence()) don't pay for it.
"""

import os

import numpy as np

from voxscribe.paths import get_bundled_models_dir as _get_bundled_models_dir
from voxscribe.transcription.audio import SAMPLE_RATE


# Model ID variant -> onnx-asr model name.
PARAKEET_VARIANTS = {"tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3"}
CANARY_VARIANTS = {"1b-v2": "nemo-canary-1b-v2"}


# onnx-asr name -> the HuggingFace repo it downloads from (as in onnx-asr
# 0.12's onnx_asr/resolver.py `model_repos`; copied, since importing onnx-asr
# loads ONNX Runtime).
ONNX_ASR_REPOS = {
    "nemo-parakeet-tdt-0.6b-v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "nemo-canary-1b-v2": "istupakov/canary-1b-v2-onnx",
}


def bundled_model_dir(family: str, variant: str) -> str | None:
    """bundled_models/<family>/<variant> if it exists (download_models.py),
    else None (use the HuggingFace cache)."""
    bundled = _get_bundled_models_dir()
    if not bundled:
        return None
    path = os.path.join(bundled, family, variant)
    return path if os.path.isdir(path) else None


def bundled_vad_dir() -> str | None:
    """Silero VAD as bundled with Parakeet, shared by every onnx-asr Backend."""
    return bundled_model_dir("parakeet", "silero-vad")


def load_vad_asr(name: str, model_dir: str | None):
    """The model behind Silero VAD: speech segments of at most 25s.

    min_silence_duration_ms: merge speech separated by short pauses into
    longer segments - these models are noticeably more accurate with a full
    sentence of context than with the VAD's default ~100ms fragments."""
    import onnx_asr

    model = onnx_asr.load_model(name, model_dir, quantization="int8")
    vad = onnx_asr.load_vad("silero", bundled_vad_dir())
    return model.with_vad(vad, batch_size=4, max_speech_duration_s=25,
                          min_silence_duration_ms=800)


def recognize_segments(asr, audio: np.ndarray, on_progress=None, on_segments=None,
                       **recognize_kwargs) -> list[dict]:
    """Runs `asr` (from load_vad_asr()) over `audio`; returns the non-empty
    segments as {"start", "end", "text"}. `on_segments([segment])` gets each
    one as it is recognized. `recognize_kwargs` go to onnx-asr's recognize()
    (e.g. Canary's language)."""
    duration = len(audio) / SAMPLE_RATE
    segments = []
    for seg in asr.recognize(audio, sample_rate=SAMPLE_RATE, **recognize_kwargs):
        text = (seg.text or "").strip()
        if text:
            segments.append({"start": float(seg.start), "end": float(seg.end), "text": text})
            if on_segments:
                on_segments([dict(segments[-1])])
        if on_progress and duration > 0:
            on_progress(min(seg.end / duration, 1.0))
    return segments
