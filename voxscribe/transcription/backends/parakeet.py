"""NVIDIA Parakeet TDT 0.6B v3 via onnx-asr on ONNX Runtime (CPU)."""

import os

import numpy as np

from voxscribe.paths import get_bundled_models_dir as _get_bundled_models_dir
from voxscribe.transcription.audio import SAMPLE_RATE
from voxscribe.transcription.backend import TranscribeRequest
from voxscribe.transcription.language import _guess_language_from_text
from voxscribe.transcription.model_ids import PARAKEET_MODEL_PREFIX


_PARAKEET_ONNX_ASR_NAMES = {"tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3"}


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


class ParakeetBackend:
    name = "Parakeet"
    skips_alignment = False

    def describe(self, model_id: str, device: str) -> str:
        return f"Device: transcription=Parakeet (ONNX Runtime, CPU), alignment/diarization={device}"

    def transcribe(self, request: TranscribeRequest) -> dict:
        print("1/3  Transcribing via Parakeet (ONNX Runtime)...")
        request.progress(0.0, "Loading audio...")
        audio = request.audio.load()
        request.progress(0.05, "Loading Parakeet model...")

        def _progress(frac):
            request.progress(0.05 + frac * 0.35, f"Transcribing... {frac * 100:.0f}%")

        return transcribe_parakeet(audio, request.language, request.model_id, on_progress=_progress)
