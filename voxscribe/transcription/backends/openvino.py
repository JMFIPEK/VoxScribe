"""Intel Arc GPU / NPU transcription via OpenVINO (Windows)."""

import os
import re

import numpy as np

from voxscribe.paths import get_bundled_models_dir as _get_bundled_models_dir
from voxscribe.transcription.audio import SAMPLE_RATE, load_audio_universal
from voxscribe.transcription.backend import TranscribeRequest
from voxscribe.transcription.model_ids import _parse_openvino_model


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
    audio: np.ndarray | None = None,
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

    `audio` is the already-decoded file, if the caller has it (saves decoding twice).
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

    if audio is None:
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


class OpenVinoBackend:
    name = "Intel OpenVINO"
    skips_alignment = False

    def describe(self, model_id: str, device: str) -> str:
        ov_device, _ = _parse_openvino_model(model_id)
        return f"Device: transcription=Intel OpenVINO ({ov_device}), alignment/diarization={device}"

    def transcribe(self, request: TranscribeRequest) -> dict:
        ov_device, whisper_size = _parse_openvino_model(request.model_id)
        print(f"1/3  Transcribing via Intel OpenVINO ({ov_device})...")
        request.progress(0.0, "Loading OpenVINO model...")

        def _progress(pct):
            request.progress(pct / 100 * 0.40, f"Transcribing... {pct:.0f}%")

        return transcribe_openvino(
            request.audio_path, request.language, ov_device, whisper_size,
            on_progress=_progress, audio=request.audio.load())
