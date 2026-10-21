"""Intel Arc GPU / NPU transcription via OpenVINO (Windows)."""

import os
import re

import numpy as np

from voxscribe.paths import get_bundled_models_dir as _get_bundled_models_dir
from voxscribe.transcription.audio import SAMPLE_RATE, load_audio_universal
from voxscribe.transcription.backend import TranscribeRequest
from voxscribe.transcription.language import _guess_language_from_text
from voxscribe.transcription.model_ids import _parse_openvino_model


_OPENVINO_LANG_TOKEN_RE = re.compile(r"<\|([a-z]{2})\|>")


def detect_chunk_language(model, processor, input_features, text: str) -> str:
    """The language of the first chunk, for auto-detect.

    generate() returns the text tokens without Whisper's <|xx|> prefix, so
    the language can't be read off its output (that silently gave "en" and
    made Whisper *translate* every later chunk of German audio, #38).
    Instead: Whisper's own language detection on the chunk, else a guess
    from the chunk's text, else English - logged either way."""
    try:
        ids = model.detect_language(input_features)
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        token = processor.tokenizer.convert_ids_to_tokens(ids[:1])[0]
        match = _OPENVINO_LANG_TOKEN_RE.fullmatch(token)
        if match:
            return match.group(1)
    except Exception as e:  # noqa: BLE001 - fall back to the text
        print(f"     Whisper language detection unavailable ({e})")
    guessed = _guess_language_from_text(text)
    if guessed:
        print(f"     Language guessed from the text: {guessed}")
        return guessed
    print("     [WARNING] Could not detect the language - assuming English")
    return "en"


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
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if detected_language is None:
            # Detected from the first 30s and forced for every subsequent
            # chunk, mirroring the CTranslate2 path.
            detected_language = detect_chunk_language(model, processor, inputs["input_features"], text)
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
