"""Local WhisperX (CTranslate2 / faster-whisper): the default Backend for Model IDs without a prefix."""

import os

from voxscribe.paths import get_bundled_models_dir
from voxscribe.transcription.backend import TranscribeRequest
from voxscribe.transcription.devices import (
    compute_type_for, describe_whisper_device, free_torch_memory, whisper_device_for,
)


class WhisperXBackend:
    name = "WhisperX"
    skips_alignment = False

    def describe(self, model_id: str, device: str) -> str:
        return describe_whisper_device(device)

    def transcribe(self, request: TranscribeRequest) -> dict:
        import whisperx

        whisper_device = whisper_device_for(request.device)
        compute_type = compute_type_for(request.device)

        print("1/3  Transcribing...")
        request.progress(0.0, "Loading Whisper model...")

        # Use the bundled model path if present
        whisper_arch = request.model_id
        whisper_local_only = False
        bundled = get_bundled_models_dir()
        if bundled:
            whisper_path = os.path.join(bundled, "whisper", request.model_id)
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
            whisper_arch, whisper_device, compute_type=compute_type, language=request.language,
            download_root=None, local_files_only=whisper_local_only,
        )

        request.progress(0.10, "Loading audio...")
        audio = request.audio.load()

        def _progress(pct_within):
            """pct_within: 0-100 from the whisperx callback -> mapped to 0.15-0.40"""
            request.progress(0.15 + (pct_within / 100) * 0.25, f"Transcribing... {pct_within:.0f}%")

        request.progress(0.15, "Transcribing...")
        result = model.transcribe(audio, batch_size=request.batch_size, progress_callback=_progress)

        # Unload the model
        del model
        free_torch_memory(whisper_device)
        return result
