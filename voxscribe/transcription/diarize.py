"""Speaker diarization (pyannote via whisperx)."""

import os

import numpy as np

from voxscribe.paths import get_bundled_models_dir
from voxscribe.transcription.devices import free_torch_memory


def prewarm_diarization() -> None:
    """Imports the whole heavy stack (torch, whisperx, pyannote.audio) ahead
    of time - for the GUI to call in an idle background thread after startup,
    so the first transcription doesn't pay these imports. Import only, no
    model is loaded (no VRAM/RAM for weights).

    voxscribe.transcriber itself no longer imports any of this at module
    load, so this is also what absorbs the cold import (1-3 minutes) that
    importing it used to cost."""
    import torch  # noqa: F401
    import whisperx  # noqa: F401
    import whisperx.diarize  # noqa: F401


def run_diarization(result: dict, audio: np.ndarray, torch_device: str, hf_token: str,
                    min_speakers: int | None, max_speakers: int | None, progress):
    """Runs the diarization model and assigns a speaker to every segment/word.

    `progress(pct, msg)` takes the overall pipeline fraction (this step covers
    0.70-0.92). Returns (result, speaker_embeddings) - embeddings per raw
    Diarization ID, for voice-print matching.
    """
    import whisperx

    progress(0.70, "Loading diarization model...")

    # Bundled diarization model path
    diarize_cache = None
    bundled = get_bundled_models_dir()
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
        progress(mapped, f"Running speaker diarization... {pct_within:.0f}%")

    progress(0.80, "Running speaker diarization...")
    diarize_segments, speaker_embeddings = diarize_model(
        audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        return_embeddings=True,
        progress_callback=_diarize_progress,
    )

    progress(0.92, "Assigning speakers...")
    result = whisperx.assign_word_speakers(
        diarize_segments, result, speaker_embeddings=speaker_embeddings)

    del diarize_model
    free_torch_memory(torch_device)
    return result, speaker_embeddings
