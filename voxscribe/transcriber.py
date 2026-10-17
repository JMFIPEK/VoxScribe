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

from voxscribe.paths import get_base_dir, get_bundled_models_dir
from voxscribe.transcription.audio import (  # noqa: F401  (re-exported)
    AudioSource, SAMPLE_RATE, _load_audio_via_container, load_audio_universal, load_audio_without_ffmpeg,
)
from voxscribe.transcription.language import (  # noqa: F401  (re-exported)
    _guess_language_from_text, _normalize_language_code,
)
from voxscribe.transcription.model_ids import (  # noqa: F401  (re-exported)
    APPLE_MODEL_PREFIX, APPLE_SPEECHANALYZER_MODEL, OPENVINO_MODEL_PREFIX, PARAKEET_MODEL,
    PARAKEET_MODEL_PREFIX, REMOTE_MODEL_PREFIX, _parse_openvino_model, default_local_model_size,
    default_model_size, is_apple_model, is_openvino_model, is_parakeet_model, is_remote_model,
    openvino_model_id, remote_model_id,
)
from voxscribe.transcription.backend import FallbackRequested, TranscribeRequest
from voxscribe.transcription import backends  # noqa: F401  (registers the built-in Backends)
from voxscribe.transcription.backends.apple import detect_apple_language, transcribe_apple  # noqa: F401  (re-exported)
from voxscribe.transcription.backends.remote import (  # noqa: F401  (re-exported)
    REMOTE_CHUNK_SECONDS, REMOTE_MIN_CHUNK_SECONDS, REMOTE_REFINE_PIECE_SECONDS,
    REMOTE_TEXT_ONLY_CHUNK_SECONDS, _find_quiet_split, transcribe_remote,
)
from voxscribe.transcription.align import align_by_language
from voxscribe.transcription.devices import resolve_device
from voxscribe.transcription.diarize import prewarm_diarization, run_diarization  # noqa: F401
from voxscribe.transcription.speakers import append_file_result, label_known_speakers, split_by_track
from voxscribe.transcription.backends.openvino import transcribe_openvino  # noqa: F401  (re-exported)
from voxscribe.transcription.backends.parakeet import transcribe_parakeet  # noqa: F401  (re-exported)
from voxscribe.transcription.registry import default_registry
from voxscribe.transcription.formats import (  # noqa: F401  (re-exported)
    _format_time, _format_time_srt, _to_srt, format_transcript, save_transcript,
)

# whisperx.diarize (pyannote.audio + lightning) is imported lazily inside
# transcribe()'s diarization step and via prewarm_diarization(): it's ~80%
# of this module's import time (~11s warm, far more on a cold start), and
# the GUI imports this module long before any diarization happens.

# Must be set before the first MPS op: pyannote/wav2vec2 use a few ops that
# aren't (yet) implemented for Apple's MPS backend - without the fallback
# flag those crash with NotImplementedError instead of transparently running
# on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


_get_base_dir = get_base_dir
_get_bundled_models_dir = get_bundled_models_dir


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
    device = resolve_device(device)
    torch_device = device
    backend = default_registry.resolve(model_size)

    print(backend.describe(model_size, device))
    print(f"Model: {model_size}")
    print(f"Language: {language or 'auto-detect'}")
    print(f"Audio: {audio_path}")
    print()

    # Progress ranges: transcription 0-40%, alignment 40-65%, diarization 65-100%
    def _prog(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    def _align_progress(pct_within):
        """pct_within: 0-100 from the whisperx callback -> mapped to 0.50-0.63"""
        mapped = 0.50 + (pct_within / 100) * 0.13
        _prog(mapped, f"Aligning... {pct_within:.0f}%")

    # --- 1. Transcription ---
    t0 = time.time()

    # Use the bundled model path if present
    bundled = _get_bundled_models_dir()

    audio_source = AudioSource(audio_path)
    request = TranscribeRequest(
        audio_path=audio_path, audio=audio_source, language=language, model_id=model_size,
        progress=_prog, batch_size=batch_size, device=torch_device,
        api_key=api_key, api_base_url=api_base_url,
    )
    try:
        result = backend.transcribe(request)
    except FallbackRequested as e:
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
    audio = audio_source.load()

    t1 = time.time()
    n_segs = len(result["segments"])
    print(f"     Transcription complete ({t1 - t0:.1f}s)")
    print(f"     {n_segs} segments detected, language: {result['language']}")
    _prog(0.40, f"Transcription done - {n_segs} segments ({t1 - t0:.0f}s)")

    # --- 2. Alignment ---
    # SpeechAnalyzer already provides word-accurate timestamps (see
    # transcribe_apple()) - the wav2vec2 alignment step is redundant for that
    # result and would additionally load another PyTorch model on macOS that
    # isn't needed at all.
    if backend.skips_alignment:
        print(f"2/3  Alignment skipped ({backend.name} already provides word timestamps)")
        _prog(0.65, f"Alignment not needed ({backend.name})")
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

        _prog(0.50, "Aligning words...")
        result = align_by_language(
            result, audio, torch_device, align_model_dir, align_cache_only,
            progress_callback=_align_progress,
        )

        t3 = time.time()
        print(f"     Alignment complete ({t3 - t2:.1f}s)")
        _prog(0.65, f"Alignment done ({t3 - t2:.0f}s)")

    # --- 3. Speaker diarization ---
    if diarize:
        if not hf_token:
            print("3/3  Diarization skipped (no HF_TOKEN set)")
            print("     Set HF_TOKEN in .env for speaker recognition")
            _prog(1.0, "Done (diarization skipped - no token)")
        else:
            print("3/3  Running speaker diarization...")
            t4 = time.time()

            result, speaker_embeddings = run_diarization(
                result, audio, torch_device, hf_token, min_speakers, max_speakers, _prog)
            split_speakers = split_by_track(result, audio_path)
            label_known_speakers(result, speaker_embeddings, split_speakers)

            t5 = time.time()
            print(f"     Diarization complete ({t5 - t4:.1f}s)")

            # Count speakers
            speakers = set()
            for seg in result.get("segments", []):
                if "speaker" in seg:
                    speakers.add(seg["speaker"])
            print(f"     {len(speakers)} speakers detected: {', '.join(sorted(speakers))}")
            _prog(0.98, f"Diarization done - {len(speakers)} speakers ({t5 - t4:.0f}s)")
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
    combined = {"segments": [], "word_segments": [], "speaker_id_map": {}, "speaker_embeddings": {}}
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

        append_file_result(combined, result, idx, n_files, cumulative_offset)

        cumulative_offset += result.get("duration", 0.0)

    final = {
        "segments": combined["segments"],
        "word_segments": combined["word_segments"],
        "language": detected_language or "en",
        "duration": cumulative_offset,
    }
    if diarize:
        final["speaker_id_map"] = combined["speaker_id_map"]
        final["speaker_embeddings"] = combined["speaker_embeddings"]
    combined = final
    return combined
