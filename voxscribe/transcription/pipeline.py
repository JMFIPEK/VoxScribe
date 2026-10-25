"""The transcription pipeline: Backend -> alignment -> diarization -> speaker labels.

transcribe() only orchestrates; each stage is a named step below. Nothing
here imports torch or whisperx at module load.
"""

import os
import time

from voxscribe.transcription import backends  # noqa: F401  (registers the built-in Backends)
from voxscribe.transcription.align import align_by_language
from voxscribe.transcription.audio import SAMPLE_RATE, AudioSource
from voxscribe.transcription.backend import Backend, FallbackRequested, TranscribeRequest
from voxscribe.transcription.devices import resolve_device
from voxscribe.transcription.diarize import run_diarization
from voxscribe.transcription.model_ids import default_local_model_size
from voxscribe.transcription.registry import default_registry
from voxscribe.transcription.speakers import (
    append_file_result, label_known_speakers, merge_stray_speakers, split_by_track)
from voxscribe.paths import get_bundled_models_dir

# Must be set before the first MPS op: pyannote/wav2vec2 use a few ops that
# aren't (yet) implemented for Apple's MPS backend - without the fallback
# flag those crash with NotImplementedError instead of transparently running
# on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _step_transcribe(backend: Backend, request: TranscribeRequest, t0: float) -> dict:
    """Step 1 (progress 0-40%): the Backend turns audio into segments."""
    result = backend.transcribe(request)
    t1 = time.time()
    n_segs = len(result["segments"])
    print(f"     Transcription complete ({t1 - t0:.1f}s)")
    print(f"     {n_segs} segments detected, language: {result['language']}")
    request.progress(0.40, f"Transcription done - {n_segs} segments ({t1 - t0:.0f}s)")
    return result


def _step_align(backend: Backend, result: dict, audio, torch_device: str, prog) -> dict:
    """Step 2 (progress 40-65%): word-level timestamps, unless the Backend already has them."""
    # SpeechAnalyzer already provides word-accurate timestamps (see
    # transcribe_apple()) - the wav2vec2 alignment step is redundant for that
    # result and would additionally load another PyTorch model on macOS that
    # isn't needed at all.
    if backend.skips_alignment:
        print(f"2/3  Alignment skipped ({backend.name} already provides word timestamps)")
        prog(0.65, f"Alignment not needed ({backend.name})")
        return result

    print("2/3  Aligning...")
    prog(0.45, "Loading alignment model...")
    t2 = time.time()

    # Bundled alignment model path
    align_model_dir = None
    align_cache_only = False
    bundled = get_bundled_models_dir()
    if bundled:
        align_path = os.path.join(bundled, "align")
        if os.path.isdir(align_path):
            align_model_dir = align_path
            align_cache_only = True

    def _align_progress(pct_within):
        """pct_within: 0-100 from the whisperx callback -> mapped to 0.50-0.63"""
        prog(0.50 + (pct_within / 100) * 0.13, f"Aligning... {pct_within:.0f}%")

    prog(0.50, "Aligning words...")
    result = align_by_language(
        result, audio, torch_device, align_model_dir, align_cache_only,
        progress_callback=_align_progress,
    )

    t3 = time.time()
    print(f"     Alignment complete ({t3 - t2:.1f}s)")
    prog(0.65, f"Alignment done ({t3 - t2:.0f}s)")
    return result


def _step_identify_speakers(result: dict, audio_source: AudioSource, torch_device: str,
                            hf_token: str | None, min_speakers: int | None,
                            max_speakers: int | None, prog) -> dict:
    """Step 3 (progress 65-100%): who speaks when - diarization, the
    microphone/system track split, the Stray speaker merge, then voice-print
    recognition."""
    if not hf_token:
        print("3/3  Diarization skipped (no HF_TOKEN set)")
        print("     Set HF_TOKEN in .env for speaker recognition")
        prog(1.0, "Done (diarization skipped - no token)")
        return result

    print("3/3  Running speaker diarization...")
    t4 = time.time()

    result, speaker_embeddings = run_diarization(
        result, audio_source.load(), torch_device, hf_token, min_speakers, max_speakers, prog)
    split_speakers = split_by_track(result, audio_source)
    for stray in merge_stray_speakers(result, speaker_embeddings):
        split_speakers.pop(stray, None)
    label_known_speakers(result, speaker_embeddings, split_speakers)

    t5 = time.time()
    print(f"     Diarization complete ({t5 - t4:.1f}s)")

    speakers = {seg["speaker"] for seg in result.get("segments", []) if "speaker" in seg}
    print(f"     {len(speakers)} speakers detected: {', '.join(sorted(speakers))}")
    prog(0.98, f"Diarization done - {len(speakers)} speakers ({t5 - t4:.0f}s)")
    return result


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

    1. Transcription (by the Backend the Model ID selects)
    2. Forced alignment (word-level timestamps)
    3. Speaker diarization (optional)

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.)
        language: Language of the audio (e.g. "en", "de"). None = automatic
            detection - WhisperX/remote providers detect from the audio itself;
            Apple's SpeechAnalyzer has no such capability, so
            transcribe_apple() runs its own German-vs-English probe instead
            (see detect_apple_language())
        model_size: Model ID: a Whisper model size ("large-v2", "large-v3",
            "medium", "base"), a provider model (prefix "server:", see
            is_remote_model()), an OpenVINO model (prefix "openvino:", see
            is_openvino_model()), "parakeet:tdt-0.6b-v3" (NVIDIA Parakeet via
            onnx-asr, see transcribe_parakeet()), or "apple:speechanalyzer"
            for Apple's SpeechAnalyzer (macOS 26+ only, see
            is_apple_model()/transcribe_apple()). The prefix selects the
            Backend through the registry (voxscribe.transcription.registry).
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
    backend = default_registry.resolve(model_size)

    print(backend.describe(model_size, device))
    print(f"Model: {model_size}")
    print(f"Language: {language or 'auto-detect'}")
    print(f"Audio: {audio_path}")
    print()

    # Progress ranges: transcription 0-40%, alignment 40-65%, diarization 65-100%
    def prog(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    t0 = time.time()
    audio_source = AudioSource(audio_path)
    request = TranscribeRequest(
        audio_path=audio_path, audio=audio_source, language=language, model_id=model_size,
        progress=prog, batch_size=batch_size, device=device,
        api_key=api_key, api_base_url=api_base_url,
    )
    try:
        result = _step_transcribe(backend, request, t0)
    except FallbackRequested as e:
        fallback_model = default_local_model_size()
        print(f"     [WARNING] Provider request failed ({e}) "
              f"- falling back to local model ({fallback_model})")
        prog(0.0, f"Provider request failed - falling back to {fallback_model}...")
        return transcribe(
            audio_path, language=language, model_size=fallback_model, diarize=diarize,
            hf_token=hf_token, min_speakers=min_speakers, max_speakers=max_speakers,
            batch_size=batch_size, device=device, on_progress=on_progress,
            api_key=api_key, api_base_url=api_base_url,
        )
    audio = audio_source.load()

    result = _step_align(backend, result, audio, device, prog)

    if diarize:
        result = _step_identify_speakers(
            result, audio_source, device, hf_token, min_speakers, max_speakers, prog)
    else:
        print("3/3  Diarization disabled")

    total = time.time() - t0
    prog(1.0, f"Done! (total: {total:.0f}s)")
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
    CLI's repeated `-i` (`python main.py transcribe -i a.wav -i b.wav`).

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
    files. See speakers.append_file_result().
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
    return final
