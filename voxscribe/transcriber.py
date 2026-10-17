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
from voxscribe.transcription.devices import resolve_device
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


def _free_torch_memory(torch_device: str) -> None:
    gc.collect()
    if torch_device == "cuda":
        torch.cuda.empty_cache()
    elif torch_device == "xpu":
        torch.xpu.empty_cache()
    elif torch_device == "mps":
        torch.mps.empty_cache()


def _align_by_language(result: dict, audio: np.ndarray, torch_device: str,
                       align_model_dir: str | None, align_cache_only: bool,
                       progress_callback=None) -> dict:
    """WhisperX forced alignment, one wav2vec2 model per segment language.

    Segments may carry their own "language" (transcribe_remote() in
    auto-detect mode, for meetings that switch languages); all others use
    result["language"]. Each language group is aligned with its own model,
    then everything is merged back in time order. If no model can be loaded
    for a minor language, its segments fall back to the main language's model.
    """
    main = result["language"]
    groups: dict[str, list[dict]] = {}
    for seg in result["segments"]:
        groups.setdefault(seg.get("language") or main, []).append(seg)
    # Main language first, so a failing minor language can fall back to it.
    order = sorted(groups, key=lambda lang: (lang != main, lang))
    if len(order) > 1:
        print("     Languages: " + ", ".join(f"{lang} ({len(groups[lang])} segments)" for lang in order))

    total = sum(len(g) for g in groups.values()) or 1
    done = 0
    segments, word_segments = [], []
    for lang in order:
        group = groups[lang]
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=lang, device=torch_device,
                model_dir=align_model_dir, model_cache_only=align_cache_only,
            )
        except Exception as e:
            if lang == main:
                raise
            print(f"     [WARNING] No alignment model for '{lang}' ({e}) - using '{main}'")
            model_a, metadata = whisperx.load_align_model(
                language_code=main, device=torch_device,
                model_dir=align_model_dir, model_cache_only=align_cache_only,
            )

        def _group_progress(pct, _start=done, _n=len(group)):
            if progress_callback:
                progress_callback((_start + pct / 100 * _n) / total * 100)

        aligned = whisperx.align(
            group, model_a, metadata, audio, torch_device,
            return_char_alignments=False,
            progress_callback=_group_progress,
        )
        for seg in aligned["segments"]:
            seg["language"] = lang
        segments.extend(aligned["segments"])
        word_segments.extend(aligned.get("word_segments", []))
        done += len(group)
        del model_a
        _free_torch_memory(torch_device)

    segments.sort(key=lambda s: s.get("start", 0.0))
    word_segments.sort(key=lambda w: w.get("start", 0.0))
    return {"segments": segments, "word_segments": word_segments, "language": main}


def prewarm_diarization() -> None:
    """Imports the diarization stack (pyannote.audio) ahead of time - for
    the GUI to call in an idle background thread after startup, so the first
    transcription doesn't pay the import at its diarization step. Import
    only, no model is loaded (no VRAM/RAM for weights)."""
    import whisperx.diarize  # noqa: F401


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
        result = _align_by_language(
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
            _prog(0.70, "Loading diarization model...")
            t4 = time.time()

            # Bundled diarization model path
            diarize_cache = None
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
                _prog(mapped, f"Running speaker diarization... {pct_within:.0f}%")

            _prog(0.80, "Running speaker diarization...")
            diarize_segments, speaker_embeddings = diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True,
                progress_callback=_diarize_progress,
            )

            _prog(0.92, "Assigning speakers...")
            result = whisperx.assign_word_speakers(
                diarize_segments, result, speaker_embeddings=speaker_embeddings)

            # "Microphone + system" recordings: use the separate tracks (see
            # tracks.py) to split speakers that pyannote merged across the
            # local/remote boundary. Runs on raw labels, before voice-print
            # matching - split-off speakers have no embedding of their own
            # and so are simply left unrecognized.
            split_speakers = {}
            try:
                from voxscribe.tracks import load_tracks, split_speakers_by_track
                tracks = load_tracks(audio_path)
                if tracks is not None:
                    split_speakers = split_speakers_by_track(result, *tracks)
                    if split_speakers:
                        print("     Split by microphone/system track: " + ", ".join(
                            f"{old} -> {old} + {new}" for new, old in split_speakers.items()))
            except Exception as e:
                print(f"     [WARNING] Track-based speaker split failed: {e}")

            # Automatically recognize known speakers (voice prints) and label
            # their segments/words directly with the recognized name.
            # speaker_id_map remembers raw diarization ID -> currently
            # displayed label, so the GUI can find the right embedding when
            # the user confirms/corrects a name via the "remember" checkbox.
            speaker_id_map = {spk: spk for spk in (speaker_embeddings or {})}
            for new_spk in split_speakers:
                speaker_id_map[new_spk] = new_spk
            if speaker_embeddings:
                try:
                    from voxscribe.speaker_profiles import match_speakers
                    suggestions = match_speakers(speaker_embeddings)
                except Exception:
                    suggestions = {}
                for spk_id, name in suggestions.items():
                    speaker_id_map[spk_id] = name
                    for seg in result.get("segments", []):
                        if seg.get("speaker") == spk_id:
                            seg["speaker"] = name
                    for word in result.get("word_segments", []) or []:
                        if word.get("speaker") == spk_id:
                            word["speaker"] = name
            result["speaker_id_map"] = speaker_id_map

            t5 = time.time()
            print(f"     Diarization complete ({t5 - t4:.1f}s)")

            # Count speakers
            speakers = set()
            for seg in result.get("segments", []):
                if "speaker" in seg:
                    speakers.add(seg["speaker"])
            print(f"     {len(speakers)} speakers detected: {', '.join(sorted(speakers))}")
            _prog(0.98, f"Diarization done - {len(speakers)} speakers ({t5 - t4:.0f}s)")

            del diarize_model
            gc.collect()
            if torch_device == "cuda":
                torch.cuda.empty_cache()
            elif torch_device == "xpu":
                torch.xpu.empty_cache()
            elif torch_device == "mps":
                torch.mps.empty_cache()
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
    combined_segments = []
    combined_word_segments = []
    combined_speaker_id_map = {}
    combined_speaker_embeddings = {}
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

        # Rename speakers not yet recognized (label == raw diarization ID) in
        # this file, to avoid collisions with other files; leave already
        # voice-print-recognized names unchanged so the same person is merged
        # across files.
        rename_map = {}
        file_speaker_id_map = result.get("speaker_id_map") or {}
        file_embeddings = result.get("speaker_embeddings") or {}
        for raw_id, current_label in file_speaker_id_map.items():
            if n_files > 1 and raw_id == current_label:
                new_label = f"File {idx + 1}: {current_label}"
                new_key = f"file{idx}:{raw_id}"
            else:
                new_label = current_label
                new_key = current_label
            rename_map[current_label] = new_label
            combined_speaker_id_map[new_key] = new_label
            if raw_id in file_embeddings:
                combined_speaker_embeddings[new_key] = file_embeddings[raw_id]

        for seg in result.get("segments", []):
            if rename_map and seg.get("speaker") in rename_map:
                seg["speaker"] = rename_map[seg["speaker"]]
            seg["start"] = seg.get("start", 0.0) + cumulative_offset
            seg["end"] = seg.get("end", 0.0) + cumulative_offset
            for w in seg.get("words", []) or []:
                if "start" in w:
                    w["start"] += cumulative_offset
                if "end" in w:
                    w["end"] += cumulative_offset
            combined_segments.append(seg)

        for w in result.get("word_segments", []) or []:
            if rename_map and w.get("speaker") in rename_map:
                w["speaker"] = rename_map[w["speaker"]]
            if "start" in w:
                w["start"] += cumulative_offset
            if "end" in w:
                w["end"] += cumulative_offset
            combined_word_segments.append(w)

        cumulative_offset += result.get("duration", 0.0)

    combined = {
        "segments": combined_segments,
        "word_segments": combined_word_segments,
        "language": detected_language or "en",
        "duration": cumulative_offset,
    }
    if diarize:
        combined["speaker_id_map"] = combined_speaker_id_map
        combined["speaker_embeddings"] = combined_speaker_embeddings
    return combined
