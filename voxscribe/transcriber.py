"""Transcription facade: WhisperX-based transcription with alignment and speaker diarization.

The code lives in the voxscribe.transcription package (see
docs/refactor-transcriber.md and docs/adr/0001); this module re-exports the
public names so main.py, voxscribe/cli.py and qt_app/ keep importing from
here. Importing it must stay cheap: torch and whisperx are only imported when
a pipeline step needs them (a cold import takes minutes).
"""

# ruff: noqa: F401  (everything below is a re-export)

from voxscribe.paths import get_base_dir as _get_base_dir
from voxscribe.paths import get_bundled_models_dir as _get_bundled_models_dir
from voxscribe.transcription.align import align_by_language as _align_by_language
from voxscribe.transcription.audio import (
    SAMPLE_RATE, AudioSource, _load_audio_via_container, load_audio_universal,
    load_audio_without_ffmpeg, probe_duration,
)
from voxscribe.transcription.backends.apple import detect_apple_language, transcribe_apple
from voxscribe.transcription.backends.openvino import transcribe_openvino
from voxscribe.transcription.backends.parakeet import transcribe_parakeet
from voxscribe.transcription.backends.remote import (
    REMOTE_CHUNK_SECONDS, REMOTE_MIN_CHUNK_SECONDS, REMOTE_REFINE_PIECE_SECONDS,
    REMOTE_TEXT_ONLY_CHUNK_SECONDS, _find_quiet_split, transcribe_remote,
)
from voxscribe.transcription.diarize import prewarm_diarization
from voxscribe.transcription.formats import (
    _format_time, _format_time_srt, _to_srt, format_transcript, save_transcript,
)
from voxscribe.transcription.language import _guess_language_from_text, _normalize_language_code
from voxscribe.transcription.model_ids import (
    APPLE_MODEL_PREFIX, APPLE_SPEECHANALYZER_MODEL, OPENVINO_MODEL_PREFIX, PARAKEET_MODEL,
    PARAKEET_MODEL_PREFIX, REMOTE_MODEL_PREFIX, _parse_openvino_model, default_local_model_size,
    default_model_size, is_apple_model, is_openvino_model, is_parakeet_model, is_remote_model,
    openvino_model_id, remote_model_id,
)
from voxscribe.transcription.pipeline import transcribe, transcribe_multi
