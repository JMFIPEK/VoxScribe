"""Constants shared across multiple pages of the PySide6 GUI."""

APP_NAME = "VoxScribe"
APP_VERSION = "1.0.0"

LANGUAGES = {
    "Auto-detect": None, "Deutsch": "de", "English": "en",
    "Français": "fr", "Español": "es", "Italiano": "it",
}
MODEL_DISPLAY_NAMES = {
    "large-v3": "large-v3",
    "large-v2": "large-v2",
    "medium": "medium",
    "base": "base",
    "openvino:GPU:medium": "medium (Intel Arc GPU)",
    "parakeet:tdt-0.6b-v3": "Parakeet v3 (fast, CPU)",
    "canary:1b-v2": "Canary 1B v2 (CPU)",
    "apple:speechanalyzer": "Apple SpeechAnalyzer (fast, less accurate)",
}
FORMATS = ["txt", "srt", "json"]

# Must match transcriber.APPLE_SPEECHANALYZER_MODEL. Duplicated here as its
# own constant instead of importing it, so that `qt_app` pages don't need
# `import transcriber` just to build (that pulls in whisperx/torch, a cold
# 1-3 minute import) - see HardwareInfoController/TranscribeController for the
# same pattern (transcriber is deliberately only imported in a background
# thread there).
APPLE_SPEECHANALYZER_MODEL = "apple:speechanalyzer"

# Prefix used for provider-backed remote models, e.g. "server:my-provider-id"
# where the suffix is a providers.py provider id. Must match
# transcriber.REMOTE_MODEL_PREFIX (same import-cost reason as above).
REMOTE_MODEL_PREFIX = "server:"

# Prefix for Intel Arc GPU/NPU models via OpenVINO, e.g. "openvino:GPU:medium".
# Must match transcriber.OPENVINO_MODEL_PREFIX (same import-cost reason as above).
OPENVINO_MODEL_PREFIX = "openvino:"

# Prefix for NVIDIA Parakeet models via onnx-asr, e.g. "parakeet:tdt-0.6b-v3".
# Must match transcriber.PARAKEET_MODEL_PREFIX (same import-cost reason as above).
PARAKEET_MODEL_PREFIX = "parakeet:"

# Must match transcriber.PARAKEET_MODEL: the local default on macOS (#60).
PARAKEET_MODEL = "parakeet:tdt-0.6b-v3"

# The local models the Transcription tab's dropdown offers per platform
# (#59) - the one place to trim them (study #58). Providers come on top.
# macOS runs no WhisperX/OpenVINO, so it only gets what runs on a Mac.
_LOCAL_MODELS_BY_PLATFORM = {
    "darwin": [PARAKEET_MODEL, APPLE_SPEECHANALYZER_MODEL],
}
_LOCAL_MODELS_ELSEWHERE = [
    "large-v3", "large-v2", "medium", "base", "openvino:GPU:medium", PARAKEET_MODEL,
]


def model_display_name(model_id: str) -> str:
    """The dropdown's name for a local Model ID (the ID itself if unknown)."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def local_models(platform: str, local_default: str | None = None) -> list[str]:
    """Local Model IDs the dropdown lists on `platform` (a sys.platform
    value), in order. `local_default` (default_local_model_size(), where
    that is cheap to ask) comes first and is added even if the static list
    lacks it, so a new local default never goes missing from the dropdown."""
    ids = list(_LOCAL_MODELS_BY_PLATFORM.get(platform, _LOCAL_MODELS_ELSEWHERE))
    if local_default:
        if local_default in ids:
            ids.remove(local_default)
        ids.insert(0, local_default)
    return ids


def format_time_short(seconds: float) -> str:
    """Formats seconds as MM:SS (or HH:MM:SS for >= 1h)."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
