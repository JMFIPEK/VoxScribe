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
}
MODEL_IDS_BY_DISPLAY = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
MODELS = list(MODEL_DISPLAY_NAMES.values())
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


def format_time_short(seconds: float) -> str:
    """Formats seconds as MM:SS (or HH:MM:SS for >= 1h)."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
