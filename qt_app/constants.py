"""Konstanten, die von mehreren Seiten der PySide6-GUI gebraucht werden."""

APP_NAME = "VoxScribe"
APP_SUBTITLE = "for ORION"
APP_VERSION = "0.2.0"

LANGUAGES = {
    "Automatisch erkennen": None, "Deutsch": "de", "English": "en",
    "Français": "fr", "Español": "es", "Italiano": "it",
}
MODEL_DISPLAY_NAMES = {
    "large-v3": "large-v3",
    "large-v2": "large-v2",
    "medium": "medium",
    "base": "base",
    "server:kit.whisper-large-v3": "KIT ToolBox (Server)",
}
MODEL_IDS_BY_DISPLAY = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
MODELS = list(MODEL_DISPLAY_NAMES.values())
FORMATS = ["txt", "srt", "json"]


def format_time_short(seconds: float) -> str:
    """Formatiert Sekunden als MM:SS (bzw. HH:MM:SS bei >= 1h)."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
