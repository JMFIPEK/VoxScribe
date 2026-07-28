"""Konstanten, die von mehreren Seiten der PySide6-GUI gebraucht werden."""

APP_NAME = "VoxScribe"
APP_SUBTITLE = "for ORION"
APP_VERSION = "1.0.0"

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

# Muss mit transcriber.APPLE_SPEECHANALYZER_MODEL uebereinstimmen. Hier als
# eigene Konstante dupliziert statt importiert, damit `qt_app`-Seiten beim
# Aufbau kein `import transcriber` brauchen (laedt whisperx/torch, kalt
# 1-3 Minuten) - siehe HardwareInfoController/TranscribeController fuer das
# gleiche Muster (transcriber wird dort bewusst erst im Background-Thread
# importiert).
APPLE_SPEECHANALYZER_MODEL = "apple:speechanalyzer"

# Muss mit transcriber.DEFAULT_API_BASE_URL uebereinstimmen - selber Grund wie
# oben bei APPLE_SPEECHANALYZER_MODEL (kein `import transcriber` fuer eine
# reine String-Konstante).
DEFAULT_API_BASE_URL_FALLBACK = "https://ki-toolbox.scc.kit.edu/api/v1"

# Chat-Modelle fuer die Live-Zusammenfassung (qt_app/pages/live_meeting_page.py) -
# ausschliesslich KIT-ToolBox-Chatmodelle, kein lokales Modell zur Auswahl,
# da dieses Feature bewusst immer online/serverseitig laeuft. Muss mit
# transcriber.DEFAULT_SUMMARY_MODEL uebereinstimmen (selber Grund wie oben).
SUMMARY_MODEL_DISPLAY_NAMES = {
    "kit.mistral-small-4-119b-a8b": "Mistral Small (schnell, empfohlen)",
    "kit.gpt-oss-120b": "GPT-OSS 120B",
    "kit.qwen3.5-397b-A17b": "Qwen 3.5 (397B, MoE)",
    "kit.minimax-m2.7-229b": "MiniMax M2.7",
    "kit.gemma4-31b-it": "Gemma 4 (31B, sehr schnell)",
}
SUMMARY_MODEL_IDS_BY_DISPLAY = {v: k for k, v in SUMMARY_MODEL_DISPLAY_NAMES.items()}
SUMMARY_MODELS = list(SUMMARY_MODEL_DISPLAY_NAMES.values())
DEFAULT_SUMMARY_MODEL_DISPLAY = SUMMARY_MODEL_DISPLAY_NAMES["kit.mistral-small-4-119b-a8b"]


def format_time_short(seconds: float) -> str:
    """Formatiert Sekunden als MM:SS (bzw. HH:MM:SS bei >= 1h)."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
