"""MainWindow: Tab-Leiste (oben) mit den drei Hauptseiten."""

import os

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QTabWidget

from qt_app import theme
from qt_app.constants import APP_NAME, APP_SUBTITLE
from qt_app.controllers import HardwareInfoController, RecorderController, TranscribeController
from qt_app.pages.record_page import RecordPage
from qt_app.pages.transcribe_page import TranscribePage
from qt_app.pages.settings_page import SettingsPage

# icon-Key (aus theme.get_icons(), None = kein Icon) + Label je Tab
TAB_ITEMS = [
    ("record", "record", "  Aufnahme"),
    ("transcribe", "speech_bubble", "  Transkription"),
    ("settings", None, "⚙  Einstellungen"),
]


class MainWindow(QMainWindow):
    """Haelt die geteilten Controller/Settings und die drei Haupt-Tabs.

    Aufnahme (Quelle/Geraet waehlen + Live-Pegel/Timer/Stop) ist eine
    einzelne Seite - wie in der alten CTk-GUI. Die Info-Inhalte (Modelle,
    Lizenzen) sind an das Ende der Einstellungen-Seite angehaengt statt
    einen eigenen Tab zu bekommen.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — {APP_SUBTITLE}")
        self.resize(980, 760)
        self.setMinimumSize(820, 620)

        self.settings = {
            "output_dir": "recordings",
            "hf_token": os.getenv("HF_TOKEN", ""),
            "api_key": os.getenv("KIT_TOOLBOX_API_KEY", ""),
            "api_base_url": None,  # per Settings-Seite mit DEFAULT_API_BASE_URL befuellt
            "batch_size": 16,
            "compute": "auto",
        }
        self.last_audio_path = None

        self.recorder_controller = RecorderController(self)
        self.transcribe_controller = TranscribeController()
        self.hardware_info_controller = HardwareInfoController(self)

        self._build_ui()

        # Erst NACH dem UI-Aufbau starten, damit die Seiten schon existieren
        # (und ihre apply_hardware_info()-Platzhalter zeigen), wenn das
        # Ergebnis (verzoegert, siehe HardwareInfoController) eintrifft.
        self.hardware_info_controller.infoReady.connect(self.pages["settings"].apply_hardware_info)
        self.hardware_info_controller.infoReady.connect(self.pages["transcribe"].apply_hardware_info)
        self.hardware_info_controller.start()

    # ------------------------------------------------------------ layout
    def _build_ui(self):
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setIconSize(QSize(18, 18))
        self.setCentralWidget(self.tabs)

        self.pages = {}
        self._add_page("record", RecordPage(self))
        self._add_page("transcribe", TranscribePage(self))
        self._add_page("settings", SettingsPage(self))

        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _add_page(self, key, widget):
        icon_key, label = next((i, lbl) for k, i, lbl in TAB_ITEMS if k == key)
        self.pages[key] = widget
        icon = QIcon(theme.get_icons()[icon_key]) if icon_key else QIcon()
        self.tabs.addTab(widget, icon, label)

    def _on_tab_changed(self, index):
        widget = self.tabs.widget(index)
        on_shown = getattr(widget, "on_page_shown", None)
        if callable(on_shown):
            on_shown()

    # --------------------------------------------------------------- nav
    def navigate_to(self, key: str):
        if key not in self.pages:
            return
        self.tabs.setCurrentWidget(self.pages[key])

    # ----------------------------------------------------- cross-page glue
    def send_to_transcription(self, audio_path: str):
        """Wird von RecordPage nach einer fertigen Aufnahme aufgerufen, um
        die Datei direkt in der Transkriptions-Seite vorzubelegen."""
        self.last_audio_path = audio_path
        self.pages["transcribe"].set_audio_file(audio_path)

    def closeEvent(self, event):  # noqa: N802 - Qt-Override
        if self.recorder_controller.is_recording:
            self.recorder_controller.stop()
        super().closeEvent(event)
