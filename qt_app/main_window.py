"""MainWindow: top tab bar with the main pages."""

import os

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QTabWidget

from qt_app import theme
from qt_app.constants import APP_NAME
from qt_app.controllers import HardwareInfoController, RecorderController, TranscribeController
from qt_app.pages.record_page import RecordPage
from qt_app.pages.transcribe_page import TranscribePage
from qt_app.pages.settings_page import SettingsPage

# icon key (from theme.get_icons(), None = no icon) + label per tab
TAB_ITEMS = [
    ("record", "record", "  Record"),
    ("transcribe", "speech_bubble", "  Transcription"),
    ("settings", None, "⚙  Settings"),
]


class MainWindow(QMainWindow):
    """Holds the shared controllers/settings and the main tabs.

    Record (pick source/device + live level/timer/stop) is a single page -
    matching the previous CustomTkinter GUI. The Info content (models,
    licenses) is appended to the end of the Settings page instead of getting
    its own tab.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(980, 760)
        self.setMinimumSize(820, 620)

        self.settings = {
            "output_dir": "recordings",
            "hf_token": os.getenv("HF_TOKEN", ""),
            "batch_size": 16,
            "compute": "auto",
        }
        self.last_audio_path = None

        self.recorder_controller = RecorderController(self)
        self.transcribe_controller = TranscribeController()
        self.hardware_info_controller = HardwareInfoController(self)

        self._build_ui()

        # Only start AFTER the UI is built, so the pages already exist (and
        # show their apply_hardware_info() placeholders) by the time the
        # result (delayed, see HardwareInfoController) arrives.
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
        """Called by RecordPage after a finished recording, to pre-fill the
        Transcription page with the file directly."""
        self.last_audio_path = audio_path
        self.pages["transcribe"].set_audio_file(audio_path)

    def closeEvent(self, event):  # noqa: N802 - Qt override
        if self.recorder_controller.is_recording:
            self.recorder_controller.stop()
        super().closeEvent(event)
