"""MainWindow: top tab bar with the main pages."""

import os

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from qt_app import theme
from qt_app.title_bar import color_title_bar
from qt_app.constants import APP_NAME
from qt_app.controllers import (HardwareInfoController, RecorderController, TranscribeController,
                                UpdateController)
from qt_app.pages.record_page import RecordPage
from qt_app.pages.transcribe_page import TranscribePage
from qt_app.pages.settings_page import SettingsPage
from qt_app.widgets.update_notice import UpdateNotice

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
        self.recorder_controller = RecorderController(self)
        self.transcribe_controller = TranscribeController()
        self.hardware_info_controller = HardwareInfoController(self)
        self.update_controller = UpdateController(self)

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
        # Paint the theme's QTabWidget background beside the (centered) tabs,
        # so the strip has the same color as the title bar above it.
        self.tabs.setAttribute(Qt.WA_StyledBackground, True)

        self.update_notice = UpdateNotice()
        self.update_notice.installRequested.connect(self._install_update)
        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.update_notice)
        column.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        # The Update button is unavailable while recording or transcribing.
        # Tracked from the signals: recordingFinished arrives while the
        # recorder thread may still be winding down.
        self._busy = {"recording": False, "transcribing": False}
        for signal, key, busy in ((self.recorder_controller.recordingStarted, "recording", True),
                                  (self.recorder_controller.recordingFinished, "recording", False),
                                  (self.transcribe_controller.started, "transcribing", True),
                                  (self.transcribe_controller.finished, "transcribing", False)):
            signal.connect(lambda *_, key=key, busy=busy: self._set_busy(key, busy))

        self.pages = {}
        self._add_page("record", RecordPage(self))
        self._add_page("transcribe", TranscribePage(self))
        self._add_page("settings", SettingsPage(self))

        self.tabs.currentChanged.connect(self._on_tab_changed)

    def showEvent(self, event):  # noqa: N802 - Qt override
        super().showEvent(event)
        if not getattr(self, "_title_bar_colored", False):
            self._title_bar_colored = True
            color_title_bar(self)
            self._start_update_check()

    # ----------------------------------------------------------- updates
    def _start_update_check(self):
        """After the window is shown: report the Update the launcher just
        installed (once), then check for a newer Release in the background."""
        from voxscribe import updates
        result = updates.take_update_result()
        if result:
            self.update_notice.report(result)
        self.update_controller.checked.connect(self._on_update_checked)
        self.update_controller.start()

    def _on_update_checked(self, result):
        if result.status in ("available", "up_to_date"):
            from datetime import datetime
            settings = QSettings("VoxScribe", "VoxScribe")
            settings.setValue("updates/last_checked", datetime.now().isoformat(timespec="minutes"))
            settings.setValue("updates/latest", result.latest)
        self.pages["settings"].show_update_check(result)
        if result.status == "available" and not self.update_notice.isVisible():
            self.update_notice.offer(result.latest)

    def _set_busy(self, key, busy):
        self._busy[key] = busy
        self.update_notice.set_busy(any(self._busy.values()))

    def _install_update(self, release):
        """Leaves the marker for the launcher and quits; the launcher
        installs the Update before starting the app again (spec #15)."""
        from voxscribe import updates
        updates.request_update(release)
        print(f"Update {release} requested - quitting for the launcher to install it")
        self.close()

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
        self.pages["transcribe"].set_audio_file(audio_path)

    def closeEvent(self, event):  # noqa: N802 - Qt override
        if self.recorder_controller.is_recording:
            self.recorder_controller.stop(wait=True)
        super().closeEvent(event)
