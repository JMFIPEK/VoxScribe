"""Quiet banner above the tabs: offers an Update, or reports how the last
one went (spec #15). Never blocks; the user can dismiss it."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from qt_app import theme


class UpdateNotice(QFrame):
    installRequested = Signal(str)  # the Release to install

    def __init__(self, parent=None):
        super().__init__(parent)
        self._release = None
        self._busy = False
        self.setObjectName("updateNotice")
        self.setStyleSheet(f"""
            QFrame#updateNotice {{ background-color: {theme.SURFACE}; border-bottom: 1px solid {theme.BORDER}; }}
            QFrame#updateNotice QLabel {{ color: {theme.TEXT}; }}
        """)
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 6, 10, 6)
        self.label = QLabel()
        row.addWidget(self.label, 1)
        self.install_button = QPushButton("Restart and install")
        self.install_button.clicked.connect(lambda: self.installRequested.emit(self._release))
        row.addWidget(self.install_button)
        dismiss = QPushButton("Later")
        dismiss.clicked.connect(self.hide)
        row.addWidget(dismiss)
        self.hide()

    def offer(self, release: str):
        self._release = release
        self.label.setText(f"Update {release} available – restart to install")
        self.install_button.show()
        self._refresh_button()
        self.show()

    def report(self, result: dict):
        """Shows how the Update the launcher installed went (once)."""
        self._release = None
        if result.get("ok"):
            self.label.setText(f"Updated to {result.get('target')}.")
        else:
            self.label.setText(f"Update to {result.get('target')} failed, still on "
                               f"{result.get('installed')}. Details are in the log.")
        self.install_button.hide()
        self.show()

    def set_busy(self, busy: bool):
        """While recording or transcribing the Update can't be started."""
        self._busy = busy
        self._refresh_button()

    def _refresh_button(self):
        self.install_button.setEnabled(not self._busy and self._release is not None)
        self.install_button.setToolTip(
            "Not available while recording or transcribing" if self._busy else "")
