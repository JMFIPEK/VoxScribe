"""Speaker list on the right of the Transcription tab (spec #40, #45): one
row per speaker, ranked by share, with in-place renaming and a check mark
for a Tentative match (#47) and a play button (#48). Shows what a
SpeakerView says; applying a name and playing are left to the page
(renameRequested, confirmRequested, playRequested)."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class _NameEdit(QLineEdit):
    """Return and focus loss apply, Esc cancels - each at most once, since
    applying rebuilds the list and deletes this field."""
    applied = Signal(str)
    cancelled = Signal()

    def __init__(self, text):
        super().__init__(text)
        self._done = False
        self.returnPressed.connect(self.apply)

    def apply(self):
        if not self._done:
            self._done = True
            self.applied.emit(self.text())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if not self._done:
                self._done = True
                self.cancelled.emit()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.apply()


class _SpeakerRow(QWidget):
    def __init__(self, speaker, hint, tentative, on_rename, on_confirm, on_play):
        super().__init__()
        self._speaker = speaker
        self._on_rename = on_rename
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(0)

        self._row = QHBoxLayout()
        self._row.setSpacing(6)
        outer.addLayout(self._row)

        dot = QLabel("●")
        dot.setStyleSheet(f"color: {speaker.color};")
        self._row.addWidget(dot)

        self._name = QLabel(speaker.label)
        self._name.setStyleSheet(f"color: {speaker.color};")
        self._name.setToolTip(speaker.label)
        if tentative:
            font = self._name.font()
            font.setItalic(True)
            self._name.setFont(font)
            self._name.setToolTip(f"Probably {tentative}")
        self._row.addWidget(self._name, 1)

        share = QLabel(f"{speaker.share * 100:.0f}%")
        share.setProperty("role", "muted")
        self._row.addWidget(share)

        play = QToolButton()
        play.setText("▶")
        play.setToolTip("Play this speaker's longest turn")
        play.setAutoRaise(True)
        play.clicked.connect(lambda: on_play(speaker.label))
        self._row.addWidget(play)

        self._check = None
        if tentative:
            self._check = QToolButton()
            self._check.setText("✓")
            self._check.setToolTip(f"Yes, this is {tentative}")
            self._check.setAutoRaise(True)
            self._check.clicked.connect(lambda: on_confirm(speaker.label))
            self._row.addWidget(self._check)

        self._pencil = QToolButton()
        self._pencil.setText("✏")
        self._pencil.setToolTip("Rename")
        self._pencil.setAutoRaise(True)
        self._pencil.clicked.connect(self._start_edit)
        self._row.addWidget(self._pencil)

        if hint:
            hint_label = QLabel(hint)
            hint_label.setProperty("role", "hint")
            outer.addWidget(hint_label)

    def _start_edit(self):
        self._name.hide()
        self._pencil.hide()
        if self._check:
            self._check.hide()
        edit = _NameEdit(self._speaker.label)
        edit.applied.connect(lambda name: self._on_rename(self._speaker.label, name))
        edit.cancelled.connect(lambda: self._end_edit(edit, save))
        # No focus for the save icon: clicking it must not count as focus
        # loss first, or the name would be applied twice.
        save = QToolButton()
        save.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        save.setToolTip("Save name")
        save.setAutoRaise(True)
        save.setFocusPolicy(Qt.NoFocus)
        save.clicked.connect(edit.apply)
        self._row.insertWidget(1, edit, 1)
        self._row.insertWidget(2, save)
        edit.setFocus()
        edit.selectAll()

    def _end_edit(self, edit, save):
        edit.deleteLater()
        save.deleteLater()
        self._name.show()
        self._pencil.show()
        if self._check:
            self._check.show()


class SpeakerList(QWidget):
    renameRequested = Signal(str, str)  # current Speaker label, typed name
    confirmRequested = Signal(str)      # Speaker label of a Tentative match
    playRequested = Signal(str)         # Speaker label: play its longest turn

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        title = QLabel("Speakers")
        title.setStyleSheet("font-weight: 600;")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)
        rows_widget = QWidget()
        self._rows = QVBoxLayout(rows_widget)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(4)
        self._rows.addStretch(1)
        scroll.setWidget(rows_widget)

    def show_view(self, view):
        """Rebuilds every row from `view` (a SpeakerView), ranked by share."""
        while self._rows.count() > 1:
            item = self._rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for speaker in view.speakers():
            row = _SpeakerRow(speaker, view.enroll_hint(speaker.label),
                              view.tentative_name(speaker.label),
                              self.renameRequested.emit, self.confirmRequested.emit,
                              self.playRequested.emit)
            self._rows.insertWidget(self._rows.count() - 1, row)
