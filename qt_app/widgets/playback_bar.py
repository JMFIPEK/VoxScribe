"""Playback of the transcribed file (spec #40, #48): play/pause button and
position/length label. Plays the file as picked - the saved .wav or the
audio of a video file - never the echo-cleaned or level-balanced mix the
transcription used. QMediaPlayer + QAudioOutput (QtMultimedia, part of
PySide6) are only created on the first load(), so building the page stays
cheap and needs no audio device."""

from PySide6.QtCore import QUrl, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from qt_app.constants import format_time_short


class PlaybackBar(QWidget):
    positionChanged = Signal(float)  # seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self._player = None
        self._audio_output = None
        self._until = None           # pause here (a speaker's longest turn)
        self._duration = 0.0

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.play_btn = QPushButton("▶")
        self.play_btn.setToolTip("Play / pause (Space)")
        self.play_btn.setFixedWidth(40)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle)
        row.addWidget(self.play_btn)
        self.position_label = QLabel("")
        self.position_label.setProperty("role", "muted")
        row.addWidget(self.position_label)

    def load(self, path: str, duration: float = 0.0):
        """Plays `path` from now on; `duration` is shown until the player
        knows the real length."""
        if self._player is None:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            self._player = QMediaPlayer(self)
            self._audio_output = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_output)
            self._player.positionChanged.connect(self._on_position)
            self._player.durationChanged.connect(self._on_duration)
            self._player.playbackStateChanged.connect(self._on_state)
        self._player.stop()
        self._until = None
        self._duration = duration
        self._player.setSource(QUrl.fromLocalFile(path))
        self.play_btn.setEnabled(True)
        self._show_position(0)

    def unload(self):
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())
        self._until = None
        self.play_btn.setEnabled(False)
        self.position_label.setText("")

    def is_loaded(self) -> bool:
        return self._player is not None and self.play_btn.isEnabled()

    def play_from(self, seconds: float, until: float | None = None):
        """Seeks to `seconds` and plays, pausing at `until` if given."""
        if not self.is_loaded():
            return
        self._until = until
        self._player.setPosition(int(max(0.0, seconds) * 1000))
        self._player.play()

    def toggle(self):
        if not self.is_loaded():
            return
        from PySide6.QtMultimedia import QMediaPlayer

        self._until = None
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    # ---------------------------------------------------------- player events
    def _on_position(self, ms):
        seconds = ms / 1000
        if self._until is not None and seconds >= self._until:
            self._until = None
            self._player.pause()
        self._show_position(seconds)
        self.positionChanged.emit(seconds)

    def _on_duration(self, ms):
        if ms > 0:
            self._duration = ms / 1000
            self._show_position(self._player.position() / 1000)

    def _on_state(self, state):
        from PySide6.QtMultimedia import QMediaPlayer

        self.play_btn.setText("⏸" if state == QMediaPlayer.PlayingState else "▶")

    def _show_position(self, seconds):
        self.position_label.setText(
            f"{format_time_short(seconds)} / {format_time_short(self._duration)}")
