"""Timeline across the full width of the Transcription tab (spec #40, #46):
one thin Speaker lane per top speaker plus Others, x axis = recording
length. No lane labels - the colours match the speaker list on the right,
as in the who-said-what reference. Paints a TimelineModel
(qt_app/timeline_model.py) and nothing else; it only appears once the
result is in (the run itself shows a plain progress bar)."""

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from qt_app import theme
from qt_app.timeline_model import TimelineModel

LANE_HEIGHT = 8
LANE_SPACING = 5
PADDING = 6
NOTE_HEIGHT = 18


class Timeline(QWidget):
    # Clicked position in seconds, only while seekable (playback, #48).
    seekRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timeline = TimelineModel(0.0, ())
        self._playhead = None    # playback position in seconds, None = none drawn
        self._seekable = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(100)
        self.setFixedHeight(self._height())

    def set_timeline(self, timeline: TimelineModel):
        self._timeline = timeline
        self.setFixedHeight(self._height())
        self.update()

    def timeline(self) -> TimelineModel:
        return self._timeline

    # ------------------------------------------------------------ playback
    def set_playhead(self, seconds: float | None):
        """Draws the playback position as a line across all lanes."""
        if seconds != self._playhead:
            self._playhead = seconds
            self.update()

    def set_seekable(self, seekable: bool):
        """Clicks only seek once there is a result to play."""
        self._seekable = seekable
        self.setCursor(Qt.PointingHandCursor if seekable else Qt.ArrowCursor)

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if self._seekable and event.button() == Qt.LeftButton:
            self.seekRequested.emit(self.seconds_at(event.position().x()))
        super().mousePressEvent(event)

    # ------------------------------------------------------------ geometry
    def _height(self) -> int:
        lanes = max(1, len(self._timeline.lanes))
        note = NOTE_HEIGHT if self._timeline.note else 0
        return 2 * PADDING + lanes * LANE_HEIGHT + (lanes - 1) * LANE_SPACING + note

    def _lanes_left(self) -> float:
        return PADDING

    def _lanes_width(self) -> float:
        return max(1.0, self.width() - 2 * PADDING)

    def lane_rect(self, index: int) -> QRectF:
        """Where lane `index` is drawn."""
        top = PADDING + index * (LANE_HEIGHT + LANE_SPACING)
        return QRectF(self._lanes_left(), top, self._lanes_width(), LANE_HEIGHT)

    def x_of(self, seconds: float) -> float:
        duration = self._timeline.duration or 1.0
        return self._lanes_left() + self._lanes_width() * max(0.0, min(seconds, duration)) / duration

    def seconds_at(self, x: float) -> float:
        """The position in the recording under widget x (for seeking)."""
        fraction = (x - self._lanes_left()) / self._lanes_width()
        return max(0.0, min(1.0, fraction)) * self._timeline.duration

    # ------------------------------------------------------------ painting
    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for index, lane in enumerate(self._timeline.lanes):
            rect = self.lane_rect(index)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.BG_ALT))
            painter.drawRoundedRect(rect, 2, 2)
            for run in lane.runs:
                left, right = self.x_of(run.start), self.x_of(run.end)
                # At least one pixel, so a short turn of a long meeting shows.
                painter.setBrush(QColor(run.color))
                painter.drawRoundedRect(QRectF(left, rect.top(), max(1.5, right - left), rect.height()), 2, 2)

        if self._playhead is not None and self._timeline.lanes:
            x = self.x_of(self._playhead)
            painter.setPen(QPen(QColor(theme.TEXT), 2))
            painter.drawLine(QPointF(x, PADDING - 3),
                             QPointF(x, self.lane_rect(len(self._timeline.lanes) - 1).bottom() + 3))

        if self._timeline.note:
            top = self._height() - PADDING - NOTE_HEIGHT
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(QRectF(self._lanes_left(), top, self._lanes_width(), NOTE_HEIGHT),
                             Qt.AlignVCenter | Qt.AlignLeft, self._timeline.note)
