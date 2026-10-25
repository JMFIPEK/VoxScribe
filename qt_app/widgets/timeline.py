"""Timeline across the full width of the Transcription tab (spec #40, #46):
one Speaker lane per top speaker plus Others, x axis = recording length.
Paints a TimelineModel (qt_app/timeline_model.py) and nothing else, so a
finished result and, later, the pipeline's progress only differ in the
model they hand to set_timeline()."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from qt_app import theme
from qt_app.timeline_model import TimelineModel

LANE_HEIGHT = 16
LANE_SPACING = 4
PADDING = 6
LABEL_WIDTH = 110
NOTE_HEIGHT = 18


class Timeline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._timeline = TimelineModel(0.0, ())
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(LABEL_WIDTH + 100)
        self.setFixedHeight(self._height())

    def set_timeline(self, timeline: TimelineModel):
        self._timeline = timeline
        self.setFixedHeight(self._height())
        self.update()

    def timeline(self) -> TimelineModel:
        return self._timeline

    # ------------------------------------------------------------ geometry
    def _height(self) -> int:
        lanes = max(1, len(self._timeline.lanes))
        note = NOTE_HEIGHT if self._timeline.note else 0
        return 2 * PADDING + lanes * LANE_HEIGHT + (lanes - 1) * LANE_SPACING + note

    def _track_left(self) -> float:
        return LABEL_WIDTH

    def _track_width(self) -> float:
        return max(1.0, self.width() - LABEL_WIDTH - PADDING)

    def lane_rect(self, index: int) -> QRectF:
        """Where lane `index` is drawn (the track part, without its label)."""
        top = PADDING + index * (LANE_HEIGHT + LANE_SPACING)
        return QRectF(self._track_left(), top, self._track_width(), LANE_HEIGHT)

    def x_of(self, seconds: float) -> float:
        duration = self._timeline.duration or 1.0
        return self._track_left() + self._track_width() * max(0.0, min(seconds, duration)) / duration

    def seconds_at(self, x: float) -> float:
        """The position in the recording under widget x (for seeking)."""
        fraction = (x - self._track_left()) / self._track_width()
        return max(0.0, min(1.0, fraction)) * self._timeline.duration

    # ------------------------------------------------------------ painting
    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        metrics = QFontMetrics(self.font())
        for index, lane in enumerate(self._timeline.lanes):
            rect = self.lane_rect(index)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.SURFACE_ALT))
            painter.drawRoundedRect(rect, 3, 3)
            for run in lane.runs:
                left, right = self.x_of(run.start), self.x_of(run.end)
                # At least one pixel, so a short turn of a long meeting shows.
                painter.fillRect(QRectF(left, rect.top(), max(1.0, right - left), rect.height()),
                                 QColor(run.color))

            label_rect = QRectF(0, rect.top(), LABEL_WIDTH - 8, rect.height())
            painter.setPen(QColor(lane.color or theme.TEXT_MUTED))
            painter.drawText(label_rect, Qt.AlignVCenter | Qt.AlignRight,
                             metrics.elidedText(lane.label, Qt.ElideRight, int(label_rect.width())))

        if self._timeline.note:
            top = self._height() - PADDING - NOTE_HEIGHT
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(QRectF(self._track_left(), top, self._track_width(), NOTE_HEIGHT),
                             Qt.AlignVCenter | Qt.AlignLeft, self._timeline.note)
