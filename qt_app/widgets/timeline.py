"""Timeline across the full width of the Transcription tab (spec #40, #46):
one Speaker lane per top speaker plus Others, x axis = recording length.
Paints a TimelineModel (qt_app/timeline_model.py) and nothing else, so a
finished result and, later, the pipeline's progress only differ in the
model they hand to set_timeline()."""

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from qt_app import theme
from qt_app.timeline_model import SPEECH, TimelineModel

# How long the grey Speech lane takes to split into the Speaker lanes (ms).
SPLIT_ANIMATION_MS = 600
# Alpha of the plain sweep and of soft (not yet aligned) segment blocks.
SWEEP_ALPHA = 90
SOFT_ALPHA = 150

LANE_HEIGHT = 16
LANE_SPACING = 4
PADDING = 6
LABEL_WIDTH = 110
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
        self.setMinimumWidth(LABEL_WIDTH + 100)
        self.setFixedHeight(self._height())
        # 0..1 while the Speech lane splits into Speaker lanes, else 1.
        self._split = 1.0
        self._split_animation = QVariantAnimation(self)
        self._split_animation.setDuration(SPLIT_ANIMATION_MS)
        self._split_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._split_animation.valueChanged.connect(self._on_split_step)

    def set_timeline(self, timeline: TimelineModel):
        splits = (_is_speech_only(self._timeline) and self._timeline.percent is not None
                  and not _is_speech_only(timeline))
        self._timeline = timeline
        self.setFixedHeight(self._height())
        if splits:
            self._split_animation.stop()
            self._split_animation.setStartValue(0.0)
            self._split_animation.setEndValue(1.0)
            self._split_animation.start()
        self.update()

    def _on_split_step(self, value):
        self._split = float(value)
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
        """Clicks only seek once there is a result to play, never while the
        transcription runs."""
        self._seekable = seekable
        self.setCursor(Qt.PointingHandCursor if seekable else Qt.ArrowCursor)

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        x = event.position().x()
        if self._seekable and event.button() == Qt.LeftButton and x >= self._lanes_left():
            self.seekRequested.emit(self.seconds_at(x))
        super().mousePressEvent(event)

    # ------------------------------------------------------------ geometry
    def _height(self) -> int:
        lanes = max(1, len(self._timeline.lanes))
        note = NOTE_HEIGHT if self._timeline.note else 0
        return 2 * PADDING + lanes * LANE_HEIGHT + (lanes - 1) * LANE_SPACING + note

    def _lanes_left(self) -> float:
        return LABEL_WIDTH

    def _lanes_width(self) -> float:
        return max(1.0, self.width() - LABEL_WIDTH - PADDING)

    def lane_rect(self, index: int) -> QRectF:
        """Where lane `index` is drawn (the part right of its label)."""
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
        metrics = QFontMetrics(self.font())
        splitting = self._split_animation.state() == QVariantAnimation.Running
        for index, lane in enumerate(self._timeline.lanes):
            rect = self.lane_rect(index)
            if splitting:
                # Every Speaker lane starts where the Speech lane was and
                # slides down into its own place.
                rect.moveTop(self.lane_rect(0).top() + (rect.top() - self.lane_rect(0).top()) * self._split)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.SURFACE_ALT))
            painter.drawRoundedRect(rect, 3, 3)
            if self._timeline.sweep is not None:
                sweep = QColor(lane.color or theme.TEXT_MUTED)
                sweep.setAlpha(SWEEP_ALPHA)
                painter.setBrush(sweep)
                painter.drawRoundedRect(QRectF(rect.left(), rect.top(),
                                               rect.width() * self._timeline.sweep, rect.height()), 3, 3)
            for run in lane.runs:
                left, right = self.x_of(run.start), self.x_of(run.end)
                # At least one pixel, so a short turn of a long meeting shows.
                run_rect = QRectF(left, rect.top(), max(1.0, right - left), rect.height())
                color = QColor(run.color)
                if self._timeline.soft:
                    # Segment blocks before alignment: softer and rounded,
                    # until the word runs sharpen them.
                    color.setAlpha(SOFT_ALPHA)
                    painter.setBrush(color)
                    painter.drawRoundedRect(run_rect, 3, 3)
                else:
                    painter.fillRect(run_rect, color)

            label_rect = QRectF(0, rect.top(), LABEL_WIDTH - 8, rect.height())
            painter.setPen(QColor(lane.color or theme.TEXT_MUTED))
            painter.drawText(label_rect, Qt.AlignVCenter | Qt.AlignRight,
                             metrics.elidedText(lane.label, Qt.ElideRight, int(label_rect.width())))

        if self._timeline.percent is not None and self._timeline.lanes:
            rect = self.lane_rect(0).adjusted(0, 0, -6, 0)
            painter.setPen(QColor(theme.TEXT))
            painter.drawText(rect, Qt.AlignVCenter | Qt.AlignRight, f"{self._timeline.percent}%")

        if self._playhead is not None and self._timeline.lanes:
            x = self.x_of(self._playhead)
            painter.setPen(QPen(QColor(theme.TEXT), 2))
            painter.drawLine(QPointF(x, PADDING), QPointF(x, self.lane_rect(len(self._timeline.lanes) - 1).bottom()))

        if self._timeline.note:
            top = self._height() - PADDING - NOTE_HEIGHT
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(QRectF(self._lanes_left(), top, self._lanes_width(), NOTE_HEIGHT),
                             Qt.AlignVCenter | Qt.AlignLeft, self._timeline.note)


def _is_speech_only(timeline: TimelineModel) -> bool:
    return len(timeline.lanes) == 1 and timeline.lanes[0].label == SPEECH
