"""Custom-gezeichnete Pegelanzeige mit Peak-Hold, fuer Live-Monitoring."""

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QLinearGradient
from PySide6.QtWidgets import QWidget

from qt_app import theme


class LevelMeter(QWidget):
    """Horizontaler Pegelbalken (0..1) mit sanftem Decay und Peak-Hold-Strich."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self._peak = 0.0
        self._peak_hold_frames = 0
        self.setMinimumHeight(22)
        self.setMinimumWidth(120)

    def set_level(self, value: float):
        value = max(0.0, min(1.0, value))
        # sanfter Abfall statt hartem Sprung, wirkt "lebendiger"
        self._level = value if value > self._level else self._level * 0.7 + value * 0.3
        if value >= self._peak:
            self._peak = value
            self._peak_hold_frames = 12
        elif self._peak_hold_frames > 0:
            self._peak_hold_frames -= 1
        else:
            self._peak = max(value, self._peak * 0.92)
        self.update()

    def reset(self):
        self._level = 0.0
        self._peak = 0.0
        self._peak_hold_frames = 0
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt-Override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        radius = rect.height() / 2

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(theme.SURFACE_ALT))
        painter.drawRoundedRect(rect, radius, radius)

        if self._level > 0.001:
            fill_width = max(rect.height(), rect.width() * self._level)
            fill_rect = QRectF(rect.left(), rect.top(), fill_width, rect.height())
            gradient = QLinearGradient(fill_rect.topLeft(), fill_rect.topRight())
            gradient.setColorAt(0.0, QColor(theme.SUCCESS))
            gradient.setColorAt(0.75, QColor(theme.ACCENT))
            gradient.setColorAt(1.0, QColor(theme.DANGER))
            painter.setBrush(gradient)
            painter.setClipPath(self._rounded_path(rect, radius))
            painter.drawRect(fill_rect)
            painter.setClipping(False)

        if self._peak > 0.001:
            peak_x = rect.left() + rect.width() * self._peak
            painter.setPen(QColor("#ffffff"))
            painter.drawLine(peak_x, rect.top(), peak_x, rect.bottom())

    def _rounded_path(self, rect, radius):
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), radius, radius)
        return path
