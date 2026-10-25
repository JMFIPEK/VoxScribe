"""Small line icons for the Transcription tab's buttons (play, pause, rename,
confirm, save), drawn with QPainter instead of Unicode glyphs - the glyphs
(▶ ⏸ ✏ ✓) come from whatever font Windows falls back to and look uneven
next to each other. Each icon is painted at 2x for high-DPI screens."""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from qt_app import theme

SIZE = 16
_SCALE = 2


def _icon(draw, color: str) -> QIcon:
    pixmap = QPixmap(SIZE * _SCALE, SIZE * _SCALE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(_SCALE, _SCALE)
    draw(painter, QColor(color))
    painter.end()
    pixmap.setDevicePixelRatio(_SCALE)
    return QIcon(pixmap)


def _line_pen(color: QColor, width: float = 1.6) -> QPen:
    return QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)


def play(color: str = theme.TEXT) -> QIcon:
    def draw(p, c):
        path = QPainterPath(QPointF(4.5, 2.5))
        path.lineTo(13.5, 8.0)
        path.lineTo(4.5, 13.5)
        path.closeSubpath()
        p.setPen(QPen(c, 1.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(c)
        p.drawPath(path)
    return _icon(draw, color)


def pause(color: str = theme.TEXT) -> QIcon:
    def draw(p, c):
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawRoundedRect(QRectF(3.5, 2.5, 3.2, 11), 1, 1)
        p.drawRoundedRect(QRectF(9.3, 2.5, 3.2, 11), 1, 1)
    return _icon(draw, color)


def pencil(color: str = theme.TEXT_MUTED) -> QIcon:
    def draw(p, c):
        p.setPen(_line_pen(c))
        p.setBrush(Qt.NoBrush)
        body = QPainterPath(QPointF(10.5, 2.5))
        body.lineTo(13.5, 5.5)
        body.lineTo(5.5, 13.5)
        body.lineTo(2.5, 13.5)
        body.lineTo(2.5, 10.5)
        body.closeSubpath()
        p.drawPath(body)
        p.drawLine(QPointF(8.8, 4.2), QPointF(11.8, 7.2))
    return _icon(draw, color)


def check(color: str = theme.SUCCESS_HOVER) -> QIcon:
    def draw(p, c):
        p.setPen(_line_pen(c, 2.0))
        path = QPainterPath(QPointF(3.0, 8.5))
        path.lineTo(6.5, 12.0)
        path.lineTo(13.0, 4.5)
        p.drawPath(path)
    return _icon(draw, color)
