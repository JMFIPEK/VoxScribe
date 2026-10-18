"""macOS app icon: Logo.png on a dark rounded-square tile.

macOS shows icons exactly as given (no automatic masking), and Logo.png is a
white outline on transparent, edge to edge - nearly invisible on a light Dock.
The same renderer feeds both the running app's Dock icon (gui_qt.py) and the
VoxScribe.app bundle's .icns (macos/make-app.sh), so the two always match."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap

from qt_app import theme

# Apple's icon grid: the tile fills 824/1024 of the canvas, the rest is margin
# for the drop shadow the Dock draws.
_TILE_FRACTION = 824 / 1024
_CORNER_FRACTION = 0.2237
_LOGO_FRACTION = 0.70


def build_icon_pixmap(logo_path: str, size: int = 1024) -> QPixmap:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)

    tile = size * _TILE_FRACTION
    margin = (size - tile) / 2
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(theme.BG))
    painter.drawRoundedRect(
        QRectF(margin, margin, tile, tile),
        tile * _CORNER_FRACTION, tile * _CORNER_FRACTION)

    logo_size = int(tile * _LOGO_FRACTION)
    logo = QPixmap(logo_path).scaled(
        logo_size, logo_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    painter.drawPixmap(
        int((size - logo.width()) / 2), int((size - logo.height()) / 2), logo)
    painter.end()
    return pix
