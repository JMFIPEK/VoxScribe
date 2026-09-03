"""Colors and stylesheet for the PySide6 GUI's dark theme.

BOTH the stylesheet AND the QPalette are set (see apply_palette()): the
stylesheet covers the box models (background/border/padding), but
Fusion-native-drawn elements like focus rings partly ignore QSS colors and
use the QPalette instead - without a matching palette they'd be dark-on-dark
and practically invisible.

Checkbox check marks and the combobox arrow are generated as PNGs at runtime
(see _generate_icons()) and wired in via QSS `image: url(...)` - Qt
stylesheets can only color subcontrols like ::indicator/::down-arrow via
image files, not reliably via a pure CSS shape trick across Qt versions."""

import os
import tempfile

BG = "#15172a"
BG_ALT = "#1c2038"
SURFACE = "#242842"
# SURFACE_ALT (Eingabefelder/Textinhalt) war zuvor #363e68 - auf echtem
# Bildschirm neben SURFACE (Karten-Hintergrund) kaum unterscheidbar (per
# Pixel-Sampling nur (36,40,66) vs (54,62,104), zu wenig Kontrast trotz
# korrekt getrennter Werte im Code). Deutlich angehoben, damit Eingabefelder/
# Textinhalt sich klar von ihrer Karte abheben.
SURFACE_ALT = "#454f8a"
BORDER = "#5c66a8"
TEXT = "#e8e9f3"
TEXT_MUTED = "#8e9aaf"
ACCENT = "#3498db"
ACCENT_HOVER = "#5dade2"
DANGER = "#c0392b"
DANGER_HOVER = "#e74c3c"
SUCCESS = "#27ae60"
SUCCESS_HOVER = "#2ecc71"

SPEAKER_COLOR_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6",
    "#1abc9c", "#e67e22", "#00bcd4", "#ff6b9d", "#95a5a6",
]


def apply_palette(app):
    """Sets a dark QPalette, so Fusion-native-drawn elements (focus rings,
    popup highlight, ...) don't end up dark-on-dark and invisible."""
    from PySide6.QtGui import QColor, QPalette

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(SURFACE_ALT))
    palette.setColor(QPalette.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.ToolTipBase, QColor(SURFACE_ALT))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(SURFACE_ALT))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.Link, QColor(ACCENT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.PlaceholderText, QColor(TEXT_MUTED))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_MUTED))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_MUTED))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(TEXT_MUTED))
    app.setPalette(palette)


def _icon_dir():
    d = os.path.join(tempfile.gettempdir(), "voxscribe_qt_icons")
    os.makedirs(d, exist_ok=True)
    return d


def _as_url(path: str) -> str:
    return path.replace(os.sep, "/")


_ICON_CACHE = None


def get_icons() -> dict:
    """Generates (once, cached) all runtime icons and returns a dict of raw
    file paths - for QIcon(...) in Python code AND (via build_stylesheet(),
    which uses the same cache) for `image: url(...)` in the stylesheet."""
    global _ICON_CACHE
    if _ICON_CACHE is None:
        _ICON_CACHE = _generate_icons()
    return _ICON_CACHE


def _generate_icons():
    """Draws all runtime icons (checkbox check mark, combobox arrow,
    record/stop/speech-bubble symbol) - 2x resolution for crisp edges when
    scaled - and saves them to the temp directory, so both QIcon(...) and the
    stylesheet's `url(...)` references have a real file path."""
    from PySide6.QtCore import Qt, QPointF, QRectF
    from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap

    d = _icon_dir()
    size = 32

    def _checkbox(path, border_color, fill_color, draw_check):
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(3, 3, size - 6, size - 6)
        p.setPen(QPen(QColor(border_color), 2.5))
        p.setBrush(QColor(fill_color))
        p.drawRoundedRect(rect, 7, 7)
        if draw_check:
            pen = QPen(QColor("#ffffff"), 3.4)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            pts = [QPointF(size * 0.26, size * 0.52),
                   QPointF(size * 0.43, size * 0.70),
                   QPointF(size * 0.76, size * 0.32)]
            p.drawPolyline(pts)
        p.end()
        pix.save(path, "PNG")

    unchecked = os.path.join(d, "checkbox_unchecked.png")
    checked = os.path.join(d, "checkbox_checked.png")
    disabled = os.path.join(d, "checkbox_disabled.png")
    _checkbox(unchecked, BORDER, SURFACE_ALT, False)
    _checkbox(checked, ACCENT, ACCENT, True)
    _checkbox(disabled, SURFACE, SURFACE, False)

    arrow = os.path.join(d, "combo_arrow.png")
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(TEXT_MUTED), 2.8)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.drawPolyline([QPointF(size * 0.24, size * 0.40),
                     QPointF(size * 0.5, size * 0.62),
                     QPointF(size * 0.76, size * 0.40)])
    p.end()
    pix.save(arrow, "PNG")

    # --- Recording: red dot/stop symbols (tab icon + record button) ---
    record = os.path.join(d, "record.png")
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(DANGER))
    p.drawEllipse(QRectF(size * 0.12, size * 0.12, size * 0.76, size * 0.76))
    p.end()
    pix.save(record, "PNG")

    stop = os.path.join(d, "stop.png")
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(DANGER))
    p.drawRoundedRect(QRectF(size * 0.16, size * 0.16, size * 0.68, size * 0.68), 4, 4)
    p.end()
    pix.save(stop, "PNG")

    # --- Transcription: speech bubble ---
    speech_bubble = os.path.join(d, "speech_bubble.png")
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(TEXT_MUTED))
    bubble_path = QPainterPath()
    bubble_path.addRoundedRect(QRectF(size * 0.08, size * 0.10, size * 0.84, size * 0.62), 7, 7)
    tail = QPainterPath()
    tail.moveTo(size * 0.24, size * 0.72)
    tail.lineTo(size * 0.20, size * 0.90)
    tail.lineTo(size * 0.40, size * 0.72)
    tail.closeSubpath()
    bubble_path.addPath(tail)
    p.drawPath(bubble_path.simplified())
    p.setCompositionMode(QPainter.CompositionMode_Clear)
    for i in range(3):
        cx = size * (0.28 + i * 0.22)
        p.drawEllipse(QPointF(cx, size * 0.41), size * 0.045, size * 0.045)
    p.end()
    pix.save(speech_bubble, "PNG")

    return {
        "checkbox_unchecked": unchecked,
        "checkbox_checked": checked,
        "checkbox_disabled": disabled,
        "combo_arrow": arrow,
        "record": record,
        "stop": stop,
        "speech_bubble": speech_bubble,
    }


def build_stylesheet() -> str:
    """Builds the full stylesheet (uses the icon cache from get_icons()).
    Must only be called AFTER the QApplication is created (QPixmap needs a
    running Qt application)."""
    icons = {k: _as_url(v) for k, v in get_icons().items()}
    return f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI";
    font-size: 13px;
}}

QMainWindow {{
    background-color: {BG};
}}

QLabel[role="title"] {{
    font-size: 18px;
    font-weight: 600;
}}

QLabel[role="subtitle"] {{
    color: {TEXT_MUTED};
    font-size: 11px;
}}

QLabel {{
    color: {TEXT};
    background-color: transparent;
}}

QLabel[role="muted"] {{
    color: {TEXT_MUTED};
}}

QLabel[role="hint"] {{
    color: {TEXT_MUTED};
    font-size: 11px;
}}

QFrame[role="card"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

QFrame[role="separator"] {{
    background-color: {BORDER};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}

QTabWidget {{
    background-color: {BG_ALT};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-top: none;
    background-color: {BG};
    top: -1px;
}}
QTabWidget::tab-bar {{
    left: 8px;
}}
QTabBar {{
    background-color: {BG_ALT};
}}
QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_MUTED};
    padding: 9px 24px 6px 24px;
    margin-right: 4px;
    border: 1px solid transparent;
    border-bottom: 3px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-size: 13px;
    min-width: 108px;
}}
QTabBar::tab:hover {{
    color: {TEXT};
    background-color: {BG_ALT};
}}
QTabBar::tab:selected {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-bottom: 3px solid {ACCENT};
}}

QPushButton {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 14px;
    outline: none;
}}
QPushButton:hover {{
    background-color: #454e82;
}}
QPushButton:pressed {{
    background-color: #2a3054;
}}
QPushButton:disabled {{
    color: {TEXT_MUTED};
    background-color: {SURFACE};
}}
QPushButton:focus {{
    border: 1px solid {ACCENT};
}}

QPushButton[role="primary"] {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px;
}}
QPushButton[role="primary"]:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton[role="primary"]:disabled {{ background-color: {SURFACE_ALT}; color: {TEXT_MUTED}; }}

QPushButton[role="danger"] {{
    background-color: {DANGER};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px;
}}
QPushButton[role="danger"]:hover {{ background-color: {DANGER_HOVER}; }}

QPushButton[role="success"] {{
    background-color: {SUCCESS};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px;
}}
QPushButton[role="success"]:hover {{ background-color: {SUCCESS_HOVER}; }}
QPushButton[role="success"]:disabled {{ background-color: {SURFACE_ALT}; color: {TEXT_MUTED}; }}

QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    min-height: 22px;
    selection-background-color: {ACCENT};
}}
QLineEdit:hover, QComboBox:hover {{
    border: 1px solid {ACCENT_HOVER};
}}
QLineEdit:focus, QComboBox:focus, QComboBox:on {{
    border: 1px solid {ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled {{
    color: {TEXT_MUTED};
    background-color: {SURFACE};
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox::down-arrow {{
    image: url({icons['combo_arrow']});
    width: 14px;
    height: 14px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    outline: none;
    padding: 4px;
    selection-background-color: {ACCENT};
    selection-color: white;
}}

QCheckBox {{
    spacing: 8px;
    background-color: transparent;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    background-color: transparent;
    border: none;
    image: url({icons['checkbox_unchecked']});
}}
QCheckBox::indicator:checked {{
    image: url({icons['checkbox_checked']});
}}
QCheckBox::indicator:disabled {{
    image: url({icons['checkbox_disabled']});
}}

QProgressBar {{
    background-color: {SURFACE_ALT};
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}

QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""
