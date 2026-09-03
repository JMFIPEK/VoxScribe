"""Shared layout constants/helpers, so all pages (Record, Transcription,
Settings) use the same outer margins, spacing, and card look - otherwise the
content "jumps" when switching tabs, since card width/position would differ
per page."""

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

PAGE_MARGINS = (28, 24, 28, 24)
PAGE_SPACING = 14

CARD_MARGINS = (16, 14, 16, 14)
CARD_SPACING = 8


def page_root(widget) -> QVBoxLayout:
    """Creates a page's outer QVBoxLayout with the shared margins/spacing."""
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(*PAGE_MARGINS)
    layout.setSpacing(PAGE_SPACING)
    return layout


def card(title: str | None = None):
    """Creates a card (QFrame[role=card]) with shared inner spacing and an
    optional section title."""
    frame = QFrame()
    frame.setProperty("role", "card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(*CARD_MARGINS)
    layout.setSpacing(CARD_SPACING)
    if title:
        lbl = QLabel(title)
        lbl.setProperty("role", "subtitle")
        layout.addWidget(lbl)
    return frame, layout
