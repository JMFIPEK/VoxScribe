"""Gemeinsame Layout-Konstanten/Helfer, damit alle Seiten (Aufnahme,
Transkription, Einstellungen) dieselben Aussenraender, Abstaende und
Karten-Optik verwenden - sonst wirkt der Inhalt beim Tab-Wechsel
"springend", weil sich Breite/Position der Karten je Seite unterscheiden."""

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

PAGE_MARGINS = (28, 24, 28, 24)
PAGE_SPACING = 14

CARD_MARGINS = (16, 14, 16, 14)
CARD_SPACING = 8


def page_root(widget) -> QVBoxLayout:
    """Erstellt das äussere QVBoxLayout einer Seite mit den einheitlichen
    Raendern/Abstaenden."""
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(*PAGE_MARGINS)
    layout.setSpacing(PAGE_SPACING)
    return layout


def card(title: str | None = None):
    """Erstellt eine Karte (QFrame[role=card]) mit einheitlichem Innenraum
    und optionalem Abschnitts-Titel."""
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
