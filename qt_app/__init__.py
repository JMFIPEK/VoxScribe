"""VoxScribe — PySide6 GUI (Nachfolger von gui.py/CustomTkinter).

Struktur:
    main_window.py   MainWindow mit Seitenleiste + QStackedWidget
    controllers.py    Thread-sichere Qt-Signal-Bruecken zu recorder.py/transcriber.py
    theme.py          Farben + Stylesheet (dunkles Theme)
    pages/            Eine Datei pro Seite (Aufnahme, Transkription, Einstellungen)
    widgets/          Wiederverwendbare Widgets (Pegelanzeige, ...)
"""
