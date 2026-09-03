"""VoxScribe — PySide6 GUI.

Structure:
    main_window.py   MainWindow with a top tab bar (QTabWidget)
    controllers.py    Thread-safe Qt signal bridges to recorder.py/transcriber.py
    theme.py          Colors + stylesheet (dark theme)
    pages/            One file per page (Record, Transcription, Settings)
    widgets/          Reusable widgets (level meter, ...)
"""
