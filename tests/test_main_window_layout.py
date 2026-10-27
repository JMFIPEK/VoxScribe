"""The pages keep their layout when the window is small and the Update
notice takes some of its height: they scroll instead of squeezing their rows
into each other."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from qt_app import main_window, theme
    from qt_app.pages import record_page

    app = QApplication.instance() or QApplication([])
    theme.apply_palette(app)
    app.setStyleSheet(theme.build_stylesheet())
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(record_page, "QSettings", lambda *a: QSettings(ini, QSettings.IniFormat))
    monkeypatch.setattr(main_window.HardwareInfoController, "start", lambda self: None)
    monkeypatch.setattr(main_window.UpdateController, "start", lambda self: None)
    monkeypatch.setattr("voxscribe.updates.take_update_result", lambda *a: None)

    window = main_window.MainWindow()
    window.update_notice.report({"ok": True, "target": "v9.0.1", "installed": "v9.0.1"})
    window.update_notice.offer("v9.0.2")
    window.resize(window.minimumSize())
    window.show()
    app.processEvents()
    yield window
    window.close()


def overlapping(widgets):
    """Pairs of visible widgets whose on-screen rectangles intersect."""
    from PySide6.QtCore import QRect
    rects = [(w, QRect(w.mapToGlobal(w.rect().topLeft()), w.size()))
             for w in widgets if w.isVisible()]
    return [(a.objectName() or type(a).__name__, b.objectName() or type(b).__name__)
            for i, (a, ra) in enumerate(rects) for b, rb in rects[i + 1:] if ra.intersects(rb)]


@pytest.mark.parametrize("mode", ["call", "playback", "room"])
def test_record_page_rows_never_overlap_under_the_update_notice(window, mode):
    page = window.pages["record"]
    page.mode_buttons[mode].click()
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()

    rows = [*page.mode_buttons.values(), page.setup_widget, page.setup_hint,
            page.playback_row, page.mic_row, page.title_edit, page.record_btn,
            page.status_label, page.auto_transcribe_check]
    assert overlapping(rows) == []
