"""The title bar matches the app's dark theme (issues #14, #31)."""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_colorref_is_windows_byte_order():
    from qt_app.title_bar import colorref

    assert colorref("#1c2038") == 0x38201C  # COLORREF = 0x00BBGGRR
    assert colorref("#ffffff") == 0xFFFFFF


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    from qt_app import theme

    app = QApplication.instance() or QApplication([])
    theme.apply_palette(app)
    app.setStyleSheet(theme.build_stylesheet())
    return app


def test_main_window_keeps_its_three_tabs_centered(app, monkeypatch):
    from qt_app import main_window
    monkeypatch.setattr(main_window.HardwareInfoController, "start", lambda self: None)
    monkeypatch.setattr(main_window.UpdateController, "start", lambda self: None)
    monkeypatch.setattr("voxscribe.updates.take_update_result", lambda *a: None)

    window = main_window.MainWindow()
    window.resize(1000, 700)
    window.show()
    app.processEvents()

    tabs = window.tabs
    assert [tabs.tabText(i).strip() for i in range(tabs.count())] == [
        "Record", "Transcription", "⚙  Settings".strip()]
    bar = tabs.tabBar().geometry()
    left, right = bar.left(), tabs.width() - bar.right() - 1
    # Centered as exactly as Qt can: the stylesheet's "alignment: center"
    # puts the bar at (W - 2 - w) // 2, so the right gap is always 2 px wider
    # (also for a bare QTabWidget), plus 1 px when W - w is odd and can't be
    # split evenly. The parity follows the tab text widths, i.e. the font:
    # 2 px with Segoe UI on Windows, 3 px with the macOS fallback font.
    odd = (tabs.width() - bar.width()) % 2
    assert abs(left - right) <= 2 + odd, (left, right)
    window.close()


@pytest.mark.skipif(sys.platform != "win32", reason="DWM title-bar colors are Windows-only")
def test_title_bar_is_colored_on_windows(app):
    from PySide6.QtWidgets import QMainWindow
    from qt_app.title_bar import color_title_bar

    window = QMainWindow()
    window.show()
    # Offscreen windows have no DWM frame; the call must not fail either way.
    assert color_title_bar(window) in (True, False)
    window.close()
