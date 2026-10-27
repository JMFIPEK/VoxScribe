"""Record page (spec #50): the Playback only label and the compact Devices
card (issue #52) - offscreen, with a fake recorder controller."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DEVICES = {
    "microphones": [{"index": 1, "name": "Headset Microphone (Jabra Evolve2 65)"},
                    {"index": 2, "name": "Mikrofonarray (Intel Smart Sound)"}],
    "loopback": [{"index": 10, "name": "Kopfhörer (Jabra Evolve2 65) [Loopback]"}],
    "default_loopback": 10,
}


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    from qt_app import theme

    app = QApplication.instance() or QApplication([])
    theme.apply_palette(app)
    app.setStyleSheet(theme.build_stylesheet())
    return app


@pytest.fixture
def page(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QObject, QSettings, Signal
    from qt_app.pages import record_page

    class FakeRecorderController(QObject):
        levelUpdated = Signal(str, float)
        recordingStarted = Signal()
        recordingStopping = Signal()
        recordingFinished = Signal(object, object, object)
        is_recording = False

        def __init__(self):
            super().__init__()
            self.refreshes = 0

        def refresh_devices(self):
            self.refreshes += 1
            return DEVICES

    class FakeWindow:
        recorder_controller = FakeRecorderController()
        settings = {"output_dir": str(tmp_path)}

    # Keep the user's real QSettings (registry) out of it.
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(record_page, "QSettings", lambda *a: QSettings(ini, QSettings.IniFormat))

    page = record_page.RecordPage(FakeWindow())
    page.resize(900, 900)
    page.show()
    app.processEvents()
    yield page
    page.close()


def _show_mode(page, app, mode):
    page.mode_buttons[mode].click()
    app.processEvents()


def _rect_in(page, widget):
    from PySide6.QtCore import QPoint, QRect

    return QRect(widget.mapTo(page, QPoint(0, 0)), widget.size())


def test_first_mode_button_reads_playback_only(page):
    first = next(iter(page.mode_buttons.values()))
    assert first.text().splitlines()[0] == "Playback only"


def test_refresh_is_an_icon_button_with_a_tooltip(page):
    assert page.refresh_btn.text() == ""
    assert not page.refresh_btn.icon().isNull()
    assert page.refresh_btn.toolTip() == "Refresh devices"


@pytest.mark.parametrize("mode", ["playback", "call", "room"])
def test_refresh_sits_beside_every_visible_picker(page, app, mode):
    if mode not in page.mode_buttons:
        pytest.skip(f"{mode} isn't offered on this OS")
    _show_mode(page, app, mode)
    pickers = [c for c in (page.playback_combo, page.mic_combo) if c.isVisible()]
    assert pickers
    refresh = _rect_in(page, page.refresh_btn)
    for combo in pickers:
        rect = _rect_in(page, combo)
        assert refresh.left() > rect.right()  # to the right of the picker ...
        assert refresh.top() < rect.bottom() and rect.top() < refresh.bottom()  # ... on its height
    # Not on a row of its own: it never reaches below the last visible picker's row.
    assert refresh.bottom() <= max(_rect_in(page, c).bottom() for c in pickers) + 2


def test_call_pickers_have_the_same_width(page, app):
    if "call" not in page.mode_buttons:
        pytest.skip("Call isn't offered on this OS")
    _show_mode(page, app, "call")
    assert page.playback_combo.width() == page.mic_combo.width()
    assert _rect_in(page, page.playback_combo).left() == _rect_in(page, page.mic_combo).left()


def test_refresh_reloads_devices_and_is_disabled_while_recording(page, app):
    rc = page.window_.recorder_controller
    before = rc.refreshes
    page.refresh_btn.click()
    assert rc.refreshes == before + 1

    rc.recordingStarted.emit()
    assert not page.refresh_btn.isEnabled()
