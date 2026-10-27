"""Record page (spec #50): the Playback only label and the compact Devices
card (issue #52), the Recording title named at Stop (issue #53) - offscreen,
with a fake recorder controller."""

import os
from datetime import datetime

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DEVICES = {
    "microphones": [{"index": 1, "name": "Headset Microphone (Jabra Evolve2 65)"},
                    {"index": 2, "name": "Mikrofonarray (Intel Smart Sound)"}],
    "loopback": [{"index": 10, "name": "Kopfhörer (Jabra Evolve2 65) [Loopback]"}],
    "default_loopback": 10,
}


class FakeClock:
    """Stands in for datetime in record_page: `now()` is what the test sets."""

    def __init__(self, now):
        self.current = now

    def now(self):
        return self.current


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

        # Like RecorderController: the path callback is asked at Stop (also
        # on window close), and what it says is where the files are saved.
        def start(self, output_path, source, device_index, loopback_index=None, setup=None):
            self.output_path = output_path
            self.is_recording = True
            self.recordingStarted.emit()

        def stop(self, wait=False):
            self.is_recording = False
            self.saved_to = self.output_path()

    class FakeWindow:
        recorder_controller = FakeRecorderController()
        settings = {"output_dir": str(tmp_path)}
        sent_to_transcription = []

        def send_to_transcription(self, path):
            self.sent_to_transcription.append(path)

    # Keep the user's real QSettings (registry) out of it.
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(record_page, "QSettings", lambda *a: QSettings(ini, QSettings.IniFormat))

    clock = FakeClock(datetime(2026, 10, 27, 14, 30, 59))
    monkeypatch.setattr(record_page, "datetime", clock)

    page = record_page.RecordPage(FakeWindow())
    page.clock = clock
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


# --- Recording title (issue #53) ----------------------------------------------

@pytest.mark.parametrize("mode, default", [("playback", "Playback"), ("call", "Meeting"), ("room", "Room")])
def test_title_placeholder_is_the_modes_default_and_follows_mode_changes(page, app, mode, default):
    if mode not in page.mode_buttons:
        pytest.skip(f"{mode} isn't offered on this OS")
    other = next(m for m in page.mode_buttons if m != mode) if len(page.mode_buttons) > 1 else mode
    _show_mode(page, app, other)
    _show_mode(page, app, mode)
    assert page.title_edit.text() == ""
    assert page.title_edit.placeholderText() == default


def test_title_line_sits_directly_above_start(page):
    title = _rect_in(page, page.title_edit)
    start = _rect_in(page, page.record_btn)
    timer = _rect_in(page, page.timer_label)
    assert timer.bottom() < title.top() and title.bottom() < start.top()
    assert start.top() - title.bottom() < 30


def _record(page, app, mode, title_before="", title_during=None):
    """Starts a recording in `mode` at 14:30:59, optionally changes the title
    while recording, stops at 15:10 and returns where it was saved."""
    if mode not in page.mode_buttons:
        pytest.skip(f"{mode} isn't offered on this OS")
    _show_mode(page, app, mode)
    page.title_edit.setText(title_before)
    page.record_btn.click()
    assert page.title_edit.isEnabled()
    if title_during is not None:
        page.title_edit.setText(title_during)
    page.clock.current = datetime(2026, 10, 27, 15, 10, 5)
    page.record_btn.click()
    return page.window_.recorder_controller.saved_to


def test_title_typed_during_a_call_names_the_file_with_the_minute_of_start(page, app, tmp_path):
    saved = _record(page, app, "call", title_during="ASAM WG")
    assert saved == os.path.join(str(tmp_path), "261027_1430_ASAM_WG.wav")


def test_title_typed_before_start_names_the_file(page, app, tmp_path):
    saved = _record(page, app, "room", title_before="Jour fixe")
    assert saved == os.path.join(str(tmp_path), "261027_1430_Jour_fixe.wav")


@pytest.mark.parametrize("mode, default", [("playback", "Playback"), ("call", "Meeting"), ("room", "Room")])
def test_empty_title_names_the_file_after_the_mode(page, app, tmp_path, mode, default):
    saved = _record(page, app, mode)
    assert saved == os.path.join(str(tmp_path), f"261027_1430_{default}.wav")


def test_recording_in_the_same_minute_gets_its_own_name(page, app, tmp_path):
    (tmp_path / "261027_1430_Meeting.tracks.flac").write_bytes(b"")
    saved = _record(page, app, "call")
    assert saved == os.path.join(str(tmp_path), "261027_1430_Meeting_2.wav")


@pytest.mark.parametrize("error", [None, "Device lost"])
def test_title_is_emptied_after_the_recording_finished(page, app, error):
    saved = _record(page, app, "call", title_during="ASAM WG")
    page.window_.recorder_controller.recordingFinished.emit(None if error else saved, 60.0, error)
    assert page.title_edit.text() == ""


def test_the_final_path_goes_to_transcription(page, app):
    saved = _record(page, app, "call", title_during="ASAM WG")
    page.window_.recorder_controller.recordingFinished.emit(saved, 60.0, None)
    assert page.window_.sent_to_transcription == [saved]


def test_controller_saves_where_the_path_callback_says_at_stop(app, tmp_path):
    from qt_app.controllers import RecorderController
    from tests.fake_capture import FakePyAudio, FakeSource
    from voxscribe.recorder import AudioRecorder, CaptureSources

    sources = CaptureSources(platform="win32", pyaudio=FakePyAudio(FakeSource(), FakeSource()))
    rc = RecorderController(recorder=AudioRecorder(sources))
    path = {"now": str(tmp_path / "at_start.wav")}
    rc.start(lambda: path["now"], "mic", None)
    path["now"] = str(tmp_path / "at_stop.wav")
    rc.stop(wait=True)

    assert os.path.isfile(tmp_path / "at_stop.wav")
    assert not os.path.exists(tmp_path / "at_start.wav")
