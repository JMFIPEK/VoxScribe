"""Transcription page on macOS (#60): no model dropdown until #59, and it
transcribes with the Mac's local default, Parakeet - offscreen, with a fake
transcribe controller."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_mac_page_has_no_dropdown_and_transcribes_with_parakeet(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QObject, Signal

    from qt_app.pages import transcribe_page
    from voxscribe.transcription.model_ids import PARAKEET_MODEL

    class FakeTranscribeController(QObject):
        progressUpdated = Signal(float, str)
        finished = Signal(object, object)
        stepStarted = Signal(str)

        def __init__(self):
            super().__init__()
            self.started = []

        def start(self, **kwargs):
            self.started.append(kwargs)

    class FakeWindow:
        settings = {"output_dir": str(tmp_path)}

        def __init__(self):
            self.transcribe_controller = FakeTranscribeController()

    monkeypatch.setattr("sys.platform", "darwin")
    window = FakeWindow()
    page = transcribe_page.TranscribePage(window)
    try:
        assert page.model_combo is None
        assert "Parakeet" in page.pipeline_info.text()

        audio = tmp_path / "call.wav"
        audio.write_bytes(b"")
        page.set_audio_file(str(audio))
        page.start_transcription()

        [kwargs] = window.transcribe_controller.started
        assert kwargs["audio_path"] == str(audio)
        assert kwargs["model_size"] == PARAKEET_MODEL
    finally:
        page.close()
