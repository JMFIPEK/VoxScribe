"""TranscribeController re-emits the pipeline's step events as Qt signals (#44)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_step_events_become_signals(monkeypatch):
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
    from qt_app.controllers import TranscribeController

    app = QCoreApplication.instance() or QCoreApplication([])  # noqa: F841
    turns = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"}]

    def fake_transcribe(on_progress=None, on_event=None, **kwargs):
        on_progress(0.1, "working")
        on_event("step", "transcription")
        on_event("segments", [{"start": 0.0, "end": 1.0, "text": "hi"}])
        on_event("step", "alignment")
        on_event("aligned", [{"start": 0.0, "end": 1.0, "text": "hi", "words": []}])
        on_event("step", "speaker_identification")
        on_event("speaker_turns", turns)
        return {"segments": turns, "step_times": {"transcription": 1.0}}

    monkeypatch.setattr("voxscribe.transcriber.transcribe", fake_transcribe)
    controller = TranscribeController()
    received = []
    controller.progressUpdated.connect(lambda pct, msg: received.append(("progress", pct)))
    controller.stepStarted.connect(lambda step: received.append(("step", step)))
    controller.segmentsStreamed.connect(lambda segs: received.append(("segments", len(segs))))
    controller.segmentsAligned.connect(lambda segs: received.append(("aligned", len(segs))))
    controller.speakerTurnsReady.connect(lambda segs: received.append(("turns", segs)))
    loop = QEventLoop()
    controller.finished.connect(lambda result, error: (received.append(("done", error)), loop.quit()))
    QTimer.singleShot(5000, loop.quit)

    controller.start(audio_path="a.wav")
    loop.exec()

    assert received == [
        ("progress", 0.1), ("step", "transcription"), ("segments", 1),
        ("step", "alignment"), ("aligned", 1),
        ("step", "speaker_identification"), ("turns", turns), ("done", None),
    ]
