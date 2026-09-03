"""Thread-safe Qt signal bridges to recorder.py and transcriber.py.

recorder.py/transcriber.py invoke their callbacks (on_level, on_done,
on_progress) from background threads. Qt widgets may only be touched from the
GUI thread - so these controllers forward the callbacks as Qt signals. When a
QObject emits a signal from a thread other than the one it lives in, Qt
automatically switches to a QueuedConnection, so the connected slot still runs
on the GUI thread - that's the only reason no manual locking is needed.
"""

import os
import threading
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from voxscribe.recorder import AudioRecorder, get_devices


class RecorderController(QObject):
    """Wraps AudioRecorder and reports level/completion as Qt signals."""

    levelUpdated = Signal(str, float)
    recordingStarted = Signal()
    recordingFinished = Signal(object, object, object)  # path, duration, error

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recorder = AudioRecorder()

    @property
    def is_recording(self) -> bool:
        return self._recorder.is_recording

    def refresh_devices(self):
        return get_devices()

    def start(self, output_path: str, source: str, device_index):
        self._recorder.start(
            output_path=output_path,
            source=source,
            device_index=device_index,
            on_level=lambda ch, rms: self.levelUpdated.emit(ch, rms),
            on_done=lambda path, dur, err: self.recordingFinished.emit(path, dur, err),
        )
        self.recordingStarted.emit()

    def stop(self):
        self._recorder.stop()


class TranscribeController(QObject):
    """Runs transcriber.transcribe() in a background thread and reports
    progress/results as Qt signals."""

    progressUpdated = Signal(float, str)
    finished = Signal(object, object)  # result dict or None, error text or None

    def start(self, **kwargs):
        def _run():
            try:
                # Import deliberately here in the background thread, not at
                # module top - `transcriber` imports whisperx/torch, which
                # can cold-take 1-3 minutes; in the GUI thread that would
                # freeze the UI for the whole time.
                from voxscribe.transcriber import transcribe
                result = transcribe(
                    on_progress=lambda pct, msg: self.progressUpdated.emit(pct, msg),
                    **kwargs,
                )
                self.finished.emit(result, None)
            except Exception as e:  # noqa: BLE001 - forwarded to the UI
                self.finished.emit(None, str(e))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def start_multi(self, **kwargs):
        """Like start(), but for multiple files (`audio_paths=[...]`) - see
        transcriber.transcribe_multi() for how the results are merged."""
        def _run():
            try:
                from voxscribe.transcriber import transcribe_multi
                result = transcribe_multi(
                    on_progress=lambda pct, msg: self.progressUpdated.emit(pct, msg),
                    **kwargs,
                )
                self.finished.emit(result, None)
            except Exception as e:  # noqa: BLE001 - forwarded to the UI
                self.finished.emit(None, str(e))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()


class HardwareInfoController(QObject):
    """Determines the hardware summary/model recommendation/bundled status in
    a background thread, since `transcriber`/`hardware_detect` import
    whisperx/torch on import - the main reason the GUI used to take 1-3
    minutes to start (SettingsPage/TranscribePage imported these modules
    synchronously during MainWindow.__init__(), i.e. before window.show()).
    Pages show placeholder text until the `infoReady` signal arrives."""

    infoReady = Signal(dict)

    def start(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        try:
            from voxscribe.hardware_detect import recommend_model, get_hardware_summary
            from voxscribe.transcriber import _get_bundled_models_dir

            hw_summary = get_hardware_summary()
            recommended_model, _device, reason = recommend_model()
            bundled = _get_bundled_models_dir()
            diarize_bundled = bool(bundled and os.path.isdir(os.path.join(bundled, "diarize")))
            info = {
                "hw_summary": hw_summary,
                "recommended_model": recommended_model,
                "reason": reason,
                "diarize_bundled": diarize_bundled,
            }
        except Exception as e:  # noqa: BLE001 - pages show an error text then
            info = {"error": str(e)}
        self.infoReady.emit(info)


def make_recording_output_path(out_dir: str) -> str:
    out_dir = out_dir or "recordings"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{timestamp}.wav")
