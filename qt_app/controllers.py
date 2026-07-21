"""Thread-sichere Qt-Signal-Bruecken zu recorder.py und transcriber.py.

recorder.py/transcriber.py rufen ihre Callbacks (on_level, on_done,
on_progress) aus Background-Threads auf. Qt-Widgets duerfen nur vom
GUI-Thread aus angefasst werden - deshalb leiten diese Controller die
Callbacks als Qt-Signale weiter. Emittiert ein QObject ein Signal aus einem
anderen Thread als dem, in dem es lebt, stellt Qt automatisch auf eine
QueuedConnection um, sodass der verbundene Slot trotzdem im GUI-Thread
laeuft - das ist der einzige Grund, warum kein manuelles Locking noetig ist.
"""

import os
import threading
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from recorder import AudioRecorder, get_devices


class RecorderController(QObject):
    """Wrapt AudioRecorder und meldet Pegel/Fertigstellung als Qt-Signale."""

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
    """Fuehrt transcriber.transcribe() in einem Background-Thread aus und
    meldet Fortschritt/Ergebnis als Qt-Signale."""

    progressUpdated = Signal(float, str)
    finished = Signal(object, object)  # result dict oder None, Fehlertext oder None

    def start(self, **kwargs):
        def _run():
            try:
                # Import bewusst erst hier im Background-Thread (nicht am
                # Modulanfang) - `transcriber` importiert whisperx/torch, was
                # kalt 1-3 Minuten dauern kann; im GUI-Thread wuerde das die
                # Oberflaeche fuer die gesamte Zeit einfrieren.
                from transcriber import transcribe
                result = transcribe(
                    on_progress=lambda pct, msg: self.progressUpdated.emit(pct, msg),
                    **kwargs,
                )
                self.finished.emit(result, None)
            except Exception as e:  # noqa: BLE001 - an die UI weiterreichen
                self.finished.emit(None, str(e))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()


class HardwareInfoController(QObject):
    """Ermittelt Hardware-Zusammenfassung/Modellempfehlung/Bundled-Status in
    einem Background-Thread, da `transcriber`/`hardware_detect` beim Import
    whisperx/torch laden - das ist der Hauptgrund, warum die GUI beim Start
    vorher 1-3 Minuten gebraucht hat (SettingsPage/TranscribePage importierten
    diese Module synchron waehrend MainWindow.__init__(), also VOR
    window.show()). Seiten zeigen bis zum `infoReady`-Signal einen
    Platzhaltertext."""

    infoReady = Signal(dict)

    def start(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        try:
            from hardware_detect import recommend_model, get_hardware_summary
            from transcriber import _get_bundled_models_dir

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
        except Exception as e:  # noqa: BLE001 - Seiten zeigen dann einen Fehlertext
            info = {"error": str(e)}
        self.infoReady.emit(info)


def make_recording_output_path(out_dir: str) -> str:
    out_dir = out_dir or "recordings"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{timestamp}.wav")
