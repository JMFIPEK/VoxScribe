"""Thread-sichere Qt-Signal-Bruecken zu recorder.py und transcriber.py.

recorder.py/transcriber.py rufen ihre Callbacks (on_level, on_done,
on_progress) aus Background-Threads auf. Qt-Widgets duerfen nur vom
GUI-Thread aus angefasst werden - deshalb leiten diese Controller die
Callbacks als Qt-Signale weiter. Emittiert ein QObject ein Signal aus einem
anderen Thread als dem, in dem es lebt, stellt Qt automatisch auf eine
QueuedConnection um, sodass der verbundene Slot trotzdem im GUI-Thread
laeuft - das ist der einzige Grund, warum kein manuelles Locking noetig ist.
"""

import difflib
import os
import threading
import time
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from recorder import AudioRecorder, get_devices

# transcribe_remote()'s default timeout (1800s/30min) is meant for one-shot
# whole-file transcription, where waiting a long time for a long recording is
# fine. For the Live-Zusammenfassung rolling-window calls (a few seconds to a
# couple minutes of audio per request), a stuck/slow server should surface as
# an error within about a minute instead of leaving "Transkribiere..." showing
# with zero feedback for up to half an hour.
LIVE_STT_TIMEOUT_SECONDS = 60.0


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

    def snapshot_recent_audio(self, seconds: float = 180.0):
        return self._recorder.snapshot_recent_audio(seconds)


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

    def start_multi(self, **kwargs):
        """Wie start(), aber fuer mehrere Dateien (`audio_paths=[...]`) - siehe
        transcriber.transcribe_multi() fuers Zusammenfuegen der Ergebnisse."""
        def _run():
            try:
                from transcriber import transcribe_multi
                result = transcribe_multi(
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


class LiveMeetingController(QObject):
    """Live-Zusammenfassung waehrend einer laufenden Aufnahme (Premium-Feature,
    siehe qt_app/pages/live_meeting_page.py) - ausschliesslich ueber KIT
    ToolBox, nie ueber lokales WhisperX: ein Hintergrund-Thread transkribiert
    periodisch ein rollierendes Zeitfenster der bisherigen Aufnahme
    (`transcriber.transcribe_remote()` auf `RecorderController.
    snapshot_recent_audio()`-Ausschnitten) und fasst das Transkript in
    groesseren Abstaenden per Chat-Modell zusammen (`transcriber.
    summarize_meeting()`). Zusaetzlich per `summarize_now()` manuell ausloesbar.

    Haelt zwei getrennte Text-Puffer: `_full_transcript` waechst nur (fuer die
    Live-Rohtranskript-Anzeige), `_pending_delta` wird nach jeder
    Zusammenfassung geleert (nur das seit der letzten Zusammenfassung NEUE
    Stueck wird zusammen mit der vorherigen Zusammenfassung an das Chat-Modell
    geschickt - haelt die Tokenkosten bei langen Meetings etwa konstant statt
    mit der Gesamtlaenge des Transkripts zu wachsen).

    `chunk_interval`/`window_seconds` sind bewusst konfigurierbar (siehe
    LiveMeetingPage): das rollierende Fenster ueberlappt sich absichtlich
    zwischen Zyklen (vermeidet abgeschnittene Woerter an Chunk-Grenzen), `_run()`
    extrahiert daher pro Zyklus nur die Segmente aus den letzten
    `chunk_interval` Sekunden des Fensters als "neu" (per Start-Zeitstempel-
    Cutoff) statt den kompletten Fenster-Text jedes Mal anzuhaengen - sonst
    wuerde sich der ueberlappende, bereits transkribierte Teil bei jedem Zyklus
    wiederholen, und das umso staerker, je kuerzer `chunk_interval` relativ zu
    `window_seconds` gewaehlt wird.
    """

    transcriptUpdated = Signal(str)   # komplettes bisheriges Live-Transkript
    summaryUpdated = Signal(str)
    statusUpdated = Signal(str)
    errorOccurred = Signal(str)

    def __init__(self, recorder_controller, parent=None):
        super().__init__(parent)
        self._recorder_controller = recorder_controller
        self._stop_event = threading.Event()
        self._thread = None
        self._summary_lock = threading.Lock()
        self._full_transcript = []
        self._pending_delta = []
        self._detected_language = None
        self._summary = ""
        self._previous_window_text = ""

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, api_key: str, base_url: str, summary_model: str,
              window_seconds: float = 180.0, chunk_interval: float = 45.0,
              summary_interval: float = 180.0):
        self._stop_event.clear()
        self._full_transcript = []
        self._pending_delta = []
        self._detected_language = None
        self._summary = ""
        self._previous_window_text = ""
        self._thread = threading.Thread(
            target=self._run,
            args=(api_key, base_url, summary_model, window_seconds, chunk_interval, summary_interval),
            daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def summarize_now(self, api_key: str, base_url: str, summary_model: str):
        """Manueller Trigger ("Jetzt zusammenfassen") - laeuft in einem
        eigenen Einweg-Thread, unabhaengig vom periodischen Zyklus."""
        thread = threading.Thread(
            target=self._run_summary_cycle,
            args=(api_key, base_url, summary_model),
            daemon=True)
        thread.start()

    def _run(self, api_key, base_url, summary_model, window_seconds, chunk_interval, summary_interval):
        # Import bewusst erst hier im Hintergrund-Thread, nicht am Modul-
        # anfang - siehe TranscribeController fuer denselben Grund
        # (transcriber importiert whisperx/torch, kalt 1-3 Minuten).
        from transcriber import transcribe_remote

        last_summary_time = time.monotonic()
        while not self._stop_event.wait(chunk_interval):
            # Der GESAMTE Zyklus-Koerper ist bewusst in einem try/except statt
            # nur der transcribe_remote()-Aufruf: eine unbehandelte Exception
            # irgendwo hier (z.B. in _run_summary_cycle(), s.u.) wuerde sonst
            # diesen Hintergrund-Thread STILL beenden - unter pythonw.exe/einer
            # fensterlosen GUI ohne sichtbare Konsole sieht das aus wie "das
            # Live-Transkript aktualisiert sich nach dem ersten Mal nicht
            # mehr", ohne jede Fehlermeldung.
            try:
                audio = self._recorder_controller.snapshot_recent_audio(window_seconds)
                if audio is None or len(audio) == 0:
                    continue

                self.statusUpdated.emit("Transkribiere...")
                result = transcribe_remote(
                    audio, self._detected_language, "kit.whisper-large-v3",
                    api_key=api_key, base_url=base_url, timeout=LIVE_STT_TIMEOUT_SECONDS)

                if self._detected_language is None and result.get("language"):
                    self._detected_language = result["language"]

                # `audio` ist das rollierende Fenster (window_seconds,
                # ueberlappt sich zwischen Zyklen absichtlich - siehe
                # snapshot_recent_audio()), NICHT nur die neuen Daten seit dem
                # letzten Zyklus. Wuerde man den kompletten zurueckgegebenen
                # Text jedes Mal anhaengen, wuerde sich der bereits
                # transkribierte, ueberlappende Teil bei jedem Zyklus
                # wiederholen. Ein Filter ueber Segment-Start-Zeitstempel
                # (fruehere Version) ging davon aus, dass der Server viele
                # kleine, feingranular getimte Segmente liefert - tatsaechlich
                # liefert KIT ToolBox pro Anfrage typischerweise NUR EIN
                # Segment ueber das gesamte Fenster (start=0.0, end=Fenster-
                # Ende, per Live-Test bestaetigt), wodurch der Zeitstempel-
                # Filter JEDES Mal den kompletten Text verworfen hat - das
                # Live-Transkript aktualisierte sich dadurch nie, obwohl die
                # Transkription serverseitig erfolgreich war (kein Fehler).
                # Robuster: den Text des aktuellen Fensters gegen den Text des
                # LETZTEN Zyklus diffen (laengste gemeinsame Teilsequenz) und
                # nur das, was danach im aktuellen Text folgt, als "neu"
                # behandeln - funktioniert unabhaengig von der Segmentierung
                # des Servers.
                current_window_text = " ".join(
                    seg.get("text", "").strip() for seg in result.get("segments", [])).strip()
                text = self._extract_new_text(self._previous_window_text, current_window_text)
                self._previous_window_text = current_window_text
                if text:
                    self._full_transcript.append(text)
                    self._pending_delta.append(text)
                    # Leerzeichen statt Zeilenumbruch: die extrahierten
                    # Fragmente sind oft nur ein paar Woerter lang (siehe
                    # _extract_new_text()), mit "\n" verbunden wuerde das
                    # Live-Transkript wie viele kurze Absaetze statt
                    # fliessendem Text aussehen.
                    self.transcriptUpdated.emit(" ".join(self._full_transcript))
                    self.statusUpdated.emit("Live-Transkript aktualisiert")
                else:
                    self.statusUpdated.emit("Keine neue Sprache erkannt.")

                if time.monotonic() - last_summary_time >= summary_interval and self._pending_delta:
                    self._run_summary_cycle(api_key, base_url, summary_model)
                    last_summary_time = time.monotonic()
            except Exception as e:  # noqa: BLE001 - Schleife MUSS weiterlaufen
                self.errorOccurred.emit(str(e))
                continue

    @staticmethod
    def _extract_new_text(previous_window_text: str, current_window_text: str) -> str:
        """Findet den Teil von `current_window_text`, der NACH der laengsten
        gemeinsamen Teilsequenz mit `previous_window_text` liegt - der
        tatsaechlich neue Teil seit dem letzten Zyklus.

        Textbasiert statt Segment-Zeitstempel-basiert, weil KIT ToolBox pro
        Anfrage typischerweise nur ein einziges Segment ueber das gesamte
        Fenster liefert (siehe Kommentar in _run()) - ein Zeitstempel-Cutoff
        kann damit nicht zwischen "alt" und "neu" unterscheiden. Funktioniert
        auch, wenn der Server denselben ueberlappenden Audio-Abschnitt beim
        naechsten Mal minimal anders formuliert/interpunktiert transkribiert -
        `find_longest_match` findet dann einen kuerzeren, aber i.d.R.
        trotzdem brauchbaren gemeinsamen Abschnitt.
        """
        if not previous_window_text:
            return current_window_text
        if not current_window_text:
            return ""
        matcher = difflib.SequenceMatcher(None, previous_window_text, current_window_text)
        match = matcher.find_longest_match(
            0, len(previous_window_text), 0, len(current_window_text))
        if match.size == 0:
            return current_window_text
        return current_window_text[match.b + match.size:].strip()

    def _run_summary_cycle(self, api_key, base_url, summary_model):
        with self._summary_lock:
            delta = " ".join(self._pending_delta)
            if not delta:
                self.statusUpdated.emit("Noch nichts Neues seit der letzten Zusammenfassung.")
                return
            try:
                from transcriber import summarize_meeting
                self.statusUpdated.emit("Fasse zusammen...")
                summary = summarize_meeting(
                    delta, api_key=api_key, base_url=base_url, model=summary_model,
                    previous_summary=self._summary or None,
                    language=self._detected_language)
                self._summary = summary
                self._pending_delta = []
                self.summaryUpdated.emit(summary)
                self.statusUpdated.emit("Zusammenfassung aktualisiert")
            except Exception as e:  # noqa: BLE001 - an die UI weiterreichen
                self.errorOccurred.emit(str(e))


def make_recording_output_path(out_dir: str) -> str:
    out_dir = out_dir or "recordings"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"{timestamp}.wav")
