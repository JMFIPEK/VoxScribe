"""Seite "Live-Zusammenfassung": KIT-ToolBox-gestuetzte Live-Transkription +
periodische Meeting-Zusammenfassung waehrend einer laufenden Aufnahme.

Bewusst KEIN Teil des normalen Aufnahme-/Transkriptions-Flows, sondern eine
eigene, in sich geschlossene Seite fuer ein Premium-Feature: Audio wird dabei
kontinuierlich (nicht nur einmal am Ende) an den KIT-ToolBox-Server gesendet,
das ist bewusst nicht der Standard-Weg. Quelle/Geraet-Auswahl spiegelt dieselbe
Plattform-Logik wie die Aufnahme-Seite (inkl. "Mikrofon + System" als Default
unter Windows) - siehe AudioRecorder.snapshot_recent_audio(), das dafuer
Mikrofon/System getrennt live mischt (_mix_sources()).

Bewusst NUR die Zusammenfassung sichtbar, kein Live-Rohtranskript und kein
"Transkribiere..."-Statustext/Ladebalken: LiveMeetingController fuehrt die
rollierende Transkription weiterhin intern durch (sie speist die
Zusammenfassung), aber die Rohtranskript-Anzeige und der Zwischen-Status
haben sich als unnoetiges Rauschen erwiesen - die Zusammenfassung allein
reicht. Aus demselben Grund sind Transkript-/Zusammenfassungs-Intervall keine
UI-Einstellungen mehr, sondern feste Werte (siehe CHUNK_INTERVAL_SECONDS/
SUMMARY_INTERVAL_SECONDS)."""

import os
import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QTextEdit,
    QWidget,
)

from qt_app import theme
from qt_app.constants import (
    DEFAULT_API_BASE_URL_FALLBACK,
    DEFAULT_SUMMARY_MODEL_DISPLAY,
    SUMMARY_MODEL_IDS_BY_DISPLAY,
    SUMMARY_MODELS,
)
from qt_app.controllers import LiveMeetingController, make_recording_output_path
from qt_app.pages.common import card, page_root

KIT_STT_MODEL = "kit.whisper-large-v3"

# Nicht mehr ueber die UI waehlbar (siehe Moduldoc) - feste interne Werte.
# CHUNK_INTERVAL etwas kuerzer als SUMMARY_INTERVAL gewaehlt, damit bei einer
# Zusammenfassung i.d.R. schon mindestens ein frischer Transkript-Ausschnitt
# vorliegt. WINDOW_BUFFER/MIN_WINDOW wie zuvor: das rollierende Fenster ist
# etwas laenger als das Transkript-Intervall (Kontext gegen abgeschnittene
# Woerter an der Fenstergrenze), mindestens aber 60s.
CHUNK_INTERVAL_SECONDS = 15.0
SUMMARY_INTERVAL_SECONDS = 20.0
WINDOW_BUFFER_SECONDS = 30.0
MIN_WINDOW_SECONDS = 60.0
WINDOW_SECONDS = max(MIN_WINDOW_SECONDS, CHUNK_INTERVAL_SECONDS + WINDOW_BUFFER_SECONDS)


class LiveMeetingPage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self.live_controller = LiveMeetingController(window.recorder_controller, self)
        self._device_map = {}
        self._all_devices = {"microphones": [], "loopback": [], "default_loopback": None}
        self._start_time = None
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self._wire()
        self.refresh_devices()

    # ------------------------------------------------------------------ ui
    def _build_ui(self):
        root = page_root(self)

        header = QLabel("Live-Zusammenfassung")
        header.setProperty("role", "title")
        root.addWidget(header)

        hint = QLabel(
            "Premium-Feature über KIT ToolBox: Audio wird während der Aufnahme "
            "kontinuierlich an den KIT-Server übertragen und transkribiert/"
            "zusammengefasst — nicht 100% lokal."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # --- Aufnahme (vereinfacht) ---
        rec_card, rec_layout = card("AUFNAHME")
        root.addWidget(rec_card)

        source_row = QHBoxLayout()
        rec_layout.addLayout(source_row)
        source_row.addWidget(QLabel("Quelle:"))
        self.source_combo = QComboBox()
        source_values, default_source, source_hint = self._platform_source_options()
        self.source_combo.addItems(source_values)
        self.source_combo.setCurrentText(default_source)
        source_row.addWidget(self.source_combo, 1)
        source_row.addWidget(QLabel("Gerät:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("Lade...")
        source_row.addWidget(self.device_combo, 1)
        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_btn.setToolTip("Geräte aktualisieren")
        self.refresh_btn.setFixedWidth(36)
        source_row.addWidget(self.refresh_btn)

        if source_hint:
            hint_lbl = QLabel(source_hint)
            hint_lbl.setProperty("role", "hint")
            hint_lbl.setWordWrap(True)
            rec_layout.addWidget(hint_lbl)

        control_row = QHBoxLayout()
        rec_layout.addLayout(control_row)
        self.record_btn = QPushButton("  Live-Aufnahme starten")
        self.record_btn.setIcon(QIcon(theme.get_icons()["record"]))
        self.record_btn.setProperty("role", "danger")
        self.record_btn.setMinimumHeight(44)
        self.record_btn.setCursor(Qt.PointingHandCursor)
        control_row.addWidget(self.record_btn, 1)
        self.timer_label = QLabel("00:00")
        self.timer_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        control_row.addWidget(self.timer_label)

        self.status_label = QLabel("Bereit")
        self.status_label.setProperty("role", "muted")
        self.status_label.setAlignment(Qt.AlignCenter)
        rec_layout.addWidget(self.status_label)

        self.post_transcribe_check = QCheckBox(
            "Nach Ende: vollständige Transkription mit Sprechererkennung starten")
        rec_layout.addWidget(self.post_transcribe_check)

        # --- Zusammenfassungs-Einstellungen ---
        settings_card, settings_layout = card("ZUSAMMENFASSUNG")
        root.addWidget(settings_card)

        settings_row = QHBoxLayout()
        settings_layout.addLayout(settings_row)
        settings_row.addWidget(QLabel("Modell:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(SUMMARY_MODELS)
        self.model_combo.setCurrentText(DEFAULT_SUMMARY_MODEL_DISPLAY)
        settings_row.addWidget(self.model_combo, 1)
        self.summarize_now_btn = QPushButton("Jetzt zusammenfassen")
        self.summarize_now_btn.setEnabled(False)
        settings_row.addWidget(self.summarize_now_btn)

        summary_hint = QLabel("Aktualisiert sich automatisch alle 20 Sekunden.")
        summary_hint.setProperty("role", "hint")
        settings_layout.addWidget(summary_hint)

        # --- Zusammenfassung (einzige Inhaltsflaeche) ---
        self.summary_view = QTextEdit()
        self.summary_view.setReadOnly(True)
        self.summary_view.setPlaceholderText(
            "Die Zusammenfassung erscheint hier, sobald genug transkribiert wurde …")
        root.addWidget(self.summary_view, 1)

    def _platform_source_options(self):
        """Identisch zu RecordPage._platform_source_options() - "Mikrofon +
        System" ist unter Windows genau wie auf der normalen Aufnahme-Seite
        der Default, damit hier nicht nur eine Seite des Gesprächs (nur
        eigene Stimme ODER nur Gegenseite) fehlt."""
        if sys.platform == "win32":
            return ["Mikrofon", "System-Audio", "Mikrofon + System"], "Mikrofon + System", None
        if sys.platform == "darwin":
            return (
                ["Mikrofon", "System-Audio", "Mikrofon + System"],
                "Mikrofon",
                "⚠ System-Audio ist auf macOS experimentell (ScreenCaptureKit) — "
                "erfordert die Berechtigung „Bildschirm- und Systemaudioaufnahme“.",
            )
        return (
            ["Mikrofon"],
            "Mikrofon",
            "System-Audio (Meeting-Mitschnitt) ist auf diesem Betriebssystem noch nicht verfügbar.",
        )

    # --------------------------------------------------------------- wire
    def _wire(self):
        self.refresh_btn.clicked.connect(self.refresh_devices)
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        self.record_btn.clicked.connect(self._toggle_recording)
        self.summarize_now_btn.clicked.connect(self._summarize_now)

        rc = self.window_.recorder_controller
        rc.recordingStarted.connect(self._on_recording_started)
        rc.recordingFinished.connect(self._on_recording_finished)

        lc = self.live_controller
        lc.summaryUpdated.connect(self._on_summary_updated)
        lc.errorOccurred.connect(self._on_live_error)

    # ----------------------------------------------------------- devices
    def refresh_devices(self):
        self._all_devices = self.window_.recorder_controller.refresh_devices()
        self._on_source_changed(self.source_combo.currentText())

    def _on_source_changed(self, value):
        self._device_map = {}
        self.device_combo.clear()

        if value == "System-Audio" and sys.platform == "darwin":
            label = "Gesamte System-Wiedergabe (keine Geräteauswahl)"
            self.device_combo.addItem(label)
            self.device_combo.setEnabled(False)
            self._device_map[label] = None
            return

        self.device_combo.setEnabled(True)
        devices = (self._all_devices.get("microphones", [])
                   if value in ("Mikrofon", "Mikrofon + System")
                   else self._all_devices.get("loopback", []))
        names = []
        seen = set()
        for d in devices:
            label = d["name"]
            if label in seen:
                host_api = d.get("host_api") or "Audio"
                label = f"{label} ({host_api}, #{d['index']})"
            seen.add(label)
            names.append(label)
            self._device_map[label] = d["index"]

        if names:
            self.device_combo.addItems(names)
        else:
            self.device_combo.addItem("Kein Gerät gefunden")

    # --------------------------------------------------------- recording
    def _toggle_recording(self):
        if self.window_.recorder_controller.is_recording:
            self.window_.recorder_controller.stop()
        else:
            self._start_recording()

    def _resolve_kit_settings(self):
        settings = self.window_.settings
        api_key = settings.get("api_key") or os.getenv("KIT_TOOLBOX_API_KEY")
        api_base_url = settings.get("api_base_url") or DEFAULT_API_BASE_URL_FALLBACK
        return api_key, api_base_url

    def _start_recording(self):
        api_key, api_base_url = self._resolve_kit_settings()
        if not api_key:
            self.status_label.setText(
                "Kein KIT-ToolBox-API-Key gesetzt (siehe Einstellungen).")
            return

        device_name = self.device_combo.currentText()
        device_index = self._device_map.get(device_name)
        source_map = {"Mikrofon": "mic", "System-Audio": "system", "Mikrofon + System": "both"}
        source = source_map.get(self.source_combo.currentText(), "mic")

        summary_model = SUMMARY_MODEL_IDS_BY_DISPLAY.get(
            self.model_combo.currentText(), self.model_combo.currentText())

        self.summary_view.clear()

        # Reihenfolge wichtig: recorder_controller.start() emittiert
        # `recordingStarted` SYNCHRON (direkte Verbindung, gleicher Thread) -
        # `_on_recording_started()` prueft `live_controller.is_running`, um zu
        # erkennen, ob DIESE Seite die Aufnahme gestartet hat (statt der
        # normalen Aufnahme-Seite, die denselben recorder_controller teilt).
        # live_controller muss also VOR recorder_controller.start() schon
        # laufen, sonst waere das Flag beim Signal-Empfang noch False.
        self.live_controller.start(
            api_key=api_key, base_url=api_base_url, summary_model=summary_model,
            window_seconds=WINDOW_SECONDS, chunk_interval=CHUNK_INTERVAL_SECONDS,
            summary_interval=SUMMARY_INTERVAL_SECONDS)

        output_path = make_recording_output_path(self.window_.settings.get("output_dir"))
        self.window_.recorder_controller.start(output_path, source, device_index)

        self.summarize_now_btn.setEnabled(True)

    def _on_recording_started(self):
        if not self.live_controller.is_running:
            return  # Aufnahme wurde auf einer anderen Seite gestartet, nicht hier
        self.record_btn.setText("  Live-Aufnahme stoppen")
        self.record_btn.setIcon(QIcon(theme.get_icons()["stop"]))
        self.source_combo.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self._start_time = time.monotonic()
        self._timer.start()

    def _on_recording_finished(self, path, duration, error):
        if not self.live_controller.is_running:
            return  # Aufnahme lief nicht ueber diese Seite
        self.live_controller.stop()
        self._timer.stop()
        self.record_btn.setText("  Live-Aufnahme starten")
        self.record_btn.setIcon(QIcon(theme.get_icons()["record"]))
        self.source_combo.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.timer_label.setText("00:00")

        if error:
            self.status_label.setText(f"Fehler: {error}")
            return
        if path:
            self.status_label.setText(f"Aufnahme beendet: {path} ({duration:.1f}s)")
            if self.post_transcribe_check.isChecked():
                self.window_.send_to_transcription(path)
                self.window_.navigate_to("transcribe")
                self.window_.pages["transcribe"].start_transcription()

    def _tick(self):
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            mins, secs = divmod(int(elapsed), 60)
            self.timer_label.setText(f"{mins:02d}:{secs:02d}")

    # ------------------------------------------------------------- live
    def _summarize_now(self):
        api_key, api_base_url = self._resolve_kit_settings()
        if not api_key:
            self.status_label.setText(
                "Kein KIT-ToolBox-API-Key gesetzt (siehe Einstellungen).")
            return
        summary_model = SUMMARY_MODEL_IDS_BY_DISPLAY.get(
            self.model_combo.currentText(), self.model_combo.currentText())
        self.live_controller.summarize_now(api_key, api_base_url, summary_model)

    def _on_summary_updated(self, summary):
        self.summary_view.setPlainText(summary)

    def _on_live_error(self, message):
        self.status_label.setText(f"Fehler: {message}")

    def on_page_shown(self):
        pass
