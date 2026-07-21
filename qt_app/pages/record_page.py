"""Seite "Aufnahme": Quelle/Geraet waehlen, Live-Pegel/Timer und Start/Stop -
alles auf einer Seite (wie in der alten CTk-GUI)."""

import sys
import time

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QWidget,
)

from qt_app import theme
from qt_app.controllers import make_recording_output_path
from qt_app.pages.common import card, page_root
from qt_app.widgets.level_meter import LevelMeter

CHANNEL_LABELS = {"mic": "Mikrofon", "system": "System"}


class RecordPage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self._device_map = {}
        self._all_devices = {"microphones": [], "loopback": [], "default_loopback": None}
        self._start_time = None
        self._current_rms = {"mic": 0.0, "system": 0.0}
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self._wire()
        self.refresh_devices()

    # ------------------------------------------------------------------ ui
    def _build_ui(self):
        root = page_root(self)

        header = QLabel("Aufnahme")
        header.setProperty("role", "title")
        root.addWidget(header)

        # --- Quelle ---
        source_card, source_layout = card("QUELLE")
        root.addWidget(source_card)

        source_row = QHBoxLayout()
        source_layout.addLayout(source_row)

        source_values, default_source, source_hint = self._platform_source_options()
        self.source_combo = QComboBox()
        self.source_combo.addItems(source_values)
        self.source_combo.setCurrentText(default_source)
        source_row.addWidget(QLabel("Quelle:"))
        source_row.addWidget(self.source_combo, 1)

        if source_hint:
            hint_lbl = QLabel(source_hint)
            hint_lbl.setProperty("role", "hint")
            hint_lbl.setWordWrap(True)
            source_layout.addWidget(hint_lbl)

        # --- Geraet ---
        device_card, device_layout = card("GERÄT")
        root.addWidget(device_card)
        device_row = QHBoxLayout()
        device_layout.addLayout(device_row)
        device_row.addWidget(QLabel("Gerät:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("Lade...")
        device_row.addWidget(self.device_combo, 1)
        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_btn.setToolTip("Geräte aktualisieren")
        self.refresh_btn.setFixedWidth(36)
        device_row.addWidget(self.refresh_btn)

        # --- Pegel + Timer ---
        meter_card, meter_layout = card("PEGEL")
        root.addWidget(meter_card)

        self._meters = {}
        self._meter_rows = {}
        for channel, label in CHANNEL_LABELS.items():
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(80)
            row.addWidget(lbl)
            meter = LevelMeter()
            row.addWidget(meter, 1)
            meter_layout.addLayout(row)
            self._meters[channel] = meter
            self._meter_rows[channel] = (lbl, meter)

        self.timer_label = QLabel("00:00")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 28px; font-weight: 700;")
        meter_layout.addWidget(self.timer_label)

        # --- Start/Stop ---
        self.record_btn = QPushButton("  Aufnahme starten")
        self.record_btn.setIcon(QIcon(theme.get_icons()["record"]))
        self.record_btn.setIconSize(QSize(18, 18))
        self.record_btn.setProperty("role", "danger")
        self.record_btn.setMinimumHeight(50)
        self.record_btn.setCursor(Qt.PointingHandCursor)
        root.addWidget(self.record_btn)

        self.status_label = QLabel("Bereit")
        self.status_label.setProperty("role", "muted")
        self.status_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status_label)

        self.auto_transcribe_check = QCheckBox("Nach Aufnahme automatisch transkribieren")
        root.addWidget(self.auto_transcribe_check, 0, Qt.AlignHCenter)

        root.addStretch(1)
        self._apply_channel_visibility()

    def _platform_source_options(self):
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
        rc = self.window_.recorder_controller
        rc.recordingStarted.connect(self._on_recording_started)
        rc.levelUpdated.connect(self._on_level_updated)
        rc.recordingFinished.connect(self._on_recording_finished)

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
        else:
            self.device_combo.setEnabled(True)
            if value in ("Mikrofon", "Mikrofon + System"):
                devices = self._all_devices.get("microphones", [])
            else:
                devices = self._all_devices.get("loopback", [])

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

        self._apply_channel_visibility()

    def _apply_channel_visibility(self):
        value = self.source_combo.currentText()
        show_mic = value in ("Mikrofon", "Mikrofon + System")
        show_sys = value in ("System-Audio", "Mikrofon + System")
        for channel, visible in (("mic", show_mic), ("system", show_sys)):
            lbl, meter = self._meter_rows[channel]
            lbl.setVisible(visible)
            meter.setVisible(visible)

    # --------------------------------------------------------- recording
    def _toggle_recording(self):
        if self.window_.recorder_controller.is_recording:
            self.window_.recorder_controller.stop()
        else:
            self._start_recording()

    def _start_recording(self):
        device_name = self.device_combo.currentText()
        device_index = self._device_map.get(device_name)
        source_map = {"Mikrofon": "mic", "System-Audio": "system", "Mikrofon + System": "both"}
        source = source_map.get(self.source_combo.currentText(), "mic")

        output_path = make_recording_output_path(self.window_.settings.get("output_dir"))
        self.window_.recorder_controller.start(output_path, source, device_index)

    def _on_recording_started(self):
        self.record_btn.setText("  Aufnahme stoppen")
        self.record_btn.setIcon(QIcon(theme.get_icons()["stop"]))
        self.status_label.setText("Aufnahme läuft...")
        self.source_combo.setEnabled(False)
        self.device_combo.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self._start_time = time.monotonic()
        self._current_rms = {"mic": 0.0, "system": 0.0}
        self._timer.start()

    def _on_level_updated(self, channel, rms):
        self._current_rms[channel] = rms

    def _on_recording_finished(self, path, duration, error):
        self._timer.stop()
        self.record_btn.setText("  Aufnahme starten")
        self.record_btn.setIcon(QIcon(theme.get_icons()["record"]))
        self.source_combo.setEnabled(True)
        self.device_combo.setEnabled(self.source_combo.currentText() != "System-Audio" or sys.platform != "darwin")
        self.refresh_btn.setEnabled(True)
        self.timer_label.setText("00:00")
        for meter in self._meters.values():
            meter.reset()

        if error:
            self.status_label.setText(f"Fehler: {error}")
            return

        if path:
            self.status_label.setText(f"Gespeichert: {path} ({duration:.1f}s)")
            self.window_.send_to_transcription(path)
            if self.auto_transcribe_check.isChecked():
                self.window_.navigate_to("transcribe")
                self.window_.pages["transcribe"].start_transcription()

    def _tick(self):
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            mins, secs = divmod(int(elapsed), 60)
            self.timer_label.setText(f"{mins:02d}:{secs:02d}")
        for channel, meter in self._meters.items():
            level = min(self._current_rms.get(channel, 0.0) / 15000, 1.0)
            meter.set_level(level)

    def on_page_shown(self):
        pass
