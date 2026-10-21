"""Page "Record": choose the Recording mode (and a Call's Setup), the devices
it needs, live levels/timer, and start/stop - all on one page.

Terms (see GLOSSARY.md): Playback audio is what the computer plays,
Microphone audio what the room microphone picks up. Recording modes:
Playback only, Call (both, as separate tracks) and In room (microphone only);
a Call's Setup is Headset or Loudspeaker.
"""

import json
import sys
import time

from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from qt_app import theme
from qt_app.controllers import make_recording_output_path
from qt_app.pages.common import card, page_root
from qt_app.widgets.level_meter import LevelMeter
from voxscribe import recording_modes as rm

CHANNEL_LABELS = {"system": "Playback", "mic": "Microphone"}

MODE_LABELS = {
    rm.MODE_PLAYBACK: "Listen only\nWhat the computer plays",
    rm.MODE_CALL: "Call\nPlayback + your microphone",
    rm.MODE_ROOM: "In room\nMicrophone only",
}
SETUP_LABELS = {
    rm.SETUP_HEADSET: "🎧  Headset",
    rm.SETUP_LOUDSPEAKER: "🔊  Loudspeaker && microphone"  # "&&": a single "&" is a Qt mnemonic,
}
SETUP_HINTS = {
    rm.SETUP_HEADSET: "Headphones with their own microphone - your voice and the "
                      "remote voices stay on separate tracks.",
    rm.SETUP_LOUDSPEAKER: "Speakers with a room microphone, mic array or speakerphone - "
                          "the microphone also picks up the remote voices.",
}

# macOS captures all playback (ScreenCaptureKit), there's no device to pick.
_MACOS_PLAYBACK_LABEL = "Entire system playback (no device selection)"

# int16 RMS (see recorder.compute_rms()). Measured: an unworn/unused headset
# mic delivers 0-0.5, a live laptop mic array in a quiet room ~3-5, speech
# hundreds to thousands.
_MIC_SILENCE_RMS = 2.0
_MIC_SILENCE_WARN_SECONDS = 8

_SETTINGS_MODE = "record/mode"
# Setup the user last chose per playback device (JSON {device name: setup}),
# so an override - e.g. a speakerphone the name heuristic takes for a
# headset - sticks for that device.
_SETTINGS_SETUP_BY_PLAYBACK = "record/setup_by_playback"


def _platform_modes():
    """(available modes, first-launch mode, hint) for this OS."""
    if sys.platform == "win32":
        return list(rm.MODES), rm.MODE_CALL, None
    if sys.platform == "darwin":
        return (
            list(rm.MODES),
            rm.MODE_CALL,
            "⚠ Recording playback audio is experimental on macOS (ScreenCaptureKit) — "
            "requires the \"Screen & System Audio Recording\" permission.",
        )
    return [rm.MODE_ROOM], rm.MODE_ROOM, "Recording playback audio isn't available on this OS yet."


def _fill_device_combo(combo: QComboBox, devices: list[dict]) -> dict:
    """Fills combo with the device names (disambiguated if duplicated) and
    returns {label: device index}."""
    combo.clear()
    device_map = {}
    for d in devices:
        label = d["name"].replace(" [Loopback]", "")
        if label in device_map:
            label = f"{label} ({d.get('host_api') or 'Audio'}, #{d['index']})"
        device_map[label] = d["index"]
    if device_map:
        combo.addItems(list(device_map))
    else:
        combo.addItem("No device found")
    return device_map


class RecordPage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self._settings = QSettings("VoxScribe", "VoxScribe")
        self._modes, default_mode, self._mode_hint = _platform_modes()
        saved_mode = self._settings.value(_SETTINGS_MODE, default_mode)
        self._mode = saved_mode if saved_mode in self._modes else default_mode
        self._setup = rm.SETUP_HEADSET
        self._mic_map = {}
        self._playback_map = {}
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

        header = QLabel("Record")
        header.setProperty("role", "title")
        root.addWidget(header)

        # --- Recording mode (+ Setup for a Call) ---
        mode_card, mode_layout = card("WHAT DO YOU WANT TO RECORD?")
        root.addWidget(mode_card)

        mode_row = QHBoxLayout()
        mode_layout.addLayout(mode_row)
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for mode in self._modes:
            btn = self._choice_button(MODE_LABELS[mode])
            btn.setMinimumHeight(56)
            self.mode_group.addButton(btn)
            self.mode_buttons[mode] = btn
            mode_row.addWidget(btn, 1)
        self.mode_buttons[self._mode].setChecked(True)

        self.setup_widget = QWidget()
        self.setup_widget.setProperty("role", "row")
        setup_row = QHBoxLayout(self.setup_widget)
        setup_row.setContentsMargins(0, 10, 0, 0)
        setup_row.addWidget(QLabel("Setup:"))
        self.setup_group = QButtonGroup(self)
        self.setup_buttons = {}
        for setup in rm.SETUPS:
            btn = self._choice_button(SETUP_LABELS[setup])
            btn.setMinimumHeight(40)
            self.setup_group.addButton(btn)
            self.setup_buttons[setup] = btn
            setup_row.addWidget(btn, 1)
        self.setup_widget.setMinimumHeight(52)
        mode_layout.addWidget(self.setup_widget)

        self.setup_hint = QLabel("")
        self.setup_hint.setProperty("role", "hint")
        self.setup_hint.setWordWrap(True)
        mode_layout.addWidget(self.setup_hint)

        if self._mode_hint:
            hint_lbl = QLabel(self._mode_hint)
            hint_lbl.setProperty("role", "hint")
            hint_lbl.setWordWrap(True)
            mode_layout.addWidget(hint_lbl)

        # --- Devices ---
        device_card, device_layout = card("DEVICES")
        root.addWidget(device_card)

        self.playback_row = QWidget()
        self.playback_row.setProperty("role", "row")
        playback_row = QHBoxLayout(self.playback_row)
        playback_row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("Playback:")
        lbl.setFixedWidth(80)
        playback_row.addWidget(lbl)
        self.playback_combo = QComboBox()
        playback_row.addWidget(self.playback_combo, 1)
        device_layout.addWidget(self.playback_row)

        self.mic_row = QWidget()
        self.mic_row.setProperty("role", "row")
        mic_row = QHBoxLayout(self.mic_row)
        mic_row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("Microphone:")
        lbl.setFixedWidth(80)
        mic_row.addWidget(lbl)
        self.mic_combo = QComboBox()
        mic_row.addWidget(self.mic_combo, 1)
        device_layout.addWidget(self.mic_row)

        refresh_row = QHBoxLayout()
        self.device_warning = QLabel("")
        self.device_warning.setWordWrap(True)
        self.device_warning.setStyleSheet(f"color: {theme.DANGER};")
        self.device_warning.hide()
        refresh_row.addWidget(self.device_warning, 1)
        self.refresh_btn = QPushButton(" Refresh devices")
        self.refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh_row.addWidget(self.refresh_btn, 0, Qt.AlignRight)
        device_layout.addLayout(refresh_row)

        # --- Level + timer ---
        meter_card, meter_layout = card("LEVEL")
        root.addWidget(meter_card)

        # Meters on the left, timer beside them - keeps the Call layout short
        # enough for the default window height.
        meters_and_timer = QHBoxLayout()
        meter_layout.addLayout(meters_and_timer)
        meters = QVBoxLayout()
        meters_and_timer.addLayout(meters, 1)

        self._meters = {}
        self._meter_rows = {}
        for channel, label in CHANNEL_LABELS.items():
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(80)
            row.addWidget(lbl)
            meter = LevelMeter()
            row.addWidget(meter, 1)
            meters.addLayout(row)
            self._meters[channel] = meter
            self._meter_rows[channel] = (lbl, meter)

        self.timer_label = QLabel("00:00")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setMinimumWidth(110)
        self.timer_label.setStyleSheet("font-size: 28px; font-weight: 700;")
        meters_and_timer.addWidget(self.timer_label)

        # --- Start/stop ---
        self.record_btn = QPushButton("  Start recording")
        self.record_btn.setIcon(QIcon(theme.get_icons()["record"]))
        self.record_btn.setIconSize(QSize(18, 18))
        self.record_btn.setProperty("role", "danger")
        self.record_btn.setMinimumHeight(50)
        self.record_btn.setCursor(Qt.PointingHandCursor)
        root.addWidget(self.record_btn)

        # Indeterminate busy bar while the WAV is being saved after stop -
        # that step scales with the recording length and can take a while.
        self.saving_bar = QProgressBar()
        self.saving_bar.setRange(0, 0)
        self.saving_bar.hide()
        root.addWidget(self.saving_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setProperty("role", "muted")
        self.status_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status_label)

        self.auto_transcribe_check = QCheckBox("Automatically transcribe after recording")
        root.addWidget(self.auto_transcribe_check, 0, Qt.AlignHCenter)

        root.addStretch(1)
        # Rows that get hidden/shown per mode must not soak up spare height.
        for row in (self.setup_widget, self.playback_row, self.mic_row):
            row.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._apply_mode()

    def _choice_button(self, text):
        btn = QPushButton(text)
        btn.setProperty("role", "choice")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    # --------------------------------------------------------------- wire
    def _wire(self):
        self.refresh_btn.clicked.connect(self.refresh_devices)
        for mode, btn in self.mode_buttons.items():
            btn.clicked.connect(lambda _=False, m=mode: self._on_mode_clicked(m))
        for setup, btn in self.setup_buttons.items():
            btn.clicked.connect(lambda _=False, s=setup: self._on_setup_clicked(s))
        self.playback_combo.currentTextChanged.connect(self._on_playback_changed)
        self.mic_combo.currentTextChanged.connect(lambda _: self._update_device_warning())
        self.record_btn.clicked.connect(self._toggle_recording)
        rc = self.window_.recorder_controller
        rc.recordingStarted.connect(self._on_recording_started)
        rc.levelUpdated.connect(self._on_level_updated)
        rc.recordingStopping.connect(self._on_recording_stopping)
        rc.recordingFinished.connect(self._on_recording_finished)

    # -------------------------------------------------------- mode/setup
    def _on_mode_clicked(self, mode):
        self._mode = mode
        self._settings.setValue(_SETTINGS_MODE, mode)
        self._apply_mode()

    def _on_setup_clicked(self, setup):
        self._set_setup(setup)
        setups = self._saved_setups()
        setups[self._playback_key()] = setup
        self._settings.setValue(_SETTINGS_SETUP_BY_PLAYBACK, json.dumps(setups))
        if setup == rm.SETUP_HEADSET:
            self._select_matching_mic()
        self._update_device_warning()

    def _set_setup(self, setup):
        self._setup = setup
        self.setup_buttons[setup].setChecked(True)
        self.setup_hint.setText(SETUP_HINTS[setup])

    def _saved_setups(self) -> dict:
        try:
            setups = json.loads(self._settings.value(_SETTINGS_SETUP_BY_PLAYBACK, "{}"))
        except (TypeError, ValueError):
            return {}
        return setups if isinstance(setups, dict) else {}

    def _apply_mode(self):
        """Shows only what the current mode needs: Setup for a Call, the
        playback and/or microphone picker, and the matching level meters."""
        has_playback = self._mode in (rm.MODE_PLAYBACK, rm.MODE_CALL)
        has_mic = self._mode in (rm.MODE_CALL, rm.MODE_ROOM)
        is_call = self._mode == rm.MODE_CALL
        self.setup_widget.setVisible(is_call)
        self.setup_hint.setVisible(is_call)
        self.playback_row.setVisible(has_playback)
        self.mic_row.setVisible(has_mic)
        for channel, visible in (("system", has_playback), ("mic", has_mic)):
            lbl, meter = self._meter_rows[channel]
            lbl.setVisible(visible)
            meter.setVisible(visible)
        self._update_device_warning()

    # ----------------------------------------------------------- devices
    def refresh_devices(self):
        self._all_devices = self.window_.recorder_controller.refresh_devices()

        self.mic_combo.blockSignals(True)
        self._mic_map = _fill_device_combo(self.mic_combo, self._all_devices.get("microphones", []))
        self.mic_combo.blockSignals(False)

        self.playback_combo.blockSignals(True)
        if sys.platform == "darwin":
            self.playback_combo.clear()
            self.playback_combo.addItem(_MACOS_PLAYBACK_LABEL)
            self._playback_map = {_MACOS_PLAYBACK_LABEL: None}
            self.playback_combo.setEnabled(False)
        else:
            self._playback_map = _fill_device_combo(self.playback_combo, self._all_devices.get("loopback", []))
            default_index = self._all_devices.get("default_loopback")
            for label, index in self._playback_map.items():
                if index == default_index:
                    self.playback_combo.setCurrentText(label)
                    break
        self.playback_combo.blockSignals(False)

        self._on_playback_changed()

    def _playback_name(self):
        """The selected playback device's full name, as recording_modes'
        device matching expects it."""
        index = self._playback_map.get(self.playback_combo.currentText())
        for d in self._all_devices.get("loopback", []):
            if d["index"] == index:
                return d["name"]
        return None

    def _playback_key(self) -> str:
        return self.playback_combo.currentText()

    def _on_playback_changed(self, *_):
        """A different playback device is a different situation: re-pick the
        Setup (the user's last choice for this device, else the guess) and,
        for a Headset, the headset's own microphone."""
        setup, match = rm.guess_setup(self._all_devices.get("microphones", []), self._playback_name())
        setup = self._saved_setups().get(self._playback_key(), setup)
        if setup not in rm.SETUPS:
            setup = rm.SETUP_LOUDSPEAKER
        self._set_setup(setup)
        if match is not None:
            self._select_mic_index(match["index"])
        self._update_device_warning()

    def _select_matching_mic(self):
        match = rm.suggest_mic_for_loopback(self._all_devices.get("microphones", []), self._playback_name())
        if match is not None:
            self._select_mic_index(match["index"])

    def _select_mic_index(self, index):
        for label, mic_index in self._mic_map.items():
            if mic_index == index:
                self.mic_combo.setCurrentText(label)
                return

    def _update_device_warning(self):
        """Headset Call whose microphone isn't part of the playback device -
        e.g. listening on one headset while the other, unworn one's mic is
        selected. Non-blocking: the user may know better."""
        warning = ""
        if self._mode == rm.MODE_CALL and self._setup == rm.SETUP_HEADSET and sys.platform == "win32":
            playback = self._playback_name()
            mic_index = self._mic_map.get(self.mic_combo.currentText())
            mic = next((d for d in self._all_devices.get("microphones", []) if d["index"] == mic_index), None)
            if playback and mic and not rm.mic_matches_playback(mic["name"], playback):
                warning = ("⚠ The microphone isn't part of the headset you're listening on. "
                           "Pick the headset's microphone, or switch the Setup to Loudspeaker.")
        self.device_warning.setText(warning)
        self.device_warning.setVisible(bool(warning))

    # --------------------------------------------------------- recording
    def _toggle_recording(self):
        if self.window_.recorder_controller.is_recording:
            self.window_.recorder_controller.stop()
        else:
            self._start_recording()

    def _start_recording(self):
        mic_index = self._mic_map.get(self.mic_combo.currentText())
        playback_index = self._playback_map.get(self.playback_combo.currentText())
        # The listed indices go stale as soon as any device is plugged in or
        # out - re-enumerate and follow the selected devices by name.
        if sys.platform == "win32":
            fresh = self.window_.recorder_controller.refresh_devices()
            mic_index = rm.current_device_index(
                self._all_devices.get("microphones", []), fresh.get("microphones", []), mic_index)
            playback_index = rm.current_device_index(
                self._all_devices.get("loopback", []), fresh.get("loopback", []), playback_index)
            has_playback = self._mode in (rm.MODE_PLAYBACK, rm.MODE_CALL)
            has_mic = self._mode in (rm.MODE_CALL, rm.MODE_ROOM)
            if (has_playback and playback_index is None and self._playback_map) or \
                    (has_mic and mic_index is None and self._mic_map):
                self.refresh_devices()
                self._set_status("The selected device is no longer connected - "
                                 "the device list has been refreshed, please check it.", theme.DANGER)
                return
        output_path = make_recording_output_path(self.window_.settings.get("output_dir"))
        rc = self.window_.recorder_controller
        if self._mode == rm.MODE_CALL:
            rc.start(output_path, rm.source_for_mode(self._mode), mic_index,
                     loopback_index=playback_index, setup=self._setup)
        elif self._mode == rm.MODE_PLAYBACK:
            rc.start(output_path, rm.source_for_mode(self._mode), playback_index)
        else:
            rc.start(output_path, rm.source_for_mode(self._mode), mic_index)

    def _set_controls_enabled(self, enabled):
        for btn in (*self.mode_buttons.values(), *self.setup_buttons.values()):
            btn.setEnabled(enabled)
        self.mic_combo.setEnabled(enabled)
        self.playback_combo.setEnabled(enabled and sys.platform != "darwin")
        self.refresh_btn.setEnabled(enabled)

    def _on_recording_started(self):
        self.record_btn.setText("  Stop recording")
        self.record_btn.setIcon(QIcon(theme.get_icons()["stop"]))
        self._set_status("Recording...")
        self._set_controls_enabled(False)
        self._start_time = time.monotonic()
        self._current_rms = {"mic": 0.0, "system": 0.0}
        self._mic_peak = 0.0
        self._mic_warning_shown = False
        self._recording_mic_name = self.mic_combo.currentText()
        self._recording_has_mic = self._mode in (rm.MODE_CALL, rm.MODE_ROOM)
        self._timer.start()

    def _on_level_updated(self, channel, rms):
        self._current_rms[channel] = rms
        if channel == "mic":
            self._mic_peak = max(self._mic_peak, rms)

    def _check_mic_signal(self, elapsed):
        """Warns if the selected microphone has delivered (near-)digital
        silence so far - typically the wrong device (an unworn second
        headset) or a hardware-muted mic. A live but quiet mic in a silent
        room still shows a noise floor well above this threshold."""
        if not self._recording_has_mic:
            return
        silent = self._mic_peak < _MIC_SILENCE_RMS
        if silent and elapsed >= _MIC_SILENCE_WARN_SECONDS and not self._mic_warning_shown:
            self._mic_warning_shown = True
            self._set_status(
                f"⚠ No signal from microphone \"{self._recording_mic_name}\" so far - "
                "wrong device selected, or muted?", theme.DANGER)
        elif not silent and self._mic_warning_shown:
            self._mic_warning_shown = False
            self._set_status("Recording...")

    def _on_recording_stopping(self):
        self._timer.stop()
        for meter in self._meters.values():
            meter.reset()
        self.record_btn.setEnabled(False)
        self.record_btn.setText("  Saving recording...")
        self.saving_bar.show()
        self._set_status(
            "Saving recording... this can take a moment for long recordings."
        )

    def _on_recording_finished(self, path, duration, error):
        self._timer.stop()
        self.saving_bar.hide()
        self.record_btn.setEnabled(True)
        self.record_btn.setText("  Start recording")
        self.record_btn.setIcon(QIcon(theme.get_icons()["record"]))
        self._set_controls_enabled(True)
        self.timer_label.setText("00:00")
        for meter in self._meters.values():
            meter.reset()

        if error:
            self._set_status(f"Error: {error}", theme.DANGER)
            return

        if path:
            mins, secs = divmod(int(round(duration or 0)), 60)
            hours, mins = divmod(mins, 60)
            length = f"{hours}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"
            self._set_status(f"✓ Done — saved: {path} ({length})", theme.SUCCESS)
            self.window_.send_to_transcription(path)
            if self.auto_transcribe_check.isChecked():
                self.window_.navigate_to("transcribe")
                self.window_.pages["transcribe"].start_transcription()

    def _set_status(self, text, color=None):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};" if color else "")

    def _tick(self):
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            mins, secs = divmod(int(elapsed), 60)
            self.timer_label.setText(f"{mins:02d}:{secs:02d}")
            self._check_mic_signal(elapsed)
        for channel, meter in self._meters.items():
            level = min(self._current_rms.get(channel, 0.0) / 15000, 1.0)
            meter.set_level(level)

    def on_page_shown(self):
        pass
