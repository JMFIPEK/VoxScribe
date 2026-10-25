"""Page "Transcription": pick a file, options, progress, then the result:
compact header, Timeline, transcript blocks, speaker list (in-place
renaming), export."""

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTextEdit,
    QWidget,
    QApplication,
)

from qt_app import theme
from qt_app.constants import (
    APPLE_SPEECHANALYZER_MODEL,
    FORMATS,
    MODEL_DISPLAY_NAMES,
    OPENVINO_MODEL_PREFIX,
    PARAKEET_MODEL_PREFIX,
    REMOTE_MODEL_PREFIX,
    format_time_short,
)
from qt_app.pages.common import card, page_root
from qt_app.speaker_view import SpeakerView
from qt_app.timeline_model import timeline_from_result
from qt_app.transcript_view import transcript_html, transcript_text
from qt_app.widgets.speaker_list import SpeakerList
from qt_app.widgets.timeline import Timeline


class TranscribePage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self._transcription_result = None
        # Colours/ranks/shares of the current result; built once when it
        # arrives, so a rename never reshuffles the colours.
        self._speaker_view = None
        self._selected_files = []
        self._model_ids_by_display = {}
        self._hw_recommended_model = None
        self._build_ui()
        self._wire()
        self._refresh_model_choices()
        self._update_pipeline_info()

    # ------------------------------------------------------------------ ui
    def _build_ui(self):
        root = page_root(self)

        header = QLabel("Transcription")
        header.setProperty("role", "title")
        root.addWidget(header)

        # --- File + options: one card, since they belong together ---
        self.opts_card, opts_layout = card("AUDIO FILE & SETTINGS")
        root.addWidget(self.opts_card)

        # Once the result is in, the card above shrinks to this one line, so
        # the Timeline and transcript get the height.
        self.result_header, header_layout = card()
        header_row = QHBoxLayout()
        header_layout.addLayout(header_row)
        self.result_summary = QLabel("")
        header_row.addWidget(self.result_summary, 1)
        self.new_btn = QPushButton("New transcription")
        header_row.addWidget(self.new_btn)
        self.result_header.hide()
        root.addWidget(self.result_header)

        file_row = QHBoxLayout()
        opts_layout.addLayout(file_row)
        self.file_label = QLabel("No file selected")
        self.file_label.setProperty("role", "muted")
        file_row.addWidget(self.file_label, 1)
        self.browse_btn = QPushButton("Choose file(s)...")
        file_row.addWidget(self.browse_btn)

        sep0 = QFrame()
        sep0.setProperty("role", "separator")
        opts_layout.addWidget(sep0)

        # Language (auto-detect), diarization, and min/max speakers are no
        # longer exposed here - auto-detect works well enough on its own,
        # diarization is always on, and speaker count is always auto-guessed.
        # This whole card only keeps the one setting worth choosing (model/
        # provider) plus a compact readout of the pipeline that will actually
        # run, so the transcript below gets as much vertical space as possible.
        opts_row = QHBoxLayout()
        opts_layout.addLayout(opts_row)

        if sys.platform == "darwin":
            # No model choice on macOS: Apple's SpeechAnalyzer (Neural
            # Engine) is the only transcription engine there, no Whisper
            # download/inference at all (see transcriber.py::
            # transcribe_apple() and CLAUDE.md). Language also isn't
            # user-facing here: transcribe_apple() runs its own German-vs-
            # English probe (transcriber.py::detect_apple_language()) instead
            # of needing a manual picker.
            self.model_combo = None
        else:
            opts_row.addWidget(QLabel("Model:"))
            self.model_combo = self._combo([], "")
            opts_row.addWidget(self.model_combo)

        # Hardware recommendation stays right behind the dropdown (short,
        # only shown when it actually differs from the current pick);
        # the full transcription/alignment/diarization chain gets its own row
        # below since it's usually too long to share a line without truncating.
        self.recommended_hint = QLabel("")
        self.recommended_hint.setProperty("role", "hint")
        opts_row.addWidget(self.recommended_hint)
        opts_row.addStretch(1)

        self.pipeline_info = QLabel("")
        self.pipeline_info.setProperty("role", "hint")
        opts_layout.addWidget(self.pipeline_info)

        # --- Start/progress ---
        self.start_btn = QPushButton("▶  Start transcription")
        self.start_btn.setProperty("role", "success")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setEnabled(False)
        root.addWidget(self.start_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.hide()
        root.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setProperty("role", "muted")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.hide()
        root.addWidget(self.status_label)

        self.timeline = Timeline()
        self.timeline.hide()
        root.addWidget(self.timeline)

        # --- Transcript + speaker list: side by side rather than stacked,
        # so neither has to fight the other for height (previously the
        # speaker card left the transcript text field almost no visible
        # space once more than 1-2 speakers were detected). Both areas use
        # the page's full available height this way. ---
        self.content_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self.content_splitter, 1)

        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.content_splitter.addWidget(self.transcript_view)

        self.speaker_frame, speaker_outer = card()
        speaker_outer.setSpacing(6)
        self.speaker_frame.hide()
        self.content_splitter.addWidget(self.speaker_frame)

        self.content_splitter.setStretchFactor(0, 3)
        self.content_splitter.setStretchFactor(1, 1)
        self.content_splitter.setSizes([700, 260])

        self.speaker_list = SpeakerList()
        speaker_outer.addWidget(self.speaker_list, 1)

        # --- Actions ---
        bottom_row = QHBoxLayout()
        root.addLayout(bottom_row)
        self.save_btn = QPushButton("Save")
        self.save_btn.setEnabled(False)
        bottom_row.addWidget(self.save_btn)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setEnabled(False)
        bottom_row.addWidget(self.copy_btn)
        bottom_row.addSpacing(12)
        bottom_row.addWidget(QLabel("Format:"))
        self.format_checks = {}
        for fmt in FORMATS:
            cb = QCheckBox(fmt.upper())
            cb.setChecked(fmt == "txt")
            bottom_row.addWidget(cb)
            self.format_checks[fmt] = cb
        bottom_row.addStretch(1)

    def apply_hardware_info(self, info: dict):
        """Slot for HardwareInfoController.infoReady (see docstring there -
        avoids the 1-3 minute whisperx/torch import while building the page)."""
        self._hw_recommended_model = None if "error" in info else info.get("recommended_model")
        self._update_pipeline_info()

    def _current_model_id(self) -> str:
        if self.model_combo is None:
            return APPLE_SPEECHANALYZER_MODEL
        model_display = self.model_combo.currentText()
        return self._model_ids_by_display.get(model_display, model_display)

    def _update_pipeline_info(self):
        """Compact, always-visible readout of what will actually run for each
        of the three pipeline phases (transcription -> alignment ->
        diarization, see CLAUDE.md), instead of exposing individual model
        names as separate dropdowns. Language is always auto-detected and
        diarization/speaker-count are always on/auto now - nothing left to
        configure there, so this label is purely informational."""
        model = self._current_model_id()

        if model == APPLE_SPEECHANALYZER_MODEL:
            # No wav2vec2 alignment step on this path - word-level timestamps
            # already come from SpeechAnalyzer itself, see CLAUDE.md.
            text = "Transcription: Apple SpeechAnalyzer (Neural Engine)  ·  Diarization: pyannote (MPS)"
        elif model.startswith(REMOTE_MODEL_PREFIX):
            text = "Transcription: remote provider  ·  Alignment: wav2vec2 (local)  ·  Diarization: pyannote (local)"
        elif model.startswith(OPENVINO_MODEL_PREFIX):
            text = "Transcription: Intel Arc GPU (OpenVINO)  ·  Alignment: wav2vec2  ·  Diarization: pyannote"
        elif model.startswith(PARAKEET_MODEL_PREFIX):
            text = "Transcription: Parakeet v3 (CPU)  ·  Alignment: wav2vec2  ·  Diarization: pyannote"
        else:
            text = f"Transcription: {model or '…'} (local)  ·  Alignment: wav2vec2  ·  Diarization: pyannote"

        self.pipeline_info.setText(text)
        self.pipeline_info.setToolTip(text)

        recommended_text = ""
        if self.model_combo is not None and self._hw_recommended_model and self._hw_recommended_model != model:
            recommended_display = MODEL_DISPLAY_NAMES.get(
                self._hw_recommended_model, self._hw_recommended_model)
            recommended_text = f"⚡ recommended: {recommended_display}"
        self.recommended_hint.setText(recommended_text)

    def _combo(self, values, default):
        from PySide6.QtWidgets import QComboBox
        combo = QComboBox()
        combo.addItems(values)
        if default:
            combo.setCurrentText(default)
        return combo

    def _refresh_model_choices(self):
        """Rebuilds the model dropdown from the local model list plus every
        configured provider (see providers.py) - called on page show since
        providers can change on the Settings page at any time."""
        if self.model_combo is None:
            return
        import voxscribe.providers as providers

        previous = self.model_combo.currentText()
        self._model_ids_by_display = dict(MODEL_DISPLAY_NAMES)
        display_names = list(MODEL_DISPLAY_NAMES.values())

        provider_list = providers.list_providers()
        default_provider = providers.get_default_provider()
        default_display = None
        for p in provider_list:
            display = f"{p['name']} (server)"
            model_id = f"{REMOTE_MODEL_PREFIX}{p['id']}"
            self._model_ids_by_display[display] = model_id
            display_names.append(display)
            if default_provider and p["id"] == default_provider["id"]:
                default_display = display

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(display_names)
        if previous in display_names:
            self.model_combo.setCurrentText(previous)
        elif default_display:
            self.model_combo.setCurrentText(default_display)
        elif display_names:
            self.model_combo.setCurrentIndex(0)
        self.model_combo.blockSignals(False)
        self._update_pipeline_info()

    def _wire(self):
        self.browse_btn.clicked.connect(self._browse_file)
        self.start_btn.clicked.connect(self.start_transcription)
        if self.model_combo is not None:
            self.model_combo.currentTextChanged.connect(self._update_pipeline_info)
        self.speaker_list.renameRequested.connect(self._rename_speaker)
        self.new_btn.clicked.connect(self._show_settings)
        self.save_btn.clicked.connect(self._save_transcript)
        self.copy_btn.clicked.connect(self._copy_transcript)

        tc = self.window_.transcribe_controller
        tc.progressUpdated.connect(self._on_progress)
        tc.finished.connect(self._on_finished)

    # --------------------------------------------------------------- file
    def set_audio_file(self, path: str):
        """Called externally (RecordPage, after a finished recording) -
        replaces the selection with exactly this one file."""
        self._selected_files = [path] if path else []
        self._update_file_display()

    def _browse_file(self):
        start_dir = self.window_.settings.get("output_dir") or "recordings"
        # One file only; several files are merged by the CLI (`-i` repeated).
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio or video file", start_dir,
            "Audio & video (*.wav *.mp3 *.m4a *.flac *.ogg *.mkv *.mp4 *.mov *.webm *.avi);;"
            "All files (*.*)")
        if path:
            self._selected_files = [path]
            self._update_file_display()

    def _update_file_display(self):
        files = self._selected_files
        if not files:
            self.file_label.setText("No file selected")
            self.file_label.setToolTip("")
        else:
            self.file_label.setText(files[0])
            self.file_label.setToolTip(files[0])
        self._on_files_changed()

    def _on_files_changed(self):
        valid = bool(self._selected_files) and all(os.path.isfile(p) for p in self._selected_files)
        self.start_btn.setEnabled(valid)
        self.status_label.hide()
        self.file_label.setStyleSheet("" if valid else f"color: {theme.TEXT_MUTED};")

    # --------------------------------------------------------- transcribe
    def start_transcription(self):
        files = self._selected_files
        if not files or not all(os.path.isfile(p) for p in files):
            self.status_label.show()
            self.status_label.setText("Please select an audio file.")
            self.status_label.setStyleSheet(f"color: {theme.DANGER};")
            return

        self.start_btn.hide()
        # Indeterminate progress (no min/max) until the first real percentage
        # comes in - see _on_progress(). Model/engine load time (whisperx/
        # torch import, model weights) can take 1-3 minutes, especially on
        # the first run, DURING which there's no progress value yet; without
        # this indicator the app looks frozen during that time (this import
        # deliberately runs in TranscribeController's background thread, not
        # here synchronously - the GUI stays responsive, but without visual
        # feedback it still looked like a freeze).
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.status_label.show()
        self.status_label.setStyleSheet(f"color: {theme.ACCENT};")
        self.status_label.setText("Initializing transcription engine... (can take 1-2 minutes on first run)")
        self.transcript_view.clear()
        self.timeline.hide()
        self.speaker_frame.hide()
        self.save_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)

        model = self._current_model_id()
        settings = self.window_.settings
        hf_token = settings.get("hf_token") or None
        compute = settings.get("compute", "auto")
        device = compute if compute in ("cuda", "xpu", "mps", "cpu") else None
        batch_size = self._parse_int(str(settings.get("batch_size", 16))) or 16

        # Language auto-detect, diarization, and speaker count auto-guessing
        # are no longer user-configurable here - see _update_pipeline_info()'s
        # docstring for why (macOS: transcriber.detect_apple_language() makes
        # None a real auto-detect there too, not just a silent English
        # fallback).
        common_kwargs = dict(
            language=None,
            model_size=model,
            diarize=True,
            hf_token=hf_token,
            min_speakers=None,
            max_speakers=None,
            batch_size=batch_size,
            device=device,
        )

        self.window_.transcribe_controller.start(audio_path=files[0], **common_kwargs)

    @staticmethod
    def _parse_int(text):
        try:
            return int(text)
        except (TypeError, ValueError):
            return None

    def _on_progress(self, pct, message):
        if self.progress_bar.maximum() == 0:
            # First real progress value - switch from the indeterminate
            # "loading bar" (see start_transcription()) to a normal percentage.
            self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(int(pct * 1000))
        self.status_label.setText(message)

    def _on_finished(self, result, error):
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 1000)
        self.start_btn.show()

        if error:
            self.status_label.setStyleSheet(f"color: {theme.DANGER};")
            self.status_label.setText(f"Error: {error}")
            return

        self.status_label.setStyleSheet(f"color: {theme.SUCCESS};")
        self.status_label.setText("Transcription complete!")
        self._transcription_result = result
        self._speaker_view = SpeakerView(result, theme.SPEAKER_COLOR_PALETTE)
        self.save_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self._render_transcript()
        self._populate_speaker_list()
        self._show_result_header()

    # ------------------------------------------------------------- header
    def _show_result_header(self):
        """Shrinks the settings card to: file name · N speakers · length."""
        name = os.path.basename(self._selected_files[0]) if self._selected_files else ""
        count = len(self._speaker_view.speakers())
        parts = [name, f"{count} speaker{'s' if count != 1 else ''}",
                 format_time_short(self.timeline.timeline().duration)]
        self.result_summary.setText("  ·  ".join(p for p in parts if p))
        self.result_summary.setToolTip(self._selected_files[0] if self._selected_files else "")
        self.opts_card.hide()
        self.start_btn.hide()
        self.status_label.hide()
        self.result_header.show()

    def _show_settings(self):
        """"New transcription": brings the settings card back; the last
        result stays visible until the next one starts."""
        self.result_header.hide()
        self.opts_card.show()
        self.start_btn.show()

    # ------------------------------------------------------------- render
    def _render_transcript(self):
        """Rebuilds Timeline and transcript entirely from
        self._transcription_result. Deliberately no targeted text replacement
        on rename - the result dict is the single source of truth here, both
        are pure read-only displays."""
        result = self._transcription_result
        if not result:
            return

        view = self._speaker_view
        note = "" if view.speakers() else "No speakers identified"
        self.timeline.set_timeline(timeline_from_result(result, view, theme.TEXT_MUTED, note))
        self.timeline.show()
        self.transcript_view.setHtml(transcript_html(result, view))

    def _copy_transcript(self):
        text = transcript_text(self._transcription_result) if self._transcription_result else ""
        if text:
            QApplication.clipboard().setText(text)
            self.status_label.show()
            self.status_label.setStyleSheet(f"color: {theme.SUCCESS};")
            self.status_label.setText("Copied to clipboard!")

    def _save_transcript(self):
        if not self._transcription_result:
            return

        from voxscribe.transcriber import save_transcript
        formats = [fmt for fmt, cb in self.format_checks.items() if cb.isChecked()] or ["txt"]
        first_file = self._selected_files[0] if self._selected_files else "transcript"
        base = os.path.splitext(first_file)[0]
        saved = save_transcript(self._transcription_result, base, formats=formats)
        if saved:
            self.status_label.show()
            self.status_label.setStyleSheet(f"color: {theme.SUCCESS};")
            self.status_label.setText(f"Saved: {', '.join(saved)}")

    # -------------------------------------------------------- speaker list
    def _populate_speaker_list(self):
        view = self._speaker_view
        if view is None or not view.speakers():
            self.speaker_frame.hide()
            return
        self.speaker_list.show_view(view)
        self.speaker_frame.show()

    def _rename_speaker(self, label, name):
        """Applies a name typed in the speaker list: relabels the result (a
        name another speaker already has merges the two, see
        SpeakerView.rename()), remembers the voice unless it is too short or
        has no embedding, and re-renders transcript and list right away."""
        name = name.strip()
        if name and name != label:
            import voxscribe.speaker_profiles as speaker_profiles

            # Only this speaker's own voice prints: on a merge the other
            # speaker's are already in the profile.
            voice_prints = self._speaker_view.voice_prints(label)
            self._speaker_view.rename(label, name)
            for embedding in voice_prints:
                speaker_profiles.enroll_speaker(name, embedding)
            self._render_transcript()
        self._populate_speaker_list()

    def on_page_shown(self):
        self._refresh_model_choices()
