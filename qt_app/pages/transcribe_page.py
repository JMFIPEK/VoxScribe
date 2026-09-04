"""Page "Transcription": pick file(s), options, progress, result, speaker
rename, export."""

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCharFormat, QColor, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from qt_app import theme
from qt_app.constants import (
    APPLE_SPEECHANALYZER_MODEL,
    FORMATS,
    MODEL_DISPLAY_NAMES,
    OPENVINO_MODEL_PREFIX,
    REMOTE_MODEL_PREFIX,
    format_time_short,
)
from qt_app.pages.common import card, page_root


class TranscribePage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self._transcription_result = None
        self._speaker_colors = {}
        self._speaker_entries = {}
        self._speaker_remember_checks = {}
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
        opts_card, opts_layout = card("AUDIO FILE & SETTINGS")
        root.addWidget(opts_card)

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
            # transcribe_apple() and CLAUDE.md).
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

        # --- Transcript + speaker rename: side by side rather than stacked,
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

        speaker_header = QHBoxLayout()
        speaker_outer.addLayout(speaker_header)
        speaker_title = QLabel("Rename speakers")
        speaker_title.setStyleSheet("font-weight: 600;")
        speaker_header.addWidget(speaker_title)
        speaker_header.addStretch(1)
        self.apply_names_btn = QPushButton("Apply")
        speaker_header.addWidget(self.apply_names_btn)

        self.speaker_scroll = QScrollArea()
        self.speaker_scroll.setWidgetResizable(True)
        speaker_outer.addWidget(self.speaker_scroll, 1)
        self.speaker_entries_widget = QWidget()
        self.speaker_entries_layout = QVBoxLayout(self.speaker_entries_widget)
        self.speaker_entries_layout.setContentsMargins(0, 0, 0, 0)
        self.speaker_entries_layout.setSpacing(4)
        self.speaker_entries_layout.addStretch(1)
        self.speaker_scroll.setWidget(self.speaker_entries_widget)

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
        self.apply_names_btn.clicked.connect(self._apply_speaker_names)
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
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select audio or video file(s)", start_dir,
            "Audio & video (*.wav *.mp3 *.m4a *.flac *.ogg *.mkv *.mp4 *.mov *.webm *.avi);;"
            "All files (*.*)")
        if paths:
            self._selected_files = paths
            self._update_file_display()

    def _update_file_display(self):
        files = self._selected_files
        if not files:
            self.file_label.setText("No file selected")
            self.file_label.setToolTip("")
        elif len(files) == 1:
            self.file_label.setText(files[0])
            self.file_label.setToolTip(files[0])
        else:
            self.file_label.setText(f"{len(files)} files selected (will be merged)")
            self.file_label.setToolTip("\n".join(files))
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
        # docstring for why.
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

        # With multiple files: transcribe_multi() (see transcriber.py)
        # transcribes them in sequence and merges the results into one
        # continuous transcript (timestamps shifted, per-file speaker label
        # disambiguation - see its docstring). With exactly one file, still
        # deliberately the normal single-file path, so behavior/labels stay
        # unchanged from before this feature existed.
        if len(files) == 1:
            self.window_.transcribe_controller.start(audio_path=files[0], **common_kwargs)
        else:
            self.window_.transcribe_controller.start_multi(audio_paths=files, **common_kwargs)

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
        self.save_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self._render_transcript()
        self._populate_speaker_entries()

    # ------------------------------------------------------------- render
    def _render_transcript(self):
        """Rebuilds the transcript text entirely from self._transcription_result.
        Deliberately no targeted text replacement on rename - the result dict
        is the single source of truth here, the text field is a pure
        read-only display."""
        result = self._transcription_result
        if not result:
            return
        speaker_id_map = result.get("speaker_id_map") or {}
        label_to_raw_id = {label: raw_id for raw_id, label in speaker_id_map.items()}

        labels = sorted({
            seg.get("speaker") for seg in result.get("segments", [])
            if seg.get("speaker")
        })
        self._speaker_colors = {
            label_to_raw_id.get(label, label): theme.SPEAKER_COLOR_PALETTE[i % len(theme.SPEAKER_COLOR_PALETTE)]
            for i, label in enumerate(labels)
        }

        self.transcript_view.clear()
        cursor = self.transcript_view.textCursor()
        default_fmt = QTextCharFormat()
        default_fmt.setForeground(QColor(theme.TEXT))

        for seg in result.get("segments", []):
            start = format_time_short(seg.get("start", 0))
            end = format_time_short(seg.get("end", 0))
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")

            cursor.movePosition(QTextCursor.End)
            cursor.setCharFormat(default_fmt)
            cursor.insertText(f"[{start} - {end}] ")

            if speaker:
                raw_id = label_to_raw_id.get(speaker, speaker)
                color = self._speaker_colors.get(raw_id, theme.TEXT)
                speaker_fmt = QTextCharFormat()
                speaker_fmt.setForeground(QColor(color))
                cursor.setCharFormat(speaker_fmt)
                cursor.insertText(f"{speaker}: ")
                cursor.setCharFormat(default_fmt)
                cursor.insertText(f"{text}\n")
            else:
                cursor.insertText(f"{text}\n")

    def _copy_transcript(self):
        text = self.transcript_view.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self.status_label.show()
            self.status_label.setStyleSheet(f"color: {theme.SUCCESS};")
            self.status_label.setText("Copied to clipboard!")

    def _save_transcript(self):
        if not self._transcription_result:
            return
        self._apply_speaker_names()

        from voxscribe.transcriber import save_transcript
        formats = [fmt for fmt, cb in self.format_checks.items() if cb.isChecked()] or ["txt"]
        first_file = self._selected_files[0] if self._selected_files else "transcript"
        base = os.path.splitext(first_file)[0]
        if len(self._selected_files) > 1:
            base += "_combined"
        saved = save_transcript(self._transcription_result, base, formats=formats)
        if saved:
            self.status_label.show()
            self.status_label.setStyleSheet(f"color: {theme.SUCCESS};")
            self.status_label.setText(f"Saved: {', '.join(saved)}")

    # ------------------------------------------------------ speaker rename
    def _populate_speaker_entries(self):
        while self.speaker_entries_layout.count() > 1:
            item = self.speaker_entries_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._speaker_entries = {}
        self._speaker_remember_checks = {}

        result = self._transcription_result
        if not result:
            self.speaker_frame.hide()
            return

        speaker_id_map = result.get("speaker_id_map")
        embeddings = result.get("speaker_embeddings") or {}

        if speaker_id_map:
            rows = sorted(speaker_id_map.items(), key=lambda kv: kv[1])
        else:
            labels = sorted({
                seg.get("speaker") for seg in result.get("segments", [])
                if seg.get("speaker")
            })
            rows = [(label, label) for label in labels]

        if not rows:
            self.speaker_frame.hide()
            return

        # Each row: label + "remember" checkbox on top, input field below
        # (full width) - fits the narrow column next to the transcript this
        # way (instead of one wide row with label+input+checkbox side by side).
        for raw_id, current_label in rows:
            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 6)
            row_layout.setSpacing(2)

            top_row = QHBoxLayout()
            row_layout.addLayout(top_row)

            color = self._speaker_colors.get(raw_id, theme.TEXT)
            lbl = QLabel(current_label)
            lbl.setStyleSheet(f"color: {color};")
            top_row.addWidget(lbl)
            top_row.addStretch(1)

            if raw_id in embeddings:
                remember_check = QCheckBox("remember")
                remember_check.setChecked(True)
                top_row.addWidget(remember_check)
                self._speaker_remember_checks[raw_id] = remember_check

            entry = QLineEdit()
            entry.setPlaceholderText("Enter name")
            if raw_id != current_label:
                entry.setText(current_label)
            row_layout.addWidget(entry)
            self._speaker_entries[raw_id] = entry

            self.speaker_entries_layout.insertWidget(
                self.speaker_entries_layout.count() - 1, row_widget)

        self.speaker_frame.show()

    def _get_speaker_mapping(self):
        mapping = {}
        for raw_id, entry in self._speaker_entries.items():
            name = entry.text().strip()
            if name:
                mapping[raw_id] = name
        return mapping

    def _apply_speaker_names(self):
        """The single place that applies speaker renames - writes to
        result['segments']/['word_segments']/['speaker_id_map'] and
        optionally remembers the voice in speaker_profiles. The full
        transcript text is then re-rendered (see _render_transcript)."""
        mapping = self._get_speaker_mapping()
        if not mapping or not self._transcription_result:
            return

        import voxscribe.speaker_profiles as speaker_profiles

        result = self._transcription_result
        speaker_id_map = result.setdefault("speaker_id_map", {})
        embeddings = result.get("speaker_embeddings") or {}

        for raw_id, new_name in mapping.items():
            old_label = speaker_id_map.get(raw_id, raw_id)
            if old_label == new_name:
                continue

            for seg in result.get("segments", []):
                if seg.get("speaker") == old_label:
                    seg["speaker"] = new_name
                for word in seg.get("words", []) or []:
                    if word.get("speaker") == old_label:
                        word["speaker"] = new_name
            for word in result.get("word_segments", []) or []:
                if word.get("speaker") == old_label:
                    word["speaker"] = new_name

            speaker_id_map[raw_id] = new_name

            remember_check = self._speaker_remember_checks.get(raw_id)
            if remember_check is not None and remember_check.isChecked() and raw_id in embeddings:
                speaker_profiles.enroll_speaker(new_name, embeddings[raw_id])

        self._render_transcript()
        self._populate_speaker_entries()
        self.status_label.show()
        self.status_label.setStyleSheet(f"color: {theme.SUCCESS};")
        self.status_label.setText("Speaker names applied!")

    def on_page_shown(self):
        self._refresh_model_choices()
