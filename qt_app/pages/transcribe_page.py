"""Seite "Transkription": Datei waehlen, Optionen, Fortschritt, Ergebnis,
Sprecher-Umbenennung, Export."""

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCharFormat, QColor, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from qt_app import theme
from qt_app.constants import (
    FORMATS,
    LANGUAGES,
    MODEL_IDS_BY_DISPLAY,
    MODELS,
    MODEL_DISPLAY_NAMES,
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
        self._build_ui()
        self._wire()
        self._on_diarize_toggled()

    # ------------------------------------------------------------------ ui
    def _build_ui(self):
        root = page_root(self)

        header = QLabel("Transkription")
        header.setProperty("role", "title")
        root.addWidget(header)

        # --- Datei + Optionen: eine Karte, da inhaltlich zusammengehoerig ---
        opts_card, opts_layout = card("AUDIO-DATEI & EINSTELLUNGEN")
        root.addWidget(opts_card)

        file_row = QHBoxLayout()
        opts_layout.addLayout(file_row)
        self.file_label = QLabel("Keine Datei ausgewählt")
        self.file_label.setProperty("role", "muted")
        file_row.addWidget(self.file_label, 1)
        self.browse_btn = QPushButton("Datei wählen...")
        file_row.addWidget(self.browse_btn)

        sep0 = QFrame()
        sep0.setProperty("role", "separator")
        opts_layout.addWidget(sep0)

        opts_row = QGridLayout()
        opts_layout.addLayout(opts_row)
        opts_row.addWidget(QLabel("Sprache:"), 0, 0)
        self.lang_combo = self._combo(list(LANGUAGES.keys()), "Automatisch erkennen")
        opts_row.addWidget(self.lang_combo, 0, 1)
        opts_row.addWidget(QLabel("Modell:"), 0, 2)
        self.model_combo = self._combo(MODELS, MODEL_DISPLAY_NAMES["server:kit.whisper-large-v3"])
        opts_row.addWidget(self.model_combo, 0, 3)
        opts_row.setColumnStretch(1, 1)
        opts_row.setColumnStretch(3, 1)

        self.hw_hint = QLabel("⚡ Ermittle Hardware-Empfehlung...")
        self.hw_hint.setProperty("role", "hint")
        opts_layout.addWidget(self.hw_hint)

        sep = QFrame()
        sep.setProperty("role", "separator")
        opts_layout.addWidget(sep)

        diar_row = QHBoxLayout()
        opts_layout.addLayout(diar_row)
        self.diarize_check = QCheckBox("Speaker Diarization")
        self.diarize_check.setChecked(True)
        diar_row.addWidget(self.diarize_check)
        diar_row.addSpacing(16)
        self.min_spk_label = QLabel("Min Sprecher:")
        diar_row.addWidget(self.min_spk_label)
        self.min_spk_edit = QLineEdit()
        self.min_spk_edit.setPlaceholderText("auto")
        self.min_spk_edit.setFixedWidth(56)
        diar_row.addWidget(self.min_spk_edit)
        diar_row.addSpacing(12)
        self.max_spk_label = QLabel("Max Sprecher:")
        diar_row.addWidget(self.max_spk_label)
        self.max_spk_edit = QLineEdit()
        self.max_spk_edit.setPlaceholderText("auto")
        self.max_spk_edit.setFixedWidth(56)
        diar_row.addWidget(self.max_spk_edit)
        diar_row.addStretch(1)

        # --- Start/Fortschritt ---
        self.start_btn = QPushButton("▶  Transkription starten")
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

        # --- Transkript ---
        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        root.addWidget(self.transcript_view, 1)

        # --- Sprecher umbenennen ---
        self.speaker_frame, speaker_outer = card()
        speaker_outer.setSpacing(6)
        self.speaker_frame.hide()
        root.addWidget(self.speaker_frame)

        speaker_header = QHBoxLayout()
        speaker_outer.addLayout(speaker_header)
        speaker_title = QLabel("Sprecher umbenennen")
        speaker_title.setStyleSheet("font-weight: 600;")
        speaker_header.addWidget(speaker_title)
        speaker_header.addStretch(1)
        self.apply_names_btn = QPushButton("Anwenden")
        speaker_header.addWidget(self.apply_names_btn)

        self.speaker_scroll = QScrollArea()
        self.speaker_scroll.setWidgetResizable(True)
        self.speaker_scroll.setFixedHeight(110)
        speaker_outer.addWidget(self.speaker_scroll)
        self.speaker_entries_widget = QWidget()
        self.speaker_entries_layout = QVBoxLayout(self.speaker_entries_widget)
        self.speaker_entries_layout.setContentsMargins(0, 0, 0, 0)
        self.speaker_entries_layout.setSpacing(4)
        self.speaker_entries_layout.addStretch(1)
        self.speaker_scroll.setWidget(self.speaker_entries_widget)

        # --- Aktionen ---
        bottom_row = QHBoxLayout()
        root.addLayout(bottom_row)
        self.save_btn = QPushButton("Speichern")
        self.save_btn.setEnabled(False)
        bottom_row.addWidget(self.save_btn)
        self.copy_btn = QPushButton("Kopieren")
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
        """Slot fuer HardwareInfoController.infoReady (siehe Docstring dort -
        vermeidet den 1-3 Minuten dauernden whisperx/torch-Import waehrend
        des Seitenaufbaus)."""
        if "error" in info:
            self.hw_hint.setText("")
        else:
            self.hw_hint.setText(f"⚡ Bei lokalem Modell empfohlen: {info['reason']}")

    def _combo(self, values, default):
        from PySide6.QtWidgets import QComboBox
        combo = QComboBox()
        combo.addItems(values)
        combo.setCurrentText(default)
        return combo

    def _wire(self):
        self.browse_btn.clicked.connect(self._browse_file)
        self.start_btn.clicked.connect(self.start_transcription)
        self.diarize_check.toggled.connect(self._on_diarize_toggled)
        self.apply_names_btn.clicked.connect(self._apply_speaker_names)
        self.save_btn.clicked.connect(self._save_transcript)
        self.copy_btn.clicked.connect(self._copy_transcript)

        tc = self.window_.transcribe_controller
        tc.progressUpdated.connect(self._on_progress)
        tc.finished.connect(self._on_finished)

    # --------------------------------------------------------------- file
    def set_audio_file(self, path: str):
        self.file_label.setText(path)
        self._on_file_changed(path)

    def _browse_file(self):
        start_dir = self.window_.settings.get("output_dir") or "recordings"
        path, _ = QFileDialog.getOpenFileName(
            self, "Audio- oder Video-Datei auswählen", start_dir,
            "Audio & Video (*.wav *.mp3 *.m4a *.flac *.ogg *.mkv *.mp4 *.mov *.webm *.avi);;"
            "Alle Dateien (*.*)")
        if path:
            self.set_audio_file(path)

    def _on_file_changed(self, path):
        valid = bool(path) and os.path.isfile(path)
        self.start_btn.setEnabled(valid)
        self.status_label.hide()
        self.file_label.setStyleSheet("" if valid else f"color: {theme.TEXT_MUTED};")

    def _on_diarize_toggled(self):
        enabled = self.diarize_check.isChecked()
        self.min_spk_edit.setEnabled(enabled)
        self.max_spk_edit.setEnabled(enabled)
        color = theme.TEXT if enabled else theme.TEXT_MUTED
        self.min_spk_label.setStyleSheet(f"color: {color};")
        self.max_spk_label.setStyleSheet(f"color: {color};")

    # --------------------------------------------------------- transcribe
    def start_transcription(self):
        audio_path = self.file_label.text()
        if not audio_path or not os.path.isfile(audio_path):
            self.status_label.show()
            self.status_label.setText("Bitte wähle eine Audio-Datei aus.")
            self.status_label.setStyleSheet(f"color: {theme.DANGER};")
            return

        self.start_btn.hide()
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.status_label.show()
        self.status_label.setStyleSheet(f"color: {theme.ACCENT};")
        self.status_label.setText("Starte Transkription...")
        self.transcript_view.clear()
        self.save_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)

        from transcriber import DEFAULT_API_BASE_URL

        lang_code = LANGUAGES.get(self.lang_combo.currentText(), "de")
        model_display = self.model_combo.currentText()
        model = MODEL_IDS_BY_DISPLAY.get(model_display, model_display)
        settings = self.window_.settings
        hf_token = settings.get("hf_token") or None
        api_key = settings.get("api_key") or os.getenv("KIT_TOOLBOX_API_KEY") or None
        api_base_url = settings.get("api_base_url") or DEFAULT_API_BASE_URL
        compute = settings.get("compute", "auto")
        device = compute if compute in ("cuda", "mps", "cpu") else None

        min_spk = self._parse_int(self.min_spk_edit.text())
        max_spk = self._parse_int(self.max_spk_edit.text())
        batch_size = self._parse_int(str(settings.get("batch_size", 16))) or 16

        self.window_.transcribe_controller.start(
            audio_path=audio_path,
            language=lang_code,
            model_size=model,
            diarize=self.diarize_check.isChecked(),
            hf_token=hf_token,
            min_speakers=min_spk,
            max_speakers=max_spk,
            batch_size=batch_size,
            device=device,
            api_key=api_key,
            api_base_url=api_base_url,
        )

    @staticmethod
    def _parse_int(text):
        try:
            return int(text)
        except (TypeError, ValueError):
            return None

    def _on_progress(self, pct, message):
        self.progress_bar.setValue(int(pct * 1000))
        self.status_label.setText(message)

    def _on_finished(self, result, error):
        self.progress_bar.hide()
        self.start_btn.show()

        if error:
            self.status_label.setStyleSheet(f"color: {theme.DANGER};")
            self.status_label.setText(f"Fehler: {error}")
            return

        self.status_label.setStyleSheet(f"color: {theme.SUCCESS};")
        self.status_label.setText("Transkription abgeschlossen!")
        self._transcription_result = result
        self.save_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self._render_transcript()
        self._populate_speaker_entries()

    # ------------------------------------------------------------- render
    def _render_transcript(self):
        """Baut den Transkript-Text komplett aus self._transcription_result
        neu auf. Bewusst kein gezieltes Text-Ersetzen bei Umbenennung (wie
        in der alten CTk-GUI) - hier ist der Ergebnis-Dict die alleinige
        Quelle der Wahrheit, das Textfeld ist reine read-only Anzeige."""
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
            self.status_label.setText("In Zwischenablage kopiert!")

    def _save_transcript(self):
        if not self._transcription_result:
            return
        self._apply_speaker_names()

        from transcriber import save_transcript
        formats = [fmt for fmt, cb in self.format_checks.items() if cb.isChecked()] or ["txt"]
        audio_path = self.file_label.text()
        base = os.path.splitext(audio_path)[0]
        saved = save_transcript(self._transcription_result, base, formats=formats)
        if saved:
            self.status_label.show()
            self.status_label.setStyleSheet(f"color: {theme.SUCCESS};")
            self.status_label.setText(f"Gespeichert: {', '.join(saved)}")

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

        for raw_id, current_label in rows:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            color = self._speaker_colors.get(raw_id, theme.TEXT)
            lbl = QLabel(f"{current_label}  →")
            lbl.setFixedWidth(150)
            lbl.setStyleSheet(f"color: {color};")
            row_layout.addWidget(lbl)

            entry = QLineEdit()
            entry.setPlaceholderText("Name eingeben")
            if raw_id != current_label:
                entry.setText(current_label)
            entry.setFixedWidth(160)
            row_layout.addWidget(entry)
            self._speaker_entries[raw_id] = entry

            if raw_id in embeddings:
                remember_check = QCheckBox("merken")
                remember_check.setChecked(True)
                row_layout.addWidget(remember_check)
                self._speaker_remember_checks[raw_id] = remember_check

            row_layout.addStretch(1)
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
        """Einzige Stelle, die Sprecher-Umbenennungen vornimmt - schreibt in
        result['segments']/['word_segments']/['speaker_id_map'] und merkt
        optional die Stimme in speaker_profiles. Danach wird der komplette
        Transkript-Text neu gerendert (siehe _render_transcript)."""
        mapping = self._get_speaker_mapping()
        if not mapping or not self._transcription_result:
            return

        import speaker_profiles

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
        self.status_label.setText("Sprecher-Namen angewendet!")

    def on_page_shown(self):
        pass
