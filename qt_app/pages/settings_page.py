"""Seite "Einstellungen": Zugangsdaten, Aufnahme-Ordner, Performance, und
(als eigener Abschnitt am Ende) die Info-Inhalte (Modelle/Lizenzen) - bewusst
kein eigener Tab mehr dafuer."""

import os
import sys

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from qt_app.pages.common import PAGE_MARGINS, PAGE_SPACING
from qt_app.pages.info_content import INFO_TEXT


class SettingsPage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self._build_ui()

    # Fallback-Konstante, damit fuer dieses Feld kein `import transcriber`
    # (laedt whisperx/torch, kalt 1-3 Minuten) beim Seitenaufbau noetig ist -
    # siehe apply_hardware_info() fuer den Rest der Hardware-/Bundled-Infos,
    # die tatsaechlich asynchron ermittelt werden muessen.
    _FALLBACK_API_BASE_URL = "https://ki-toolbox.scc.kit.edu/api/v1"

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(*PAGE_MARGINS)
        root.setSpacing(PAGE_SPACING)

        header = QLabel("Einstellungen")
        header.setProperty("role", "title")
        root.addWidget(header)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)
        row = 0

        grid.addWidget(self._bold("HuggingFace Token:"), row, 0)
        self.hf_entry = QLineEdit(self.window_.settings.get("hf_token", ""))
        self.hf_entry.setEchoMode(QLineEdit.Password)
        self.hf_entry.textChanged.connect(lambda v: self.window_.settings.__setitem__("hf_token", v))
        grid.addWidget(self.hf_entry, row, 1)
        row += 1

        show_hf = QCheckBox("Token anzeigen")
        show_hf.toggled.connect(
            lambda checked: self.hf_entry.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password))
        grid.addWidget(show_hf, row, 1)
        row += 1

        self.hf_hint_label = self._muted("Für Speaker Diarization benötigt (wird ermittelt...).")
        grid.addWidget(self.hf_hint_label, row, 0, 1, 2)
        row += 1

        grid.addWidget(self._separator(), row, 0, 1, 2)
        row += 1

        grid.addWidget(self._bold("KIT ToolBox API-Key:"), row, 0)
        self.api_key_entry = QLineEdit(self.window_.settings.get("api_key", ""))
        self.api_key_entry.setEchoMode(QLineEdit.Password)
        self.api_key_entry.textChanged.connect(lambda v: self.window_.settings.__setitem__("api_key", v))
        grid.addWidget(self.api_key_entry, row, 1)
        row += 1

        show_api = QCheckBox("API-Key anzeigen")
        show_api.toggled.connect(
            lambda checked: self.api_key_entry.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password))
        grid.addWidget(show_api, row, 1)
        row += 1

        grid.addWidget(self._bold("KIT ToolBox Basis-URL:"), row, 0)
        default_url = (self.window_.settings.get("api_base_url") or os.getenv("KIT_TOOLBOX_BASE_URL")
                       or self._FALLBACK_API_BASE_URL)
        self.window_.settings["api_base_url"] = default_url
        self.api_url_entry = QLineEdit(default_url)
        self.api_url_entry.textChanged.connect(lambda v: self.window_.settings.__setitem__("api_base_url", v))
        grid.addWidget(self.api_url_entry, row, 1)
        row += 1

        grid.addWidget(self._muted(
            "Für das Server-Modell „KIT ToolBox“ benötigt. Audio wird dafür\n"
            "an den KIT-Server übertragen (nicht mehr 100% lokal)."), row, 0, 1, 2)
        row += 1

        grid.addWidget(self._separator(), row, 0, 1, 2)
        row += 1

        grid.addWidget(self._bold("Aufnahme-Ordner:"), row, 0)
        self.output_dir_entry = QLineEdit(self.window_.settings.get("output_dir", "recordings"))
        self.output_dir_entry.textChanged.connect(lambda v: self.window_.settings.__setitem__("output_dir", v))
        grid.addWidget(self.output_dir_entry, row, 1)
        row += 1

        grid.addWidget(self._bold("Batch Size:"), row, 0)
        self.batch_size_entry = QLineEdit(str(self.window_.settings.get("batch_size", 16)))
        self.batch_size_entry.setFixedWidth(80)
        self.batch_size_entry.textChanged.connect(self._on_batch_size_changed)
        grid.addWidget(self.batch_size_entry, row, 1)
        row += 1

        grid.addWidget(self._bold("Compute:"), row, 0)
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["Auto (CUDA/MPS wenn verfügbar)", "cuda", "mps", "cpu"])
        self.compute_combo.currentTextChanged.connect(self._on_compute_changed)
        grid.addWidget(self.compute_combo, row, 1)
        row += 1

        self.hw_summary_label = self._muted("Hardware: wird ermittelt...")
        root.addWidget(self.hw_summary_label)
        if sys.platform == "darwin":
            # Auf macOS gibt es keine Whisper-Modellwahl mehr (siehe
            # Kommentar bei self.model_combo in transcribe_page.py) - eine
            # "empfohlene Modellgroesse" ergibt hier keinen Sinn.
            self.recommended_model_label = None
        else:
            self.recommended_model_label = self._muted("Empfohlenes Modell: wird ermittelt...")
            root.addWidget(self.recommended_model_label)

        root.addWidget(self._separator())
        info_header = QLabel("Info")
        info_header.setProperty("role", "title")
        root.addWidget(info_header)
        info_sub = QLabel("Funktionsumfang, verwendete Modelle & Lizenzen")
        info_sub.setProperty("role", "muted")
        root.addWidget(info_sub)

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setFontFamily("Consolas")
        info_text.setPlainText(INFO_TEXT)
        info_text.setMinimumHeight(320)
        root.addWidget(info_text)

    def apply_hardware_info(self, info: dict):
        """Slot fuer HardwareInfoController.infoReady - fuellt die
        Platzhalter, sobald die (im Hintergrund ermittelte) Hardware-/
        Bundled-Info da ist."""
        if "error" in info:
            self.hw_summary_label.setText("Hardware: konnte nicht ermittelt werden")
            if self.recommended_model_label is not None:
                self.recommended_model_label.setText("Empfohlenes Modell: ?")
            return

        self.hw_summary_label.setText(f"Hardware: {info['hw_summary']}")
        if self.recommended_model_label is not None:
            self.recommended_model_label.setText(f"Empfohlenes Modell: {info['recommended_model']}")
        if info["diarize_bundled"]:
            self.hf_hint_label.setText("Diarization-Modelle sind integriert — kein Token nötig.")
        else:
            self.hf_hint_label.setText(
                "Für Speaker Diarization benötigt.\nErstelle einen Read-Token auf huggingface.co/settings/tokens")

    def _on_batch_size_changed(self, value):
        try:
            self.window_.settings["batch_size"] = int(value)
        except ValueError:
            pass

    def _on_compute_changed(self, value):
        self.window_.settings["compute"] = value if value in ("cuda", "mps", "cpu") else "auto"

    @staticmethod
    def _bold(text):
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: 600;")
        return lbl

    @staticmethod
    def _muted(text):
        lbl = QLabel(text)
        lbl.setProperty("role", "muted")
        lbl.setWordWrap(True)
        return lbl

    @staticmethod
    def _separator():
        sep = QFrame()
        sep.setProperty("role", "separator")
        return sep

    def on_page_shown(self):
        pass
