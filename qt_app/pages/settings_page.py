"""Settings page: providers, HuggingFace token, recording folder, performance,
and (as its own section at the bottom) the Info content (models/licenses) -
deliberately not its own tab."""

import sys
import threading

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from qt_app import theme
from qt_app.pages.common import PAGE_MARGINS, PAGE_SPACING
from qt_app.pages.info_content import INFO_TEXT


class ProviderDialog(QDialog):
    """Add/edit dialog for a single transcription provider (OpenAI-compatible
    endpoint: name, base URL, API key, model)."""

    # Runs on a background thread (network call) - Qt auto-marshals a signal
    # emitted off the GUI thread to the connected slot, same pattern as
    # qt_app/controllers.py.
    _test_finished = Signal(bool, str)

    def __init__(self, parent=None, provider=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Provider" if provider else "Add Provider")
        self.setMinimumWidth(420)

        form = QFormLayout(self)

        self.name_entry = QLineEdit(provider["name"] if provider else "")
        self.name_entry.setPlaceholderText("e.g. My Whisper Server")
        form.addRow("Name:", self.name_entry)

        self.base_url_entry = QLineEdit(provider["base_url"] if provider else "")
        self.base_url_entry.setPlaceholderText("https://example.com/api/v1")
        form.addRow("Base URL:", self.base_url_entry)

        self.api_key_entry = QLineEdit(provider["api_key"] if provider else "")
        self.api_key_entry.setEchoMode(QLineEdit.Password)
        form.addRow("API key:", self.api_key_entry)

        show_key = QCheckBox("Show API key")
        show_key.toggled.connect(
            lambda checked: self.api_key_entry.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password))
        form.addRow("", show_key)

        self.model_entry = QLineEdit(provider["model"] if provider else "")
        self.model_entry.setPlaceholderText("e.g. whisper-large-v3")
        form.addRow("Model:", self.model_entry)

        test_row = QWidget()
        test_row_layout = QHBoxLayout(test_row)
        test_row_layout.setContentsMargins(0, 0, 0, 0)
        self.test_btn = QPushButton("Test")
        self.test_btn.setToolTip(
            "Sends a tiny silent test clip to the endpoint above to check "
            "the URL/API key/model actually work")
        test_row_layout.addWidget(self.test_btn)
        self.test_status = QLabel("")
        self.test_status.setWordWrap(True)
        test_row_layout.addWidget(self.test_status, 1)
        form.addRow("", test_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.test_btn.clicked.connect(self._on_test)
        self._test_finished.connect(self._on_test_finished)

    def _on_test(self):
        base_url = self.base_url_entry.text().strip()
        api_key = self.api_key_entry.text().strip()
        model = self.model_entry.text().strip()

        self.test_btn.setEnabled(False)
        self.test_status.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self.test_status.setText("Testing...")

        def _run():
            import voxscribe.providers as providers
            ok, msg = providers.test_provider(base_url, api_key, model)
            self._test_finished.emit(ok, msg)

        threading.Thread(target=_run, daemon=True).start()

    def _on_test_finished(self, ok: bool, message: str):
        self.test_btn.setEnabled(True)
        self.test_status.setStyleSheet(f"color: {theme.SUCCESS if ok else theme.DANGER};")
        self.test_status.setText(message)

    def _on_accept(self):
        if not self.name_entry.text().strip() or not self.base_url_entry.text().strip():
            QMessageBox.warning(self, "Missing information", "Name and Base URL are required.")
            return
        self.accept()

    def values(self) -> dict:
        return {
            "name": self.name_entry.text().strip(),
            "base_url": self.base_url_entry.text().strip(),
            "api_key": self.api_key_entry.text().strip(),
            "model": self.model_entry.text().strip(),
        }


class SettingsPage(QWidget):
    def __init__(self, window):
        super().__init__()
        self.window_ = window
        self._build_ui()
        self._reload_providers()

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

        header = QLabel("Settings")
        header.setProperty("role", "title")
        root.addWidget(header)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)
        row = 0

        grid.addWidget(self._bold("HuggingFace token:"), row, 0)
        self.hf_entry = QLineEdit(self.window_.settings.get("hf_token", ""))
        self.hf_entry.setEchoMode(QLineEdit.Password)
        self.hf_entry.textChanged.connect(lambda v: self.window_.settings.__setitem__("hf_token", v))
        grid.addWidget(self.hf_entry, row, 1)
        row += 1

        show_hf = QCheckBox("Show token")
        show_hf.toggled.connect(
            lambda checked: self.hf_entry.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password))
        grid.addWidget(show_hf, row, 1)
        row += 1

        self.hf_hint_label = self._muted("Required for speaker diarization (checking...).")
        grid.addWidget(self.hf_hint_label, row, 0, 1, 2)
        row += 1

        grid.addWidget(self._bold("Recording folder:"), row, 0)
        self.output_dir_entry = QLineEdit(self.window_.settings.get("output_dir", "recordings"))
        self.output_dir_entry.textChanged.connect(lambda v: self.window_.settings.__setitem__("output_dir", v))
        grid.addWidget(self.output_dir_entry, row, 1)
        row += 1

        grid.addWidget(self._bold("Batch size:"), row, 0)
        self.batch_size_entry = QLineEdit(str(self.window_.settings.get("batch_size", 16)))
        self.batch_size_entry.setFixedWidth(80)
        self.batch_size_entry.textChanged.connect(self._on_batch_size_changed)
        grid.addWidget(self.batch_size_entry, row, 1)
        row += 1

        grid.addWidget(self._bold("Compute:"), row, 0)
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["Auto (CUDA/XPU/MPS if available)", "cuda", "xpu", "mps", "cpu"])
        self.compute_combo.currentTextChanged.connect(self._on_compute_changed)
        grid.addWidget(self.compute_combo, row, 1)
        row += 1

        self.hw_summary_label = self._muted("Hardware: checking...")
        root.addWidget(self.hw_summary_label)
        if sys.platform == "darwin":
            # No local Whisper model choice on macOS (see the matching
            # comment on self.model_combo in transcribe_page.py) - a
            # "recommended model size" doesn't make sense here.
            self.recommended_model_label = None
        else:
            self.recommended_model_label = self._muted("Recommended model: checking...")
            root.addWidget(self.recommended_model_label)

        root.addWidget(self._separator())

        # --- Providers ---------------------------------------------------
        prov_header = QLabel("Providers")
        prov_header.setProperty("role", "title")
        root.addWidget(prov_header)
        prov_sub = self._muted(
            "Add any OpenAI-compatible transcription endpoint (name, base URL, "
            "API key, model). Audio sent to a provider leaves this machine - "
            "the local models above never do. Mark one provider as default; "
            "if it fails at transcription time, the app automatically falls "
            "back to the best local model for your hardware."
        )
        root.addWidget(prov_sub)

        self.provider_list = QListWidget()
        self.provider_list.setMaximumHeight(140)
        root.addWidget(self.provider_list)

        prov_btn_row = QHBoxLayout()
        root.addLayout(prov_btn_row)
        self.add_provider_btn = QPushButton("Add...")
        self.edit_provider_btn = QPushButton("Edit...")
        self.delete_provider_btn = QPushButton("Delete")
        self.set_default_provider_btn = QPushButton("Set as default")
        prov_btn_row.addWidget(self.add_provider_btn)
        prov_btn_row.addWidget(self.edit_provider_btn)
        prov_btn_row.addWidget(self.delete_provider_btn)
        prov_btn_row.addWidget(self.set_default_provider_btn)
        prov_btn_row.addStretch(1)

        self.add_provider_btn.clicked.connect(self._add_provider)
        self.edit_provider_btn.clicked.connect(self._edit_provider)
        self.delete_provider_btn.clicked.connect(self._delete_provider)
        self.set_default_provider_btn.clicked.connect(self._set_default_provider)

        root.addWidget(self._separator())
        info_header = QLabel("Info")
        info_header.setProperty("role", "title")
        root.addWidget(info_header)
        info_sub = QLabel("Features, models used & licenses")
        info_sub.setProperty("role", "muted")
        root.addWidget(info_sub)

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setFontFamily("Consolas")
        info_text.setPlainText(INFO_TEXT)
        info_text.setMinimumHeight(320)
        root.addWidget(info_text)

    # ------------------------------------------------------------ providers
    def _reload_providers(self):
        import voxscribe.providers as providers

        providers.migrate_from_env()
        self._providers = providers.list_providers()
        default = providers.get_default_provider()
        default_id = default["id"] if default else None

        self.provider_list.clear()
        for p in self._providers:
            label = f"{p['name']} ({p['model'] or 'no model set'})"
            if p["id"] == default_id:
                label += "  [default]"
            item = QListWidgetItem(label)
            item.setData(1, p["id"])
            self.provider_list.addItem(item)

    def _selected_provider_id(self):
        item = self.provider_list.currentItem()
        return item.data(1) if item else None

    def _add_provider(self):
        import voxscribe.providers as providers

        dlg = ProviderDialog(self)
        if dlg.exec() == QDialog.Accepted:
            v = dlg.values()
            providers.add_provider(v["name"], v["base_url"], v["api_key"], v["model"])
            self._reload_providers()

    def _edit_provider(self):
        import voxscribe.providers as providers

        pid = self._selected_provider_id()
        if not pid:
            return
        provider = providers.get_provider(pid)
        dlg = ProviderDialog(self, provider=provider)
        if dlg.exec() == QDialog.Accepted:
            v = dlg.values()
            providers.update_provider(pid, **v)
            self._reload_providers()

    def _delete_provider(self):
        import voxscribe.providers as providers

        pid = self._selected_provider_id()
        if not pid:
            return
        providers.delete_provider(pid)
        self._reload_providers()

    def _set_default_provider(self):
        import voxscribe.providers as providers

        pid = self._selected_provider_id()
        if not pid:
            return
        providers.set_default_provider(pid)
        self._reload_providers()

    # ------------------------------------------------------------ hardware
    def apply_hardware_info(self, info: dict):
        """Slot for HardwareInfoController.infoReady - fills in the
        placeholders once the (background-thread-determined) hardware/
        bundled info is available."""
        if "error" in info:
            self.hw_summary_label.setText("Hardware: could not be determined")
            if self.recommended_model_label is not None:
                self.recommended_model_label.setText("Recommended model: ?")
            return

        self.hw_summary_label.setText(f"Hardware: {info['hw_summary']}")
        if self.recommended_model_label is not None:
            self.recommended_model_label.setText(f"Recommended model: {info['recommended_model']}")
        if info["diarize_bundled"]:
            self.hf_hint_label.setText("Diarization models are bundled - no token needed.")
        else:
            self.hf_hint_label.setText(
                "Required for speaker diarization.\nCreate a read token at huggingface.co/settings/tokens")

    def _on_batch_size_changed(self, value):
        try:
            self.window_.settings["batch_size"] = int(value)
        except ValueError:
            pass

    def _on_compute_changed(self, value):
        self.window_.settings["compute"] = value if value in ("cuda", "xpu", "mps", "cpu") else "auto"

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
        self._reload_providers()
