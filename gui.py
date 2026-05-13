"""WhisperX Recorder & Transcriber — CustomTkinter GUI."""

import os
import ssl
import warnings

# --- Warnungen unterdruecken ---
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_LOGS"] = "-all"
os.environ["TORCHAUDIO_NO_BACKEND_CHECK"] = "1"

# --- Firmen-Proxy: SSL-Verifikation deaktivieren ---
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import logging
logging.disable(logging.WARNING)
logging.getLogger().setLevel(logging.ERROR)

import requests
_original_send = requests.adapters.HTTPAdapter.send
def _patched_send(self, request, *args, **kwargs):
    kwargs["verify"] = False
    return _original_send(self, request, *args, **kwargs)
requests.adapters.HTTPAdapter.send = _patched_send

import threading
import time
from datetime import datetime
from tkinter import filedialog

import customtkinter as ctk
from dotenv import load_dotenv

from recorder import AudioRecorder, get_devices
from transcriber import transcribe, format_transcript, save_transcript

load_dotenv()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

LANGUAGES = {"Deutsch": "de", "English": "en", "Français": "fr",
             "Español": "es", "Italiano": "it"}
MODELS = ["large-v2", "large-v3", "medium", "base"]
FORMATS = ["txt", "srt", "json"]


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("WhisperX Recorder & Transcriber")
        self.geometry("900x700")
        self.minsize(800, 600)

        self.recorder = AudioRecorder()
        self._record_start_time = None
        self._timer_id = None
        self._current_rms = 0.0
        self._last_audio_path = None

        self._build_ui()
        self._refresh_devices()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self._build_record_tab()
        self._build_transcribe_tab()
        self._build_settings_tab()

    # --- Tab: Aufnahme ---
    def _build_record_tab(self):
        tab = self.tabview.add("Aufnahme")
        tab.grid_columnconfigure(0, weight=1)

        # Source selection
        source_frame = ctk.CTkFrame(tab)
        source_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        source_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(source_frame, text="Quelle:").grid(
            row=0, column=0, padx=10, pady=10)
        self.source_var = ctk.StringVar(value="Mikrofon + System")
        self.source_menu = ctk.CTkSegmentedButton(
            source_frame, values=["Mikrofon", "System-Audio", "Mikrofon + System"],
            variable=self.source_var, command=self._on_source_changed)
        self.source_menu.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Device selection
        device_frame = ctk.CTkFrame(tab)
        device_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        device_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(device_frame, text="Gerät:").grid(
            row=0, column=0, padx=10, pady=10)
        self.device_var = ctk.StringVar()
        self.device_menu = ctk.CTkOptionMenu(
            device_frame, variable=self.device_var, values=["Lade..."])
        self.device_menu.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self._device_map = {}

        btn_refresh = ctk.CTkButton(
            device_frame, text="↻", width=35, command=self._refresh_devices)
        btn_refresh.grid(row=0, column=2, padx=(0, 10), pady=10)

        # Level meter + Timer
        meter_frame = ctk.CTkFrame(tab)
        meter_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        meter_frame.grid_columnconfigure(0, weight=1)

        self.level_bar = ctk.CTkProgressBar(meter_frame)
        self.level_bar.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        self.level_bar.set(0)

        self.timer_label = ctk.CTkLabel(
            meter_frame, text="00:00", font=ctk.CTkFont(size=28, weight="bold"))
        self.timer_label.grid(row=1, column=0, pady=(0, 10))

        # Record button
        self.record_btn = ctk.CTkButton(
            tab, text="⏺  Aufnahme starten", height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#c0392b", hover_color="#e74c3c",
            command=self._toggle_recording)
        self.record_btn.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        # Status
        self.record_status = ctk.CTkLabel(tab, text="Bereit", text_color="gray")
        self.record_status.grid(row=4, column=0, padx=10, pady=(0, 5))

        # Auto-transcribe checkbox
        self.auto_transcribe_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            tab, text="Nach Aufnahme automatisch transkribieren",
            variable=self.auto_transcribe_var
        ).grid(row=5, column=0, padx=10, pady=(0, 10))

    # --- Tab: Transkription ---
    def _build_transcribe_tab(self):
        tab = self.tabview.add("Transkription")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(6, weight=1)

        # File selection
        file_frame = ctk.CTkFrame(tab)
        file_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        file_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(file_frame, text="Audio:").grid(
            row=0, column=0, padx=10, pady=10)
        self.file_var = ctk.StringVar(value="Keine Datei ausgewählt")
        self.file_label = ctk.CTkLabel(
            file_frame, textvariable=self.file_var, anchor="w")
        self.file_label.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        btn_browse = ctk.CTkButton(
            file_frame, text="Datei wählen", width=120,
            command=self._browse_file)
        btn_browse.grid(row=0, column=2, padx=(0, 10), pady=10)

        # Options row
        opts_frame = ctk.CTkFrame(tab)
        opts_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        for i in range(4):
            opts_frame.grid_columnconfigure(i, weight=1)

        ctk.CTkLabel(opts_frame, text="Sprache:").grid(
            row=0, column=0, padx=(10, 2), pady=10, sticky="e")
        self.lang_var = ctk.StringVar(value="Deutsch")
        ctk.CTkOptionMenu(
            opts_frame, variable=self.lang_var,
            values=list(LANGUAGES.keys()), width=120
        ).grid(row=0, column=1, padx=5, pady=10, sticky="w")

        ctk.CTkLabel(opts_frame, text="Modell:").grid(
            row=0, column=2, padx=(10, 2), pady=10, sticky="e")
        self.model_var = ctk.StringVar(value="large-v2")
        ctk.CTkOptionMenu(
            opts_frame, variable=self.model_var,
            values=MODELS, width=120
        ).grid(row=0, column=3, padx=(5, 10), pady=10, sticky="w")

        # Diarization + speakers
        diar_frame = ctk.CTkFrame(tab)
        diar_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.diarize_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            diar_frame, text="Speaker Diarization",
            variable=self.diarize_var
        ).grid(row=0, column=0, padx=10, pady=10)

        ctk.CTkLabel(diar_frame, text="Min Sprecher:").grid(
            row=0, column=1, padx=(20, 2), pady=10)
        self.min_spk_var = ctk.StringVar(value="")
        ctk.CTkEntry(diar_frame, textvariable=self.min_spk_var, width=50
                     ).grid(row=0, column=2, padx=5, pady=10)

        ctk.CTkLabel(diar_frame, text="Max Sprecher:").grid(
            row=0, column=3, padx=(20, 2), pady=10)
        self.max_spk_var = ctk.StringVar(value="")
        ctk.CTkEntry(diar_frame, textvariable=self.max_spk_var, width=50
                     ).grid(row=0, column=4, padx=(5, 10), pady=10)

        # Transcribe button + progress
        self.transcribe_btn = ctk.CTkButton(
            tab, text="Transkription starten", height=45,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._start_transcription)
        self.transcribe_btn.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(tab)
        self.progress_bar.grid(row=4, column=0, padx=10, pady=(0, 2), sticky="ew")
        self.progress_bar.set(0)
        self.progress_bar.grid_remove()

        self.transcribe_status = ctk.CTkLabel(
            tab, text="", text_color="gray")
        self.transcribe_status.grid(row=5, column=0, padx=10, pady=(0, 5))
        self.transcribe_status.grid_remove()

        # Transcript output
        self.transcript_text = ctk.CTkTextbox(
            tab, font=ctk.CTkFont(size=13), wrap="word")
        self.transcript_text.grid(
            row=6, column=0, padx=10, pady=(5, 5), sticky="nsew")

        # Bottom buttons
        bottom_frame = ctk.CTkFrame(tab, fg_color="transparent")
        bottom_frame.grid(row=7, column=0, padx=10, pady=(0, 10), sticky="ew")
        bottom_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.save_btn = ctk.CTkButton(
            bottom_frame, text="Speichern", state="disabled",
            command=self._save_transcript)
        self.save_btn.grid(row=0, column=0, padx=5, sticky="ew")

        self.copy_btn = ctk.CTkButton(
            bottom_frame, text="Kopieren", state="disabled",
            command=self._copy_transcript)
        self.copy_btn.grid(row=0, column=1, padx=5, sticky="ew")

        fmt_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        fmt_frame.grid(row=0, column=2, padx=5, sticky="ew")
        ctk.CTkLabel(fmt_frame, text="Format:").pack(side="left", padx=(0, 5))
        self.format_vars = {}
        for fmt in FORMATS:
            var = ctk.BooleanVar(value=(fmt == "txt"))
            cb = ctk.CTkCheckBox(fmt_frame, text=fmt.upper(), variable=var,
                                 width=60)
            cb.pack(side="left", padx=2)
            self.format_vars[fmt] = var

    # --- Tab: Einstellungen ---
    def _build_settings_tab(self):
        tab = self.tabview.add("Einstellungen")
        tab.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(tab, text="HuggingFace Token:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=(20, 5), sticky="w")
        self.hf_token_var = ctk.StringVar(value=os.getenv("HF_TOKEN", ""))
        self.hf_entry = ctk.CTkEntry(
            tab, textvariable=self.hf_token_var, show="•", width=400)
        self.hf_entry.grid(row=0, column=1, padx=10, pady=(20, 5), sticky="ew")

        self.show_token_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            tab, text="Token anzeigen", variable=self.show_token_var,
            command=self._toggle_token_visibility
        ).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(tab, text="Für Speaker Diarization benötigt.\n"
                     "Erstelle einen Read-Token auf huggingface.co/settings/tokens",
                     text_color="gray").grid(
            row=2, column=0, columnspan=2, padx=10, pady=(0, 20), sticky="w")

        # Separator
        sep = ctk.CTkFrame(tab, height=2)
        sep.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(tab, text="Aufnahme-Ordner:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=4, column=0, padx=10, pady=5, sticky="w")
        self.output_dir_var = ctk.StringVar(value="recordings")
        ctk.CTkEntry(tab, textvariable=self.output_dir_var).grid(
            row=4, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(tab, text="Batch Size:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=5, column=0, padx=10, pady=5, sticky="w")
        self.batch_size_var = ctk.StringVar(value="16")
        ctk.CTkEntry(tab, textvariable=self.batch_size_var, width=80).grid(
            row=5, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(tab, text="Compute:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=6, column=0, padx=10, pady=5, sticky="w")
        self.compute_var = ctk.StringVar(value="Auto (CUDA wenn verfügbar)")
        ctk.CTkOptionMenu(
            tab, variable=self.compute_var,
            values=["Auto (CUDA wenn verfügbar)", "cuda", "cpu"], width=250
        ).grid(row=6, column=1, padx=10, pady=5, sticky="w")

        # Version info
        ctk.CTkLabel(tab, text="WhisperX Recorder v0.1.0 — 100% lokal",
                     text_color="gray").grid(
            row=10, column=0, columnspan=2, padx=10, pady=(40, 10), sticky="w")

    # ------------------------------------------------------------ Devices
    def _refresh_devices(self):
        devs = get_devices()
        self._all_devices = devs
        self._on_source_changed(self.source_var.get())

    def _on_source_changed(self, value):
        if not hasattr(self, '_all_devices'):
            return
        self._device_map = {}
        if value == "Mikrofon + System":
            # Mikrofon wählen — System-Audio wird automatisch genutzt
            devices = self._all_devices["microphones"]
        elif value == "Mikrofon":
            devices = self._all_devices["microphones"]
        else:
            devices = self._all_devices["loopback"]

        names = []
        for d in devices:
            label = d["name"]
            names.append(label)
            self._device_map[label] = d["index"]

        if names:
            self.device_menu.configure(values=names)
            self.device_var.set(names[0])
        else:
            self.device_menu.configure(values=["Kein Gerät gefunden"])
            self.device_var.set("Kein Gerät gefunden")

    # ---------------------------------------------------------- Recording
    def _toggle_recording(self):
        if self.recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        device_name = self.device_var.get()
        device_index = self._device_map.get(device_name)
        source_map = {"Mikrofon": "mic", "System-Audio": "system",
                      "Mikrofon + System": "both"}
        source = source_map.get(self.source_var.get(), "mic")

        out_dir = self.output_dir_var.get() or "recordings"
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(out_dir, f"{timestamp}.wav")

        self._record_start_time = time.time()
        self._current_rms = 0.0

        self.recorder.start(
            output_path=output_path,
            source=source,
            device_index=device_index,
            on_level=self._on_level_update,
            on_done=self._on_recording_done,
        )

        self.record_btn.configure(
            text="⏹  Aufnahme stoppen",
            fg_color="#2c3e50", hover_color="#34495e")
        self.record_status.configure(text="Aufnahme läuft...", text_color="#e74c3c")
        self._update_timer()

    def _stop_recording(self):
        self.recorder.stop()
        if self._timer_id:
            self.after_cancel(self._timer_id)
            self._timer_id = None
        self.record_btn.configure(
            text="⏺  Aufnahme starten",
            fg_color="#c0392b", hover_color="#e74c3c")
        self.level_bar.set(0)

    def _on_level_update(self, rms):
        self._current_rms = rms

    def _on_recording_done(self, path, duration, error):
        def _update():
            if error:
                self.record_status.configure(
                    text=f"Fehler: {error}", text_color="#e74c3c")
                return

            self._last_audio_path = path
            self.record_status.configure(
                text=f"Gespeichert: {path} ({duration:.1f}s)",
                text_color="#27ae60")

            # Set file in transcription tab
            self.file_var.set(path)

            if self.auto_transcribe_var.get():
                self.tabview.set("Transkription")
                self.after(300, self._start_transcription)

        self.after(0, _update)

    def _update_timer(self):
        if not self.recorder.is_recording:
            return
        elapsed = time.time() - self._record_start_time
        mins, secs = divmod(int(elapsed), 60)
        self.timer_label.configure(text=f"{mins:02d}:{secs:02d}")

        # Update level bar
        level = min(self._current_rms / 15000, 1.0)
        self.level_bar.set(level)

        self._timer_id = self.after(100, self._update_timer)

    # ------------------------------------------------------- Transcription
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Audio-Datei auswählen",
            filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                       ("Alle Dateien", "*.*")],
            initialdir=self.output_dir_var.get() or "recordings")
        if path:
            self.file_var.set(path)
            self.transcribe_status.grid_remove()

    def _start_transcription(self):
        audio_path = self.file_var.get()
        if not audio_path or not os.path.isfile(audio_path):
            self.transcribe_status.grid()
            self.transcribe_status.configure(
                text="Bitte wähle eine Audio-Datei aus.", text_color="#e74c3c")
            return

        self.transcribe_btn.grid_remove()
        self.progress_bar.grid()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(0)
        self.transcribe_status.grid()
        self.transcribe_status.configure(
            text="Starte Transkription...", text_color="#3498db")
        self.transcript_text.delete("1.0", "end")
        self.save_btn.configure(state="disabled")
        self.copy_btn.configure(state="disabled")

        lang_code = LANGUAGES.get(self.lang_var.get(), "de")
        model = self.model_var.get()
        do_diarize = self.diarize_var.get()
        hf_token = self.hf_token_var.get().strip() or None
        compute = self.compute_var.get()
        device = None
        if compute in ("cuda", "cpu"):
            device = compute

        min_spk = None
        max_spk = None
        try:
            min_spk = int(self.min_spk_var.get())
        except ValueError:
            pass
        try:
            max_spk = int(self.max_spk_var.get())
        except ValueError:
            pass

        try:
            batch_size = int(self.batch_size_var.get())
        except ValueError:
            batch_size = 16

        thread = threading.Thread(
            target=self._run_transcription,
            args=(audio_path, lang_code, model, do_diarize, hf_token,
                  min_spk, max_spk, batch_size, device),
            daemon=True)
        thread.start()

    def _run_transcription(self, audio_path, lang_code, model, do_diarize,
                           hf_token, min_spk, max_spk, batch_size, device):
        try:
            result = transcribe(
                audio_path=audio_path,
                language=lang_code,
                model_size=model,
                diarize=do_diarize,
                hf_token=hf_token,
                min_speakers=min_spk,
                max_speakers=max_spk,
                batch_size=batch_size,
                device=device,
                on_progress=self._on_transcribe_progress,
            )
            text = format_transcript(result, include_speakers=do_diarize)
            self._transcription_result = result
            self.after(0, lambda: self._on_transcription_done(text, None))
        except Exception as e:
            self.after(0, lambda: self._on_transcription_done(None, str(e)))

    def _on_transcribe_progress(self, pct, message):
        """Callback from transcriber thread — schedule UI update on main thread."""
        self.after(0, lambda: self._update_progress(pct, message))

    def _update_progress(self, pct, message):
        self.progress_bar.set(pct)
        self.transcribe_status.configure(text=message, text_color="#3498db")

    def _on_transcription_done(self, text, error):
        self.progress_bar.grid_remove()
        self.transcribe_btn.grid()

        if error:
            self.transcribe_status.configure(
                text=f"Fehler: {error}", text_color="#e74c3c")
            return

        self.transcribe_status.configure(
            text="Transkription abgeschlossen!", text_color="#27ae60")
        self.transcript_text.delete("1.0", "end")
        self.transcript_text.insert("1.0", text)
        self.save_btn.configure(state="normal")
        self.copy_btn.configure(state="normal")

    def _copy_transcript(self):
        text = self.transcript_text.get("1.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.transcribe_status.configure(
                text="In Zwischenablage kopiert!", text_color="#27ae60")

    def _save_transcript(self):
        if not hasattr(self, '_transcription_result'):
            return

        formats = [fmt for fmt, var in self.format_vars.items() if var.get()]
        if not formats:
            formats = ["txt"]

        audio_path = self.file_var.get()
        base = os.path.splitext(audio_path)[0]
        saved = save_transcript(self._transcription_result, base, formats=formats)
        if saved:
            self.transcribe_status.configure(
                text=f"Gespeichert: {', '.join(saved)}", text_color="#27ae60")

    # ---------------------------------------------------------- Settings
    def _toggle_token_visibility(self):
        if self.show_token_var.get():
            self.hf_entry.configure(show="")
        else:
            self.hf_entry.configure(show="•")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
