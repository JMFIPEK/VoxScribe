"""VoxScribe — GUI für ORION (Obsidian Retrieval for Information and Organized Notes)."""

import os
import ssl
import warnings
import threading
import time
import tkinter as tk

# ═══════════════════════════════════════════════════════════════════════
#  App-Bezeichnung — hier zentral änderbar
# ═══════════════════════════════════════════════════════════════════════
APP_NAME = "VoxScribe"
APP_SUBTITLE = "for ORION"
APP_VERSION = "0.1.0"
APP_ID = "orion.voxscribe"

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

# Windows: eigene AppUserModelID setzen, damit Taskleiste eigenes Icon zeigt
import ctypes
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)


# ═══════════════════════════════════════════════════════════════════════
#  Splash Screen — wird sofort angezeigt während Module laden
# ═══════════════════════════════════════════════════════════════════════

class SplashScreen:
    """Borderless splash window with logo and animated loading indicator."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # Kein Fensterrahmen
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#1a1a2e")

        # Größe und zentrieren
        width, height = 480, 380
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # Logo laden
        self._logo_image = None
        logo_path = os.path.join(os.path.dirname(__file__), "Logo.png")
        if os.path.exists(logo_path):
            try:
                from PIL import Image, ImageTk
                img = Image.open(logo_path)
                img = img.resize((180, 180), Image.LANCZOS)
                self._logo_image = ImageTk.PhotoImage(img)
            except Exception:
                pass

        # Canvas für schickes Layout
        canvas = tk.Canvas(self.root, width=width, height=height,
                           bg="#1a1a2e", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Abgerundeter Rahmen (Border-Effekt)
        canvas.create_rectangle(2, 2, width - 2, height - 2,
                                outline="#3498db", width=2)

        # Logo
        if self._logo_image:
            canvas.create_image(width // 2, 120, image=self._logo_image)
        else:
            canvas.create_text(width // 2, 120, text="\U0001f399",
                               font=("Segoe UI", 60), fill="white")

        # Titel
        canvas.create_text(width // 2, 240, text=APP_NAME,
                           font=("Segoe UI Semibold", 20), fill="white")
        canvas.create_text(width // 2, 270, text=APP_SUBTITLE,
                           font=("Segoe UI", 14), fill="#8e9aaf")

        # Status-Text
        self._status_id = canvas.create_text(
            width // 2, 320, text="Module werden geladen...",
            font=("Segoe UI", 10), fill="#7f8c8d")
        self._canvas = canvas

        # Lade-Animation (pulsierende Punkte)
        self._dots = 0
        self._animate()

        self.root.update()

    def _animate(self):
        """Pulsierende Punkte als Lade-Indikator."""
        self._dots = (self._dots % 3) + 1
        dots_text = "\u25cf" * self._dots + "\u25cb" * (3 - self._dots)
        self._canvas.itemconfig(self._status_id,
                                text=f"Module werden geladen  {dots_text}")
        self._anim_id = self.root.after(500, self._animate)

    def update_status(self, text):
        """Status-Text aktualisieren."""
        self._canvas.itemconfig(self._status_id, text=text)
        self.root.update()

    def destroy(self):
        """Splash schließen."""
        try:
            self.root.after_cancel(self._anim_id)
        except Exception:
            pass
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════
#  Splash starten und schwere Module laden
# ═══════════════════════════════════════════════════════════════════════

_splash = None
if __name__ == "__main__":
    _splash = SplashScreen()

# --- Schwere Imports mit Splash-Updates ---
if _splash:
    _splash.update_status("Netzwerk-Module...")

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

if _splash:
    _splash.update_status("GUI-Framework...")

from datetime import datetime
from tkinter import filedialog

import customtkinter as ctk
from dotenv import load_dotenv

if _splash:
    _splash.update_status("Audio-Recorder...")

from recorder import AudioRecorder, get_devices

if _splash:
    _splash.update_status("Transcriber-Engine...")

from transcriber import transcribe, format_transcript, save_transcript

if _splash:
    _splash.update_status("Hardware-Erkennung...")

from hardware_detect import recommend_model, get_hardware_summary

load_dotenv()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

LANGUAGES = {"Deutsch": "de", "English": "en", "Français": "fr",
             "Español": "es", "Italiano": "it"}
MODELS = ["large-v3", "large-v2", "medium", "base"]
FORMATS = ["txt", "srt", "json"]

# Hardware-basierte Modellempfehlung
_recommended_model, _recommended_device, _hw_reason = recommend_model()

if _splash:
    _splash.update_status("Bereit!")
    time.sleep(0.3)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} — {APP_SUBTITLE}")
        self.geometry("900x700")
        self.minsize(800, 600)

        # App-Icon setzen
        icon_path = os.path.join(os.path.dirname(__file__), "Logo.png")
        if os.path.exists(icon_path):
            from PIL import Image as PILImage
            import tempfile
            img = PILImage.open(icon_path)
            # ICO mit mehreren Größen für Titelleiste + Taskleiste
            ico_path = os.path.join(tempfile.gettempdir(), "whisperx_icon.ico")
            img.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
            self.iconbitmap(ico_path)
            self.after(200, lambda: self.iconbitmap(ico_path))

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
        self._build_info_tab()

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
        tab.grid_rowconfigure(7, weight=1)

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
        self.model_var = ctk.StringVar(value=_recommended_model)
        ctk.CTkOptionMenu(
            opts_frame, variable=self.model_var,
            values=MODELS, width=120
        ).grid(row=0, column=3, padx=(5, 10), pady=10, sticky="w")

        # Hardware-Empfehlung anzeigen
        hw_hint = ctk.CTkLabel(
            tab, text=f"⚡ {_hw_reason}",
            text_color="gray", font=ctk.CTkFont(size=11))
        hw_hint.grid(row=1, column=0, padx=10, pady=(0, 0), sticky="w")
        # Shift subsequent rows down
        opts_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        hw_hint.grid(row=2, column=0, padx=10, pady=(0, 2), sticky="w")

        # Diarization + speakers
        diar_frame = ctk.CTkFrame(tab)
        diar_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

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
        self.transcribe_btn.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(tab)
        self.progress_bar.grid(row=5, column=0, padx=10, pady=(0, 2), sticky="ew")
        self.progress_bar.set(0)
        self.progress_bar.grid_remove()

        self.transcribe_status = ctk.CTkLabel(
            tab, text="", text_color="gray")
        self.transcribe_status.grid(row=6, column=0, padx=10, pady=(0, 5))
        self.transcribe_status.grid_remove()

        # Transcript output
        self.transcript_text = ctk.CTkTextbox(
            tab, font=ctk.CTkFont(size=13), wrap="word")
        self.transcript_text.grid(
            row=7, column=0, padx=10, pady=(5, 5), sticky="nsew")

        # Speaker renaming frame (hidden until transcription done)
        self.speaker_frame = ctk.CTkFrame(tab)
        self.speaker_frame.grid(row=8, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.speaker_frame.grid_remove()
        self.speaker_frame.grid_columnconfigure(0, weight=1)

        speaker_header = ctk.CTkFrame(self.speaker_frame, fg_color="transparent")
        speaker_header.grid(row=0, column=0, sticky="ew")
        speaker_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(speaker_header, text="Sprecher umbenennen",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=5, sticky="w")
        self.apply_names_btn = ctk.CTkButton(
            speaker_header, text="Anwenden", width=100,
            command=self._apply_speaker_names)
        self.apply_names_btn.grid(row=0, column=1, padx=10, pady=5)

        self.speaker_entries_frame = ctk.CTkFrame(
            self.speaker_frame, fg_color="transparent")
        self.speaker_entries_frame.grid(
            row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        self._speaker_name_entries = {}

        # Bottom buttons
        bottom_frame = ctk.CTkFrame(tab, fg_color="transparent")
        bottom_frame.grid(row=9, column=0, padx=10, pady=(0, 10), sticky="ew")
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
        ctk.CTkLabel(tab, text=f"{APP_NAME} v{APP_VERSION} — 100% lokal",
                     text_color="gray").grid(
            row=10, column=0, columnspan=2, padx=10, pady=(40, 5), sticky="w")

        # Hardware info
        hw_summary = get_hardware_summary()
        ctk.CTkLabel(tab, text=f"Hardware: {hw_summary}",
                     text_color="gray", justify="left").grid(
            row=11, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="w")
        ctk.CTkLabel(tab, text=f"Empfohlenes Modell: {_recommended_model}",
                     text_color="gray").grid(
            row=12, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="w")

    # --- Tab: Info ---
    def _build_info_tab(self):
        tab = self.tabview.add("Info")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        info_text = ctk.CTkTextbox(
            tab, font=ctk.CTkFont(size=13), wrap="word")
        info_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        content = (
            "═══════════════════════════════════════════════════════════\n"
            f"  {APP_NAME} {APP_SUBTITLE} — Übersicht\n"
            "═══════════════════════════════════════════════════════════\n\n"
            "Diese Anwendung ermöglicht die lokale Aufnahme und automatische\n"
            "Transkription von Audio mit Sprechererkennung (Speaker Diarization).\n"
            "Alle Verarbeitungsschritte laufen vollständig lokal auf dem eigenen\n"
            "Rechner — es werden keine Audio-Daten an externe Server gesendet.\n\n"
            "─────────────────────────────────────────────────────────────\n"
            "  Funktionsumfang\n"
            "─────────────────────────────────────────────────────────────\n\n"
            "• Aufnahme von Mikrofon, System-Audio oder beidem gleichzeitig\n"
            "  (WASAPI Loopback unter Windows)\n"
            "• Automatische Transkription mit wortgenauem Zeitstempel-Alignment\n"
            "• Speaker Diarization — Erkennung und Zuordnung einzelner Sprecher\n"
            "• Sprecher-Umbenennung nach der Transkription\n"
            "• Export in TXT, SRT und JSON\n"
            "• GPU-Beschleunigung via CUDA (falls verfügbar)\n\n\n"
            "═══════════════════════════════════════════════════════════\n"
            "  Verwendete Modelle & Lizenzen\n"
            "═══════════════════════════════════════════════════════════\n\n"
            "┌─────────────────────────────────────────────────────────┐\n"
            "│  1. OpenAI Whisper (via WhisperX)                       │\n"
            "├─────────────────────────────────────────────────────────┤\n"
            "│  Zweck:    Sprache-zu-Text Transkription                │\n"
            "│  Modelle:  large-v2, large-v3, medium, base             │\n"
            "│  Autor:    OpenAI                                       │\n"
            "│  Lizenz:   MIT License                                  │\n"
            "│  Quelle:   https://github.com/openai/whisper            │\n"
            "│                                                         │\n"
            "│  Hinweis: Die Whisper-Modelle stehen unter der MIT-     │\n"
            "│  Lizenz. Die Nutzung ist für kommerzielle und nicht-    │\n"
            "│  kommerzielle Zwecke gestattet.                         │\n"
            "└─────────────────────────────────────────────────────────┘\n\n"
            "┌─────────────────────────────────────────────────────────┐\n"
            "│  2. WhisperX                                            │\n"
            "├─────────────────────────────────────────────────────────┤\n"
            "│  Zweck:    Batched Inference, Forced Alignment,         │\n"
            "│            Integration der Diarization-Pipeline         │\n"
            "│  Autor:    Max Bain                                     │\n"
            "│  Lizenz:   BSD 4-Clause License                         │\n"
            "│  Quelle:   https://github.com/m-bain/whisperX           │\n"
            "│                                                         │\n"
            "│  Hinweis: WhisperX erweitert Whisper um wortgenaues     │\n"
            "│  Alignment und Speaker Diarization. BSD-4-Clause        │\n"
            "│  erfordert Namensnennung bei Weitergabe.                │\n"
            "└─────────────────────────────────────────────────────────┘\n\n"
            "┌─────────────────────────────────────────────────────────┐\n"
            "│  3. pyannote.audio (Speaker Diarization)                │\n"
            "├─────────────────────────────────────────────────────────┤\n"
            "│  Zweck:    Erkennung & Zuordnung von Sprechern          │\n"
            "│  Modell:   pyannote/speaker-diarization-3.1             │\n"
            "│  Autor:    Hervé Bredin (CNRS)                          │\n"
            "│  Lizenz:   MIT License                                  │\n"
            "│  Quelle:   https://github.com/pyannote/pyannote-audio   │\n"
            "│                                                         │\n"
            "│  ⚠ WICHTIG: Die Nutzung der pyannote-Modelle erfordert │\n"
            "│  die Zustimmung zu den Nutzungsbedingungen auf          │\n"
            "│  HuggingFace sowie einen gültigen HF-Token.            │\n"
            "│  Für kommerzielle Nutzung gelten ggf. separate          │\n"
            "│  Lizenzbedingungen — siehe pyannote.ai.                 │\n"
            "└─────────────────────────────────────────────────────────┘\n\n"
            "┌─────────────────────────────────────────────────────────┐\n"
            "│  4. PyTorch                                             │\n"
            "├─────────────────────────────────────────────────────────┤\n"
            "│  Zweck:    Deep-Learning-Framework für Inferenz         │\n"
            "│  Autor:    Meta AI (Facebook)                           │\n"
            "│  Lizenz:   BSD 3-Clause License                         │\n"
            "│  Quelle:   https://github.com/pytorch/pytorch           │\n"
            "└─────────────────────────────────────────────────────────┘\n\n"
            "┌─────────────────────────────────────────────────────────┐\n"
            "│  5. Forced Alignment Modelle                            │\n"
            "├─────────────────────────────────────────────────────────┤\n"
            "│  Zweck:    Wortgenaue Zeitstempel-Zuordnung             │\n"
            "│  Modelle:  WAV2VEC2-basierte Alignment-Modelle          │\n"
            "│            (sprachspezifisch, z.B. für DE, EN, FR)      │\n"
            "│  Autor:    Meta AI / HuggingFace Community              │\n"
            "│  Lizenz:   Apache 2.0 / MIT (modellabhängig)            │\n"
            "│  Quelle:   HuggingFace Model Hub                        │\n"
            "└─────────────────────────────────────────────────────────┘\n\n\n"
            "═══════════════════════════════════════════════════════════\n"
            "  Weitere Bibliotheken\n"
            "═══════════════════════════════════════════════════════════\n\n"
            "• CustomTkinter (MIT) — GUI-Framework\n"
            "• PyAudioWPatch (MIT) — Audio-Aufnahme unter Windows\n"
            "• SoundFile (BSD 3-Clause) — Audio-Datei I/O\n"
            "• SciPy (BSD 3-Clause) — Signal-Resampling\n"
            "• NumPy (BSD 3-Clause) — Numerische Berechnungen\n\n\n"
            "═══════════════════════════════════════════════════════════\n"
            "  Hinweise zur Nutzung\n"
            "═══════════════════════════════════════════════════════════\n\n"
            "• Alle KI-Modelle werden beim ersten Start automatisch von\n"
            "  HuggingFace heruntergeladen und lokal zwischengespeichert.\n"
            "• Für Speaker Diarization wird ein kostenloser HuggingFace-\n"
            "  Account mit akzeptierten Modell-Bedingungen benötigt.\n"
            "• Die Transkriptionsqualität hängt von der Modellgröße ab:\n"
            "  large-v3 > large-v2 > medium > base (Qualität vs. Geschwindigkeit)\n"
            "• GPU (NVIDIA CUDA) wird empfohlen für large-Modelle.\n"
            "  CPU-Inferenz ist möglich, aber deutlich langsamer.\n\n"
            "─────────────────────────────────────────────────────────────\n"
            f"  {APP_NAME} v{APP_VERSION}\n"
            "  Alle Verarbeitung erfolgt lokal — keine Cloud-Dienste.\n"
            "─────────────────────────────────────────────────────────────\n"
        )

        info_text.insert("1.0", content)
        info_text.configure(state="disabled")

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
        seen_labels = set()
        for d in devices:
            label = d["name"]
            if label in seen_labels:
                host_api = d.get("host_api") or "Audio"
                label = f"{label} ({host_api}, #{d['index']})"
            seen_labels.add(label)
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
                if self._timer_id:
                    self.after_cancel(self._timer_id)
                    self._timer_id = None
                self.record_btn.configure(
                    text="⏺  Aufnahme starten",
                    fg_color="#c0392b", hover_color="#e74c3c")
                self.level_bar.set(0)
                self.timer_label.configure(text="00:00")
                self._record_start_time = None
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
            error = str(e)
            self.after(
                0,
                lambda error=error: self._on_transcription_done(None, error),
            )

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

        # Sprecher-Einträge aufbauen
        self._populate_speaker_entries()

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

        # Sprecher-Namen anwenden bevor gespeichert wird
        self._apply_speaker_names_to_result()

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
    # ------------------------------------------------- Speaker Renaming
    def _populate_speaker_entries(self):
        """Erstellt Eingabefelder für jeden erkannten Sprecher."""
        # Alte Einträge löschen
        for w in self.speaker_entries_frame.winfo_children():
            w.destroy()
        self._speaker_name_entries = {}

        if not hasattr(self, '_transcription_result'):
            self.speaker_frame.grid_remove()
            return

        speakers = set()
        for seg in self._transcription_result.get("segments", []):
            if "speaker" in seg:
                speakers.add(seg["speaker"])

        if not speakers:
            self.speaker_frame.grid_remove()
            return

        for col, spk in enumerate(sorted(speakers)):
            frame = ctk.CTkFrame(self.speaker_entries_frame, fg_color="transparent")
            frame.pack(side="left", padx=(0, 15), pady=2)
            ctk.CTkLabel(frame, text=f"{spk}  \u2192",
                         font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 5))
            entry = ctk.CTkEntry(frame, width=140,
                                 placeholder_text="Name eingeben")
            entry.pack(side="left")
            self._speaker_name_entries[spk] = entry

        self.speaker_frame.grid()

    def _get_speaker_mapping(self):
        """Gibt das aktuelle Mapping {SPEAKER_XX: Name} zurück."""
        mapping = {}
        for spk, entry in self._speaker_name_entries.items():
            name = entry.get().strip()
            if name:
                mapping[spk] = name
        return mapping

    def _apply_speaker_names(self):
        """Wendet die Sprecher-Namen auf den Text im Textfeld an."""
        mapping = self._get_speaker_mapping()
        if not mapping:
            return

        text = self.transcript_text.get("1.0", "end")
        for spk_id, name in mapping.items():
            text = text.replace(spk_id, name)
        self.transcript_text.delete("1.0", "end")
        self.transcript_text.insert("1.0", text.rstrip())

        self.transcribe_status.configure(
            text="Sprecher-Namen angewendet!", text_color="#27ae60")

    def _apply_speaker_names_to_result(self):
        """Wendet die Sprecher-Namen auf das interne Ergebnis an (für Speichern)."""
        mapping = self._get_speaker_mapping()
        if not mapping or not hasattr(self, '_transcription_result'):
            return
        for seg in self._transcription_result.get("segments", []):
            spk = seg.get("speaker", "")
            if spk in mapping:
                seg["speaker"] = mapping[spk]
            for word in seg.get("words", []) or []:
                word_spk = word.get("speaker", "")
                if word_spk in mapping:
                    word["speaker"] = mapping[word_spk]

        for word in self._transcription_result.get("word_segments", []) or []:
            word_spk = word.get("speaker", "")
            if word_spk in mapping:
                word["speaker"] = mapping[word_spk]

def main():
    global _splash
    # Splash schließen
    if _splash:
        _splash.destroy()
        _splash = None

    # Hauptanwendung starten
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
