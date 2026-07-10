"""VoxScribe — GUI für ORION (Obsidian Retrieval for Information and Organized Notes)."""

import os
import ssl
import sys
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
# (ctypes.windll existiert nur unter Windows - auf macOS/Linux ueberspringen)
if sys.platform == "win32":
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

from transcriber import (
    DEFAULT_API_BASE_URL,
    transcribe,
    save_transcript,
    _get_bundled_models_dir,
)

if _splash:
    _splash.update_status("Hardware-Erkennung...")

from hardware_detect import recommend_model, get_hardware_summary
import speaker_profiles

load_dotenv()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

LANGUAGES = {"Automatisch erkennen": None, "Deutsch": "de", "English": "en",
             "Français": "fr", "Español": "es", "Italiano": "it"}
MODEL_DISPLAY_NAMES = {
    "large-v3": "large-v3",
    "large-v2": "large-v2",
    "medium": "medium",
    "base": "base",
    "server:kit.whisper-large-v3": "KIT ToolBox (Server)",
}
MODEL_IDS_BY_DISPLAY = {v: k for k, v in MODEL_DISPLAY_NAMES.items()}
MODELS = list(MODEL_DISPLAY_NAMES.values())
FORMATS = ["txt", "srt", "json"]

# Gut unterscheidbare Farben fuer die Sprecher-Einfaerbung im Transkript (dunkles Theme)
SPEAKER_COLOR_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6",
    "#1abc9c", "#e67e22", "#00bcd4", "#ff6b9d", "#95a5a6",
]


def _format_time_short(seconds: float) -> str:
    """Formatiert Sekunden als MM:SS (bzw. HH:MM:SS bei >= 1h) fuer die Anzeige."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

# Hardware-basierte Modellempfehlung
_recommended_model, _recommended_device, _hw_reason = recommend_model()

if _splash:
    _splash.update_status("Bereit!")
    time.sleep(0.3)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} — {APP_SUBTITLE}")
        self.geometry("900x760")
        self.minsize(800, 640)

        # App-Icon setzen
        icon_path = os.path.join(os.path.dirname(__file__), "Logo.png")
        if os.path.exists(icon_path):
            from PIL import Image as PILImage
            if sys.platform == "win32":
                import tempfile
                # ICO mit mehreren Größen für Titelleiste + Taskleiste
                ico_path = os.path.join(tempfile.gettempdir(), "whisperx_icon.ico")
                img = PILImage.open(icon_path)
                img.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
                self.iconbitmap(ico_path)
                self.after(200, lambda: self.iconbitmap(ico_path))
            else:
                # macOS/Linux: iconbitmap() erwartet .ico (Windows) bzw. .xbm
                # (X11) und schlaegt fuer unser PNG fehl - iconphoto ist der
                # plattformuebergreifende Tk-Weg fuer beliebige Bildformate.
                icon_img = tk.PhotoImage(file=icon_path)
                self.iconphoto(True, icon_img)
                self._icon_img_ref = icon_img  # Referenz halten (sonst GC'd)

        self.recorder = AudioRecorder()
        self._record_start_time = None
        self._timer_id = None
        self._last_audio_path = None

        self._build_ui()
        self._refresh_devices()
        self._on_diarize_toggled()

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
        # System-Audio: unter Windows vollstaendig (WASAPI Loopback, inkl.
        # Kombination mit Mikrofon). Unter macOS experimentell via
        # ScreenCaptureKit (siehe macos/README.md) - nur einzeln, noch nicht
        # kombiniert mit Mikrofon. Unter Linux gar nicht verfuegbar.
        if sys.platform == "win32":
            source_values = ["Mikrofon", "System-Audio", "Mikrofon + System"]
            default_source = "Mikrofon + System"
            source_hint = None
        elif sys.platform == "darwin":
            source_values = ["Mikrofon", "System-Audio", "Mikrofon + System"]
            default_source = "Mikrofon"
            source_hint = ("⚠ System-Audio ist auf macOS experimentell (ScreenCaptureKit) - "
                          "erfordert die Berechtigung „Bildschirm- und Systemaudioaufnahme“.")
        else:
            source_values = ["Mikrofon"]
            default_source = "Mikrofon"
            source_hint = "System-Audio (Meeting-Mitschnitt) ist auf diesem Betriebssystem noch nicht verfügbar."

        self.source_var = ctk.StringVar(value=default_source)
        self.source_menu = ctk.CTkSegmentedButton(
            source_frame, values=source_values,
            variable=self.source_var, command=self._on_source_changed)
        self.source_menu.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        if source_hint:
            ctk.CTkLabel(
                source_frame, text=source_hint,
                text_color="gray", font=ctk.CTkFont(size=11)
            ).grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="w")

        # Device selection
        device_frame = ctk.CTkFrame(tab)
        device_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        device_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(device_frame, text="Gerät:").grid(
            row=0, column=0, padx=10, pady=10)
        self.device_var = ctk.StringVar()
        self.device_menu = ctk.CTkOptionMenu(
            device_frame, variable=self.device_var, values=["Lade..."],
            width=460, dynamic_resizing=False, anchor="w")
        self.device_menu.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self._device_map = {}

        btn_refresh = ctk.CTkButton(
            device_frame, text="↻", width=35, command=self._refresh_devices)
        btn_refresh.grid(row=0, column=2, padx=(0, 10), pady=10)

        # Eine feste Breite (z.B. 460px) reicht je nach Schriftart/DPI/
        # Skalierung des jeweiligen Systems nicht immer, um lange
        # Geraetenamen vollstaendig zu zeigen - deshalb die Dropdown-Breite
        # bei jeder Groessenaenderung des Frames neu an den tatsaechlich
        # verfuegbaren Platz anpassen, statt einen festen Wert zu raten.
        def _on_device_frame_configure(event):
            available = event.width - 180  # Platz fuer Label + Refresh-Button + Paddings
            if available > 150:
                self.device_menu.configure(width=available)
        device_frame.bind("<Configure>", _on_device_frame_configure)

        # Level meter (pro Kanal) + Timer
        meter_frame = ctk.CTkFrame(tab)
        meter_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        meter_frame.grid_columnconfigure(1, weight=1)

        self._level_bars = {}
        self._level_rows = {}
        self._current_rms = {"mic": 0.0, "system": 0.0}

        for row, (channel, label) in enumerate((("mic", "Mikrofon"), ("system", "System"))):
            lbl = ctk.CTkLabel(meter_frame, text=label, width=70, anchor="w")
            lbl.grid(row=row, column=0, padx=(10, 5), pady=(10 if row == 0 else 2, 2), sticky="w")
            bar = ctk.CTkProgressBar(meter_frame)
            bar.grid(row=row, column=1, padx=(0, 10), pady=(10 if row == 0 else 2, 2), sticky="ew")
            bar.set(0)
            self._level_bars[channel] = bar
            self._level_rows[channel] = (lbl, bar)

        self.timer_label = ctk.CTkLabel(
            meter_frame, text="00:00", font=ctk.CTkFont(size=28, weight="bold"))
        self.timer_label.grid(row=2, column=0, columnspan=2, pady=(5, 10))

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
        tab.grid_rowconfigure(5, weight=1)

        # --- Karte: Audio-Datei ---
        file_card = ctk.CTkFrame(tab, corner_radius=10)
        file_card.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        file_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            file_card, text="AUDIO-DATEI", text_color="gray",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=0, padx=14, pady=(6, 0), sticky="w")

        file_row = ctk.CTkFrame(file_card, fg_color="transparent")
        file_row.grid(row=1, column=0, padx=14, pady=(3, 8), sticky="ew")
        file_row.grid_columnconfigure(0, weight=1)

        self.file_var = ctk.StringVar(value="Keine Datei ausgewählt")
        self.file_var.trace_add("write", self._on_file_selected)
        self.file_label = ctk.CTkLabel(
            file_row, textvariable=self.file_var, anchor="w",
            text_color="gray")
        self.file_label.grid(row=0, column=0, padx=(0, 10), sticky="ew")

        btn_browse = ctk.CTkButton(
            file_row, text="Datei wählen...", width=140,
            command=self._browse_file)
        btn_browse.grid(row=0, column=1)

        # --- Karte: Optionen (Sprache/Modell + Diarization) ---
        opts_card = ctk.CTkFrame(tab, corner_radius=10)
        opts_card.grid(row=1, column=0, padx=10, pady=4, sticky="ew")
        opts_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            opts_card, text="EINSTELLUNGEN", text_color="gray",
            font=ctk.CTkFont(size=11, weight="bold")
        ).grid(row=0, column=0, padx=14, pady=(6, 2), sticky="w")

        opts_frame = ctk.CTkFrame(opts_card, fg_color="transparent")
        opts_frame.grid(row=1, column=0, padx=14, pady=0, sticky="ew")
        for i in range(4):
            opts_frame.grid_columnconfigure(i, weight=1)

        ctk.CTkLabel(opts_frame, text="Sprache:").grid(
            row=0, column=0, padx=(0, 2), pady=4, sticky="e")
        self.lang_var = ctk.StringVar(value="Automatisch erkennen")
        ctk.CTkOptionMenu(
            opts_frame, variable=self.lang_var,
            values=list(LANGUAGES.keys()), width=140
        ).grid(row=0, column=1, padx=5, pady=4, sticky="w")

        ctk.CTkLabel(opts_frame, text="Modell:").grid(
            row=0, column=2, padx=(10, 2), pady=4, sticky="e")
        self.model_var = ctk.StringVar(
            value=MODEL_DISPLAY_NAMES["server:kit.whisper-large-v3"])
        ctk.CTkOptionMenu(
            opts_frame, variable=self.model_var,
            values=MODELS, width=170
        ).grid(row=0, column=3, padx=(5, 0), pady=4, sticky="w")

        # Hardware-Empfehlung anzeigen (nur relevant bei lokalen Modellen)
        ctk.CTkLabel(
            opts_card, text=f"⚡ Bei lokalem Modell empfohlen: {_hw_reason}",
            text_color="gray", font=ctk.CTkFont(size=11)
        ).grid(row=2, column=0, padx=14, pady=(0, 4), sticky="w")

        sep = ctk.CTkFrame(opts_card, height=1, fg_color=("gray80", "gray30"))
        sep.grid(row=3, column=0, padx=14, pady=(2, 4), sticky="ew")

        diar_frame = ctk.CTkFrame(opts_card, fg_color="transparent")
        diar_frame.grid(row=4, column=0, padx=14, pady=(0, 6), sticky="ew")

        self.diarize_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            diar_frame, text="Speaker Diarization",
            variable=self.diarize_var, command=self._on_diarize_toggled
        ).grid(row=0, column=0, padx=(0, 10), pady=2)

        self.min_spk_label = ctk.CTkLabel(diar_frame, text="Min Sprecher:")
        self.min_spk_label.grid(row=0, column=1, padx=(20, 2), pady=2)
        self.min_spk_var = ctk.StringVar(value="")
        self.min_spk_entry = ctk.CTkEntry(
            diar_frame, textvariable=self.min_spk_var, width=50,
            placeholder_text="auto")
        self.min_spk_entry.grid(row=0, column=2, padx=5, pady=2)

        self.max_spk_label = ctk.CTkLabel(diar_frame, text="Max Sprecher:")
        self.max_spk_label.grid(row=0, column=3, padx=(20, 2), pady=2)
        self.max_spk_var = ctk.StringVar(value="")
        self.max_spk_entry = ctk.CTkEntry(
            diar_frame, textvariable=self.max_spk_var, width=50,
            placeholder_text="auto")
        self.max_spk_entry.grid(row=0, column=4, padx=5, pady=2)

        # --- Start-Button + Fortschritt ---
        self.transcribe_btn = ctk.CTkButton(
            tab, text="▶  Transkription starten", height=38,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#27ae60", hover_color="#2ecc71", text_color="white",
            state="disabled",
            command=self._start_transcription)
        self.transcribe_btn.grid(row=2, column=0, padx=10, pady=(4, 3), sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(tab)
        self.progress_bar.grid(row=3, column=0, padx=10, pady=(0, 2), sticky="ew")
        self.progress_bar.set(0)
        self.progress_bar.grid_remove()

        self.transcribe_status = ctk.CTkLabel(
            tab, text="", text_color="gray")
        self.transcribe_status.grid(row=4, column=0, padx=10, pady=(0, 5))
        self.transcribe_status.grid_remove()

        # Transcript output
        self.transcript_text = ctk.CTkTextbox(
            tab, font=ctk.CTkFont(size=13), wrap="word", corner_radius=10)
        self.transcript_text.grid(
            row=5, column=0, padx=10, pady=(5, 5), sticky="nsew")

        # Speaker renaming frame (hidden until transcription done)
        self.speaker_frame = ctk.CTkFrame(tab, corner_radius=10)
        self.speaker_frame.grid(row=6, column=0, padx=10, pady=(0, 5), sticky="ew")
        self.speaker_frame.grid_remove()
        self.speaker_frame.grid_columnconfigure(0, weight=1)

        speaker_header = ctk.CTkFrame(self.speaker_frame, fg_color="transparent")
        speaker_header.grid(row=0, column=0, sticky="ew")
        speaker_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(speaker_header, text="Sprecher umbenennen",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=3, sticky="w")
        self.apply_names_btn = ctk.CTkButton(
            speaker_header, text="Anwenden", width=100,
            command=self._apply_speaker_names)
        self.apply_names_btn.grid(row=0, column=1, padx=10, pady=3)

        # Beliebig viele Sprecher (nicht auf eine feste Spalten-/Zeilenzahl
        # hartkodiert): ein fest-hoher, scrollbarer Bereich mit einer Zeile
        # pro Sprecher. CTkScrollableFrame wurde bewusst NICHT verwendet - es
        # bringt die Geometrie des restlichen Tabs durcheinander und macht den
        # Transkript-Textbereich unsichtbar (siehe Kommentar dort). Stattdessen
        # ein simples, manuelles Canvas+Scrollbar-Konstrukt: die Hoehe ist
        # fest (SPEAKER_SCROLL_HEIGHT), damit der Transkript-Bereich beim
        # Zuordnen der Sprecher immer sichtbar bleibt, egal wie viele
        # Sprecher es sind. Bewusst knapp bemessen (Platz fuer ~2-3 Zeilen),
        # damit moeglichst viel Hoehe beim Transkript-Bereich bleibt - bei
        # mehr Sprechern wird einfach gescrollt.
        SPEAKER_SCROLL_HEIGHT = 100
        scroll_outer = ctk.CTkFrame(self.speaker_frame, fg_color="transparent")
        scroll_outer.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="ew")
        scroll_outer.grid_columnconfigure(0, weight=1)

        canvas_bg = self.speaker_frame.cget("fg_color")
        if isinstance(canvas_bg, (list, tuple)):
            canvas_bg = canvas_bg[1 if ctk.get_appearance_mode() == "Dark" else 0]
        self._speaker_canvas = tk.Canvas(
            scroll_outer, height=SPEAKER_SCROLL_HEIGHT,
            highlightthickness=0, bg=canvas_bg)
        self._speaker_canvas.grid(row=0, column=0, sticky="ew")
        # CTkScrollbar defaults to height=200 fuer orientation="vertical" -
        # ohne explizite Hoehe wuerde das die Zeile (und damit die ganze
        # Sprecher-Karte) auf mindestens 200px aufblasen, egal wie klein der
        # Canvas ist.
        speaker_scrollbar = ctk.CTkScrollbar(
            scroll_outer, orientation="vertical", command=self._speaker_canvas.yview,
            height=SPEAKER_SCROLL_HEIGHT)
        speaker_scrollbar.grid(row=0, column=1, sticky="ns")
        self._speaker_canvas.configure(yscrollcommand=speaker_scrollbar.set)

        self.speaker_entries_frame = ctk.CTkFrame(
            self._speaker_canvas, fg_color="transparent")
        self._speaker_canvas_window = self._speaker_canvas.create_window(
            (0, 0), window=self.speaker_entries_frame, anchor="nw")

        def _on_entries_configure(_event=None):
            self._speaker_canvas.configure(scrollregion=self._speaker_canvas.bbox("all"))
        self.speaker_entries_frame.bind("<Configure>", _on_entries_configure)

        def _on_canvas_configure(event):
            self._speaker_canvas.itemconfig(self._speaker_canvas_window, width=event.width)
        self._speaker_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            self._speaker_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._speaker_canvas.bind(
            "<Enter>", lambda e: self._speaker_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self._speaker_canvas.bind(
            "<Leave>", lambda e: self._speaker_canvas.unbind_all("<MouseWheel>"))

        self._speaker_name_entries = {}

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

        # HF Token Hinweis anpassen je nach Bundled-Status
        _bundled = _get_bundled_models_dir()
        _diarize_bundled = _bundled and os.path.isdir(os.path.join(_bundled, "diarize"))

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

        if _diarize_bundled:
            hf_hint = "Diarization-Modelle sind integriert — kein Token nötig."
        else:
            hf_hint = ("Für Speaker Diarization benötigt.\n"
                       "Erstelle einen Read-Token auf huggingface.co/settings/tokens")
        ctk.CTkLabel(tab, text=hf_hint,
                     text_color="gray").grid(
            row=2, column=0, columnspan=2, padx=10, pady=(0, 20), sticky="w")

        # Separator
        sep = ctk.CTkFrame(tab, height=2)
        sep.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # KIT ToolBox (Server-Modell) Zugangsdaten
        ctk.CTkLabel(tab, text="KIT ToolBox API-Key:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=4, column=0, padx=10, pady=5, sticky="w")
        self.api_key_var = ctk.StringVar(value=os.getenv("KIT_TOOLBOX_API_KEY", ""))
        self.api_key_entry = ctk.CTkEntry(
            tab, textvariable=self.api_key_var, show="•", width=400)
        self.api_key_entry.grid(row=4, column=1, padx=10, pady=5, sticky="ew")

        self.show_api_key_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            tab, text="API-Key anzeigen", variable=self.show_api_key_var,
            command=self._toggle_api_key_visibility
        ).grid(row=5, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(tab, text="KIT ToolBox Basis-URL:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=6, column=0, padx=10, pady=5, sticky="w")
        self.api_base_url_var = ctk.StringVar(
            value=os.getenv("KIT_TOOLBOX_BASE_URL", DEFAULT_API_BASE_URL))
        ctk.CTkEntry(tab, textvariable=self.api_base_url_var, width=400).grid(
            row=6, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(
            tab,
            text="Für das Server-Modell „KIT ToolBox“ benötigt. Audio wird\n"
                 "dafür an den KIT-Server übertragen (nicht mehr 100% lokal).",
            text_color="gray").grid(
            row=7, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="w")

        # Separator
        sep2 = ctk.CTkFrame(tab, height=2)
        sep2.grid(row=8, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(tab, text="Aufnahme-Ordner:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=9, column=0, padx=10, pady=5, sticky="w")
        self.output_dir_var = ctk.StringVar(value="recordings")
        ctk.CTkEntry(tab, textvariable=self.output_dir_var).grid(
            row=9, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(tab, text="Batch Size:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=10, column=0, padx=10, pady=5, sticky="w")
        self.batch_size_var = ctk.StringVar(value="16")
        ctk.CTkEntry(tab, textvariable=self.batch_size_var, width=80).grid(
            row=10, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(tab, text="Compute:",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=11, column=0, padx=10, pady=5, sticky="w")
        self.compute_var = ctk.StringVar(value="Auto (CUDA/MPS wenn verfügbar)")
        ctk.CTkOptionMenu(
            tab, variable=self.compute_var,
            values=["Auto (CUDA/MPS wenn verfügbar)", "cuda", "mps", "cpu"], width=250
        ).grid(row=11, column=1, padx=10, pady=5, sticky="w")

        # Version info
        ctk.CTkLabel(tab, text=f"{APP_NAME} v{APP_VERSION} — 100% lokal",
                     text_color="gray").grid(
            row=20, column=0, columnspan=2, padx=10, pady=(40, 5), sticky="w")

        # Hardware info
        hw_summary = get_hardware_summary()
        ctk.CTkLabel(tab, text=f"Hardware: {hw_summary}",
                     text_color="gray", justify="left").grid(
            row=21, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="w")
        ctk.CTkLabel(tab, text=f"Empfohlenes Modell: {_recommended_model}",
                     text_color="gray").grid(
            row=22, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="w")

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

        # macOS System-Audio (ScreenCaptureKit) hat keine Geraeteauswahl - es
        # wird immer die gesamte System-Wiedergabe aufgenommen.
        if value == "System-Audio" and sys.platform == "darwin":
            label = "Gesamte System-Wiedergabe (keine Geräteauswahl)"
            self.device_menu.configure(values=[label], state="disabled")
            self.device_var.set(label)
            self._device_map[label] = None
        else:
            self.device_menu.configure(state="normal")
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

        # Pegelanzeige an gewaehlte Quelle anpassen (nur relevante Kanaele zeigen)
        show_mic = value in ("Mikrofon", "Mikrofon + System")
        show_sys = value in ("System-Audio", "Mikrofon + System")
        for channel, visible in (("mic", show_mic), ("system", show_sys)):
            lbl, bar = self._level_rows[channel]
            if visible:
                lbl.grid()
                bar.grid()
            else:
                lbl.grid_remove()
                bar.grid_remove()

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
        self._current_rms = {"mic": 0.0, "system": 0.0}

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
        for bar in self._level_bars.values():
            bar.set(0)

    def _on_level_update(self, channel, rms):
        self._current_rms[channel] = rms

    def _on_recording_done(self, path, duration, error):
        def _update():
            if error:
                if self._timer_id:
                    self.after_cancel(self._timer_id)
                    self._timer_id = None
                self.record_btn.configure(
                    text="⏺  Aufnahme starten",
                    fg_color="#c0392b", hover_color="#e74c3c")
                for bar in self._level_bars.values():
                    bar.set(0)
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

        # Pegel pro Kanal aktualisieren
        for channel, bar in self._level_bars.items():
            level = min(self._current_rms.get(channel, 0.0) / 15000, 1.0)
            bar.set(level)

        self._timer_id = self.after(100, self._update_timer)

    # ------------------------------------------------------- Transcription
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Audio- oder Video-Datei auswählen",
            filetypes=[("Audio & Video", "*.wav *.mp3 *.m4a *.flac *.ogg *.mkv *.mp4 *.mov *.webm *.avi"),
                       ("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg"),
                       ("Video", "*.mkv *.mp4 *.mov *.webm *.avi"),
                       ("Alle Dateien", "*.*")],
            initialdir=self.output_dir_var.get() or "recordings")
        if path:
            self.file_var.set(path)

    def _on_file_selected(self, *_args):
        """Wird bei jeder Aenderung von file_var aufgerufen — steuert, ob der
        Start-Button aktiv ist. Die Transkription startet NIE automatisch,
        nur per Klick auf 'Transkription starten'."""
        path = self.file_var.get()
        valid = bool(path) and os.path.isfile(path)
        self.transcribe_btn.configure(state="normal" if valid else "disabled")
        self.transcribe_status.grid_remove()
        self.file_label.configure(text_color=("black", "white") if valid else "gray")

    def _on_diarize_toggled(self):
        """Min/Max-Sprecher-Felder nur aktiv, wenn Diarization eingeschaltet ist."""
        state = "normal" if self.diarize_var.get() else "disabled"
        self.min_spk_entry.configure(state=state)
        self.max_spk_entry.configure(state=state)
        text_color = ("gray10", "gray90") if state == "normal" else "gray"
        self.min_spk_label.configure(text_color=text_color)
        self.max_spk_label.configure(text_color=text_color)

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
        model_display = self.model_var.get()
        model = MODEL_IDS_BY_DISPLAY.get(model_display, model_display)
        do_diarize = self.diarize_var.get()
        hf_token = self.hf_token_var.get().strip() or None
        api_key = self.api_key_var.get().strip() or os.getenv("KIT_TOOLBOX_API_KEY") or None
        api_base_url = self.api_base_url_var.get().strip() or DEFAULT_API_BASE_URL
        compute = self.compute_var.get()
        device = None
        if compute in ("cuda", "mps", "cpu"):
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
                  min_spk, max_spk, batch_size, device, api_key, api_base_url),
            daemon=True)
        thread.start()

    def _run_transcription(self, audio_path, lang_code, model, do_diarize,
                           hf_token, min_spk, max_spk, batch_size, device,
                           api_key, api_base_url):
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
                api_key=api_key,
                api_base_url=api_base_url,
                on_progress=self._on_transcribe_progress,
            )
            self._transcription_result = result
            self.after(0, lambda: self._on_transcription_done(result, None))
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

    def _on_transcription_done(self, result, error):
        self.progress_bar.grid_remove()
        self.transcribe_btn.grid()

        if error:
            self.transcribe_status.configure(
                text=f"Fehler: {error}", text_color="#e74c3c")
            return

        self.transcribe_status.configure(
            text="Transkription abgeschlossen!", text_color="#27ae60")
        self._render_transcript(result)
        self.save_btn.configure(state="normal")
        self.copy_btn.configure(state="normal")

        # Sprecher-Einträge aufbauen
        self._populate_speaker_entries()

    def _render_transcript(self, result, speaker_names=None):
        """Baut den Transkript-Text auf und faerbt jede Zeile nach Sprecher ein.

        Farben/Tags sind an die rohe Diarization-ID gekoppelt (ueber
        result["speaker_id_map"]), nicht an den gerade angezeigten Namen —
        dadurch bleibt die Farbe erhalten, auch wenn ein Sprecher schon
        automatisch per Voice-Print erkannt oder spaeter umbenannt wurde.
        """
        speaker_names = speaker_names or {}
        speaker_id_map = result.get("speaker_id_map") or {}
        label_to_raw_id = {label: raw_id for raw_id, label in speaker_id_map.items()}

        labels = sorted({
            seg.get("speaker") for seg in result.get("segments", [])
            if seg.get("speaker")
        })
        self._speaker_colors = {
            label_to_raw_id.get(label, label): SPEAKER_COLOR_PALETTE[i % len(SPEAKER_COLOR_PALETTE)]
            for i, label in enumerate(labels)
        }

        self.transcript_text.delete("1.0", "end")
        for seg in result.get("segments", []):
            start = _format_time_short(seg.get("start", 0))
            end = _format_time_short(seg.get("end", 0))
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "")

            if speaker:
                display_name = speaker_names.get(speaker, speaker)
                line = f"[{start} - {end}] {display_name}: {text}\n"
            else:
                line = f"[{start} - {end}] {text}\n"

            insert_start = self.transcript_text.index("end-1c")
            self.transcript_text.insert("end", line)
            if speaker:
                raw_id = label_to_raw_id.get(speaker, speaker)
                tag = f"speaker_{raw_id}"
                color = self._speaker_colors.get(raw_id, "#ffffff")
                self.transcript_text.tag_config(tag, foreground=color)
                self.transcript_text.tag_add(
                    tag, insert_start, self.transcript_text.index("end-1c"))

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
        self._apply_speaker_names()

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

    def _toggle_api_key_visibility(self):
        if self.show_api_key_var.get():
            self.api_key_entry.configure(show="")
        else:
            self.api_key_entry.configure(show="•")
    # ------------------------------------------------- Speaker Renaming
    def _populate_speaker_entries(self):
        """Erstellt Eingabefelder fuer jeden erkannten Sprecher.

        Eintraege sind ueber die rohe Diarization-ID (nicht den angezeigten
        Namen) indiziert, damit "merken" das richtige Voice-Print-Embedding
        findet, auch wenn der Sprecher schon automatisch erkannt und
        umbenannt wurde.
        """
        for w in self.speaker_entries_frame.winfo_children():
            w.destroy()
        self._speaker_name_entries = {}
        self._speaker_remember_vars = {}

        if not hasattr(self, '_transcription_result'):
            self.speaker_frame.grid_remove()
            return

        result = self._transcription_result
        speaker_id_map = result.get("speaker_id_map")
        embeddings = result.get("speaker_embeddings") or {}

        if speaker_id_map:
            rows = sorted(speaker_id_map.items(), key=lambda kv: kv[1])
        else:
            # Kein Voice-Print verfuegbar (z.B. Diarization ohne HF-Token) -
            # Fallback auf die reinen Labels aus den Segmenten.
            labels = sorted({
                seg.get("speaker") for seg in result.get("segments", [])
                if seg.get("speaker")
            })
            rows = [(label, label) for label in labels]

        if not rows:
            self.speaker_frame.grid_remove()
            return

        # Eine Zeile pro Sprecher, im scrollbaren Bereich - egal wie viele
        # es sind (kein hartkodiertes Spalten-/Zeilenlimit), der sichtbare
        # Bereich bleibt durch die feste Canvas-Hoehe konstant.
        for row_idx, (raw_id, current_label) in enumerate(rows):
            frame = ctk.CTkFrame(self.speaker_entries_frame, fg_color="transparent")
            frame.grid(row=row_idx, column=0, sticky="w", pady=2)

            color = self._speaker_colors.get(raw_id) if hasattr(self, "_speaker_colors") else None
            ctk.CTkLabel(frame, text=f"{current_label}  →", width=160, anchor="w",
                         font=ctk.CTkFont(size=12),
                         text_color=color or ("gray10", "gray90")).pack(side="left", padx=(0, 5))

            entry = ctk.CTkEntry(frame, width=160, placeholder_text="Name eingeben")
            if raw_id != current_label:
                entry.insert(0, current_label)
            entry.pack(side="left")
            self._speaker_name_entries[raw_id] = entry

            if raw_id in embeddings:
                remember_var = ctk.BooleanVar(value=True)
                ctk.CTkCheckBox(frame, text="merken", variable=remember_var,
                                width=20, font=ctk.CTkFont(size=11)
                                ).pack(side="left", padx=(10, 0))
                self._speaker_remember_vars[raw_id] = remember_var

        self.speaker_frame.grid()
        self._speaker_canvas.update_idletasks()
        self._speaker_canvas.configure(scrollregion=self._speaker_canvas.bbox("all"))
        self._speaker_canvas.yview_moveto(0)

    def _get_speaker_mapping(self):
        """Gibt das aktuelle Mapping {rohe_Diarization_ID: neuer_Name} zurueck."""
        mapping = {}
        for raw_id, entry in self._speaker_name_entries.items():
            name = entry.get().strip()
            if name:
                mapping[raw_id] = name
        return mapping


    def _apply_speaker_names(self):
        """Wendet die eingegebenen Sprecher-Namen auf Textfeld UND internes
        Ergebnis an und merkt sich optional die Stimme fuer zukuenftige
        Erkennung.

        Einzige Stelle, die Umbenennungen vornimmt - wird sowohl vom
        "Anwenden"-Button als auch automatisch vor dem Speichern aufgerufen,
        damit Textfeld-Anzeige und das gespeicherte Ergebnis nie auseinander-
        laufen koennen. Ersetzt im Textfeld gezielt nur die Label-Vorkommen
        (per Suche), statt den ganzen Text neu aufzubauen - dadurch bleiben
        sowohl die Sprecherfarben als auch etwaige manuelle Korrekturen im
        Transkript-Text erhalten.
        """
        mapping = self._get_speaker_mapping()  # {rohe_ID: neuer_Name}
        if not mapping or not hasattr(self, "_transcription_result"):
            return

        result = self._transcription_result
        speaker_id_map = result.setdefault("speaker_id_map", {})
        embeddings = result.get("speaker_embeddings") or {}

        for raw_id, new_name in mapping.items():
            old_label = speaker_id_map.get(raw_id, raw_id)
            if old_label == new_name:
                continue

            # Textfeld: gezielt ersetzen (Farbe/Tag bleibt erhalten)
            tag = f"speaker_{raw_id}"
            search_from = "1.0"
            while True:
                pos = self.transcript_text.search(old_label, search_from, stopindex="end")
                if not pos:
                    break
                end_pos = f"{pos}+{len(old_label)}c"
                self.transcript_text.delete(pos, end_pos)
                self.transcript_text.insert(pos, new_name, tag)
                search_from = f"{pos}+{len(new_name)}c"

            # Internes Ergebnis (fuer Speichern) synchron mitziehen
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

            remember_var = self._speaker_remember_vars.get(raw_id)
            if remember_var is not None and remember_var.get() and raw_id in embeddings:
                speaker_profiles.enroll_speaker(new_name, embeddings[raw_id])

        self.transcribe_status.configure(
            text="Sprecher-Namen angewendet!", text_color="#27ae60")

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
