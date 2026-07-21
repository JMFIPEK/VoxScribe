# VoxScribe — ORION

**VoxScribe** ist die Spracheingabe-Komponente für **ORION** (Obsidian Retrieval for Information and Organized Notes).

Lokale Audio-Aufnahme und Transkription mit [WhisperX](https://github.com/m-bain/whisperX). Unterstützt Mikrofon- und System-Audio-Aufnahme (z.B. Teams/Zoom via WASAPI Loopback) sowie Video-Dateien (MKV, MP4, ...) als Eingabe, mit anschließender Transkription inkl. Speaker Diarization und Sprecher-Wiedererkennung. Läuft standardmäßig 100 % lokal nach einmaligem Modell-Download — optional kann die Transkription auch an einen gehosteten Server (KIT ToolBox) ausgelagert werden.

## Features

- **Zwei GUIs** — **PySide6/Qt** (`gui_qt.py`, primäre GUI, aktiv weiterentwickelt) und die ursprüngliche **CustomTkinter**-GUI (`gui.py`, funktioniert weiterhin, bekommt aber keine neuen Features mehr)
- **Mikrofon-Aufnahme** — direktes Aufnehmen von Gesprächen (alle Plattformen)
- **System-Audio (Loopback)** — Aufnahme von Teams/Zoom/Webex über WASAPI (Windows) bzw. ScreenCaptureKit (macOS, experimentell)
- **Mikrofon + System-Audio** — beide Quellen gleichzeitig für vollständige Meeting-Aufnahmen (Windows; auf macOS implementiert, aber noch nicht auf echter Hardware verifiziert)
- **Video-Dateien als Eingabe** — MKV, MP4, MOV, WebM, AVI werden direkt transkribiert (nur die Audiospur wird extrahiert, kein ffmpeg nötig)
- **Transkription** — WhisperX (`large-v2`/`large-v3` auf GPU) unter Windows/Linux; unter macOS läuft die Transkription stattdessen über **Apples SpeechAnalyzer** (Speech-Framework, Neural Engine, macOS 26+) — kein Modell-Download, keine Modellwahl nötig, siehe [macOS-Hinweise](#macos-installation)
- **KIT ToolBox (Server)** — alternativ (alle Plattformen): Transkription über einen gehosteten Whisper-Endpunkt statt lokal (siehe [Server-Modell](#kit-toolbox-server-modell))
- **Automatische Spracherkennung** — Sprache muss nicht manuell gewählt werden ("Automatisch erkennen"); auf macOS/SpeechAnalyzer nicht verfügbar, dort wird "Deutsch" angenommen, falls keine Sprache gewählt ist
- **Detaillierter Fortschritt** — Fortschrittsbalken pro Pipeline-Schritt (Transkription, Alignment, Diarization), auch in der CLI
- **Word-Level Timestamps** — exakte Wort-Zeitstempel via Forced Alignment (wav2vec2) unter Windows/Linux, bzw. direkt von SpeechAnalyzer unter macOS (dort entfällt der Alignment-Schritt komplett)
- **Speaker Diarization** — Sprecherzuordnung via pyannote-audio, farbcodiert im Transkript (alle Plattformen unverändert, auch unter macOS)
- **Sprecher-Wiedererkennung (Voice-Prints)** — einmal benannte Sprecher werden bei zukünftigen Aufnahmen automatisch anhand ihrer Stimme wiedererkannt
- **Sprecher umbenennen** — nach Transkription können SPEAKER_00 etc. durch echte Namen ersetzt werden
- **Ausgabeformate** — TXT, SRT (Untertitel), JSON
- **Live-Pegel** — Lautstärke-Anzeige während der Aufnahme
- **CLI** — vollständige Kommandozeilen-Bedienung alternativ zur GUI

## Voraussetzungen

### Windows (primäre Zielplattform, voller Funktionsumfang)

- Windows 10/11
- Python 3.11
- NVIDIA GPU mit CUDA 12.8 (z.B. RTX 5000, RTX 4090, ...)
- [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive)
- **[uv](https://docs.astral.sh/uv/)** — schneller Python-Paketmanager (optional, aber empfohlen)

### macOS (Apple Silicon)

- macOS 26 (Tahoe) oder neuer — SpeechAnalyzer (Transkription) und ScreenCaptureKit (System-Audio) sind beides macOS-26+-APIs
- Apple Silicon (M1 oder neuer)
- Python 3.11+
- Xcode Command Line Tools (`xcode-select --install`) — zum Bauen der beiden Swift-Helfer in `macos/`
- **[uv](https://docs.astral.sh/uv/)** — empfohlen, siehe [Installation](#macos-installation) unten
- Keine NVIDIA-GPU/CUDA nötig — Transkription läuft über die Neural Engine (SpeechAnalyzer), Alignment/Diarization nutzen MPS (Apple GPU)

### Linux (experimentell)

- Python 3.11+
- Nur Mikrofon-Aufnahme, kein System-Audio-Weg
- NVIDIA GPU mit CUDA 12.8 empfohlen für WhisperX (sonst CPU)

## Installation

### Windows

#### Option 1: Mit uv (empfohlen)

`pyproject.toml` pinnt `torch`/`torchaudio`/`torchvision` bereits auf den CUDA-12.8-Index, daher reicht ein einziger Befehl:

```bash
# 1. uv installieren (falls noch nicht vorhanden)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Abhängigkeiten installieren (erstellt .venv automatisch, inkl. CUDA-PyTorch)
uv sync
```

> Falls `torch.__version__` danach trotzdem mit `+cpu` endet (z.B. nach einem manuellen `uv pip install torch` davor), hilft ein erzwungenes Neuinstallieren:
> ```bash
> uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 --reinstall
> ```
> (`uv pip install` ohne `--reinstall` hält ein bereits installiertes Paket für "erfüllt" und prüft die Build-Variante nicht.)

#### Option 2: Mit venv (Standard Python)

```bash
# 1. Virtuelle Umgebung erstellen
python -m venv .venv
.\.venv\Scripts\activate

# 2. PyTorch mit CUDA 12.8 installieren
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Weitere Abhängigkeiten installieren
pip install -r requirements.txt

# 4. PyTorch CUDA-Version sicherstellen
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

### macOS-Installation

Es gibt kein CUDA auf dem Mac — `pyproject.toml`/`requirements.txt` markieren den CUDA-Torch-Index bereits als `win32`/`linux`-only, macOS bekommt automatisch normales PyPI-`torch` mit MPS-Unterstützung. Ein einziger `uv sync` reicht daher auch hier:

```bash
# 1. uv installieren (falls noch nicht vorhanden)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Xcode Command Line Tools installieren (falls noch nicht vorhanden) -
#    noetig, um die beiden Swift-Helfer in macos/ zu bauen
xcode-select --install

# 3. Abhängigkeiten installieren (erstellt .venv automatisch, inkl. PySide6)
uv sync

# 4. Swift-Helfer bauen (System-Audio-Aufnahme + SpeechAnalyzer-Transkription)
cd macos
./build.sh
cd ..
```

Danach venv aktivieren mit `source .venv/bin/activate` (oder `uv run python ...` ohne Aktivierung).

**Wichtig für macOS:**
- Ohne den `macos/build.sh`-Schritt funktionieren weder System-Audio-Aufnahme noch Transkription — beide rufen kompilierte Binaries in `macos/` als Subprozess auf (`macos/SystemAudioCapture`, `macos/SpeechAnalyzerTranscribe`), die nicht im Repo mitgeliefert werden (siehe `.gitignore`).
- Die Transkription läuft über Apples `SpeechAnalyzer` (Speech-Framework) statt WhisperX — es gibt daher **keine Modellwahl** in der GUI unter macOS und `download_models.py`/das GPU-Speicher-Kapitel weiter unten sind für macOS nicht relevant.
- System-Audio-Aufnahme erfordert die Berechtigung „Bildschirm- und Systemaudioaufnahme“ (Systemeinstellungen → Datenschutz & Sicherheit) für den Prozess, der VoxScribe ausführt (Terminal beim Testen, später die gepackte App) — macOS fragt das beim ersten Versuch ab.
- Speaker Diarization (pyannote) läuft unverändert wie unter Windows/Linux, nur eben mit MPS statt CUDA — siehe [HuggingFace Token](#huggingface-token-für-speaker-diarization) unten, der wird weiterhin gebraucht.
- Details/Hintergrund zu beiden Swift-Helfern: `macos/README.md`.

### HuggingFace Token (für Speaker Diarization)

1. Token erstellen: https://huggingface.co/settings/tokens → **Read**-Berechtigung
2. Modelllizenzen akzeptieren:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Token in `.env` eintragen:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
   ```

> Nach dem ersten Modell-Download (~3 GB) läuft alles vollständig offline.

### KIT ToolBox (Server-Modell)

Optional: statt eines lokalen Whisper-Modells kann die Transkription an einen gehosteten,
OpenAI-kompatiblen Endpunkt geschickt werden (Modellauswahl "KIT ToolBox (Server)"). Dafür
in `.env` eintragen:

```
KIT_TOOLBOX_API_KEY=xxx             # erforderlich für das Server-Modell
KIT_TOOLBOX_BASE_URL=https://...    # optional, Default: https://ki-toolbox.scc.kit.edu/api/v1
```

Alignment und Speaker Diarization laufen weiterhin lokal auf den zurückgegebenen
Segmenten — nur die reine Transkription verlässt in diesem Fall den Rechner.

## Verwendung

### GUI starten

```bash
.\.venv\Scripts\activate     # Windows
source .venv/bin/activate    # macOS/Linux
python gui_qt.py              # PySide6-GUI (primär, empfohlen)
python gui.py                 # CustomTkinter-GUI (legacy)
```

Oder mit uv (ohne manuelle Aktivierung):
```bash
uv run python gui_qt.py
```

Beide GUIs teilen sich denselben Backend-Code (`recorder.py`, `transcriber.py`, ...) und bieten funktional dasselbe — Tab-/Seitenaufteilung und Bedienung sind im Folgenden am Beispiel von `gui_qt.py` beschrieben, `gui.py` ist nahezu identisch aufgebaut. Auf macOS zeigt die Transkriptions-Seite **keine** Modellwahl (siehe unten) — das ist kein Bug, sondern weil Apples SpeechAnalyzer dort die einzige Transkriptions-Engine ist.

Die GUI hat drei Tabs:

#### Tab: Aufnahme

- **Quelle wählen**: Mikrofon, System-Audio oder Mikrofon + System (für vollständige Meetings)
- **Gerät auswählen**: Dropdown mit erkannten Audio-Geräten
- **Start/Stop-Button**: Aufnahme starten und stoppen
- **Live-Pegelanzeige + Timer**: Lautstärke und Aufnahmedauer in Echtzeit
- **Auto-Transkription**: Optional nach Aufnahme automatisch transkribieren

#### Tab: Transkription

- **Audio-/Video-Datei wählen**: Datei-Picker (WAV, MP3, MKV, MP4, ...) oder automatisch nach Aufnahme
- **Sprache**: "Automatisch erkennen" (Default, nicht verfügbar unter macOS), Deutsch, Englisch, Französisch, ...
- **Modell** (nur Windows/Linux): lokale Whisper-Modelle (`large-v3`, `large-v2`, `medium`, `base`) oder "KIT ToolBox (Server)". **Unter macOS entfällt diese Auswahl komplett** — dort wird immer Apples SpeechAnalyzer verwendet
- **Speaker Diarization**: Ein/Aus + Min/Max Sprecheranzahl (Felder deaktivieren sich automatisch, wenn Diarization aus ist) — auf allen Plattformen identisch, auch unter macOS
- **Start-Button**: Die Transkription startet ausschließlich per Klick auf "Transkription starten" — nie automatisch beim Auswählen einer Datei
- **Fortschrittsanzeige**: Detaillierter Balken pro Pipeline-Schritt
- **Farbcodiertes Transkript**: jeder Sprecher bekommt automatisch eine eigene Farbe
- **Sprecher umbenennen & merken**: SPEAKER_00 → echter Name zuweisen; bereits erkannte Sprecher (Voice-Print-Abgleich) werden mit vorausgefülltem Namen angezeigt. Die Checkbox "merken" (Standard: an) speichert/aktualisiert das Stimmprofil, damit dieselbe Person in zukünftigen Aufnahmen automatisch erkannt wird
- **Speichern**: TXT, SRT und/oder JSON + Kopieren in Zwischenablage

#### Tab: Einstellungen

- HuggingFace Token
- KIT ToolBox API-Key + Basis-URL (für das Server-Modell)
- Aufnahme-Ordner, Batch Size, Compute-Device

### CLI

Virtuelle Umgebung aktivieren:
```bash
.\.venv\Scripts\activate      # Windows
source .venv/bin/activate     # macOS/Linux
```

Oder mit uv (ohne Aktivierung):
```bash
uv run python main.py <befehl>
```

### Audio-Geräte anzeigen

```bash
python main.py devices
```

### Aufnahme

```bash
# Mikrofon (stoppt mit ENTER)
python main.py record --source mic

# System-Audio / Teams / Zoom (WASAPI Loopback)
python main.py record --source system

# Bestimmtes Mikrofon auswählen (Index aus 'devices')
python main.py record --source mic --device 15
```

### Transkription

```bash
# Standard (Deutsch, mit Diarization)
python main.py transcribe -i recordings/aufnahme.wav

# Englisch
python main.py transcribe -i recordings/meeting.wav -l en

# Ohne Speaker Diarization (schneller)
python main.py transcribe -i recordings/aufnahme.wav --no-diarize

# Mehrere Ausgabeformate
python main.py transcribe -i recordings/aufnahme.wav -f txt,srt,json

# Sprecheranzahl eingrenzen
python main.py transcribe -i recordings/meeting.wav --min-speakers 2 --max-speakers 4

# Kleineres Modell (weniger VRAM, schneller, weniger genau)
python main.py transcribe -i recordings/aufnahme.wav -m medium

# Automatische Spracherkennung statt fester Sprache
python main.py transcribe -i recordings/aufnahme.wav -l auto

# Video-Datei direkt transkribieren (nur Audiospur wird extrahiert)
python main.py transcribe -i recordings/meeting.mkv

# Über KIT ToolBox (Server) statt lokalem Modell transkribieren
python main.py transcribe -i recordings/aufnahme.wav -m server:kit.whisper-large-v3
```

### Aufnahme + sofortige Transkription

```bash
# Mikrofon → direkt transkribieren
python main.py run --source mic

# Teams-Meeting aufnehmen und transkribieren
python main.py run --source system

# Mit allen Optionen
python main.py run --source system -l de -m large-v2 --min-speakers 2 --max-speakers 2 -f txt,srt
```

## Projektstruktur

```
├── gui_qt.py              # GUI (PySide6/Qt) — primär, aktiv weiterentwickelt
├── qt_app/                # PySide6-GUI-Code: main_window.py, controllers.py, theme.py, pages/, widgets/
├── gui.py                 # GUI (CustomTkinter) — legacy, funktioniert weiterhin
├── main.py                # CLI Entry Point
├── recorder.py            # Audio-Aufnahme (Mikrofon + WASAPI Loopback/ScreenCaptureKit + kombiniert)
├── transcriber.py         # WhisperX/Apple-SpeechAnalyzer Transkription + Alignment + Diarization + Server-Modell + Video-Input
├── speaker_profiles.py    # Persistente Sprecher-Profile (Voice-Prints) fuer Wiedererkennung
├── hardware_detect.py     # GPU/CPU-Erkennung + Modellempfehlung (Windows/Linux)
├── download_models.py     # Modelle fuer Offline-/Bundled-Betrieb vorladen (Windows/Linux, WhisperX)
├── build_exe.py           # PyInstaller-Build fuer eigenstaendige .exe (Windows)
├── macos/                 # macOS-Swift-Helfer: SystemAudioCapture.swift, SpeechAnalyzerTranscribe.swift, build.sh
├── pyproject.toml         # Python-Abhängigkeiten (inkl. CUDA-PyTorch-Pinning), primäre Quelle für `uv`
├── requirements.txt       # Python-Abhängigkeiten (alternativ für `pip`/Option 2)
├── .env                   # HuggingFace Token, KIT ToolBox API-Key (nicht committen!)
├── .gitignore
└── recordings/            # Aufnahmen + Transkripte
```

## CLI-Referenz

| Befehl | Beschreibung |
|--------|-------------|
| `devices` | Verfügbare Audio-Geräte auflisten |
| `record` | Audio aufnehmen |
| `transcribe` | Audio-Datei transkribieren |
| `run` | Aufnahme + sofortige Transkription |

### Optionen

| Option | Default | Beschreibung |
|--------|---------|-------------|
| `--source` | `mic` | `mic`, `system` (WASAPI Loopback) oder `both` (Mikrofon + System) |
| `--language`, `-l` | `de` | Sprache (de, en, fr, es, ...) oder `auto` für automatische Erkennung |
| `--model`, `-m` | `large-v2` (Windows/Linux), `apple:speechanalyzer` (macOS) | Whisper-Modell (large-v2, large-v3, medium, base), Server-Modell (`server:kit.whisper-large-v3`) oder `apple:speechanalyzer` (nur macOS) |
| `--diarize` / `--no-diarize` | `--diarize` | Speaker Diarization an/aus |
| `--min-speakers` | – | Minimale Sprecheranzahl |
| `--max-speakers` | – | Maximale Sprecheranzahl |
| `--batch-size` | `16` | Batch-Größe (kleiner = weniger VRAM) |
| `--device-compute` | auto | `cuda` oder `cpu` |
| `--api-key` | – | API-Key für Server-Modelle (alternativ: `KIT_TOOLBOX_API_KEY` in `.env`) |
| `--api-base-url` | – | Basis-URL für Server-Modelle (alternativ: `KIT_TOOLBOX_BASE_URL` in `.env`) |
| `--format`, `-f` | `txt` | Ausgabeformate: txt, srt, json (kommagetrennt) |
| `--output`, `-o` | auto | Ausgabepfad |

## GPU-Speicher

Gilt für Windows/Linux (WhisperX). Unter macOS läuft die Transkription über Apples SpeechAnalyzer (Neural Engine) statt lokaler Whisper-Modelle — diese Tabelle ist dort nicht relevant, nur Alignment/Diarization brauchen dort (MPS-)Speicher, deutlich weniger als ein Whisper-Modell.

| Modell | VRAM |
|--------|------|
| `base` | ~1 GB |
| `medium` | ~5 GB |
| `large-v2` | ~8 GB |
| `large-v3` | ~8 GB |
| KIT ToolBox (Server) | kein lokales VRAM für die Transkription selbst nötig — Alignment und Diarization laufen aber weiterhin lokal (deutlich kleinerer Bedarf, ca. wie `base`) |

Die lokalen Modelle werden sequenziell geladen und entladen (Whisper → Alignment → Diarization), um den VRAM optimal zu nutzen.

## Hinweise

- Beim **ersten Start** werden Modelle von HuggingFace heruntergeladen (~3 GB, nur Windows/Linux/WhisperX). Danach läuft alles offline (außer bei Nutzung des KIT-ToolBox-Server-Modells). Unter macOS gibt es keinen Whisper-Modell-Download — Apples SpeechAnalyzer-Sprachmodell wird bei Bedarf einmalig über das Betriebssystem selbst geladen (`AssetInventory`, siehe `macos/SpeechAnalyzerTranscribe.swift`).
- **Mikrofon + System** nimmt beide Quellen gleichzeitig auf und mischt sie — ideal für vollständige Meeting-Transkription. Unter macOS implementiert, aber noch nicht auf echter Hardware verifiziert (siehe `macos/README.md`).
- **System-Audio** nimmt auf, was über die Lautsprecher/Kopfhörer ausgegeben wird — ideal für Teams/Zoom. Unter macOS experimentell (ScreenCaptureKit) und erfordert die Berechtigung „Bildschirm- und Systemaudioaufnahme“.
- In der GUI stoppt die Aufnahme per **Button**, in der CLI mit **ENTER** oder **Ctrl+C**.
- Transkripte werden im `recordings/`-Ordner gespeichert.
- **Video-Dateien** (MKV, MP4, MOV, WebM, AVI) können direkt ausgewählt werden — nur die Audiospur wird extrahiert, kein separates ffmpeg nötig.
- **Sprecher umbenennen & merken**: Nach der Transkription kann man in der GUI SPEAKER_00 etc. durch echte Namen ersetzen. Ist die Checkbox "merken" aktiv, wird die Stimme als Profil gespeichert (`~/.voxscribe/speaker_profiles.json`) und bei zukünftigen Aufnahmen automatisch wiedererkannt — das ist ein heuristischer Stimmabgleich, gelegentliche Fehlzuordnungen (v.a. bei kurzen/leisen Segmenten) sind möglich und sollten vor dem Speichern geprüft werden.
- **macOS**: keine automatische Spracherkennung bei der Transkription (SpeechAnalyzer kennt keine Sprach-Auto-Detection) — ohne explizite Sprachwahl wird Deutsch angenommen. Diarization (Sprechererkennung) funktioniert unverändert, da sie weiterhin über pyannote läuft, nicht über SpeechAnalyzer.

## Firmen-Proxy / SSL

Bei SSL-Problemen im Firmennetz wird die SSL-Verifikation automatisch deaktiviert (konfiguriert in `main.py`). Für pip-Installationen ggf. `--trusted-host pypi.org --trusted-host files.pythonhosted.org` ergänzen.
