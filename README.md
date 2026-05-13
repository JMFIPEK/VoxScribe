# VoxScribe — ORION

**VoxScribe** ist die Spracheingabe-Komponente für **ORION** (Obsidian Retrieval for Information and Organized Notes).

Lokale Audio-Aufnahme und Transkription mit [WhisperX](https://github.com/m-bain/whisperX). Unterstützt Mikrofon- und System-Audio-Aufnahme (z.B. Teams/Zoom via WASAPI Loopback) mit anschließender Transkription inkl. Speaker Diarization. Läuft 100 % lokal nach einmaligem Modell-Download.

## Features

- **GUI (CustomTkinter)** — native Desktop-App mit Dark Mode
- **Mikrofon-Aufnahme** — direktes Aufnehmen von Gesprächen
- **System-Audio (Loopback)** — Aufnahme von Teams/Zoom/Webex über WASAPI
- **Mikrofon + System-Audio** — beide Quellen gleichzeitig für vollständige Meeting-Aufnahmen
- **WhisperX-Transkription** — schnelle Batch-Inference mit `large-v2` auf GPU
- **Detaillierter Fortschritt** — Fortschrittsbalken pro Pipeline-Schritt (Transkription, Alignment, Diarization)
- **Word-Level Timestamps** — exakte Wort-Zeitstempel via Forced Alignment (wav2vec2)
- **Speaker Diarization** — Sprecherzuordnung via pyannote-audio
- **Sprecher umbenennen** — nach Transkription können SPEAKER_00 etc. durch echte Namen ersetzt werden
- **Ausgabeformate** — TXT, SRT (Untertitel), JSON
- **Live-Pegel** — Lautstärke-Anzeige während der Aufnahme
- **CLI** — vollständige Kommandozeilen-Bedienung alternativ zur GUI

## Voraussetzungen

- Windows 10/11
- Python 3.11
- NVIDIA GPU mit CUDA 12.8 (z.B. RTX 5000, RTX 4090, ...)
- [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive)
- [Miniconda](https://docs.anaconda.com/miniconda/)

## Installation

```bash
# 1. Conda-Umgebung erstellen
conda create -n whisperx python=3.11 -y
conda activate whisperx

# 2. PyTorch mit CUDA 12.8 installieren
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Weitere Abhängigkeiten installieren
pip install whisperx PyAudioWPatch soundfile scipy numpy python-dotenv customtkinter

# 4. PyTorch CUDA-Version sicherstellen (whisperx überschreibt manchmal mit CPU-Version)
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

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

## Verwendung

### GUI starten

```bash
conda activate whisperx
python gui.py
```

Die GUI hat drei Tabs:

#### Tab: Aufnahme

- **Quelle wählen**: Mikrofon, System-Audio oder Mikrofon + System (für vollständige Meetings)
- **Gerät auswählen**: Dropdown mit erkannten Audio-Geräten
- **Start/Stop-Button**: Aufnahme starten und stoppen
- **Live-Pegelanzeige + Timer**: Lautstärke und Aufnahmedauer in Echtzeit
- **Auto-Transkription**: Optional nach Aufnahme automatisch transkribieren

#### Tab: Transkription

- **Audio-Datei wählen**: Datei-Picker oder automatisch nach Aufnahme
- **Sprache / Modell**: Deutsch, Englisch, Französisch, ... + Modellauswahl
- **Speaker Diarization**: Ein/Aus + Min/Max Sprecheranzahl
- **Fortschrittsanzeige**: Detaillierter Balken pro Pipeline-Schritt
- **Sprecher umbenennen**: SPEAKER_00 → echter Name zuweisen und auf Text anwenden
- **Speichern**: TXT, SRT und/oder JSON + Kopieren in Zwischenablage

#### Tab: Einstellungen

- HuggingFace Token
- Aufnahme-Ordner, Batch Size, Compute-Device

### CLI

```bash
conda activate whisperx
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
├── gui.py               # GUI (CustomTkinter)
├── main.py              # CLI Entry Point
├── recorder.py          # Audio-Aufnahme (Mikrofon + WASAPI Loopback + kombiniert)
├── transcriber.py       # WhisperX Transkription + Alignment + Diarization
├── requirements.txt     # Python-Abhängigkeiten
├── .env                 # HuggingFace Token (nicht committen!)
├── .gitignore
└── recordings/          # Aufnahmen + Transkripte
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
| `--language`, `-l` | `de` | Sprache (de, en, fr, es, ...) |
| `--model`, `-m` | `large-v2` | Whisper-Modell (large-v2, large-v3, medium, base) |
| `--diarize` / `--no-diarize` | `--diarize` | Speaker Diarization an/aus |
| `--min-speakers` | – | Minimale Sprecheranzahl |
| `--max-speakers` | – | Maximale Sprecheranzahl |
| `--batch-size` | `16` | Batch-Größe (kleiner = weniger VRAM) |
| `--device-compute` | auto | `cuda` oder `cpu` |
| `--format`, `-f` | `txt` | Ausgabeformate: txt, srt, json (kommagetrennt) |
| `--output`, `-o` | auto | Ausgabepfad |

## GPU-Speicher

| Modell | VRAM |
|--------|------|
| `base` | ~1 GB |
| `medium` | ~5 GB |
| `large-v2` | ~8 GB |
| `large-v3` | ~8 GB |

Die Modelle werden sequenziell geladen und entladen (Whisper → Alignment → Diarization), um den VRAM optimal zu nutzen.

## Hinweise

- Beim **ersten Start** werden Modelle von HuggingFace heruntergeladen (~3 GB). Danach läuft alles offline.
- **Mikrofon + System** nimmt beide Quellen gleichzeitig auf und mischt sie — ideal für vollständige Meeting-Transkription.
- **System-Audio** nimmt auf, was über die Lautsprecher/Kopfhörer ausgegeben wird — ideal für Teams/Zoom.
- In der GUI stoppt die Aufnahme per **Button**, in der CLI mit **ENTER** oder **Ctrl+C**.
- Transkripte werden im `recordings/`-Ordner gespeichert.
- **Sprecher umbenennen**: Nach der Transkription kann man in der GUI SPEAKER_00 etc. durch echte Namen ersetzen.

## Firmen-Proxy / SSL

Bei SSL-Problemen im Firmennetz wird die SSL-Verifikation automatisch deaktiviert (konfiguriert in `main.py`). Für pip-Installationen ggf. `--trusted-host pypi.org --trusted-host files.pythonhosted.org` ergänzen.
