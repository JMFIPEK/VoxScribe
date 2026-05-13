# WhisperX Recorder & Transcriber

Lokale Audio-Aufnahme und Transkription mit [WhisperX](https://github.com/m-bain/whisperX). Unterstützt Mikrofon- und System-Audio-Aufnahme (z.B. Teams/Zoom via WASAPI Loopback) mit anschließender Transkription inkl. Speaker Diarization. Läuft 100 % lokal nach einmaligem Modell-Download.

## Features

- **Mikrofon-Aufnahme** — direktes Aufnehmen von Gesprächen
- **System-Audio (Loopback)** — Aufnahme von Teams/Zoom/Webex über WASAPI
- **WhisperX-Transkription** — schnelle Batch-Inference mit `large-v2` auf GPU
- **Word-Level Timestamps** — exakte Wort-Zeitstempel via Forced Alignment (wav2vec2)
- **Speaker Diarization** — Sprecherzuordnung via pyannote-audio
- **Ausgabeformate** — TXT, SRT (Untertitel), JSON
- **Live-Pegel** — Lautstärke-Anzeige während der Aufnahme

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
pip install whisperx PyAudioWPatch soundfile numpy python-dotenv

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
├── main.py              # CLI Entry Point
├── recorder.py          # Audio-Aufnahme (Mikrofon + WASAPI Loopback)
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
| `--source` | `mic` | `mic` (Mikrofon) oder `system` (WASAPI Loopback) |
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
- **System-Audio** nimmt auf, was über die Lautsprecher/Kopfhörer ausgegeben wird — ideal für Teams/Zoom.
- Die Aufnahme stoppt mit **ENTER** oder **Ctrl+C**.
- Transkripte werden im `recordings/`-Ordner gespeichert.

## Firmen-Proxy / SSL

Bei SSL-Problemen im Firmennetz wird die SSL-Verifikation automatisch deaktiviert (konfiguriert in `main.py`). Für pip-Installationen ggf. `--trusted-host pypi.org --trusted-host files.pythonhosted.org` ergänzen.
