# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VoxScribe** is a German-language desktop application for local audio recording and transcription using WhisperX. It supports microphone recording, system audio (WASAPI Loopback for Teams/Zoom), or both simultaneously, with speaker diarization via pyannote-audio.

## Quick Commands

```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Or use uv (no activation needed)
uv run python <script>

# List available audio devices
python main.py devices

# Record from microphone (stop with ENTER)
python main.py record --source mic

# Record system audio (WASAPI Loopback)
python main.py record --source system

# Transcribe an audio file
python main.py transcribe -i recordings/audio.wav

# Full pipeline: record + transcribe
python main.py run --source system

# Start GUI
python gui.py

# Download models for offline/bundled operation
python download_models.py

# Build standalone .exe (requires PyInstaller)
python build_exe.py
```

## Architecture

### Core Modules

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point with subcommands: `devices`, `record`, `transcribe`, `run` |
| `gui.py` | CustomTkinter GUI with tabs: Aufnahme, Transkription, Einstellungen, Info |
| `recorder.py` | Audio recording via PyAudioWPatch: microphone, WASAPI loopback, or mixed |
| `transcriber.py` | WhisperX pipeline: transcription → alignment → diarization (sequential loading) |
| `hardware_detect.py` | GPU/CPU detection and model recommendations based on VRAM/RAM |
| `download_models.py` | Pre-download models to `bundled_models/` for offline operation |
| `build_exe.py` | PyInstaller build script for creating standalone .exe |

### Pipeline Flow

```
Recording (recorder.py)
    ↓
Audio WAV file (16kHz mono)
    ↓
Transcription (transcriber.py)
    ├── 1. WhisperX inference (0-40%)
    ├── 2. Forced Alignment with wav2vec2 (40-65%)
    └── 3. Speaker Diarization with pyannote (65-100%)
    ↓
Output: TXT, SRT, JSON
```

### Key Design Patterns

- **Sequential model loading**: Models are loaded/unloaded sequentially (Whisper → Alignment → Diarization) to minimize VRAM usage
- **SSL bypass for corporate proxy**: All HTTPS verification is disabled at module load to work behind corporate proxies
- **Bundled models support**: Models can be pre-downloaded to `bundled_models/` and used offline without HF_TOKEN
- **Threaded recording**: Audio recording runs in background threads with callbacks for GUI updates

## Configuration

### Environment Variables (.env)

```
HF_TOKEN=hf_xxx  # Required for speaker diarization (pyannote models)
```

### Hardware Requirements

- Windows 10/11
- NVIDIA GPU with CUDA 12.8 (recommended for large-v2/v3 models)
- Python 3.11+ (managed via `uv` or system installation)

### Model Selection by VRAM

| VRAM | Recommended Model |
|------|-------------------|
| ≥8 GB | large-v3 |
| ≥5 GB | medium |
| <5 GB | base |

## Output Formats

- **TXT**: Timestamped transcript with speaker labels
- **SRT**: Subtitle format for video players
- **JSON**: Full WhisperX result with word-level timestamps

## Important Notes

- First run downloads ~3GB of models from HuggingFace
- For diarization, accept model licenses at:
  - https://huggingface.co/pyannote/speaker-diarization-3.1
  - https://huggingface.co/pyannote/segmentation-3.0
- The app uses `PyAudioWPatch` (not standard PyAudio) for Windows WASAPI support
- All processing is local; no data leaves the machine after initial model download
