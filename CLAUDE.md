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
| `transcriber.py` | WhisperX pipeline: transcription → alignment → diarization (sequential loading); also handles video-container audio extraction (PyAV) and the remote/server model |
| `hardware_detect.py` | GPU/CPU detection and model recommendations based on VRAM/RAM |
| `speaker_profiles.py` | Persistent voice-print storage (`~/.voxscribe/speaker_profiles.json`) and cosine-similarity speaker matching/enrollment |
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
- **Universal audio input**: `transcriber.load_audio_universal()` tries `soundfile` first (fast path for WAV/FLAC/OGG), then falls back to PyAV (`av` package) to demux just the audio stream out of video containers (MKV, MP4, MOV, ...) — no ffmpeg CLI required
- **Remote/server model**: model IDs prefixed with `server:` (e.g. `server:kit.whisper-large-v3`) skip local WhisperX inference and instead POST the audio to an OpenAI-compatible `/audio/transcriptions` endpoint (`transcriber.transcribe_remote`); alignment and diarization still run locally on top of the returned segments. Long recordings are chunked (FLAC-encoded, ~180s each) with automatic halving on HTTP 413 to stay under server upload limits. Selectable in the GUI as "KIT ToolBox (Server)" — note this sends audio off-machine, breaking the "100% local" guarantee for that model only
- **Automatic language detection**: passing `language=None` to `transcribe()` makes WhisperX auto-detect from the first 30s of audio (local models) or omits the `language` field so the remote server auto-detects (which is then normalized from a full name like `"german"` to an ISO code like `"de"` for the alignment step). Exposed in the GUI/CLI as "Automatisch erkennen" / `--language auto`
- **Speaker recognition (voice-prints)**: diarization requests per-speaker embeddings (`DiarizationPipeline(..., return_embeddings=True)`); `transcriber.py` matches them against `speaker_profiles.py`'s stored profiles and relabels recognized speakers' segments directly with their name instead of `SPEAKER_00` (CLI output benefits automatically). `result["speaker_id_map"]` tracks raw diarization ID → currently-displayed label so the GUI can find the right embedding when the user confirms/corrects a name via the "merken" (remember) checkbox, which enrolls/updates that speaker's profile as a running average
- **Speaker color-coding**: `gui.py`'s `_render_transcript()` tags each line in the transcript textbox by the *raw* diarization ID (not the display label), so colors and renames stay correctly associated even after a speaker gets auto-recognized or manually renamed. `_apply_speaker_names()` is the single place that mutates the textbox, `result["segments"]`, and `speaker_id_map` together — it must stay merged; a previous version had this split across two functions and the two representations could silently drift out of sync

## Configuration

### Environment Variables (.env)

```
HF_TOKEN=hf_xxx                     # Required for speaker diarization (pyannote models)
KIT_TOOLBOX_API_KEY=xxx             # Required for the "KIT ToolBox (Server)" remote model
KIT_TOOLBOX_BASE_URL=https://...    # Optional override (default: https://ki-toolbox.scc.kit.edu/api/v1)
```

### Hardware Requirements

- Windows 10/11
- NVIDIA GPU with CUDA 12.8 (recommended for large-v2/v3 models)
- Python 3.11+ (managed via `uv` or system installation)

### PyTorch / CUDA install

PyPI's default `torch` wheel is CPU-only. `pyproject.toml` pins `torch`/`torchaudio`/`torchvision` to the `https://download.pytorch.org/whl/cu128` index via `[tool.uv.sources]`, so a plain `uv sync` installs a matched CUDA build. If you ever see `torch.__version__` end in `+cpu` (or a `torchvision` import fails with a circular-import/`_meta_registrations` error from a version mismatch), the fix is `uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 --reinstall` — a bare `uv pip install` without `--reinstall` treats an already-installed package name as satisfied and won't re-fetch the correct build.

### Speaker profiles

Voice-print profiles for the speaker-recognition feature are stored outside the repo at `~/.voxscribe/speaker_profiles.json` (created lazily on first "merken"/enrollment). Not part of project config, but worth knowing when debugging why a speaker is/isn't auto-recognized.

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
- All processing is local by default; the only exception is the optional "KIT ToolBox (Server)" model, which sends audio to a remote endpoint for transcription (see Key Design Patterns above)
