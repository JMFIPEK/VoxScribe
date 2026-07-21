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

# Start GUI (legacy CustomTkinter — being phased out, see below)
python gui.py

# Start GUI (new PySide6 GUI — in-progress replacement for gui.py)
python gui_qt.py

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
| `gui.py` | Legacy CustomTkinter GUI with tabs: Aufnahme, Transkription, Einstellungen, Info — being phased out in favor of `gui_qt.py` |
| `gui_qt.py` | New PySide6 GUI entry point (in progress) — thin bootstrap (splash, SSL/proxy setup) that builds `qt_app.main_window.MainWindow`; app logic lives in `qt_app/` |
| `qt_app/` | PySide6 GUI package: `main_window.py` (sidebar nav + `QStackedWidget`), `controllers.py` (Qt-signal bridges to `recorder.py`/`transcriber.py` for thread-safe UI updates), `theme.py` (dark QSS stylesheet), `constants.py` (shared language/model/format constants), `pages/` (one file per page: Aufnahme, Monitoring, Transkription, Einstellungen, Info), `widgets/` (custom widgets, e.g. `level_meter.py`) |
| `recorder.py` | Audio recording: PyAudioWPatch (Windows: microphone, WASAPI loopback, or mixed), `sounddevice` (macOS/Linux microphone), experimental ScreenCaptureKit subprocess (macOS system-audio, see `macos/`) |
| `transcriber.py` | WhisperX pipeline: transcription → alignment → diarization (sequential loading); also handles video-container audio extraction (PyAV), the remote/server model, and (macOS only) Apple's SpeechAnalyzer engine via `transcribe_apple()` |
| `hardware_detect.py` | GPU/CPU detection and model recommendations based on VRAM/RAM |
| `speaker_profiles.py` | Persistent voice-print storage (`~/.voxscribe/speaker_profiles.json`) and cosine-similarity speaker matching/enrollment |
| `download_models.py` | Pre-download models to `bundled_models/` for offline operation |
| `build_exe.py` | PyInstaller build script for creating standalone .exe |
| `macos/` | macOS-only Swift helpers: `SystemAudioCapture` (experimental system-audio recording via ScreenCaptureKit) and `SpeechAnalyzerTranscribe` (transcription via Apple's Speech framework) — see `macos/README.md` for build steps and known open issues |

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
- **Apple Silicon (MPS) acceleration is partial by design**: `transcriber.transcribe()` resolves two separate device variables — `whisper_device` (only ever `"cuda"` or `"cpu"`) and `torch_device` (`"cuda"`, `"mps"`, or `"cpu"`). This split exists because WhisperX's transcription backend (CTranslate2/faster-whisper) has no MPS support at all — passing it `"mps"` would crash — so Whisper inference always runs on CPU on a Mac regardless of the resolved device. Alignment (wav2vec2) and diarization (pyannote) are plain PyTorch, so they use `torch_device` and do get real MPS acceleration. `hardware_detect.recommend_model()` mirrors this: on Apple Silicon it picks model size from CPU/RAM (since that's what actually bounds Whisper's speed there) rather than pretending GPU compute helps the transcription step, but still returns `device="mps"` so alignment/diarization benefit. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set at import time in `transcriber.py` because pyannote/wav2vec2 use a few ops MPS doesn't implement yet, and without the fallback flag those crash instead of transparently running on CPU.
- **macOS transcription runs on Apple's SpeechAnalyzer, not WhisperX**: on macOS, `transcriber.transcribe()` defaults to `model_size="apple:speechanalyzer"` (`transcriber.default_model_size()`) and no Whisper model is ever downloaded or run there — Windows/Linux are unaffected and keep WhisperX (plus the "KIT ToolBox (Server)" option) exactly as before. `transcribe_apple()` shells out to a one-shot Swift helper (`macos/SpeechAnalyzerTranscribe`, built via `macos/build.sh`) that runs the audio file through Apple's Speech framework (`SpeechAnalyzer`/`SpeechTranscriber`, macOS 26+) on the Neural Engine and streams one NDJSON line per recognized segment to stdout (word list included) — same subprocess-helper pattern as `SystemAudioCapture`, but a one-shot batch call instead of a live stream, so there's no watcher thread and no SIGTERM handling. Verified at end-to-end on real Apple Silicon hardware (macOS 26.5.1) against synthetic German test audio: **word-level timestamps are already present** in the `SpeechTranscriber.Result.text` `AttributedString`'s per-run `audioTimeRange` attribute, confirmed word-accurate against the source audio — so `transcribe()` **skips the wav2vec2 forced-alignment step entirely** on this path (see the `apple` branch guarding "--- 2. Alignment ---"), since there's nothing left for it to add. This makes macOS transcription meaningfully faster than the old WhisperX-on-CPU path too: the "Apple Silicon (MPS) acceleration is partial by design" bullet above explains that Whisper/CTranslate2 has no MPS support at all and always ran on the CPU on a Mac — SpeechAnalyzer instead runs on the Neural Engine. Diarization is **unchanged**: SpeechAnalyzer has no speaker-diarization capability, so pyannote-audio still runs afterwards exactly as today (MPS-accelerated per the same bullet), consuming `transcribe_apple()`'s word/segment output the same way it would consume WhisperX's aligned output. SpeechAnalyzer also has no automatic language-detection API (unlike WhisperX/the server model), so `transcribe_apple()` maps the requested ISO language code to a BCP-47 locale (`_APPLE_LOCALE_BY_LANGUAGE`, e.g. `"de"` → `"de-DE"`) and falls back to `"de-DE"` if none was given. Because there is no model choice left to make on macOS, `gui.py`'s transcription tab hides the "Modell" dropdown and the Whisper-hardware-recommendation hint there entirely on `sys.platform == "darwin"` (everything else — language, diarization, min/max speakers — stays); the settings tab likewise hides the "Empfohlenes Modell" line there, since it was always a Whisper-model-size recommendation.
- **SSL bypass for corporate proxy**: All HTTPS verification is disabled at module load to work behind corporate proxies
- **Bundled models support**: Models can be pre-downloaded to `bundled_models/` and used offline without HF_TOKEN
- **Threaded recording**: Audio recording runs in background threads with callbacks for GUI updates
- **Universal audio input**: `transcriber.load_audio_universal()` tries `soundfile` first (fast path for WAV/FLAC/OGG), then falls back to PyAV (`av` package) to demux just the audio stream out of video containers (MKV, MP4, MOV, ...) — no ffmpeg CLI required
- **Remote/server model**: model IDs prefixed with `server:` (e.g. `server:kit.whisper-large-v3`) skip local WhisperX inference and instead POST the audio to an OpenAI-compatible `/audio/transcriptions` endpoint (`transcriber.transcribe_remote`); alignment and diarization still run locally on top of the returned segments. Long recordings are chunked (FLAC-encoded, ~180s each) with automatic halving on HTTP 413 to stay under server upload limits. Selectable in the GUI as "KIT ToolBox (Server)" — note this sends audio off-machine, breaking the "100% local" guarantee for that model only
- **Automatic language detection**: passing `language=None` to `transcribe()` makes WhisperX auto-detect from the first 30s of audio (local models) or omits the `language` field so the remote server auto-detects (which is then normalized from a full name like `"german"` to an ISO code like `"de"` for the alignment step). Exposed in the GUI/CLI as "Automatisch erkennen" / `--language auto`
- **Speaker recognition (voice-prints)**: diarization requests per-speaker embeddings (`DiarizationPipeline(..., return_embeddings=True)`); `transcriber.py` matches them against `speaker_profiles.py`'s stored profiles and relabels recognized speakers' segments directly with their name instead of `SPEAKER_00` (CLI output benefits automatically). `result["speaker_id_map"]` tracks raw diarization ID → currently-displayed label so the GUI can find the right embedding when the user confirms/corrects a name via the "merken" (remember) checkbox, which enrolls/updates that speaker's profile as a running average
- **Speaker color-coding**: `gui.py`'s `_render_transcript()` tags each line in the transcript textbox by the *raw* diarization ID (not the display label), so colors and renames stay correctly associated even after a speaker gets auto-recognized or manually renamed. `_apply_speaker_names()` is the single place that mutates the textbox, `result["segments"]`, and `speaker_id_map` together — it must stay merged; a previous version had this split across two functions and the two representations could silently drift out of sync
- **GUI migration to PySide6 (in progress)**: `gui.py` (CustomTkinter) is being replaced by `gui_qt.py` + `qt_app/` (PySide6/Qt), chosen over an Electron rewrite because the whole backend (`recorder.py`, `transcriber.py`) is Python — PySide6 keeps it as direct in-process calls/threads instead of needing an IPC boundary to a subprocess. Both GUIs currently coexist and share the same backend modules unchanged; `gui.py` stays until `gui_qt.py` reaches feature parity. Key differences from the old GUI: (1) recording is split into two separate pages instead of one tab — `qt_app/pages/record_page.py` (source/device selection, start button) and `qt_app/pages/monitor_page.py` (live per-channel level meters, timer, stop button); starting a recording auto-navigates from Aufnahme to Monitoring. (2) Since `AudioRecorder`'s `on_level`/`on_done` and `transcriber.transcribe()`'s `on_progress` callbacks fire from background threads, and Qt widgets may only be touched from the GUI thread, `qt_app/controllers.py` wraps them in `QObject` subclasses (`RecorderController`, `TranscribeController`) that re-emit them as Qt signals — Qt automatically marshals a signal emitted from a non-GUI thread to a `QueuedConnection` when the receiving `QObject` lives in the GUI thread, so no manual locking is needed. (3) The transcript view (`qt_app/pages/transcribe_page.py`) always fully re-renders from `result["segments"]`/`speaker_id_map` on every change (including after a speaker rename) instead of doing targeted text search-and-replace like `gui.py`'s `_apply_speaker_names()` — simpler and safe since the `QTextEdit` is read-only and the result dict is the single source of truth.
- **Cross-platform recording (partial)**: `recorder.py` branches on `IS_WINDOWS = sys.platform == "win32"` and `IS_MACOS = sys.platform == "darwin"`. Windows uses `PyAudioWPatch` for both microphone and WASAPI-loopback system-audio capture (including combined "Mikrofon + System"). Non-Windows falls back to `sounddevice` for **microphone-only** recording. macOS additionally has an **experimental** system-audio path: `AudioRecorder._record_system_macos()` shells out to a compiled helper at `macos/SystemAudioCapture` (source: `macos/SystemAudioCapture.swift`, built via `macos/build.sh`) that captures system audio through ScreenCaptureKit (requires macOS 13+ and the "Bildschirm- und Systemaudioaufnahme" permission) and streams raw 16kHz mono Int16 PCM over stdout; Python reads it like any other audio stream. A dedicated watcher thread calls `proc.terminate()` the moment `stop()` is requested so the blocking `stdout.read()` unblocks via EOF instead of hanging — same class of fix as the WASAPI stall bug below, applied proactively here. Combined "Mikrofon + System" is NOT implemented on macOS yet (deliberate scope cut — see `macos/README.md` for open issues). `gui.py`'s source selector reflects this: Windows gets all three options, macOS gets "Mikrofon"/"System-Audio" (with an experimental-status hint label, and the device dropdown disabled for System-Audio since ScreenCaptureKit captures the whole system, not a selectable device), Linux gets "Mikrofon" only. `pyproject.toml`/`requirements.txt` use PEP 508 markers (`sys_platform == 'win32'` / `!= 'win32'`) so the right backend installs per platform. The CUDA torch index in `[tool.uv.sources]` is similarly marker-gated to win32/linux only — macOS has no CUDA and gets plain PyPI torch (MPS backend) instead. Note: this was originally implemented on Windows without access to real Apple Silicon hardware (non-Windows Python logic validated only by mocking `sounddevice`/`subprocess.Popen`), then built and debugged on real macOS hardware — fixed a Swift compile error (string/`Data` type mismatch in the error-logging path) and a Python import-time crash (`recorder.py` referenced `pyaudio.PyAudio` in a type annotation even on macOS, where `pyaudio` is never imported — annotations are evaluated eagerly at def-time and crashed on import; fixed via `from __future__ import annotations`). Verified end-to-end on macOS 26/Apple Silicon: builds cleanly, requests/respects the "Bildschirm- und Systemaudioaufnahme" TCC permission, captures real non-silent system audio through the full `recorder.py` pipeline, and `stop()` terminates the subprocess in ~10ms with no orphaned process.

## Configuration

### Environment Variables (.env)

```
HF_TOKEN=hf_xxx                     # Required for speaker diarization (pyannote models)
KIT_TOOLBOX_API_KEY=xxx             # Required for the "KIT ToolBox (Server)" remote model
KIT_TOOLBOX_BASE_URL=https://...    # Optional override (default: https://ki-toolbox.scc.kit.edu/api/v1)
```

### Hardware Requirements

- Windows 10/11 — primary target, full feature set (mic + system-audio/WASAPI recording)
- macOS (Apple Silicon) — microphone and system-audio recording both verified working on real hardware (ScreenCaptureKit helper, see `macos/README.md`); no combined mic+system yet; transcription runs via Apple's SpeechAnalyzer on the Neural Engine (macOS 26+, see Key Design Patterns above), not WhisperX; diarization still runs via pyannote on CPU/MPS instead of CUDA
- Linux — experimental, microphone-only, no system-audio path at all
- NVIDIA GPU with CUDA 12.8 (recommended for large-v2/v3 models on Windows/Linux)
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
