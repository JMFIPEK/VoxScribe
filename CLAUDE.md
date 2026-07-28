# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VoxScribe** is a German-language desktop application for local audio recording and transcription using WhisperX. It supports microphone recording, system audio (WASAPI Loopback for Teams/Zoom), or both simultaneously, with speaker diarization via pyannote-audio.

## Quick Commands

```bash
# Activate virtual environment
.\.venv\Scripts\activate      # Windows
source .venv/bin/activate     # macOS/Linux

# Or use uv (no activation needed)
uv run python <script>

# macOS only, once: build the Swift helpers (system-audio capture +
# SpeechAnalyzer transcription) - see macos/README.md
cd macos && ./build.sh && cd ..

# List available audio devices
python main.py devices

# Record from microphone (stop with ENTER)
python main.py record --source mic

# Record system audio (WASAPI Loopback on Windows, ScreenCaptureKit on macOS)
python main.py record --source system

# Transcribe an audio file
python main.py transcribe -i recordings/audio.wav

# Full pipeline: record + transcribe
python main.py run --source system

# Start GUI (PySide6 — primary GUI going forward)
python gui_qt.py

# Start GUI (legacy CustomTkinter — kept working, no longer where new features land first)
python gui.py

# Download models for offline/bundled operation (Windows/Linux WhisperX only —
# macOS uses Apple's SpeechAnalyzer, no download needed for transcription)
python download_models.py

# Build standalone .exe (Windows, requires PyInstaller)
python build_exe.py
```

## Architecture

### Core Modules

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point with subcommands: `devices`, `record`, `transcribe`, `run` |
| `gui.py` | Legacy CustomTkinter GUI with tabs: Aufnahme, Transkription, Einstellungen, Info — kept working but `gui_qt.py` is where new features land |
| `gui_qt.py` | Primary PySide6 GUI entry point — thin bootstrap (splash, SSL/proxy setup) that builds `qt_app.main_window.MainWindow`; app logic lives in `qt_app/` |
| `qt_app/` | PySide6 GUI package: `main_window.py` (top tab bar, `QTabWidget`), `controllers.py` (Qt-signal bridges to `recorder.py`/`transcriber.py` for thread-safe UI updates), `theme.py` (dark QSS stylesheet), `constants.py` (shared language/model/format constants), `pages/` (one file per tab: Aufnahme, Transkription, Live-Zusammenfassung, Einstellungen — Info is folded into the bottom of Einstellungen, not its own tab), `widgets/` (custom widgets, e.g. `level_meter.py`) |
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
- **macOS transcription runs on Apple's SpeechAnalyzer, not WhisperX**: on macOS, `transcriber.transcribe()` defaults to `model_size="apple:speechanalyzer"` (`transcriber.default_model_size()`) and no Whisper model is ever downloaded or run there — Windows/Linux are unaffected and keep WhisperX (plus the "KIT ToolBox (Server)" option) exactly as before. `transcribe_apple()` shells out to a one-shot Swift helper (`macos/SpeechAnalyzerTranscribe`, built via `macos/build.sh`) that runs the audio file through Apple's Speech framework (`SpeechAnalyzer`/`SpeechTranscriber`, macOS 26+) on the Neural Engine and streams one NDJSON line per recognized segment to stdout (word list included) — same subprocess-helper pattern as `SystemAudioCapture`, but a one-shot batch call instead of a live stream, so there's no watcher thread and no SIGTERM handling. Verified at end-to-end on real Apple Silicon hardware (macOS 26.5.1) against synthetic German test audio: **word-level timestamps are already present** in the `SpeechTranscriber.Result.text` `AttributedString`'s per-run `audioTimeRange` attribute, confirmed word-accurate against the source audio — so `transcribe()` **skips the wav2vec2 forced-alignment step entirely** on this path (see the `apple` branch guarding "--- 2. Alignment ---"), since there's nothing left for it to add. This makes macOS transcription meaningfully faster than the old WhisperX-on-CPU path too: the "Apple Silicon (MPS) acceleration is partial by design" bullet above explains that Whisper/CTranslate2 has no MPS support at all and always ran on the CPU on a Mac — SpeechAnalyzer instead runs on the Neural Engine. Diarization is **unchanged**: SpeechAnalyzer has no speaker-diarization capability, so pyannote-audio still runs afterwards exactly as today (MPS-accelerated per the same bullet), consuming `transcribe_apple()`'s word/segment output the same way it would consume WhisperX's aligned output. SpeechAnalyzer also has no automatic language-detection API (unlike WhisperX/the server model), so `transcribe_apple()` maps the requested ISO language code to a BCP-47 locale (`_APPLE_LOCALE_BY_LANGUAGE`, e.g. `"de"` → `"de-DE"`) and falls back to `"de-DE"` if none was given. Because there is no model choice left to make on macOS, both GUIs hide the "Modell" dropdown and the Whisper-hardware-recommendation hint entirely on `sys.platform == "darwin"` (everything else — language, diarization, min/max speakers — stays), and hide "Empfohlenes Modell" on the settings page, since it was always a Whisper-model-size recommendation: `gui.py`'s transcription/settings tabs, and `qt_app/pages/transcribe_page.py`/`qt_app/pages/settings_page.py` (`self.model_combo`/`self.hw_hint`/`self.recommended_model_label` are `None` on macOS instead of being built, and every call site checks for that before touching them). `qt_app/constants.py` duplicates the `"apple:speechanalyzer"` sentinel as `APPLE_SPEECHANALYZER_MODEL` rather than importing it from `transcriber` — `qt_app` pages are built eagerly during `MainWindow.__init__()`, and `transcriber` pulls in whisperx/torch at import time (a cold multi-minute import), which is exactly what `HardwareInfoController`/`TranscribeController` already defer into background threads for the same reason.
- **SSL bypass for corporate proxy**: All HTTPS verification is disabled at module load to work behind corporate proxies
- **Bundled models support**: Models can be pre-downloaded to `bundled_models/` and used offline without HF_TOKEN
- **Threaded recording**: Audio recording runs in background threads with callbacks for GUI updates
- **Universal audio input**: `transcriber.load_audio_universal()` tries `soundfile` first (fast path for WAV/FLAC/OGG), then falls back to PyAV (`av` package) to demux just the audio stream out of video containers (MKV, MP4, MOV, ...) — no ffmpeg CLI required
- **Remote/server model**: model IDs prefixed with `server:` (e.g. `server:kit.whisper-large-v3`) skip local WhisperX inference and instead POST the audio to an OpenAI-compatible `/audio/transcriptions` endpoint (`transcriber.transcribe_remote`); alignment and diarization still run locally on top of the returned segments. Long recordings are chunked (FLAC-encoded, ~180s each) with automatic halving on HTTP 413 to stay under server upload limits. Selectable in the GUI as "KIT ToolBox (Server)" — note this sends audio off-machine, breaking the "100% local" guarantee for that model only
- **Automatic language detection**: passing `language=None` to `transcribe()` makes WhisperX auto-detect from the first 30s of audio (local models) or omits the `language` field so the remote server auto-detects (which is then normalized from a full name like `"german"` to an ISO code like `"de"` for the alignment step). Exposed in the GUI/CLI as "Automatisch erkennen" / `--language auto`
- **Speaker recognition (voice-prints)**: diarization requests per-speaker embeddings (`DiarizationPipeline(..., return_embeddings=True)`); `transcriber.py` matches them against `speaker_profiles.py`'s stored profiles and relabels recognized speakers' segments directly with their name instead of `SPEAKER_00` (CLI output benefits automatically). `result["speaker_id_map"]` tracks raw diarization ID → currently-displayed label so the GUI can find the right embedding when the user confirms/corrects a name via the "merken" (remember) checkbox, which enrolls/updates that speaker's profile as a running average
- **Speaker color-coding**: `gui.py`'s `_render_transcript()` tags each line in the transcript textbox by the *raw* diarization ID (not the display label), so colors and renames stay correctly associated even after a speaker gets auto-recognized or manually renamed. `_apply_speaker_names()` is the single place that mutates the textbox, `result["segments"]`, and `speaker_id_map` together — it must stay merged; a previous version had this split across two functions and the two representations could silently drift out of sync
- **Live-Zusammenfassung (Premium, KIT-ToolBox-only)**: `qt_app/pages/live_meeting_page.py` is a 4th, self-contained tab — deliberately *not* part of the normal Aufnahme/Transkription flow, since it continuously streams audio off-machine rather than once at the end. It has its own simplified recording controls, mirroring `RecordPage`'s platform-conditional source options exactly (including "Mikrofon + System" defaulting on Windows — a one-sided summary missing either your own voice or the other party otherwise defeats the point) and sharing the same `RecorderController`/`AudioRecorder` instance as the Aufnahme tab (only one recording at a time makes sense). Only the summary is shown in the UI — no live-transcript pane and no "Transkribiere..."/busy-indicator, since that turned out to be noise the user didn't want to see; the intervals are fixed constants (`CHUNK_INTERVAL_SECONDS=15`, `SUMMARY_INTERVAL_SECONDS=20`) rather than user-facing dropdowns (an earlier version exposed both as configurable — removed for simplicity, not because configurability was wrong).
  - Driven by `LiveMeetingController` (`qt_app/controllers.py`) running a background-thread loop: every `chunk_interval` it takes a rolling-window snapshot of the audio recorded *so far* via `AudioRecorder.snapshot_recent_audio(window_seconds)` (`window_seconds` scales with `chunk_interval`, `+30s` buffer, floor 60s, so a shorter interval doesn't multiply the redundant re-upload cost) and transcribes it via the *existing* `transcriber.transcribe_remote()` (KIT ToolBox), reusing its FLAC-encoding/413-backoff logic unchanged. For `source == "both"`, `snapshot_recent_audio()` reads `self._mic_frames`/`self._sys_frames` — instance attributes `_record_both()`/`_record_both_macos()` now expose alongside the local variables they already append to (same list objects) — and mixes them live via the extracted `_mix_sources()` helper (same formula the end-of-recording mix uses).
  - **Text-diffing, not timestamp-cutoff, for delta extraction**: an earlier version filtered `transcribe_remote()`'s returned segments by start-timestamp (keep only segments starting in the last `chunk_interval` seconds of the window) to avoid re-appending the overlapping portion every cycle. This was **broken in practice** — confirmed live against the real KIT ToolBox endpoint, it returns **one single coarse segment spanning the entire window** (`start=0.0, end=window_duration`), not many small timestamped ones, so the cutoff filter discarded that one segment on literally every cycle (no error, no update — the STT call was succeeding, the result was just thrown away). Fixed by diffing each cycle's full window text against the *previous* cycle's window text (`difflib.SequenceMatcher.find_longest_match`) and keeping only the tail after their longest common match as "new" — works regardless of the server's segmentation granularity. `LiveMeetingController._extract_new_text()`.
  - Every `summary_interval` or on manual "Jetzt zusammenfassen" click, the accumulated transcript-since-last-summary (kept in `_pending_delta`, cleared after each summary — separate from the ever-growing `_full_transcript` used internally to feed the diff) is sent to `transcriber.summarize_meeting()`, which POSTs to KIT ToolBox's `/chat/completions` with the *previous* summary as context and asks the model to update it rather than regenerate from scratch, keeping token cost roughly flat as the meeting gets longer. Language is locked to whatever `transcribe_remote()` first detects and passed to `summarize_meeting()` so it responds in the meeting's language. Default chat model `kit.mistral-small-4-119b-a8b` (MoE, ~8B active params — chosen for low latency over raw capability; summarization doesn't need frontier-scale reasoning and this runs repeatedly during a live meeting).
  - `transcribe_remote()`/`_post_audio_chunk()` take an optional `timeout` (default unchanged at 1800s for the normal one-shot Transkription-tab path); `LiveMeetingController` passes a 60s timeout (`LIVE_STT_TIMEOUT_SECONDS`) instead, so a stuck/slow request surfaces as a visible error within about a minute rather than an indefinite silent wait.
  - The `_run()` background-thread loop's entire per-cycle body is wrapped in one try/except (not just the `transcribe_remote()` call) — an unhandled exception anywhere in the cycle (e.g. inside `_run_summary_cycle()`) would otherwise kill the thread permanently with no visible error under a windowed/`pythonw.exe` build (no console), which looked exactly like "the live transcript stops updating after the first time."
- **Multi-file transcription with stitching**: the Transkription tab's file picker (`QFileDialog.getOpenFileNames`) allows selecting multiple audio/video files at once; `transcriber.transcribe_multi()` runs `transcribe()` on each in sequence (unchanged, one call per file) and merges the results into one continuous transcript — each file's segment/word timestamps are shifted by the cumulative duration of the preceding files (`result["duration"]`, now returned by `transcribe()` for exactly this purpose). Speaker labels are **not** blindly unified across files: diarization runs per-file independently, so `SPEAKER_00` in file 1 isn't necessarily the same person as `SPEAKER_00` in file 2. `transcribe_multi()` disambiguates by checking whether a file's `speaker_id_map` entry is still the raw, unrecognized label (`raw_id == current_label`) — if so it gets a file-prefixed label (`"Datei 2: SPEAKER_00"`); already-recognized speakers (voice-print matched to a real name, `raw_id != current_label`) are left unprefixed and merge naturally across files, since the same name should refer to the same person regardless of which file it came from. `TranscribeController.start_multi()` mirrors `start()` but calls `transcribe_multi()`; the GUI (`qt_app/pages/transcribe_page.py`) branches on file count — exactly one file still goes through the original single-file `start()` path unchanged (so single-file behavior/labels are byte-identical to before this feature), only 2+ files route through `start_multi()`.
- **PySide6 is now the primary GUI**: `gui_qt.py` + `qt_app/` is the actively developed GUI going forward; `gui.py` (CustomTkinter) is kept working but is legacy and no longer where new features land first. Chosen over an Electron rewrite because the whole backend (`recorder.py`, `transcriber.py`) is Python — PySide6 keeps it as direct in-process calls/threads instead of needing an IPC boundary to a subprocess. Both GUIs share the same backend modules unchanged. Key differences from the old GUI: (1) `qt_app/pages/record_page.py` puts source/device selection, live per-channel level meters, timer, and start/stop all on one page (`QTabWidget`'s "Aufnahme" tab) — same single-page layout as the old CTk GUI, just re-implemented in Qt widgets. (2) Since `AudioRecorder`'s `on_level`/`on_done` and `transcriber.transcribe()`'s `on_progress` callbacks fire from background threads, and Qt widgets may only be touched from the GUI thread, `qt_app/controllers.py` wraps them in `QObject` subclasses (`RecorderController`, `TranscribeController`) that re-emit them as Qt signals — Qt automatically marshals a signal emitted from a non-GUI thread to a `QueuedConnection` when the receiving `QObject` lives in the GUI thread, so no manual locking is needed. (3) The transcript view (`qt_app/pages/transcribe_page.py`) always fully re-renders from `result["segments"]`/`speaker_id_map` on every change (including after a speaker rename) instead of doing targeted text search-and-replace like `gui.py`'s `_apply_speaker_names()` — simpler and safe since the `QTextEdit` is read-only and the result dict is the single source of truth. (4) `HardwareInfoController`/`TranscribeController` defer `import transcriber`/`hardware_detect` into background threads specifically because those modules import whisperx/torch at module load, which can cold-take 1-3 minutes — `qt_app/constants.py` duplicates small constants (e.g. `APPLE_SPEECHANALYZER_MODEL`) instead of importing them from `transcriber` for the same reason, since pages are built eagerly in `MainWindow.__init__()`.
- **Cross-platform recording (partial)**: `recorder.py` branches on `IS_WINDOWS = sys.platform == "win32"` and `IS_MACOS = sys.platform == "darwin"`. Windows uses `PyAudioWPatch` for both microphone and WASAPI-loopback system-audio capture (including combined "Mikrofon + System"). Non-Windows falls back to `sounddevice` for **microphone-only** recording. macOS additionally has an **experimental** system-audio path: `AudioRecorder._record_system_macos()` shells out to a compiled helper at `macos/SystemAudioCapture` (source: `macos/SystemAudioCapture.swift`, built via `macos/build.sh`) that captures system audio through ScreenCaptureKit (requires macOS 13+ and the "Bildschirm- und Systemaudioaufnahme" permission) and streams raw 16kHz mono Int16 PCM over stdout; Python reads it like any other audio stream. A dedicated watcher thread calls `proc.terminate()` the moment `stop()` is requested so the blocking `stdout.read()` unblocks via EOF instead of hanging — same class of fix as the WASAPI stall bug below, applied proactively here. Combined "Mikrofon + System" is implemented on macOS too (`AudioRecorder._record_both_macos()`: microphone via `sounddevice`'s own callback thread, system audio read blocking on the current thread identical to `_record_system_macos()`, both frame lists mixed after stop) but — unlike the standalone system-audio path — has **not yet been verified on real hardware** (see `macos/README.md` open issues); watch for sync drift between the two sources and behavior when only one side has recording permission when first testing it for real. Both GUIs' source selectors reflect the platform split: Windows gets all three options ("Mikrofon"/"System-Audio"/"Mikrofon + System"), macOS gets all three too (with an experimental-status hint label, and the device dropdown disabled for System-Audio since ScreenCaptureKit captures the whole system, not a selectable device), Linux gets "Mikrofon" only. `pyproject.toml`/`requirements.txt` use PEP 508 markers (`sys_platform == 'win32'` / `!= 'win32'`) so the right backend installs per platform. The CUDA torch index in `[tool.uv.sources]` is similarly marker-gated to win32/linux only — macOS has no CUDA and gets plain PyPI torch (MPS backend) instead. Note: this was originally implemented on Windows without access to real Apple Silicon hardware (non-Windows Python logic validated only by mocking `sounddevice`/`subprocess.Popen`), then built and debugged on real macOS hardware — fixed a Swift compile error (string/`Data` type mismatch in the error-logging path) and a Python import-time crash (`recorder.py` referenced `pyaudio.PyAudio` in a type annotation even on macOS, where `pyaudio` is never imported — annotations are evaluated eagerly at def-time and crashed on import; fixed via `from __future__ import annotations`). Verified end-to-end on macOS 26/Apple Silicon: builds cleanly, requests/respects the "Bildschirm- und Systemaudioaufnahme" TCC permission, captures real non-silent system audio through the full `recorder.py` pipeline, and `stop()` terminates the subprocess in ~10ms with no orphaned process.

## Configuration

### Environment Variables (.env)

```
HF_TOKEN=hf_xxx                     # Required for speaker diarization (pyannote models)
KIT_TOOLBOX_API_KEY=xxx             # Required for the "KIT ToolBox (Server)" remote model
KIT_TOOLBOX_BASE_URL=https://...    # Optional override (default: https://ki-toolbox.scc.kit.edu/api/v1)
```

### Hardware Requirements

- Windows 10/11 — primary target, full feature set (mic + system-audio/WASAPI recording)
- macOS (Apple Silicon) — microphone and system-audio recording both verified working on real hardware (ScreenCaptureKit helper, see `macos/README.md`); combined mic+system is implemented but not yet verified on real hardware; transcription runs via Apple's SpeechAnalyzer on the Neural Engine (macOS 26+, see Key Design Patterns above), not WhisperX; diarization still runs via pyannote on CPU/MPS instead of CUDA
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
