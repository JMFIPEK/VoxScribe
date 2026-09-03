# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VoxScribe** is a desktop application for local audio recording and transcription using WhisperX. It supports microphone recording, system audio (WASAPI Loopback for Teams/Zoom), or both simultaneously, with speaker diarization via pyannote-audio. Transcription can use a configurable remote provider (any OpenAI-compatible endpoint) or run fully locally.

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

# Start GUI (PySide6)
python gui_qt.py

# Download models for offline/bundled operation (Windows/Linux WhisperX only —
# macOS uses Apple's SpeechAnalyzer, no download needed for transcription)
python voxscribe/download_models.py

# Build standalone .exe (Windows, requires PyInstaller)
python voxscribe/build_exe.py
```

## Architecture

### Core Modules

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point with subcommands: `devices`, `record`, `transcribe`, `run` |
| `gui_qt.py` | PySide6 GUI entry point — thin bootstrap (splash, SSL/proxy setup, `gpu_setup` check) that builds `qt_app.main_window.MainWindow`; app logic lives in `qt_app/` |
| `qt_app/` | PySide6 GUI package: `main_window.py` (top tab bar, `QTabWidget`), `controllers.py` (Qt-signal bridges to `voxscribe/recorder.py`/`voxscribe/transcriber.py` for thread-safe UI updates), `theme.py` (dark QSS stylesheet), `constants.py` (shared language/model/format constants), `pages/` (one file per tab: Record, Transcription, Settings — Info is folded into the bottom of Settings, not its own tab), `widgets/` (custom widgets, e.g. `level_meter.py`) |
| `voxscribe/recorder.py` | Audio recording: PyAudioWPatch (Windows: microphone, WASAPI loopback, or mixed), `sounddevice` (macOS/Linux microphone), experimental ScreenCaptureKit subprocess (macOS system-audio, see `macos/`) |
| `voxscribe/transcriber.py` | WhisperX pipeline: transcription → alignment → diarization (sequential loading); also handles video-container audio extraction (PyAV), provider-backed remote models, the Intel Arc/OpenVINO path, and (macOS only) Apple's SpeechAnalyzer engine via `transcribe_apple()` |
| `voxscribe/providers.py` | Persistent configuration for remote transcription providers (`~/.voxscribe/providers.json`) — name/base URL/API key/model per provider, one marked default; includes one-time migration from the old single-server env-var config |
| `voxscribe/hardware_detect.py` | GPU/CPU detection and model recommendations based on VRAM/RAM, including Intel Arc GPU via OpenVINO |
| `voxscribe/gpu_setup.py` | Windows-only: detects a CUDA/XPU mismatch between the installed `torch` build and the actual GPU present, and automatically reinstalls the correct build (see Key Design Patterns below) |
| `voxscribe/speaker_profiles.py` | Persistent voice-print storage (`~/.voxscribe/speaker_profiles.json`) and cosine-similarity speaker matching/enrollment |
| `voxscribe/download_models.py` | Pre-download models to `bundled_models/` for offline operation |
| `voxscribe/build_exe.py` | PyInstaller build script for creating a standalone .exe |
| `macos/` | macOS-only Swift helpers: `SystemAudioCapture` (experimental system-audio recording via ScreenCaptureKit) and `SpeechAnalyzerTranscribe` (transcription via Apple's Speech framework) — see `macos/README.md` for build steps and known open issues |

### Pipeline Flow

```
Recording (voxscribe/recorder.py)
    ↓
Audio WAV file (16kHz mono)
    ↓
Transcription (voxscribe/transcriber.py)
    ├── 1. WhisperX inference (0-40%)
    ├── 2. Forced Alignment with wav2vec2 (40-65%)
    └── 3. Speaker Diarization with pyannote (65-100%)
    ↓
Output: TXT, SRT, JSON
```

### Key Design Patterns

- **Sequential model loading**: Models are loaded/unloaded sequentially (Whisper → Alignment → Diarization) to minimize VRAM usage
- **Apple Silicon (MPS) acceleration is partial by design**: `transcriber.transcribe()` resolves two separate device variables — `whisper_device` (only ever `"cuda"` or `"cpu"`) and `torch_device` (`"cuda"`, `"xpu"`, `"mps"`, or `"cpu"`). This split exists because WhisperX's transcription backend (CTranslate2/faster-whisper) has no MPS support at all — passing it `"mps"` would crash — so Whisper inference always runs on CPU on a Mac regardless of the resolved device. Alignment (wav2vec2) and diarization (pyannote) are plain PyTorch, so they use `torch_device` and do get real MPS acceleration. `hardware_detect.recommend_model()` mirrors this: on Apple Silicon it picks model size from CPU/RAM (since that's what actually bounds Whisper's speed there) rather than pretending GPU compute helps the transcription step, but still returns `device="mps"` so alignment/diarization benefit. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set at import time in `voxscribe/transcriber.py` because pyannote/wav2vec2 use a few ops MPS doesn't implement yet, and without the fallback flag those crash instead of transparently running on CPU.
- **macOS transcription runs on Apple's SpeechAnalyzer, not WhisperX**: on macOS, `transcriber.default_local_model_size()` returns `"apple:speechanalyzer"` and no Whisper model is ever downloaded or run there — Windows/Linux are unaffected and keep WhisperX exactly as before. `transcribe_apple()` shells out to a one-shot Swift helper (`macos/SpeechAnalyzerTranscribe`, built via `macos/build.sh`) that runs the audio file through Apple's Speech framework (`SpeechAnalyzer`/`SpeechTranscriber`, macOS 26+) on the Neural Engine and streams one NDJSON line per recognized segment to stdout (word list included) — same subprocess-helper pattern as `SystemAudioCapture`, but a one-shot batch call instead of a live stream, so there's no watcher thread and no SIGTERM handling. Verified end-to-end on real Apple Silicon hardware (macOS 26.5.1): **word-level timestamps are already present** in the `SpeechTranscriber.Result.text` `AttributedString`'s per-run `audioTimeRange` attribute, confirmed word-accurate against the source audio — so `transcribe()` **skips the wav2vec2 forced-alignment step entirely** on this path (see the `apple` branch guarding "--- 2. Alignment ---"), since there's nothing left for it to add. This makes macOS transcription meaningfully faster than the old WhisperX-on-CPU path too: the "Apple Silicon (MPS) acceleration is partial by design" bullet above explains that Whisper/CTranslate2 has no MPS support at all and always ran on the CPU on a Mac — SpeechAnalyzer instead runs on the Neural Engine. Diarization is **unchanged**: SpeechAnalyzer has no speaker-diarization capability, so pyannote-audio still runs afterwards exactly as today (MPS-accelerated per the same bullet), consuming `transcribe_apple()`'s word/segment output the same way it would consume WhisperX's aligned output. SpeechAnalyzer also has no automatic language-detection API (unlike WhisperX/provider models), so `transcribe_apple()` maps the requested ISO language code to a BCP-47 locale (`_APPLE_LOCALE_BY_LANGUAGE`, e.g. `"en"` → `"en-US"`) and falls back to `"en-US"` if none was given. Because there is no model choice left to make on macOS, the GUI hides the "Model" dropdown and the Whisper-hardware-recommendation hint entirely on `sys.platform == "darwin"` (everything else — language, diarization, min/max speakers — stays), and hides "Recommended model" on the settings page, since it was always a Whisper-model-size recommendation: `qt_app/pages/transcribe_page.py`/`qt_app/pages/settings_page.py` (`self.model_combo`/`self.hw_hint`/`self.recommended_model_label` are `None` on macOS instead of being built, and every call site checks for that before touching them). `qt_app/constants.py` duplicates the `"apple:speechanalyzer"` sentinel as `APPLE_SPEECHANALYZER_MODEL` rather than importing it from `transcriber` — `qt_app` pages are built eagerly during `MainWindow.__init__()`, and `transcriber` pulls in whisperx/torch at import time (a cold multi-minute import), which is exactly what `HardwareInfoController`/`TranscribeController` already defer into background threads for the same reason.
- **SSL bypass for corporate proxy**: All HTTPS verification is disabled at module load to work behind corporate proxies
- **Bundled models support**: Models can be pre-downloaded to `bundled_models/` and used offline without HF_TOKEN
- **Threaded recording**: Audio recording runs in background threads with callbacks for GUI updates
- **Universal audio input**: `transcriber.load_audio_universal()` tries `soundfile` first (fast path for WAV/FLAC/OGG), then falls back to PyAV (`av` package) to demux just the audio stream out of video containers (MKV, MP4, MOV, ...) — no ffmpeg CLI required
- **Configurable remote providers** (`voxscribe/providers.py`): the transcription server is no longer a single hardcoded endpoint. Users add one or more providers in Settings (name, base URL, API key, model) and mark one as default; model IDs prefixed with `server:` (e.g. `server:my-provider-id`, see `transcriber.remote_model_id()`/`is_remote_model()`) resolve to a stored provider via `providers.get_provider()` and POST the audio to that OpenAI-compatible `/audio/transcriptions` endpoint (`transcriber.transcribe_remote()`); alignment and diarization still run locally on top of the returned segments. Long recordings are chunked (FLAC-encoded, ~180s each) with automatic halving on HTTP 413 to stay under server upload limits. The CLI's `--api-key`/`--api-base-url` flags override a stored provider's credentials for ad-hoc one-off use without saving a provider. Note that using any remote provider sends audio off-machine, breaking the "100% local" guarantee for that request only — local transcription always remains fully available.
  - **Migration from the old single-server config**: `providers.migrate_from_env()` runs once at every app start (before providers.json exists) and, if the legacy `KIT_TOOLBOX_API_KEY`/`KIT_TOOLBOX_BASE_URL` env vars are set, creates a provider from them automatically — so upgrading from a pre-provider-system install doesn't silently drop a working setup. See `docs/providers.md`.
- **Automatic language detection**: passing `language=None` to `transcribe()` makes WhisperX auto-detect from the first 30s of audio (local models) or omits the `language` field so the remote provider auto-detects (which is then normalized from a full name like `"german"` to an ISO code like `"de"` for the alignment step). Exposed in the GUI/CLI as "Auto-detect" / `--language auto` (the CLI's default)
- **Speaker recognition (voice-prints)**: diarization requests per-speaker embeddings (`DiarizationPipeline(..., return_embeddings=True)`); `voxscribe/transcriber.py` matches them against `voxscribe/speaker_profiles.py`'s stored profiles and relabels recognized speakers' segments directly with their name instead of `SPEAKER_00` (CLI output benefits automatically). `result["speaker_id_map"]` tracks raw diarization ID → currently-displayed label so the GUI can find the right embedding when the user confirms/corrects a name via the "remember" checkbox, which enrolls/updates that speaker's profile as a running average
- **Multi-file transcription with stitching**: the Transcription tab's file picker (`QFileDialog.getOpenFileNames`) allows selecting multiple audio/video files at once; `transcriber.transcribe_multi()` runs `transcribe()` on each in sequence (unchanged, one call per file) and merges the results into one continuous transcript — each file's segment/word timestamps are shifted by the cumulative duration of the preceding files (`result["duration"]`, now returned by `transcribe()` for exactly this purpose). Speaker labels are **not** blindly unified across files: diarization runs per-file independently, so `SPEAKER_00` in file 1 isn't necessarily the same person as `SPEAKER_00` in file 2. `transcribe_multi()` disambiguates by checking whether a file's `speaker_id_map` entry is still the raw, unrecognized label (`raw_id == current_label`) — if so it gets a file-prefixed label (`"File 2: SPEAKER_00"`); already-recognized speakers (voice-print matched to a real name, `raw_id != current_label`) are left unprefixed and merge naturally across files, since the same name should refer to the same person regardless of which file it came from. `TranscribeController.start_multi()` mirrors `start()` but calls `transcribe_multi()`; the GUI (`qt_app/pages/transcribe_page.py`) branches on file count — exactly one file still goes through the original single-file `start()` path unchanged (so single-file behavior/labels are byte-identical to before this feature), only 2+ files route through `start_multi()`.
- **PySide6/Qt GUI**: `gui_qt.py` + `qt_app/`. Chosen over an Electron rewrite because the whole backend (`voxscribe/recorder.py`, `voxscribe/transcriber.py`) is Python — PySide6 keeps it as direct in-process calls/threads instead of needing an IPC boundary to a subprocess. Key points: (1) `qt_app/pages/record_page.py` puts source/device selection, live per-channel level meters, timer, and start/stop all on one page (`QTabWidget`'s "Record" tab). (2) Since `AudioRecorder`'s `on_level`/`on_done` and `transcriber.transcribe()`'s `on_progress` callbacks fire from background threads, and Qt widgets may only be touched from the GUI thread, `qt_app/controllers.py` wraps them in `QObject` subclasses (`RecorderController`, `TranscribeController`) that re-emit them as Qt signals — Qt automatically marshals a signal emitted from a non-GUI thread to a `QueuedConnection` when the receiving `QObject` lives in the GUI thread, so no manual locking is needed. (3) The transcript view (`qt_app/pages/transcribe_page.py`) always fully re-renders from `result["segments"]`/`speaker_id_map` on every change (including after a speaker rename) instead of doing targeted text search-and-replace — simpler and safe since the `QTextEdit` is read-only and the result dict is the single source of truth. (4) `HardwareInfoController`/`TranscribeController` defer `from voxscribe import transcriber`/`hardware_detect` into background threads specifically because those modules import whisperx/torch at module load, which can cold-take 1-3 minutes — `qt_app/constants.py` duplicates small constants (e.g. `APPLE_SPEECHANALYZER_MODEL`) instead of importing them from `voxscribe.transcriber` for the same reason, since pages are built eagerly in `MainWindow.__init__()`.
- **Cross-platform recording (partial)**: `voxscribe/recorder.py` branches on `IS_WINDOWS = sys.platform == "win32"` and `IS_MACOS = sys.platform == "darwin"`. Windows uses `PyAudioWPatch` for both microphone and WASAPI-loopback system-audio capture (including combined "microphone + system"). Non-Windows falls back to `sounddevice` for **microphone-only** recording. macOS additionally has an **experimental** system-audio path: `AudioRecorder._record_system_macos()` shells out to a compiled helper at `macos/SystemAudioCapture` (source: `macos/SystemAudioCapture.swift`, built via `macos/build.sh`) that captures system audio through ScreenCaptureKit (requires macOS 13+ and the "Screen & System Audio Recording" permission) and streams raw 16kHz mono Int16 PCM over stdout; Python reads it like any other audio stream. A dedicated watcher thread calls `proc.terminate()` the moment `stop()` is requested so the blocking `stdout.read()` unblocks via EOF instead of hanging — same class of fix as the WASAPI stall bug below, applied proactively here. Combined "microphone + system" is implemented on macOS too (`AudioRecorder._record_both_macos()`: microphone via `sounddevice`'s own callback thread, system audio read blocking on the current thread identical to `_record_system_macos()`, both frame lists mixed after stop) but — unlike the standalone system-audio path — has **not yet been verified on real hardware** (see `macos/README.md` open issues); watch for sync drift between the two sources and behavior when only one side has recording permission when first testing it for real. The GUI's source selector reflects the platform split: Windows gets all three options ("Microphone"/"System audio"/"Microphone + system"), macOS gets all three too (with an experimental-status hint label, and the device dropdown disabled for System audio since ScreenCaptureKit captures the whole system, not a selectable device), Linux gets "Microphone" only. `pyproject.toml`/`requirements.txt` use PEP 508 markers (`sys_platform == 'win32'` / `!= 'win32'`) so the right backend installs per platform. The CUDA torch index in `[tool.uv.sources]` is similarly marker-gated to win32/linux only — macOS has no CUDA and gets plain PyPI torch (MPS backend) instead. Note: this was originally implemented on Windows without access to real Apple Silicon hardware (non-Windows Python logic validated only by mocking `sounddevice`/`subprocess.Popen`), then built and debugged on real macOS hardware — fixed a Swift compile error (string/`Data` type mismatch in the error-logging path) and a Python import-time crash (`voxscribe/recorder.py` referenced `pyaudio.PyAudio` in a type annotation even on macOS, where `pyaudio` is never imported — annotations are evaluated eagerly at def-time and crashed on import; fixed via `from __future__ import annotations`). Verified end-to-end on macOS 26/Apple Silicon: builds cleanly, requests/respects the "Screen & System Audio Recording" TCC permission, captures real non-silent system audio through the full `voxscribe/recorder.py` pipeline, and `stop()` terminates the subprocess in ~10ms with no orphaned process.
- **Intel Arc GPU transcription (Windows, via OpenVINO)**: CTranslate2 (WhisperX' default inference backend) has zero Intel GPU/NPU support at all, so this is a parallel opt-in path rather than a config flag on the existing one. `transcriber.transcribe_openvino()` loads a Whisper model pre-exported to OpenVINO IR (INT8-quantized, `voxscribe/download_models.py::export_openvino_model()`, step 4/4 — a separate model format from the CTranslate2 one `download_whisper_model()` downloads, needing its own HF transformers-checkpoint-based export via `optimum-cli export openvino`) and runs it via `optimum.intel.openvino.OVModelForSpeechSeq2Seq` targeting OpenVINO's `"GPU"` device (the Arc iGPU/dGPU; `"NPU"`/`"CPU"` are also valid OpenVINO device strings but unused here). Selected via the `"openvino:<device>:<size>"` model_size prefix (e.g. `"openvino:GPU:medium"`, `transcriber.OPENVINO_MODEL_PREFIX`/`is_openvino_model()`/`openvino_model_id()`), matching the existing `"server:"`/`"apple:"` prefix convention. Benchmarked on a Core Ultra 7 258V (Arc 140V GPU): ~10-16x realtime vs. ~1.6-1.9x on CPU int8 — CTranslate2's CPU path was already using int8 quantization (the fastest CPU option it has), so this isn't a "CPU was misconfigured" fix, the Arc GPU is just genuinely far more throughput for this workload.
  - **Manual 30s-window chunking, not HuggingFace's own ASR pipeline**: `transformers.pipeline("automatic-speech-recognition", chunk_length_s=...)` was tried first and rejected — its internal chunking iterator unconditionally imports `torchcodec` (ffmpeg-based) regardless of whether the input is a file path or a pre-loaded array, and `torchcodec` isn't installed in this project (same ffmpeg-avoidance reason `load_audio_universal()`/`load_audio_without_ffmpeg()` exist elsewhere in `voxscribe/transcriber.py` — confirmed by hitting the exact same `RuntimeError: Could not load libtorchcodec` those already work around). `transcribe_openvino()` instead loads the whole file via the existing `load_audio_universal()` and manually slices it into 30s windows, calling `model.generate()` per window directly.
  - Produces the same `{"segments": [{"start", "end", "text"}], "language": ...}` shape the CTranslate2 path returns, specifically so the subsequent Alignment step (which adds the word-level timestamps this path doesn't produce on its own, unlike the Apple SpeechAnalyzer path which skips Alignment entirely) and Diarization run completely unmodified afterward — `transcribe()`'s `elif openvino:` branch sits alongside the existing `apple`/`remote` branches and only replaces step 1.
  - **Language auto-detect**: when `language=None` ("Auto-detect"), the first chunk is generated without a forced `language` kwarg, then the actual detected language is read back off the generated token IDs (`_detect_language_from_token_ids()`: `processor.tokenizer.convert_ids_to_tokens()` on the first few tokens, regex-matching Whisper's `<|xx|>` language-token format) and forced explicitly for every subsequent chunk — mirrors WhisperX's own documented behavior of auto-detecting once from the first 30s and reusing that language throughout, rather than re-detecting (and potentially flip-flopping) every chunk.
  - `hardware_detect.get_openvino_devices()` (guarded `from openvino import Core; Core().available_devices` — returns `None` if `optimum-intel[openvino]` isn't installed, which is a Windows-only dependency in `pyproject.toml`) and `_bundled_openvino_model_exists()` (checks for the exported IR files) together gate `recommend_model()`'s new branch: an Arc GPU is only ever recommended if both the hardware is present *and* the export step has actually been run — recommending an unusable model would just move the error from "no GPU" to "model not found" at transcribe time.
  - **Alignment/Diarization on the same Arc GPU is a separate, independent axis** from the above: those are plain PyTorch (wav2vec2/pyannote), so they follow `hardware_detect.get_gpu_info()`/`transcriber.transcribe()`'s device auto-detect, extended from `cuda > mps > cpu` to **`cuda > xpu > mps > cpu`** (`torch.xpu.is_available()`, PyTorch's native Intel GPU backend since 2.5). This only activates if the *installed* `torch` build itself was compiled with XPU support — a plain `uv sync` installs the CUDA build (`[tool.uv.sources]` pins `cu128`), which reports `torch.xpu.is_available() == False` unconditionally regardless of what hardware is actually present.
  - **This is now handled automatically by `voxscribe/gpu_setup.py`** (see its own bullet below) rather than being a purely manual step — `uv sync --extra xpu` was tried and rejected: uv's `[tool.uv.sources]` marker system doesn't support switching a package's index based on which extra was requested at install time the way it does for platform/python-version markers — every attempted marker split (`extra == 'xpu'` vs `extra != 'xpu'`) either broke `uv lock`'s universal cross-Python resolution (the XPU package index's `triton-xpu` dependency isn't published for every Python ABI `requires-python`'s unbounded range implies) or produced a `"conflicting indexes"` resolution error, because uv wants one deterministic index per python/platform split, not one that also forks on a requested extra.
  - Both `qt_app/constants.py` (`MODEL_DISPLAY_NAMES["openvino:GPU:medium"] = "medium (Intel Arc GPU)"`) and `main.py`'s `--device-compute` CLI flag (`choices=[..., "xpu", ...]`) expose this — the model dropdown entry is unconditionally visible (matching how e.g. `large-v3` is selectable even without a big-enough CUDA GPU; the hardware-recommendation hint is informational, not a hard gate), it just errors clearly at transcribe time via `transcribe_openvino()`'s `FileNotFoundError` if the export step was never run.
- **`voxscribe/gpu_setup.py` automatically fixes a CUDA/XPU torch mismatch on every app start (Windows only)**: since `uv run`/`uv sync` (without `--no-sync`) silently reverts a manually-installed XPU torch build back to the `pyproject.toml`-pinned CUDA build every time (confirmed: `torch.xpu.is_available()` flips back to `False` immediately after a plain `uv run python -c "..."`), a purely-documented manual fix would keep silently breaking itself. `gpu_setup.ensure_correct_torch_backend()` runs at the very top of both entry points (`gui_qt.py`, `main.py`'s `cmd_transcribe()`), *before* anything imports torch anywhere in the process (torch only loads its native libraries on first import in a process — a fix on disk before that point still takes effect in the same process, a fix after it does not). The check itself (`Get-CimInstance Win32_VideoController` via PowerShell for GPU names, `importlib.metadata.version("torch")` for the installed build) deliberately imports neither torch nor whisperx, so it's fast enough to run unconditionally on every launch (unlike the `hardware_detect`/`transcriber` imports elsewhere, which are the actual reason those are deferred to background threads). Only the actual fix (`uv pip install --reinstall` against the matching index) is slow, and only runs on an actual mismatch — most launches are a no-op. Skipped entirely on non-Windows (this issue doesn't apply to MPS/CPU) and inside the frozen `.exe` (`sys.frozen`; there's no `.venv` to reinstall into there — the bundled torch variant is fixed at build time). `_find_uv()` prefers `~/.local/bin/uv.exe` (the official installer's location, matching what the README documents) over a bare `shutil.which("uv")`, since the latter can find an unrelated tool's bundled `uv.exe` first on a dev machine's PATH (observed in practice).

## Configuration

### Environment Variables (.env)

```
HF_TOKEN=hf_xxx                     # Required for speaker diarization (pyannote models)
```

Remote transcription providers are configured in the GUI (Settings → Providers) or via `voxscribe/providers.py`, not via `.env` — see `docs/providers.md`. The old single-server env vars (`KIT_TOOLBOX_API_KEY`/`KIT_TOOLBOX_BASE_URL`) are still read once, automatically, by `providers.migrate_from_env()` to create a provider from a pre-provider-system install, but aren't the primary configuration path anymore.

### Hardware Requirements

- Windows 10/11 — primary target, full feature set (mic + system-audio/WASAPI recording)
- macOS (Apple Silicon) — microphone and system-audio recording both verified working on real hardware (ScreenCaptureKit helper, see `macos/README.md`); combined mic+system is implemented but not yet verified on real hardware; transcription runs via Apple's SpeechAnalyzer on the Neural Engine (macOS 26+, see Key Design Patterns above), not WhisperX; diarization still runs via pyannote on CPU/MPS instead of CUDA
- Linux — experimental, microphone-only, no system-audio path at all
- NVIDIA GPU with CUDA 12.8 (recommended for large-v2/v3 models on Windows/Linux)
- Intel Arc GPU (Windows, no NVIDIA GPU present) — alternative transcription path via OpenVINO, handled automatically by `voxscribe/gpu_setup.py`, see Key Design Patterns above and `docs/intel-arc-gpu.md`
- Python 3.11+ (managed via `uv` or system installation)

### PyTorch / CUDA install

PyPI's default `torch` wheel is CPU-only. `pyproject.toml` pins `torch`/`torchaudio`/`torchvision` to the `https://download.pytorch.org/whl/cu128` index via `[tool.uv.sources]`, so a plain `uv sync` installs a matched CUDA build. If you ever see `torch.__version__` end in `+cpu` (or a `torchvision` import fails with a circular-import/`_meta_registrations` error from a version mismatch), the fix is `uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 --reinstall` — a bare `uv pip install` without `--reinstall` treats an already-installed package name as satisfied and won't re-fetch the correct build. On an Intel Arc GPU machine, this happens automatically instead, see `voxscribe/gpu_setup.py` above.

### Speaker profiles

Voice-print profiles for the speaker-recognition feature are stored outside the repo at `~/.voxscribe/speaker_profiles.json` (created lazily on first "remember"/enrollment). Not part of project config, but worth knowing when debugging why a speaker is/isn't auto-recognized.

### Provider configuration

Similarly stored outside the repo at `~/.voxscribe/providers.json` (created on first provider added, or via the legacy-env-var migration, see above). See `docs/providers.md` for the schema.

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
- **The user's configured default provider is the default transcription model** (`transcriber.default_model_size()`), on all platforms — not local-by-default, since a provider needs no model download and works regardless of local hardware. If no provider is configured, or the request fails (network, missing/invalid API key, server down, ...), `transcribe()` catches the exception and automatically retries once with the best local model for the current platform/hardware (`transcriber.default_local_model_size()`: Apple SpeechAnalyzer on macOS, otherwise whatever `hardware_detect.recommend_model()` picks — CUDA large-v3/medium/base on an NVIDIA GPU, the OpenVINO Arc-GPU model if no NVIDIA GPU but an Intel Arc GPU is present, else a RAM-sized CPU model). This fallback only triggers on an actual failure, not proactively — if the provider succeeds, everything else about the pipeline is unaffected. Explicitly picking a local model still works exactly as before and is unaffected by any of this.
