# macOS setup

## Requirements

- macOS 26 (Tahoe) or newer - SpeechAnalyzer (transcription) and
  ScreenCaptureKit (system audio) are both macOS-26+ APIs
- Apple Silicon (M1 or newer)
- Python 3.11+
- Xcode Command Line Tools (`xcode-select --install`) - to build the two
  Swift helpers in `macos/`
- [uv](https://docs.astral.sh/uv/) - recommended
- No NVIDIA GPU/CUDA needed - transcription runs on the Neural Engine
  (SpeechAnalyzer), alignment/diarization use MPS (Apple GPU)

## Install

There's no CUDA on a Mac - `pyproject.toml` already marks
the CUDA torch index as `win32`/`linux`-only, so macOS automatically gets
plain PyPI `torch` with MPS support:

```bash
# 1. Get the code
git clone https://gitlab.kit.edu/kit/ipek/acm/voxscribe.git
cd voxscribe
# Switch to the newest Release (Git's "detached HEAD" note is expected)
git checkout $(git describe --tags --abbrev=0 --match "v*")

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install Xcode Command Line Tools (if not already installed) -
#    needed to build the two Swift helpers in macos/
xcode-select --install

# 4. Install dependencies (creates .venv automatically, incl. PySide6)
uv sync

# 5. Build the Swift helpers (system-audio recording + SpeechAnalyzer transcription)
cd macos
./build.sh
cd ..
```

Then activate the venv with `source .venv/bin/activate` (or use
`uv run python ...` without activating).

## Dock icon (VoxScribe.app)

```bash
macos/make-app.sh          # creates ~/Applications/VoxScribe.app
```

This builds a small launcher app with the VoxScribe icon that starts the GUI
from this repo in place (the counterpart of `VoxScribe.bat` on Windows). Open
it from Spotlight or Launchpad, then right-click its Dock icon > Options >
Keep in Dock. It re-syncs the environment when `uv.lock` changes and builds
the Swift helpers if they are missing; output goes to
`~/.voxscribe/voxscribe.log`.

- The repo path is baked in - re-run `macos/make-app.sh` if you move the repo.
- macOS attributes the microphone and "Screen & System Audio Recording"
  permissions to **VoxScribe** (not Terminal) when started this way, so grant
  them once for the app on first use.
- The icon (Logo.png on a dark rounded tile) is rendered by
  `qt_app/app_icon.py`; running `python gui_qt.py` from a terminal shows the
  same icon in the Dock while it runs.

## Important notes

- Without the `macos/build.sh` step, neither system-audio recording nor
  transcription work - both call compiled binaries in `macos/` as a
  subprocess (`macos/SystemAudioCapture`, `macos/SpeechAnalyzerTranscribe`),
  which aren't shipped in the repo (see `.gitignore`).
- Transcription runs via Apple's `SpeechAnalyzer` (Speech framework) instead
  of WhisperX - there is therefore **no model choice** in the GUI on macOS,
  and `download_models.py`/the GPU memory table in the main README aren't
  relevant there.
- System-audio recording requires the "Screen & System Audio Recording"
  permission (System Settings → Privacy & Security) for the process running
  VoxScribe (VoxScribe.app, or Terminal when started from there) - macOS
  prompts for this on first use.
- Speaker diarization (pyannote) works unchanged from Windows/Linux, just
  with MPS instead of CUDA - see the main README's HuggingFace token section,
  still needed here.
- Details/background on both Swift helpers: `macos/README.md`.
