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

There's no CUDA on a Mac - `pyproject.toml`/`requirements.txt` already mark
the CUDA torch index as `win32`/`linux`-only, so macOS automatically gets
plain PyPI `torch` with MPS support. A single `uv sync` is enough here too:

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Xcode Command Line Tools (if not already installed) -
#    needed to build the two Swift helpers in macos/
xcode-select --install

# 3. Install dependencies (creates .venv automatically, incl. PySide6)
uv sync

# 4. Build the Swift helpers (system-audio recording + SpeechAnalyzer transcription)
cd macos
./build.sh
cd ..
```

Then activate the venv with `source .venv/bin/activate` (or use
`uv run python ...` without activating).

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
  VoxScribe (Terminal while testing, later the packaged app) - macOS prompts
  for this on first use.
- Speaker diarization (pyannote) works unchanged from Windows/Linux, just
  with MPS instead of CUDA - see the main README's HuggingFace token section,
  still needed here.
- Details/background on both Swift helpers: `macos/README.md`.
