# Development

For working on VoxScribe or running it from a terminal. Users start it from
the shortcut (Windows) or VoxScribe.app (macOS), see the [README](../README.md).

## Environment

```bash
uv sync
```

`pyproject.toml` pins `torch`/`torchaudio`/`torchvision` to the CUDA 12.8 index
on Windows/Linux (macOS gets plain PyPI torch with MPS). On an Intel Arc GPU,
`voxscribe/gpu_setup.py` switches torch to the XPU build at start, see
[Intel Arc GPU](intel-arc-gpu.md).

## Running from a terminal

```bash
uv run --no-sync python gui_qt.py
```

Use `--no-sync` (or `VoxScribe.bat`, which does this for you): a plain `uv run`
re-syncs the venv first, which on an Intel Arc machine reverts torch to the
CUDA build and makes the next start re-install the XPU build. Run `uv sync`
yourself only after pulling dependency changes.

`VoxScribe.bat` syncs the venv only when `uv.lock` changed since its last
successful sync (or the venv is missing), installs a pending Update, and starts
the GUI with `pythonw.exe` (no console). Its output goes to
`%USERPROFILE%\.voxscribe\voxscribe.log`. `Create-Shortcuts.ps1` points the
Desktop/Start menu shortcuts at it; re-run it only if you move the folder.

CLI: see [CLI reference](cli-reference.md), e.g.

```bash
uv run --no-sync python main.py transcribe -i recordings/audio.wav
```

## Tests

```bash
uv run --no-sync python -m pytest
```

## Offline models and the `.exe`

```bash
uv run --no-sync python download_models.py
```

downloads the models into `bundled_models/` (diarization models only with
`HF_TOKEN` set) for offline operation without a token.
`python build_exe.py` builds a standalone Windows `.exe` with PyInstaller.

## Configuration files

| File | Content |
|---|---|
| `.env` (VoxScribe folder, not committed) | `HF_TOKEN=...`, written by Settings → HuggingFace token |
| `~/.voxscribe/providers.json` | Providers, see [Providers](providers.md) |
| `~/.voxscribe/speaker_profiles.json` | Voice profiles for speaker recognition |
| `~/.voxscribe/voxscribe.log` | GUI log |

## Project structure

```
├── gui_qt.py                    # GUI entry point (PySide6/Qt) - thin, delegates to qt_app/
├── main.py                      # CLI entry point - thin, delegates to voxscribe/cli.py
├── download_models.py           # Thin entry point -> voxscribe/download_models.py
├── build_exe.py                 # Thin entry point -> voxscribe/build_exe.py
├── release.py                   # Thin entry point -> voxscribe/release.py (maintainer: cut a Release)
├── VoxScribe.bat                # Windows launcher (sync when needed, apply Updates, start GUI)
├── Create-Shortcuts.ps1         # Desktop + Start menu shortcuts to VoxScribe.bat
├── qt_app/                      # GUI code: main_window.py, controllers.py, theme.py, pages/, widgets/
├── voxscribe/                   # Backend package
│   ├── cli.py                   # CLI command implementations
│   ├── recorder.py              # Audio recording (microphone + WASAPI loopback/ScreenCaptureKit + combined)
│   ├── transcriber.py           # Facade for the transcription API (re-exports from transcription/)
│   ├── transcription/           # Pipeline: Backends (WhisperX, Provider, OpenVINO, Parakeet, Apple), alignment, diarization, speakers
│   ├── providers.py             # Persistent Provider configuration
│   ├── env_file.py              # Writes HF_TOKEN to .env from Settings
│   ├── speaker_profiles.py      # Persistent voice prints for recognition
│   ├── hardware_detect.py       # GPU/CPU detection + model recommendation
│   ├── gpu_setup.py             # Automatic CUDA/XPU torch backend correction (Windows)
│   ├── updates.py               # Installed Release + Updates
│   ├── release.py               # Release script: bump version, tag vX.Y.Z, push
│   ├── download_models.py       # Pre-download models for offline/bundled operation
│   └── build_exe.py             # PyInstaller build (Windows)
├── macos/                       # macOS Swift helpers + build.sh, make-app.sh
├── docs/                        # Detailed docs
├── tests/
└── recordings/                  # Recordings + transcripts (not committed)
```

Architecture and design decisions: `CLAUDE.md`, `GLOSSARY.md`, `docs/adr/`.

## Corporate proxy / SSL

VoxScribe turns off HTTPS certificate verification at start (`gui_qt.py`,
`voxscribe/cli.py`) so model downloads work behind intercepting proxies. For
`pip` installs you may also need
`--trusted-host pypi.org --trusted-host files.pythonhosted.org`.
