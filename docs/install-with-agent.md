# Installing VoxScribe: instructions for an AI agent

You are a coding agent (Claude Code, Codex, ...) asked to install VoxScribe on
the user's computer. Follow these steps in order. Run each check and stop
when one fails: report the exact output to the user instead of improvising.

## Rules

- **Never ask for, type, read or write the HuggingFace token or any API key.**
  Don't create or edit `.env` or `~/.voxscribe/providers.json`. The user enters
  secrets in the app themselves (see "Hand over" below).
- **Ask before installing software** (Git, uv, Xcode Command Line Tools). Say
  what you will install and the command you'll use.
- **Don't overwrite anything.** If the target folder already exists, ask the
  user whether to use it, pick another one, or stop.
- Don't download models ahead of time: they load on first use.
- Don't change system settings, PATH entries or privacy permissions.
- Default install location: a `voxscribe` folder in the user's home directory,
  unless the user named another one.
- Clone URL: `https://github.com/JMFIPEK/VoxScribe.git`. If the user said they
  are at KIT or gave the KIT GitLab, use
  `https://gitlab.kit.edu/kit/ipek/acm/voxscribe.git` instead.

## 1. Find the platform

| Platform | Continue with |
|---|---|
| Windows 10/11 | 2W |
| macOS 26+ on Apple Silicon | 2M |
| macOS older than 26, or Intel Mac | Stop: not supported. Tell the user. |
| Linux | 2L (experimental, microphone recording only, tell the user) |

## 2W. Windows

Run these in PowerShell.

1. **Git**: `git --version`. If missing, ask, then
   `winget install --id Git.Git -e`. Open a new shell afterwards.
2. **uv**: `& "$env:USERPROFILE\.local\bin\uv.exe" --version`, else
   `uv --version`. If missing, ask, then
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
   Use `$env:USERPROFILE\.local\bin\uv.exe` as `uv` from here on if `uv` isn't
   on PATH yet.
3. **Clone and check out the newest Release:**
   ```powershell
   git clone <clone URL> "$env:USERPROFILE\voxscribe"
   cd "$env:USERPROFILE\voxscribe"
   git checkout $(git describe --tags --abbrev=0 --match "v*")
   ```
   Check: `git describe --tags --exact-match` prints `vX.Y.Z`. The "detached
   HEAD" note is expected.
4. **Install the environment** (several GB incl. PyTorch, can take 5-15
   minutes; run it with a long timeout or in the background):
   ```powershell
   uv sync --locked
   ```
   Check: `.venv\Scripts\python.exe` exists.
5. **Smoke test:**
   ```powershell
   uv run --no-sync python main.py devices
   ```
   Check: it lists at least one input device and exits without a traceback.
   Always use `--no-sync` with `uv run` here.
6. **Shortcuts:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File Create-Shortcuts.ps1
   ```
   Check: a `VoxScribe` shortcut exists on the Desktop.

The first start from the shortcut may sync once more and, on an Intel Arc GPU
laptop, switch PyTorch to the Intel build (a few minutes). That's expected;
mention it to the user.

Continue with "Hand over".

## 2M. macOS

1. **Git and Xcode Command Line Tools**: `xcode-select -p`. If it fails, ask,
   then run `xcode-select --install`. That opens a dialog **the user must
   confirm**; wait until they say it finished. Check: `git --version` and
   `swiftc --version` both work.
2. **uv**: `uv --version` or `~/.local/bin/uv --version`. If missing, ask,
   then `curl -LsSf https://astral.sh/uv/install.sh | sh`.
3. **Clone and check out the newest Release:**
   ```bash
   git clone <clone URL> ~/voxscribe
   cd ~/voxscribe
   git checkout $(git describe --tags --abbrev=0 --match "v*")
   ```
   Check: `git describe --tags --exact-match` prints `vX.Y.Z`.
4. **Environment:** `uv sync --locked`. Check: `.venv/bin/python` exists.
5. **Swift helpers:** `cd macos && ./build.sh && cd ..`. Check:
   `macos/SystemAudioCapture` and `macos/SpeechAnalyzerTranscribe` exist.
6. **App launcher:** `macos/make-app.sh`. Check:
   `~/Applications/VoxScribe.app` exists.
7. **Smoke test:** `uv run --no-sync python main.py devices` lists at least
   one input device.

Continue with "Hand over". Add for macOS: on first use, macOS asks for the
**Microphone** and **Screen & System Audio Recording** permissions for
VoxScribe; the user must allow both.

## 2L. Linux

1. Git and uv as on macOS (step 1: `git --version`, install via the
   distribution's package manager after asking; step 2 as above).
2. Clone and check out the Release as on macOS.
3. `uv sync --locked`, then `uv run --no-sync python main.py devices`.
4. Start command for the user: `cd ~/voxscribe && uv run --no-sync python gui_qt.py`.

## Hand over

Tell the user, in this order, that the install is done and what's left for
them (link them to the README sections):

1. **Start VoxScribe** from the Desktop/Start menu shortcut (Windows), from
   Spotlight/Launchpad as *VoxScribe* (macOS), or with the command above
   (Linux).
2. **HuggingFace token** (needed for speaker names), README section 2:
   create a *Read* token at <https://huggingface.co/settings/tokens>, accept
   the licenses on
   <https://huggingface.co/pyannote/speaker-diarization-3.1> and
   <https://huggingface.co/pyannote/segmentation-3.0>, then paste the token
   into **Settings → HuggingFace token**.
3. **Provider** (optional), README section 3: **Settings → Providers →
   Add...**, Preset *OpenRouter*, paste their own API key, **Set as default**.
   Without one, transcription runs locally.

Also report: the install folder, the installed Release (`vX.Y.Z`), and
anything that failed or was skipped.
