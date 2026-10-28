# VoxScribe

Record a meeting or call and get a transcript with word-accurate timestamps and speaker names, from one desktop app. Name a speaker once and VoxScribe recognizes their voice in later recordings.

Transcription runs either on a **provider** (any OpenAI-compatible transcription service you add, e.g. OpenRouter) or **fully locally** on your computer. Without a provider, or when it's unreachable, VoxScribe uses the best local model for your hardware automatically.

![VoxScribe — Record tab](docs/images/screenshot.png)

**Platforms:** Windows 10/11 (full feature set) · macOS 26+ on Apple Silicon ([macOS setup](docs/macos.md)) · Linux (experimental, microphone only)

> 🤖 **Let an AI agent install it.** Paste this into Claude Code, Codex or a similar coding agent:
>
> ```
> Install VoxScribe on this computer by following https://github.com/JMFIPEK/VoxScribe/blob/main/docs/install-with-agent.md exactly.
> ```
>
> The agent installs everything and then tells you the two things only you can do: paste your HuggingFace token and add a provider (steps 2 and 3 below).

## 1. Install (Windows)

**Prerequisites** (once, in PowerShell). Skip whichever you already have:

```powershell
winget install --id Git.Git -e
```

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell afterwards so `git` and `uv` are found.

**Get VoxScribe** and switch to the newest Release:

```powershell
git clone https://github.com/JMFIPEK/VoxScribe.git voxscribe
```

(KIT internal: `git clone https://gitlab.kit.edu/kit/ipek/acm/voxscribe.git voxscribe`)

```powershell
cd voxscribe
```

```powershell
git checkout $(git describe --tags --abbrev=0 --match "v*")
```

Git's "detached HEAD" note is expected: Releases are tags, not a branch.

**Create the shortcuts and start it:**

```powershell
powershell -ExecutionPolicy Bypass -File Create-Shortcuts.ps1
```

This puts a **VoxScribe** entry on the Desktop and in the Start menu (to pin it to the taskbar: right-click the Start menu entry → *Pin to taskbar*). Start VoxScribe from there.

The **first start installs everything** (Python packages incl. PyTorch, several GB) in a console window and takes a few minutes. Later starts open the app directly. On an Intel Arc GPU laptop, the first start may also switch PyTorch to the Intel build once, see [Intel Arc GPU](docs/intel-arc-gpu.md).

## 2. HuggingFace token (for speaker names)

Without it, transcription works but the transcript has no speakers. It's free:

1. Create an account at [huggingface.co](https://huggingface.co) and a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **Create new token** → type **Read** → copy it (`hf_...`).
2. While logged in, open both model pages and click **Agree and access repository** on each:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. In VoxScribe, open **Settings**, paste the token into **HuggingFace token**, and click anywhere else. It is saved and used from then on.

Step 2 is the one people miss: a token without the two accepted licenses fails at the speaker step.

<details>
<summary>Alternative: put the token in <code>.env</code> yourself</summary>

Create a file named `.env` in the VoxScribe folder (next to `VoxScribe.bat`) with this line, then restart VoxScribe:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

The Settings field writes this same file, and the CLI reads it too.
</details>

## 3. Add a provider (optional, recommended)

A provider transcribes on a server: no model download, and fast on any laptop. The audio then leaves your computer for that request. Skip this to transcribe fully locally.

1. Get an API key from [OpenRouter](https://openrouter.ai/settings/keys) (or use any other OpenAI-compatible transcription service).
2. In VoxScribe: **Settings → Providers → Add...**
3. Pick **Preset: OpenRouter**. Name, base URL and model are filled in. Paste your **API key** and save.
4. Select the provider and click **Set as default**.

Details, other services and troubleshooting: [Providers](docs/providers.md).

## 4. Record and transcribe

- **Record** tab: choose a **Recording mode**: **Call** (Teams/Zoom: what you hear + your microphone, as separate tracks so you and the remote side are told apart), **Playback only** (what the computer plays) or **In room** (microphone only). For a Call, pick the **Setup**: **Headset** or **Loudspeaker**. VoxScribe preselects it from your devices. Give it a title if you like, then **Start** / **Stop**.
- **Transcription** tab: pick the recording (or any audio/video file: WAV, MP3, MKV, MP4, MOV, ...), then **Transcribe**. Click a speaker's name to rename them. From 10 s of speech on, their voice is remembered and recognized next time. Export as TXT, SRT or JSON.

Recordings and transcripts are saved in the `recordings/` folder. Voice profiles live in `%USERPROFILE%\.voxscribe\speaker_profiles.json`. Recognition is a heuristic, so check the names before you share a transcript. A name with a question mark ("Jonas?") is a guess you can confirm with one click.

## Updates

When a newer Release is out, VoxScribe shows a notice. Choose **Install and relaunch** (or **Install and quit**). Installs cloned before v1.1.0 or still on the `main` branch move to a Release once by hand: see [Updating](docs/updating.md).

## More

- [macOS setup](docs/macos.md) · [Providers](docs/providers.md) · [Intel Arc GPU](docs/intel-arc-gpu.md) · [Known issues](docs/known-issues.md)
- [Command line (CLI)](docs/cli-reference.md): record and transcribe from a terminal or script
- [Development](docs/development.md): running from a terminal, project structure, tests, building the `.exe`
- Logs: `%USERPROFILE%\.voxscribe\voxscribe.log` (previous session: `voxscribe.log.1`)
- Behind a corporate proxy, VoxScribe turns off SSL certificate checks for its own downloads automatically.
