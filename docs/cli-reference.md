# CLI reference

## Commands

| Command | Description |
|--------|-------------|
| `devices` | List available audio devices |
| `record` | Record audio |
| `transcribe` | Transcribe an audio file |
| `run` | Record + transcribe immediately |

## `record` / `run` recording options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `room` | Recording mode: `room` (microphone only), `playback` (what the computer plays, e.g. Teams/Zoom), or `call` (playback + microphone, saved as separate tracks too) |
| `--setup` | guessed | Call only: `headset` or `loudspeaker` (speakers + room mic/speakerphone). Guessed from the devices if omitted, and stored with the recording |
| `--device` | default input | Microphone device index (see `devices`) |
| `--playback-device` | default output | Playback (loopback) device index (Windows) |
| `--name` | `Playback`, `Meeting` or `Room` after the mode | Recording title, the end of the file name: `recordings/YYMMDD_HHMM_<title>.wav`. Ignored, with a note, when `-o`/`--output` gives the path |
| `--source` | – | Deprecated alias: `mic` = `--mode room`, `system` = `--mode playback`, `both` = `--mode call` |

## `transcribe` options

| Option | Default | Description |
|--------|---------|-------------|
| `--language`, `-l` | `auto` | Language (en, de, fr, es, ...) or `auto` for automatic detection |
| `--model`, `-m` | your default provider, if configured (falls back automatically to the recommended local model on failure) | Whisper model (large-v2, large-v3, medium, base), `openvino:GPU:medium` (Intel Arc GPU, Windows only, see [Intel Arc GPU](intel-arc-gpu.md)), a configured provider (`server:<provider-id>`, see [Providers](providers.md)), or `apple:speechanalyzer` (macOS only) |
| `--diarize` / `--no-diarize` | `--diarize` | Speaker diarization on/off |
| `--min-speakers` | – | Minimum number of speakers |
| `--max-speakers` | – | Maximum number of speakers |
| `--batch-size` | `16` | Batch size (smaller = less VRAM) |
| `--device-compute` | auto | `cuda`, `xpu` (Intel Arc GPU), `mps` (Apple Silicon), or `cpu` — only controls alignment/diarization, not Whisper transcription itself (which for `openvino:...` models always runs independently via OpenVINO) |
| `--api-key` | – | API key for ad-hoc provider use, overriding a saved provider's key |
| `--api-base-url` | – | Base URL for ad-hoc provider use, overriding a saved provider's URL |
| `--format`, `-f` | `txt` | Output format(s): txt, srt, json (comma-separated) |
| `--output`, `-o` | auto | Output path |

## Examples

```bash
# Record from microphone (stop with ENTER)
python main.py record --mode room

# Playback audio only - what Teams/Zoom plays (WASAPI loopback)
python main.py record --mode playback

# A call on a headset: playback + the headset's microphone
python main.py record --mode call --setup headset

# Give the recording a title: recordings/YYMMDD_HHMM_ASAM_WG.wav
python main.py record --mode call --name "ASAM WG"

# Pick a specific microphone (index from 'devices')
python main.py record --mode room --device 15

# Transcribe with automatic language detection (default)
python main.py transcribe -i recordings/meeting.wav

# Force a language
python main.py transcribe -i recordings/meeting.wav -l en

# Without speaker diarization (faster)
python main.py transcribe -i recordings/meeting.wav --no-diarize

# Multiple output formats
python main.py transcribe -i recordings/meeting.wav -f txt,srt,json

# Constrain the speaker count
python main.py transcribe -i recordings/meeting.wav --min-speakers 2 --max-speakers 4

# Smaller local model (less VRAM, faster, less accurate)
python main.py transcribe -i recordings/meeting.wav -m medium

# Transcribe a video file directly (only the audio track is extracted)
python main.py transcribe -i recordings/meeting.mkv

# Transcribe via a specific configured provider instead of the default
python main.py transcribe -i recordings/meeting.wav -m server:my-provider

# Record + transcribe immediately
python main.py run --mode call -l en -m large-v2 --min-speakers 2 --max-speakers 2 -f txt,srt
```
