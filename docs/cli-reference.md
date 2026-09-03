# CLI reference

## Commands

| Command | Description |
|--------|-------------|
| `devices` | List available audio devices |
| `record` | Record audio |
| `transcribe` | Transcribe an audio file |
| `run` | Record + transcribe immediately |

## `transcribe` options

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | `mic` | `mic`, `system` (WASAPI loopback), or `both` (microphone + system) |
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
python main.py record --source mic

# System audio / Teams / Zoom (WASAPI loopback)
python main.py record --source system

# Pick a specific microphone (index from 'devices')
python main.py record --source mic --device 15

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
python main.py run --source system -l en -m large-v2 --min-speakers 2 --max-speakers 2 -f txt,srt
```
