# Intel Arc GPU (Windows)

On machines with Intel Arc graphics (e.g. Core Ultra laptops) and **no**
NVIDIA GPU, transcription can run via **Intel OpenVINO** instead of
CTranslate2 - CTranslate2 (WhisperX' default backend) has no Intel GPU
support, so this is a separate, parallel path. Benchmarked at roughly
**10-16x realtime** on a Core Ultra 7 258V (Arc 140V), vs. ~1.6-1.9x on the CPU.

## Setup

1. `optimum-intel[openvino]` is installed automatically on Windows (`uv sync`)
2. Export the model to OpenVINO IR (one-time, ~3 GB download): `python download_models.py` (step 4/4)
3. In the GUI's Transcription tab, pick **"medium (Intel Arc GPU)"** as the model, or via CLI: `python main.py transcribe -i recordings/meeting.wav -m openvino:GPU:medium`

This part is fully automatic: `voxscribe/gpu_setup.py` detects an Arc GPU on every app
start and, if needed, switches `torch` to the correct build for you (see
below) - no manual steps required for the common case.

## Alignment/diarization on the Arc GPU too (instead of CPU)

Transcription via OpenVINO above is independent of which `torch` build is
installed. But alignment (wav2vec2) and diarization (pyannote) are plain
PyTorch, so they need `torch` itself built with Intel XPU support to use the
Arc GPU - the standard install uses the CUDA build.

**This is automatic**: `voxscribe/gpu_setup.py` runs at every app start (GUI and CLI),
detects the GPU present via a fast WMI query, and switches `torch` to the
matching build automatically if it doesn't match - CUDA for a dedicated
NVIDIA GPU, XPU for an Intel Arc GPU (dGPU takes priority if both are
present), or leaves it alone if neither is detected. The switch itself (the
one time it's actually needed) runs `uv pip install --reinstall` with the
correct index and can take a few minutes; every day-to-day start after that
is a fast no-op check.

To do the switch manually instead:

```bash
uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/xpu --reinstall
```

**Important:** a single `torch` build only ever supports one GPU backend
(CUDA **or** XPU, never both) - after this command, `torch` no longer runs
with CUDA. A plain `uv sync`/`uv run` (without `--no-sync`) reverts this back
to CUDA, since `pyproject.toml` still declares CUDA as the default; the
automatic check above exists specifically to self-heal that.

If an NVIDIA GPU is present, it automatically takes priority (CUDA > Intel
Arc GPU > Apple MPS > CPU) - this applies to both Whisper transcription and
alignment/diarization (`hardware_detect.recommend_model()`).
