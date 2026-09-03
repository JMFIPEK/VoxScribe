"""Automatically ensures the installed torch build (CUDA/XPU) matches the GPU
actually present - Windows-only.

Background: a torch build only ever supports ONE GPU backend (CUDA OR Intel
XPU, never both - see pyproject.toml). `pyproject.toml` pins the CUDA build
as the project default; on a machine with an Intel Arc GPU (and no NVIDIA
GPU), torch instead has to be manually switched to the XPU build so that
alignment/diarization (plain PyTorch) can use the Arc GPU instead of falling
back to the CPU. On top of that, every `uv run`/`uv sync` (without
`--no-sync`) automatically reverts that manual switch back to CUDA, since
`pyproject.toml` still declares CUDA.

This module makes the necessary correction automatic instead of documenting
it as a manual step: on every app start (CLI and GUI, see main.py/gui_qt.py)
it quickly checks whether the installed torch build matches the detected GPU
(priority dGPU > iGPU, see hardware_detect.py) - and if not, fixes it
automatically via `uv pip install --reinstall`, BEFORE anything imports torch
anywhere (torch only loads its native libraries on the first import in this
process - a change on disk before that point still affects this same
process, a change after it no longer does, see the call order in
main()/gui_qt.py).

The actual check (WMI query + `importlib.metadata`) deliberately doesn't
import torch/whisperx (which have a 1-3 minute cold-start import, see
CLAUDE.md) and is therefore unobtrusively fast on every start too - only when
a switch is actually needed does it take longer (download, one-time).
"""

import importlib.metadata
import os
import shutil
import subprocess
import sys

PYTORCH_XPU_INDEX = "https://download.pytorch.org/whl/xpu"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
_TORCH_PACKAGES = ["torch", "torchaudio", "torchvision"]


def _detect_gpu_names() -> list[str]:
    """Returns the names of all detected GPUs via WMI - no torch/openvino
    needed, so fast enough to check on every program start."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=10,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _current_torch_variant() -> str | None:
    """'cu128', 'xpu', 'cpu', or None (torch not installed) - only reads the
    package metadata (PEP 440 local version suffix), doesn't import torch itself."""
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
    if "+" not in version:
        return "cpu"
    return version.split("+", 1)[1]


def _find_uv() -> str | None:
    """Prefers the standard path used by astral.sh's official installer (see
    README) over a generic PATH hit - on development machines,
    `shutil.which("uv")` can otherwise find a `uv.exe` bundled with some
    completely different tool first (observed in practice)."""
    standard = os.path.expanduser(r"~\.local\bin\uv.exe")
    if os.path.isfile(standard):
        return standard
    return shutil.which("uv")


def ensure_correct_torch_backend(on_status=print) -> None:
    """Checks GPU vs. installed torch build and fixes it automatically if
    needed. `on_status(msg)` is called for progress messages (default: print
    - CLI; the GUI can use this to e.g. update the splash screen text)."""
    if sys.platform != "win32":
        return  # macOS (MPS) / Linux aren't affected by this CUDA/XPU issue

    if getattr(sys, "frozen", False):
        # There's no `.venv` inside the PyInstaller .exe for `uv pip install`
        # to meaningfully reinstall into - the torch build bundled there is
        # fixed at build time (see build_exe.py). This auto-correction only
        # applies to the source/uv install.
        return

    gpu_names = _detect_gpu_names()
    has_nvidia = any("NVIDIA" in n.upper() for n in gpu_names)
    has_arc = any("ARC" in n.upper() and "INTEL" in n.upper() for n in gpu_names)

    # Priority dGPU > iGPU > none, see hardware_detect.recommend_model()
    if has_nvidia:
        desired = "cu128"
    elif has_arc:
        desired = "xpu"
    else:
        return  # no dGPU/Arc iGPU detected - leave the current build alone

    current = _current_torch_variant()
    if current is None or current == desired:
        return  # torch isn't installed at all (not our case here), or already matches

    uv = _find_uv()
    if uv is None:
        on_status(
            f"[GPU setup] {gpu_names[0] if gpu_names else 'GPU'} detected, but "
            f"torch is on '{current}' instead of '{desired}' - 'uv' not found, "
            "automatic correction skipped. See README ('Intel Arc GPU')."
        )
        return

    index = PYTORCH_XPU_INDEX if desired == "xpu" else PYTORCH_CUDA_INDEX
    label = "Intel Arc GPU" if desired == "xpu" else "NVIDIA GPU"
    on_status(
        f"[GPU setup] {label} detected, but torch is running with '{current}' "
        f"instead of '{desired}' - setting up GPU acceleration (one-time "
        "download, can take a few minutes)..."
    )
    try:
        result = subprocess.run(
            [uv, "pip", "install", *_TORCH_PACKAGES,
             "--index-url", index, "--reinstall"],
            timeout=1800,
        )
    except Exception as e:
        on_status(f"[GPU setup] Automatic correction failed: {e}")
        return

    if result.returncode == 0:
        on_status(f"[GPU setup] torch successfully switched to '{desired}'.")
    else:
        on_status(
            f"[GPU setup] Automatic correction failed (exit code "
            f"{result.returncode}) - GPU acceleration may not be available."
        )
