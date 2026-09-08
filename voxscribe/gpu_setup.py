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

To stop that reinstall from repeating on every launch (the `uv run` re-sync
above would otherwise revert it again each time), the first correction also
writes `UV_NO_SYNC=1` into the project's `.env` - `uv run` auto-loads that
file, so later launches skip the sync entirely and this check becomes a true
no-op. Only happens when the GPU needs a build other than pyproject.toml's
pinned one (the Intel Arc case); an NVIDIA machine already matches the pin and
is left untouched. Run `uv sync` by hand after changing dependencies.
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

# The torch build pyproject.toml pins via [tool.uv.sources]. On a machine whose
# GPU needs a *different* build (the Intel Arc / "xpu" case), a plain `uv run`
# re-syncs the venv and reverts torch back to this build on every launch -
# _persist_uv_no_sync() writes a guard so that stops happening.
_PYPROJECT_TORCH_VARIANT = "cu128"


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


def _uv_no_sync_is_set(env_text: str) -> bool:
    """True if the .env text already sets UV_NO_SYNC to a truthy value."""
    for line in env_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() != "UV_NO_SYNC":
            continue
        return value.strip().strip("\"'").lower() not in ("", "0", "false", "no")
    return False


def _persist_uv_no_sync(on_status=print) -> None:
    """Append UV_NO_SYNC=1 to the project's .env (which `uv run` auto-loads), so
    subsequent launches don't re-sync the venv and revert the torch build this
    module just corrected. Idempotent - does nothing (and prints nothing) once
    the guard is in place, which is what makes ensure_correct_torch_backend() a
    genuine no-op on every launch after the first."""
    # `uv run` reads .env from its invocation directory, which for this project
    # is the repo root - match that rather than this file's location.
    env_path = os.path.join(os.getcwd(), ".env")
    try:
        existing = ""
        if os.path.isfile(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                existing = f.read()
        if _uv_no_sync_is_set(existing):
            return  # already guarded - nothing to change

        block = (
            "\n# Added automatically by voxscribe/gpu_setup.py: this machine's GPU\n"
            "# needs a torch build that differs from the one pinned in\n"
            "# pyproject.toml, and a plain `uv run` would re-sync the venv and\n"
            "# revert it on every launch. `uv run` auto-loads this file, so\n"
            "# UV_NO_SYNC=1 keeps that from happening. After changing\n"
            "# dependencies, run `uv sync` by hand (or delete this line).\n"
            "UV_NO_SYNC=1\n"
        )
        with open(env_path, "a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(block)
        on_status(
            "[GPU setup] Wrote UV_NO_SYNC=1 to .env so future launches keep this "
            "torch build instead of re-installing it every time. Run `uv sync` "
            "manually after changing dependencies."
        )
    except Exception as e:
        on_status(
            f"[GPU setup] Could not add UV_NO_SYNC to .env ({e}) - the torch "
            "check may re-run on the next launch."
        )


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
    if current is None:
        return  # torch isn't installed at all - not this module's job to bootstrap

    # If the GPU needs a build other than pyproject.toml's pinned one, keep
    # `uv run` from silently reverting it on every future launch. Idempotent, so
    # once the guard exists this is a no-op and the early return below makes the
    # whole function do nothing on subsequent starts.
    if desired != _PYPROJECT_TORCH_VARIANT:
        _persist_uv_no_sync(on_status)

    if current == desired:
        return  # already the right build - nothing to install

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
