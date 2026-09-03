"""Sorgt automatisch dafuer, dass die installierte torch-Variante (CUDA/XPU)
zur tatsaechlich vorhandenen GPU passt - Windows-only.

Hintergrund: ein torch-Build unterstuetzt immer nur EIN GPU-Backend (CUDA ODER
Intel-XPU, nie beide - siehe pyproject.toml). `pyproject.toml` pinnt den
CUDA-Build als Projekt-Default; auf einem Rechner mit Intel Arc GPU (und ohne
NVIDIA-GPU) muss torch stattdessen manuell auf den XPU-Build umgestellt werden,
damit Alignment/Diarization (plain PyTorch) die Arc GPU nutzen koennen statt
auf die CPU zu fallen. Ausserdem setzt jedes `uv run`/`uv sync` (ohne
`--no-sync`) diese manuelle Umstellung automatisch wieder auf CUDA zurueck,
da `pyproject.toml` weiterhin CUDA deklariert.

Dieses Modul macht die noetige Korrektur selbst automatisch statt sie als
manuellen Schritt zu dokumentieren: bei jedem App-Start (CLI und GUI, siehe
main.py/gui_qt.py) wird schnell geprueft, ob die installierte torch-Variante
zur erkannten GPU passt (Prioritaet dGPU > iGPU, siehe hardware_detect.py) -
und falls nicht, automatisch per `uv pip install --reinstall` korrigiert,
BEVOR irgendwo `import torch` passiert (torch laedt seine nativen Bibliotheken
nur beim ersten Import in diesem Prozess - ein Wechsel auf der Festplatte
davor wirkt sich noch auf denselben Prozess aus, ein Wechsel danach nicht mehr,
siehe main()/gui_qt.py's Aufrufreihenfolge).

Der eigentliche Check (WMI-Abfrage + `importlib.metadata`) importiert bewusst
kein torch/whisperx (die haben einen 1-3 Minuten dauernden Kaltstart-Import,
siehe CLAUDE.md) und ist daher auch bei jedem Start unauffaellig schnell -
nur wenn tatsaechlich ein Wechsel noetig ist, dauert es laenger (Download,
einmalig).
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
    """Liefert die Namen aller erkannten GPUs via WMI - kein torch/openvino
    noetig, daher schnell genug fuer einen Check bei jedem Programmstart."""
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
    """'cu128', 'xpu', 'cpu' oder None (torch nicht installiert) - liest nur
    die Paket-Metadaten (PEP 440 Local-Version-Suffix), importiert torch
    selbst nicht."""
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None
    if "+" not in version:
        return "cpu"
    return version.split("+", 1)[1]


def _find_uv() -> str | None:
    """Bevorzugt den von astral.sh's offiziellem Installer genutzten Standardpfad
    (siehe README) vor einem generischen PATH-Treffer - auf Entwicklungsrechnern
    kann `shutil.which("uv")` sonst ein von einem ganz anderen Tool mitgebrachtes
    `uv.exe` zuerst finden (in der Praxis beobachtet)."""
    standard = os.path.expanduser(r"~\.local\bin\uv.exe")
    if os.path.isfile(standard):
        return standard
    return shutil.which("uv")


def ensure_correct_torch_backend(on_status=print) -> None:
    """Prueft GPU vs. installierte torch-Variante und korrigiert bei Bedarf
    automatisch. `on_status(msg)` wird fuer Fortschrittsmeldungen aufgerufen
    (Default: print - CLI; die GUI kann hier z.B. den Splash-Screen-Text
    aktualisieren)."""
    if sys.platform != "win32":
        return  # macOS (MPS) / Linux betrifft dieses CUDA/XPU-Problem nicht

    if getattr(sys, "frozen", False):
        # In der PyInstaller-.exe gibt es kein `.venv`, in das `uv pip install`
        # sinnvoll reinstallieren koennte - die dort gebuendelte torch-Variante
        # ist zur Build-Zeit fixiert (siehe build_exe.py). Diese Auto-Korrektur
        # gilt nur fuer die Source-/uv-Installation.
        return

    gpu_names = _detect_gpu_names()
    has_nvidia = any("NVIDIA" in n.upper() for n in gpu_names)
    has_arc = any("ARC" in n.upper() and "INTEL" in n.upper() for n in gpu_names)

    # Prioritaet dGPU > iGPU > keine, siehe hardware_detect.recommend_model()
    if has_nvidia:
        desired = "cu128"
    elif has_arc:
        desired = "xpu"
    else:
        return  # keine dGPU/Arc-iGPU erkannt - aktuelle Variante nicht anfassen

    current = _current_torch_variant()
    if current is None or current == desired:
        return  # torch fehlt komplett (kein Fall hier fuer uns) oder passt schon

    uv = _find_uv()
    if uv is None:
        on_status(
            f"[GPU-Setup] {gpu_names[0] if gpu_names else 'GPU'} erkannt, aber "
            f"torch ist auf '{current}' statt '{desired}' - 'uv' nicht gefunden, "
            "automatische Korrektur uebersprungen. Siehe README ('Intel Arc GPU')."
        )
        return

    index = PYTORCH_XPU_INDEX if desired == "xpu" else PYTORCH_CUDA_INDEX
    label = "Intel Arc GPU" if desired == "xpu" else "NVIDIA-GPU"
    on_status(
        f"[GPU-Setup] {label} erkannt, torch laeuft aber mit '{current}' statt "
        f"'{desired}' - richte GPU-Beschleunigung einmalig ein (Download, kann "
        "einige Minuten dauern)..."
    )
    try:
        result = subprocess.run(
            [uv, "pip", "install", *_TORCH_PACKAGES,
             "--index-url", index, "--reinstall"],
            timeout=1800,
        )
    except Exception as e:
        on_status(f"[GPU-Setup] Automatische Korrektur fehlgeschlagen: {e}")
        return

    if result.returncode == 0:
        on_status(f"[GPU-Setup] torch erfolgreich auf '{desired}' umgestellt.")
    else:
        on_status(
            f"[GPU-Setup] Automatische Korrektur fehlgeschlagen (Exit-Code "
            f"{result.returncode}) - GPU-Beschleunigung evtl. nicht verfuegbar."
        )
