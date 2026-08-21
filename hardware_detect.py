"""Hardware-Erkennung und automatische Modellauswahl fuer WhisperX."""

import os
import platform

import torch


def get_gpu_info() -> dict | None:
    """Ermittelt GPU-Informationen (Name, VRAM, Backend) falls CUDA, Intel XPU oder
    MPS verfuegbar sind. Prioritaet dGPU > iGPU > integrierte Alternative: CUDA
    (dedizierte NVIDIA-GPU) wird zuerst geprueft, dann Intel XPU (Arc-iGPU/dGPU -
    nur verfuegbar, wenn torch mit einem XPU-faehigen Build installiert wurde,
    siehe pyproject.toml's "xpu"-Extra; die staerker verbreitete CUDA-Standard-
    Installation hat torch.xpu.is_available() == False), dann Apple MPS."""
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            # PyTorch >= 2.11 uses total_memory, older versions use total_mem
            vram_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            vram_total = vram_bytes / (1024 ** 3)  # GB
            return {"name": gpu_name, "vram_gb": vram_total, "backend": "cuda"}
        except Exception:
            return None

    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        try:
            gpu_name = torch.xpu.get_device_name(0)
            props = torch.xpu.get_device_properties(0)
            vram_bytes = getattr(props, "total_memory", 0)
            vram_total = vram_bytes / (1024 ** 3)  # GB
            return {"name": gpu_name, "vram_gb": vram_total, "backend": "xpu"}
        except Exception:
            return None

    if torch.backends.mps.is_available():
        # Apple Silicon: keine dedizierte VRAM-Abfrage moeglich (unified memory
        # wird mit der CPU geteilt) - RAM-Gesamtgroesse als Naeherung nutzen.
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            return {"name": platform.processor() or "Apple GPU (MPS)",
                    "vram_gb": ram_gb, "backend": "mps"}
        except Exception:
            return {"name": "Apple GPU (MPS)", "vram_gb": 0.0, "backend": "mps"}

    return None


def get_cpu_info() -> dict:
    """Ermittelt CPU-Informationen (Name, Kernanzahl, RAM)."""
    import psutil

    cpu_count = os.cpu_count() or 4
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    # CPU-Name ermitteln
    cpu_name = platform.processor() or "Unknown CPU"
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            winreg.CloseKey(key)
        except Exception:
            pass

    return {"name": cpu_name, "cores": cpu_count, "ram_gb": ram_gb}


def get_openvino_devices() -> list[str] | None:
    """Ermittelt die von OpenVINO erkannten Inferenz-Geraete (z.B. ['CPU','GPU','NPU']
    fuer eine Intel Arc GPU + NPU). Gibt None zurueck, wenn das optionale
    `optimum-intel[openvino]`-Paket nicht installiert ist (nur auf Windows in
    pyproject.toml vorgesehen, siehe transcriber.transcribe_openvino())."""
    try:
        from openvino import Core
    except ImportError:
        return None
    try:
        return Core().available_devices
    except Exception:
        return None


def _bundled_openvino_model_exists(whisper_size: str = "medium") -> bool:
    """Prueft, ob download_models.py::export_openvino_model() bereits gelaufen ist
    (bundled_models/openvino/whisper-<size>/) - ohne dieses Modell kann
    transcribe_openvino() nichts laden, auch wenn eine Arc GPU/NPU erkannt wird."""
    base = os.path.dirname(os.path.abspath(__file__))
    marker = os.path.join(base, "bundled_models", "openvino", f"whisper-{whisper_size}",
                           "openvino_encoder_model.bin")
    return os.path.isfile(marker)


def recommend_model() -> tuple[str, str, str]:
    """Empfiehlt das optimale Modell basierend auf der verfuegbaren Hardware.

    Returns:
        Tuple (model_name, device, reason)
        - model_name: "large-v3", "large-v2", "medium", "base" oder
          "openvino:GPU:medium" (Intel Arc GPU via OpenVINO)
        - device: "cuda", "xpu", "mps" oder "cpu" - Geraet fuer Alignment/Diarization
          (plain PyTorch). Bei "openvino:..."-Modellen laeuft die eigentliche
          Whisper-Transkription unabhaengig davon immer ueber OpenVINO auf dem im
          model_name kodierten Geraet, siehe transcriber.transcribe_openvino()
        - reason: Begruendung der Empfehlung
    """
    gpu = get_gpu_info()
    cpu = get_cpu_info()

    # CUDA-Pfad: Modell anhand VRAM waehlen (WhisperX-Inferenz laeuft direkt auf
    # der GPU ueber CTranslate2)
    if gpu is not None and gpu["backend"] == "cuda":
        vram = gpu["vram_gb"]
        if vram >= 8:
            return ("large-v3", "cuda",
                    f"GPU {gpu['name']} mit {vram:.1f} GB VRAM — large-v3 optimal")
        elif vram >= 5:
            return ("medium", "cuda",
                    f"GPU {gpu['name']} mit {vram:.1f} GB VRAM — medium empfohlen")
        else:
            return ("base", "cuda",
                    f"GPU {gpu['name']} mit {vram:.1f} GB VRAM — base empfohlen")

    # Keine CUDA-GPU: Intel Arc GPU via OpenVINO pruefen, bevor auf reine
    # CPU-Transkription (CTranslate2) zurueckgefallen wird - im Benchmark auf
    # einem Core Ultra 7 258V (Arc 140V) rund 6x schneller als CPU-int8.
    ov_devices = get_openvino_devices() or []
    if "GPU" in ov_devices and _bundled_openvino_model_exists("medium"):
        # Alignment/Diarization (plain PyTorch) koennen dieselbe Arc GPU nur
        # nutzen, wenn torch selbst mit einem XPU-faehigen Build installiert
        # ist (siehe pyproject.toml's "xpu"-Extra) - der Standard-Install
        # (CUDA-Build) hat torch.xpu.is_available() == False, dann laeuft nur
        # die Whisper-Transkription selbst (via OpenVINO) auf der Arc GPU.
        if gpu is not None and gpu["backend"] == "xpu":
            return ("openvino:GPU:medium", "xpu",
                    f"Intel Arc GPU erkannt — Transkription (OpenVINO) und "
                    "Alignment/Diarization (torch XPU) laufen beide auf der GPU")
        return ("openvino:GPU:medium", "cpu",
                "Intel Arc GPU erkannt (OpenVINO) — deutlich schneller als CPU-Transkription "
                "(CTranslate2 unterstuetzt keine Intel-GPUs, siehe CLAUDE.md). "
                "Alignment/Diarization laufen auf der CPU (torch ohne XPU-Build installiert)")

    # Kein CUDA (auch Apple Silicon/MPS): Modellgroesse anhand RAM/Kerne
    # waehlen, da die eigentliche Whisper-Transkription ueber CTranslate2
    # laeuft, das kein MPS unterstuetzt und daher immer auf der CPU rechnet -
    # MPS beschleunigt hier nur Alignment und Diarization (siehe transcriber.py).
    ram = cpu["ram_gb"]
    cores = cpu["cores"]

    if ram >= 16 and cores >= 8:
        model, size_reason = "medium", f"{cpu['name']} mit {ram:.0f} GB RAM, {cores} Kerne — medium moeglich"
    elif ram >= 8:
        model, size_reason = "base", f"{ram:.0f} GB RAM — base empfohlen"
    else:
        model, size_reason = "base", "begrenzte Ressourcen — base empfohlen"

    if gpu is not None and gpu["backend"] == "mps":
        return (model, "mps",
                f"Apple Silicon (MPS) — {size_reason}. Whisper-Transkription "
                "laeuft CPU-basiert (CTranslate2 unterstuetzt kein MPS), "
                "Alignment/Diarization nutzen die Apple-GPU")

    return (model, "cpu", f"Keine GPU — {size_reason}")


def get_hardware_summary() -> str:
    """Gibt eine lesbare Zusammenfassung der Hardware zurueck."""
    cpu = get_cpu_info()
    gpu = get_gpu_info()

    lines = [f"CPU: {cpu['name']} ({cpu['cores']} Kerne, {cpu['ram_gb']:.1f} GB RAM)"]
    if gpu and gpu["backend"] == "cuda":
        lines.append(f"GPU: {gpu['name']} ({gpu['vram_gb']:.1f} GB VRAM, CUDA)")
    elif gpu and gpu["backend"] == "mps":
        lines.append(f"GPU: Apple Silicon (MPS, {gpu['vram_gb']:.1f} GB unified memory) "
                      "— beschleunigt Alignment/Diarization, Whisper-Transkription läuft auf der CPU")
    else:
        lines.append("GPU: Keine CUDA- oder MPS-fähige GPU erkannt")

    # Zusaetzlich zu CUDA/MPS: Intel Arc GPU/NPU via OpenVINO (siehe
    # get_openvino_devices()) - unabhaengig von obigem gpu-Wert, da
    # get_gpu_info() nur CUDA/MPS kennt und Arc-Hardware separat erkannt wird.
    ov_devices = get_openvino_devices()
    if ov_devices:
        intel_devices = [d for d in ov_devices if d != "CPU"]
        if intel_devices:
            bundled = _bundled_openvino_model_exists("medium")
            status = "" if bundled else " (Modell noch nicht exportiert - 'python download_models.py' ausfuehren)"
            lines.append(f"Intel OpenVINO: {', '.join(intel_devices)} erkannt{status}")

    return "\n".join(lines)
