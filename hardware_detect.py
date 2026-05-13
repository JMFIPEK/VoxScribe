"""Hardware-Erkennung und automatische Modellauswahl fuer WhisperX."""

import os
import platform

import torch


def get_gpu_info() -> dict | None:
    """Ermittelt GPU-Informationen (Name, VRAM) falls CUDA verfuegbar."""
    if not torch.cuda.is_available():
        return None

    try:
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)  # GB
        return {"name": gpu_name, "vram_gb": vram_total}
    except Exception:
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


def recommend_model() -> tuple[str, str, str]:
    """Empfiehlt das optimale Modell basierend auf der verfuegbaren Hardware.

    Returns:
        Tuple (model_name, device, reason)
        - model_name: "large-v3", "large-v2", "medium", oder "base"
        - device: "cuda" oder "cpu"
        - reason: Begruendung der Empfehlung
    """
    gpu = get_gpu_info()
    cpu = get_cpu_info()

    # GPU-Pfad: Modell anhand VRAM waehlen
    if gpu is not None:
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

    # CPU-Pfad: Modell anhand RAM und Kerne waehlen
    ram = cpu["ram_gb"]
    cores = cpu["cores"]

    if ram >= 16 and cores >= 8:
        return ("medium", "cpu",
                f"Keine GPU — {cpu['name']} mit {ram:.0f} GB RAM, {cores} Kerne — medium moeglich")
    elif ram >= 8:
        return ("base", "cpu",
                f"Keine GPU — {ram:.0f} GB RAM — base empfohlen")
    else:
        return ("base", "cpu",
                f"Keine GPU — begrenzte Ressourcen — base empfohlen")


def get_hardware_summary() -> str:
    """Gibt eine lesbare Zusammenfassung der Hardware zurueck."""
    cpu = get_cpu_info()
    gpu = get_gpu_info()

    lines = [f"CPU: {cpu['name']} ({cpu['cores']} Kerne, {cpu['ram_gb']:.1f} GB RAM)"]
    if gpu:
        lines.append(f"GPU: {gpu['name']} ({gpu['vram_gb']:.1f} GB VRAM)")
    else:
        lines.append("GPU: Keine CUDA-fähige GPU erkannt")

    return "\n".join(lines)
