"""Hardware detection and automatic model selection for WhisperX."""

import os
import platform

import torch


def get_gpu_info() -> dict | None:
    """Detects GPU info (name, VRAM, backend) if CUDA, Intel XPU, or MPS is
    available. Priority dGPU > iGPU > integrated alternative: CUDA (dedicated
    NVIDIA GPU) is checked first, then Intel XPU (Arc iGPU/dGPU - only
    available if torch was installed with an XPU-capable build; the more
    common CUDA-standard install has torch.xpu.is_available() == False),
    then Apple MPS."""
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
        # Apple Silicon: no dedicated VRAM query available (unified memory is
        # shared with the CPU) - use total RAM as an approximation.
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            return {"name": platform.processor() or "Apple GPU (MPS)",
                    "vram_gb": ram_gb, "backend": "mps"}
        except Exception:
            return {"name": "Apple GPU (MPS)", "vram_gb": 0.0, "backend": "mps"}

    return None


def get_cpu_info() -> dict:
    """Detects CPU info (name, core count, RAM)."""
    import psutil

    cpu_count = os.cpu_count() or 4
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

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
    """Returns the inference devices OpenVINO detects (e.g. ['CPU','GPU','NPU']
    for an Intel Arc GPU + NPU). Returns None if the optional
    `optimum-intel[openvino]` package isn't installed (Windows-only in
    pyproject.toml, see transcriber.transcribe_openvino())."""
    try:
        from openvino import Core
    except ImportError:
        return None
    try:
        return Core().available_devices
    except Exception:
        return None


def _bundled_openvino_model_exists(whisper_size: str = "medium") -> bool:
    """Checks whether download_models.py::export_openvino_model() has already
    run (bundled_models/openvino/whisper-<size>/) - without this model,
    transcribe_openvino() has nothing to load, even if an Arc GPU/NPU is
    detected."""
    base = os.path.dirname(os.path.abspath(__file__))
    marker = os.path.join(base, "bundled_models", "openvino", f"whisper-{whisper_size}",
                           "openvino_encoder_model.bin")
    return os.path.isfile(marker)


def recommend_model() -> tuple[str, str, str]:
    """Recommends the best model based on the available hardware.

    Returns:
        Tuple (model_name, device, reason)
        - model_name: "large-v3", "large-v2", "medium", "base", or
          "openvino:GPU:medium" (Intel Arc GPU via OpenVINO)
        - device: "cuda", "xpu", "mps", or "cpu" - device for alignment/
          diarization (plain PyTorch). For "openvino:..." models, the actual
          Whisper transcription always runs via OpenVINO on the device
          encoded in model_name, independent of this value - see
          transcriber.transcribe_openvino()
        - reason: rationale for the recommendation
    """
    gpu = get_gpu_info()
    cpu = get_cpu_info()

    # CUDA path: pick model size by VRAM (WhisperX inference runs directly on
    # the GPU via CTranslate2)
    if gpu is not None and gpu["backend"] == "cuda":
        vram = gpu["vram_gb"]
        if vram >= 8:
            return ("large-v3", "cuda",
                    f"GPU {gpu['name']} with {vram:.1f} GB VRAM — large-v3 is optimal")
        elif vram >= 5:
            return ("medium", "cuda",
                    f"GPU {gpu['name']} with {vram:.1f} GB VRAM — medium recommended")
        else:
            return ("base", "cuda",
                    f"GPU {gpu['name']} with {vram:.1f} GB VRAM — base recommended")

    # No CUDA GPU: check for an Intel Arc GPU via OpenVINO before falling
    # back to plain CPU transcription (CTranslate2) - about 6x faster than
    # CPU int8 in benchmarks on a Core Ultra 7 258V (Arc 140V).
    ov_devices = get_openvino_devices() or []
    if "GPU" in ov_devices and _bundled_openvino_model_exists("medium"):
        # Alignment/diarization (plain PyTorch) can only use the same Arc GPU
        # if torch itself was installed with an XPU-capable build - the
        # standard install (CUDA build) has torch.xpu.is_available() ==
        # False, in which case only the Whisper transcription itself (via
        # OpenVINO) runs on the Arc GPU.
        if gpu is not None and gpu["backend"] == "xpu":
            return ("openvino:GPU:medium", "xpu",
                    "Intel Arc GPU detected — transcription (OpenVINO) and "
                    "alignment/diarization (torch XPU) both run on the GPU")
        return ("openvino:GPU:medium", "cpu",
                "Intel Arc GPU detected (OpenVINO) — much faster than CPU transcription "
                "(CTranslate2 has no Intel GPU support). "
                "Alignment/diarization run on the CPU (torch installed without the XPU build)")

    # No CUDA (including Apple Silicon/MPS): pick model size by RAM/cores,
    # since the actual Whisper transcription runs via CTranslate2, which has
    # no MPS support and therefore always runs on the CPU - MPS here only
    # accelerates alignment and diarization (see transcriber.py).
    ram = cpu["ram_gb"]
    cores = cpu["cores"]

    if ram >= 16 and cores >= 8:
        model, size_reason = "medium", f"{cpu['name']} with {ram:.0f} GB RAM, {cores} cores — medium possible"
    elif ram >= 8:
        model, size_reason = "base", f"{ram:.0f} GB RAM — base recommended"
    else:
        model, size_reason = "base", "limited resources — base recommended"

    if gpu is not None and gpu["backend"] == "mps":
        return (model, "mps",
                f"Apple Silicon (MPS) — {size_reason}. Whisper transcription "
                "runs on the CPU (CTranslate2 has no MPS support), "
                "alignment/diarization use the Apple GPU")

    return (model, "cpu", f"No GPU — {size_reason}")


def get_hardware_summary() -> str:
    """Returns a readable summary of the hardware."""
    cpu = get_cpu_info()
    gpu = get_gpu_info()

    lines = [f"CPU: {cpu['name']} ({cpu['cores']} cores, {cpu['ram_gb']:.1f} GB RAM)"]
    if gpu and gpu["backend"] == "cuda":
        lines.append(f"GPU: {gpu['name']} ({gpu['vram_gb']:.1f} GB VRAM, CUDA)")
    elif gpu and gpu["backend"] == "mps":
        lines.append(f"GPU: Apple Silicon (MPS, {gpu['vram_gb']:.1f} GB unified memory) "
                      "— accelerates alignment/diarization, Whisper transcription runs on the CPU")
    else:
        lines.append("GPU: No CUDA- or MPS-capable GPU detected")

    # In addition to CUDA/MPS: Intel Arc GPU/NPU via OpenVINO (see
    # get_openvino_devices()) - independent of the gpu value above, since
    # get_gpu_info() only knows CUDA/MPS and Arc hardware is detected separately.
    ov_devices = get_openvino_devices()
    if ov_devices:
        intel_devices = [d for d in ov_devices if d != "CPU"]
        if intel_devices:
            bundled = _bundled_openvino_model_exists("medium")
            status = "" if bundled else " (model not exported yet - run 'python download_models.py')"
            lines.append(f"Intel OpenVINO: {', '.join(intel_devices)} detected{status}")

    return "\n".join(lines)
