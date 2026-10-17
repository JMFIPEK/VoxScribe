"""Device handling shared by the pipeline and the Backends.

Importing this module is cheap; torch is only imported inside functions.
"""


def whisper_device_for(device: str) -> str:
    """CTranslate2 (WhisperX's inference backend) only knows "cuda"/"cpu" -
    neither MPS nor XPU. Alignment (wav2vec2) and diarization (pyannote) are
    plain PyTorch and can use MPS/XPU instead, hence two separate devices."""
    return device if device == "cuda" else "cpu"


def compute_type_for(device: str) -> str:
    return "float16" if whisper_device_for(device) == "cuda" else "int8"


def describe_whisper_device(device: str) -> str:
    """The "Device: ..." line for the Whisper-based paths (local and Provider)."""
    whisper_device, compute_type = whisper_device_for(device), compute_type_for(device)
    if device == "mps":
        return f"Device: Whisper={whisper_device} ({compute_type}), alignment/diarization={device} (Apple MPS)"
    if device == "xpu":
        return f"Device: Whisper={whisper_device} ({compute_type}), alignment/diarization={device} (Intel Arc GPU)"
    return f"Device: {device} ({compute_type})"


def resolve_device(device: str | None) -> str:
    """The torch device to use: `device` if given, else cuda > xpu > mps > cpu."""
    if device is not None:
        return device
    import torch

    # Priority dGPU > iGPU > CPU: CUDA (dedicated NVIDIA GPU) first, then
    # Intel XPU (Arc GPU - only available if torch was installed with an
    # XPU-capable build, see gpu_setup.py; the standard CUDA install has
    # torch.xpu.is_available() == False and falls through here
    # transparently), then Apple MPS.
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_torch_memory(torch_device: str) -> None:
    """Garbage-collects and empties the device's allocator cache (after a model is dropped)."""
    import gc

    gc.collect()
    if torch_device == "cuda":
        import torch
        torch.cuda.empty_cache()
    elif torch_device == "xpu":
        import torch
        torch.xpu.empty_cache()
    elif torch_device == "mps":
        import torch
        torch.mps.empty_cache()
