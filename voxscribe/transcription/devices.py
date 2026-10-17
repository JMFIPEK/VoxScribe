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
