"""Model ID helpers: the prefix convention that selects a Backend, and defaults."""

import sys


REMOTE_MODEL_PREFIX = "server:"


def is_remote_model(model_size: str) -> bool:
    """Checks whether this refers to a server-hosted (provider) model."""
    return model_size.startswith(REMOTE_MODEL_PREFIX)


def remote_model_id(provider_id: str) -> str:
    """Builds the model_size string for a configured provider, e.g. 'server:my-provider'."""
    return f"{REMOTE_MODEL_PREFIX}{provider_id}"


OPENVINO_MODEL_PREFIX = "openvino:"


def is_openvino_model(model_size: str) -> bool:
    """Checks whether transcription should run via the Intel OpenVINO backend
    (Arc GPU or NPU) instead of CTranslate2 (WhisperX' default backend, which
    has no Intel GPU/NPU support at all - see CLAUDE.md)."""
    return model_size.startswith(OPENVINO_MODEL_PREFIX)


def openvino_model_id(ov_device: str, whisper_size: str) -> str:
    """Builds the model_size string for transcribe(), e.g. 'openvino:GPU:medium'."""
    return f"{OPENVINO_MODEL_PREFIX}{ov_device}:{whisper_size}"


def _parse_openvino_model(model_size: str) -> tuple[str, str]:
    """'openvino:GPU:medium' -> ('GPU', 'medium')."""
    rest = model_size[len(OPENVINO_MODEL_PREFIX):]
    ov_device, whisper_size = rest.split(":", 1)
    return ov_device, whisper_size


PARAKEET_MODEL_PREFIX = "parakeet:"


PARAKEET_MODEL = "parakeet:tdt-0.6b-v3"


def is_parakeet_model(model_size: str) -> bool:
    """Checks whether transcription should run via NVIDIA Parakeet (onnx-asr)."""
    return model_size.startswith(PARAKEET_MODEL_PREFIX)


APPLE_MODEL_PREFIX = "apple:"


APPLE_SPEECHANALYZER_MODEL = "apple:speechanalyzer"


def is_apple_model(model_size: str) -> bool:
    """Checks whether the Apple SpeechAnalyzer engine (macOS only) should be used."""
    return model_size.startswith(APPLE_MODEL_PREFIX)


def default_model_size() -> str:
    """Platform-wide default for model_size when the caller has no explicit
    preference: the user's configured default provider (see providers.py),
    if one is set up - otherwise falls back to the best local model for this
    platform/hardware (default_local_model_size()). transcribe() also falls
    back to a local model automatically if a chosen provider call fails
    (network, server down, ...), see there.
    """
    from voxscribe.providers import get_default_provider
    provider = get_default_provider()
    if provider:
        return remote_model_id(provider["id"])
    return default_local_model_size()


def default_local_model_size() -> str:
    """Recommended *local* model - used as a fallback whenever a remote
    provider call fails or isn't configured, and anywhere a local model is
    explicitly wanted. Platform-/hardware-dependent:
    - macOS: always Apple's SpeechAnalyzer (Neural Engine) - the only local
      transcription engine there, no Whisper model download/inference at
      all, see CLAUDE.md ("Apple Silicon (MPS) acceleration is partial by
      design").
    - Windows/Linux: whatever hardware_detect.recommend_model() picks - a
      dedicated NVIDIA GPU (CUDA) recommends large-v3/medium/base depending
      on VRAM, an Intel Arc GPU (no CUDA present) recommends the OpenVINO
      model, otherwise it falls back to a RAM-sized CPU model.
    """
    if sys.platform == "darwin":
        return APPLE_SPEECHANALYZER_MODEL
    from voxscribe.hardware_detect import recommend_model
    model_size, _device, _reason = recommend_model()
    return model_size
