"""Model ID helpers: the prefix convention that selects a Backend, and defaults."""

import sys


REMOTE_MODEL_PREFIX = "server:"


def is_remote_model(model_size: str) -> bool:
    """Checks whether this refers to a server-hosted (provider) model."""
    return model_size.startswith(REMOTE_MODEL_PREFIX)


def remote_model_id(provider_id: str) -> str:
    """Builds the model_size string for a configured provider, e.g. 'server:my-provider'."""
    return f"{REMOTE_MODEL_PREFIX}{provider_id}"


# Benchmark only (#57): "server:<provider-id>#<model>" runs a Provider with
# another model than the one it stores, so one Provider (e.g. OpenRouter)
# can compare several models. The app never builds such a Model ID.
REMOTE_MODEL_OVERRIDE_SEPARATOR = "#"


def split_remote_model(model_id: str) -> tuple[str, str | None]:
    """'server:openrouter#qwen/qwen3-asr-1.7b' -> ('server:openrouter',
    'qwen/qwen3-asr-1.7b'); anything else -> (model_id, None)."""
    if is_remote_model(model_id) and REMOTE_MODEL_OVERRIDE_SEPARATOR in model_id:
        base, model = model_id.split(REMOTE_MODEL_OVERRIDE_SEPARATOR, 1)
        return base, (model or None)
    return model_id, None


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


# Candidate (#57/#58): reachable by Model ID from the CLI and the benchmark,
# not offered in the GUI.
CANARY_MODEL_PREFIX = "canary:"


CANARY_MODEL = "canary:1b-v2"


def is_canary_model(model_size: str) -> bool:
    """Checks whether transcription should run via NVIDIA Canary (onnx-asr)."""
    return model_size.startswith(CANARY_MODEL_PREFIX)


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
    - macOS: NVIDIA Canary-1B-v2 (onnx-asr, CPU, int8). In the Mac study
      (#58) it was the most accurate local model (5.5 % WER vs. 6.7 % for
      Parakeet, 11.5 % for SpeechAnalyzer) and ~2x faster than Parakeet,
      also on German-then-English meetings, since auto-detect picks the
      language per speech segment (backends/canary.py). Parakeet (#60) and
      Apple's SpeechAnalyzer stay selectable; SpeechAnalyzer drops whole
      passages when two voices alternate, as in a Call mix.
    - Windows/Linux: whatever hardware_detect.recommend_model() picks - a
      dedicated NVIDIA GPU (CUDA) recommends large-v3/medium/base depending
      on VRAM, an Intel Arc GPU (no CUDA present) recommends the OpenVINO
      model, otherwise it falls back to a RAM-sized CPU model.
    """
    if sys.platform == "darwin":
        return CANARY_MODEL
    from voxscribe.hardware_detect import recommend_model
    model_size, _device, _reason = recommend_model()
    return model_size
