"""Built-in Backends, registered on import.

Each Backend lives in its own module and is imported on first use, so the
heavy dependencies (torch, onnx-asr, optimum, ...) are only loaded for the
Backend actually selected. Importing this package stays cheap.
"""

from voxscribe.transcription.model_ids import (
    APPLE_MODEL_PREFIX, OPENVINO_MODEL_PREFIX, PARAKEET_MODEL_PREFIX, REMOTE_MODEL_PREFIX,
)
from voxscribe.transcription.registry import default_registry


def _parakeet():
    from voxscribe.transcription.backends.parakeet import ParakeetBackend
    return ParakeetBackend()


def _openvino():
    from voxscribe.transcription.backends.openvino import OpenVinoBackend
    return OpenVinoBackend()


def _apple():
    from voxscribe.transcription.backends.apple import AppleBackend
    return AppleBackend()


def _remote():
    from voxscribe.transcription.backends.remote import RemoteBackend
    return RemoteBackend()


default_registry.register(REMOTE_MODEL_PREFIX, _remote)
default_registry.register(APPLE_MODEL_PREFIX, _apple)
default_registry.register(OPENVINO_MODEL_PREFIX, _openvino)
default_registry.register(PARAKEET_MODEL_PREFIX, _parakeet)
