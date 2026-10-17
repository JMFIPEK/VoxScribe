"""Built-in Backends, registered on import.

Each Backend lives in its own module and is imported on first use, so the
heavy dependencies (torch, onnx-asr, optimum, ...) are only loaded for the
Backend actually selected. Importing this package stays cheap.
"""

from voxscribe.transcription.model_ids import PARAKEET_MODEL_PREFIX
from voxscribe.transcription.registry import default_registry


def _parakeet():
    from voxscribe.transcription.backends.parakeet import ParakeetBackend
    return ParakeetBackend()


default_registry.register(PARAKEET_MODEL_PREFIX, _parakeet)
