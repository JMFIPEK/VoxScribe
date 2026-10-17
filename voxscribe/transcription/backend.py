"""The Backend interface: anything that turns audio into text segments."""

from typing import Callable, Protocol

import numpy as np


class Backend(Protocol):
    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None,
        model_id: str,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict:
        """Transcribes 16kHz mono float32 `audio`.

        `model_id` is the full Model ID (prefix included). `language` is an
        ISO-639-1 code, or None for auto-detect. `on_progress` receives a
        fraction from 0 to 1. Returns {"segments": [{"start", "end", "text"}],
        "language": <ISO code>}; alignment and diarization run on top of this.
        """
        ...
