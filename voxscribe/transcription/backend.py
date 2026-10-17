"""The Backend interface: anything that turns audio into text segments."""

from dataclasses import dataclass
from typing import Callable, Protocol

from voxscribe.transcription.audio import AudioSource


class FallbackRequested(Exception):
    """A Backend raises this when its Model ID failed in a way that warrants
    retrying with the best local model (e.g. a Provider that is unreachable).
    Any other exception propagates to the caller unchanged."""


@dataclass
class TranscribeRequest:
    audio_path: str
    audio: AudioSource
    language: str | None  # ISO-639-1 code, or None for auto-detect
    model_id: str  # the full Model ID, prefix included
    progress: Callable[[float, str], None]  # overall pipeline progress, 0-1, plus a message
    batch_size: int = 16
    device: str = "cpu"  # resolved torch device: cuda, xpu, mps or cpu
    api_key: str | None = None  # ad-hoc Provider credentials, see transcribe()
    api_base_url: str | None = None


class Backend(Protocol):
    name: str
    # True if segments already carry word timestamps (the alignment step is skipped).
    skips_alignment: bool

    def describe(self, model_id: str, device: str) -> str:
        """The "Device: ..." line printed before the run."""
        ...

    def transcribe(self, request: TranscribeRequest) -> dict:
        """Returns {"segments": [{"start", "end", "text"}], "language": <ISO code>};
        alignment and diarization run on top of this. The Backend owns its part
        of the progress range (0 to 0.40) and reports through request.progress."""
        ...
