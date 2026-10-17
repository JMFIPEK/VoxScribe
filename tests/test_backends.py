"""Each built-in Backend, up to the hardware/network boundary (no real model runs)."""

import numpy as np
import soundfile as sf

from voxscribe.transcription import backends  # noqa: F401  (registers the built-ins)
from voxscribe.transcription.audio import AudioSource
from voxscribe.transcription.backend import TranscribeRequest
from voxscribe.transcription.registry import default_registry


def _request(tmp_path, model_id, language=None, **kw):
    path = str(tmp_path / "a.wav")
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    events = []
    request = TranscribeRequest(
        audio_path=path, audio=AudioSource(path), language=language, model_id=model_id,
        progress=lambda pct, msg: events.append((round(pct, 4), msg)), **kw)
    return request, events


def test_parakeet_backend_is_registered_and_describes_itself():
    backend = default_registry.resolve("parakeet:tdt-0.6b-v3")

    assert backend.skips_alignment is False
    assert backend.describe("parakeet:tdt-0.6b-v3", "mps") == (
        "Device: transcription=Parakeet (ONNX Runtime, CPU), alignment/diarization=mps")


def test_parakeet_backend_maps_progress_into_its_slice(tmp_path, monkeypatch):
    from voxscribe.transcription.backends import parakeet

    def fake(audio, language, model_size, on_progress=None):
        assert len(audio) == 16000 and (language, model_size) == ("de", "parakeet:tdt-0.6b-v3")
        on_progress(0.5)
        return {"segments": [], "language": "de"}

    monkeypatch.setattr(parakeet, "transcribe_parakeet", fake)
    request, events = _request(tmp_path, "parakeet:tdt-0.6b-v3", language="de")

    result = default_registry.resolve(request.model_id).transcribe(request)

    assert result == {"segments": [], "language": "de"}
    assert events == [(0.0, "Loading audio..."), (0.05, "Loading Parakeet model..."),
                      (0.225, "Transcribing... 50%")]
