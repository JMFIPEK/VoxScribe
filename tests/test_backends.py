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


def test_openvino_backend_describes_itself():
    backend = default_registry.resolve("openvino:GPU:medium")

    assert backend.skips_alignment is False
    assert backend.describe("openvino:GPU:medium", "xpu") == (
        "Device: transcription=Intel OpenVINO (GPU), alignment/diarization=xpu")


def test_openvino_backend_passes_device_and_size_and_maps_progress(tmp_path, monkeypatch):
    from voxscribe.transcription.backends import openvino

    seen = {}

    def fake(audio_path, language, ov_device, whisper_size, on_progress=None, audio=None):
        seen.update(path=audio_path, language=language, device=ov_device, size=whisper_size,
                    samples=len(audio))
        on_progress(50.0)
        return {"segments": [], "language": "en"}

    monkeypatch.setattr(openvino, "transcribe_openvino", fake)
    request, events = _request(tmp_path, "openvino:GPU:medium", language="en")

    default_registry.resolve(request.model_id).transcribe(request)

    assert seen == {"path": request.audio_path, "language": "en", "device": "GPU",
                    "size": "medium", "samples": 16000}
    assert events == [(0.0, "Loading OpenVINO model..."), (0.2, "Transcribing... 50%")]


def test_apple_backend_skips_alignment_and_describes_itself():
    backend = default_registry.resolve("apple:speechanalyzer")

    assert backend.skips_alignment is True
    assert backend.name == "Apple SpeechAnalyzer"
    assert backend.describe("apple:speechanalyzer", "mps") == (
        "Device: transcription=Apple Neural Engine (SpeechAnalyzer), diarization=mps")


def test_apple_backend_uses_the_file_path_and_maps_progress(tmp_path, monkeypatch):
    from voxscribe.transcription.backends import apple

    seen = {}

    def fake(audio_path, language=None, on_progress=None):
        seen.update(path=audio_path, language=language)
        on_progress(0.5)
        return {"segments": [], "word_segments": [], "language": "de"}

    monkeypatch.setattr(apple, "transcribe_apple", fake)
    request, events = _request(tmp_path, "apple:speechanalyzer")

    default_registry.resolve(request.model_id).transcribe(request)

    assert seen == {"path": request.audio_path, "language": None}
    assert events == [(0.0, "SpeechAnalyzer transcribing..."), (0.2, "Transcribing... 50%")]


def test_apple_locale_identifier():
    from voxscribe.transcription.backends.apple import _apple_locale_identifier

    assert _apple_locale_identifier("de") == "de-DE"
    assert _apple_locale_identifier("DE") == "de-DE"
    assert _apple_locale_identifier(None) == "en-US"
    assert _apple_locale_identifier("xx") == "en-US"
