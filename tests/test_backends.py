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


# --- remote Provider ---------------------------------------------------------

import pytest  # noqa: E402

from voxscribe.transcription.backend import FallbackRequested  # noqa: E402

PROVIDER = {"id": "p1", "name": "My Provider", "base_url": "https://x/v1",
            "api_key": "secret", "model": "whisper-large"}


def _remote_calls(monkeypatch, provider=PROVIDER, error=None):
    from voxscribe.transcription.backends import remote

    calls = []
    monkeypatch.setattr("voxscribe.providers.get_provider", lambda pid: provider)

    def fake(audio, language, model_name, api_key, base_url, on_progress=None, **kw):
        calls.append(dict(samples=len(audio), language=language, model=model_name,
                          key=api_key, url=base_url))
        if error:
            raise error
        on_progress(0.5)
        return {"segments": [], "language": "de"}

    monkeypatch.setattr(remote, "transcribe_remote", fake)
    return calls


def test_remote_backend_describes_like_local_whisper():
    backend = default_registry.resolve("server:p1")

    assert backend.skips_alignment is False
    assert backend.describe("server:p1", "cpu") == "Device: cpu (int8)"
    assert backend.describe("server:p1", "cuda") == "Device: cuda (float16)"
    assert backend.describe("server:p1", "mps") == (
        "Device: Whisper=cpu (int8), alignment/diarization=mps (Apple MPS)")
    assert backend.describe("server:p1", "xpu") == (
        "Device: Whisper=cpu (int8), alignment/diarization=xpu (Intel Arc GPU)")


def test_remote_backend_uses_the_stored_provider(tmp_path, monkeypatch):
    calls = _remote_calls(monkeypatch)
    request, events = _request(tmp_path, "server:p1", language="de")

    default_registry.resolve(request.model_id).transcribe(request)

    assert calls == [dict(samples=16000, language="de", model="whisper-large",
                          key="secret", url="https://x/v1")]
    assert events == [(0.0, "Loading audio..."), (0.1, "Sending audio to server (My Provider)..."),
                      (0.25, "Sending audio to server... 50%")]


def test_remote_backend_explicit_credentials_override_the_provider(tmp_path, monkeypatch):
    calls = _remote_calls(monkeypatch)
    request, _ = _request(tmp_path, "server:p1", api_key="k2", api_base_url="https://other")

    default_registry.resolve(request.model_id).transcribe(request)

    assert (calls[0]["key"], calls[0]["url"], calls[0]["model"]) == ("k2", "https://other", "whisper-large")


def test_remote_backend_unknown_provider_is_a_plain_error(tmp_path, monkeypatch):
    _remote_calls(monkeypatch, provider=None)
    request, _ = _request(tmp_path, "server:nope")

    with pytest.raises(ValueError, match="Unknown provider 'nope'"):
        default_registry.resolve(request.model_id).transcribe(request)


def test_remote_backend_request_failure_asks_for_the_local_fallback(tmp_path, monkeypatch):
    _remote_calls(monkeypatch, error=RuntimeError("HTTP 502"))
    request, _ = _request(tmp_path, "server:p1")

    with pytest.raises(FallbackRequested, match="HTTP 502"):
        default_registry.resolve(request.model_id).transcribe(request)


def test_transcribe_remote_shifts_chunks_and_tags_the_language(monkeypatch):
    from voxscribe.transcription.backends import remote

    def fake_chunk(chunk, language, model_name, api_key, base_url, timeout=0):
        return [{"start": 0.0, "end": 1.0, "text": "hello"}], None, True

    monkeypatch.setattr(remote, "_transcribe_chunk_with_backoff", fake_chunk)
    audio = np.zeros(40 * 16000, dtype=np.float32)

    result = remote.transcribe_remote(audio, "de", "m", api_key="k", base_url="u")

    # The first 30s chunk is cut at the quietest 100ms frame of its last 4s
    # (all-silent audio: the first one); the rest starts from there.
    assert [(s["start"], s["language"]) for s in result["segments"]] == [
        (0.0, "de"), (pytest.approx(26.05), "de")]
    assert result["language"] == "de"


def test_transcribe_remote_requires_an_api_key():
    from voxscribe.transcription.backends import remote

    with pytest.raises(ValueError, match="No API key"):
        remote.transcribe_remote(np.zeros(10, dtype=np.float32), "de", "m", api_key="", base_url="u")


# --- local WhisperX ----------------------------------------------------------

class _FakeWhisperModel:
    def __init__(self):
        self.batch_size = None

    def transcribe(self, audio, batch_size, progress_callback):
        self.batch_size = batch_size
        progress_callback(50.0)
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "x"}], "language": "de"}


def _patch_whisperx(monkeypatch, bundled):
    import whisperx
    from voxscribe.transcription.backends import whisperx_local

    loaded = {}
    model = _FakeWhisperModel()

    def fake_load_model(arch, device, **kw):
        loaded.update(arch=arch, device=device, **kw)
        return model

    # raising=False: whisperx loads its attributes lazily, don't trigger that import
    monkeypatch.setattr(whisperx, "load_model", fake_load_model, raising=False)
    monkeypatch.setattr(whisperx_local, "get_bundled_models_dir", lambda: bundled)
    return loaded, model


def test_unprefixed_model_ids_use_local_whisperx():
    backend = default_registry.resolve("large-v3")

    assert backend.name == "WhisperX"
    assert backend.skips_alignment is False
    assert backend.describe("large-v3", "cuda") == "Device: cuda (float16)"


def test_local_backend_loads_the_named_model_and_maps_progress(tmp_path, monkeypatch):
    loaded, model = _patch_whisperx(monkeypatch, bundled=None)
    request, events = _request(tmp_path, "medium", language="de", batch_size=4, device="mps")

    result = default_registry.resolve("medium").transcribe(request)

    assert loaded == dict(arch="medium", device="cpu", compute_type="int8", language="de",
                          download_root=None, local_files_only=False)
    assert model.batch_size == 4
    assert result["language"] == "de"
    assert events == [(0.0, "Loading Whisper model..."), (0.1, "Loading audio..."),
                      (0.15, "Transcribing..."), (0.275, "Transcribing... 50%")]


def test_local_backend_prefers_a_bundled_model_directory(tmp_path, monkeypatch):
    (tmp_path / "bundled" / "whisper" / "medium").mkdir(parents=True)
    loaded, _ = _patch_whisperx(monkeypatch, bundled=str(tmp_path / "bundled"))
    request, _ = _request(tmp_path, "medium")

    default_registry.resolve("medium").transcribe(request)

    assert loaded["arch"] == str(tmp_path / "bundled" / "whisper" / "medium")
    assert loaded["local_files_only"] is True
