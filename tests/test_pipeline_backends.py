"""transcribe() runs whatever Backend the registry resolves for the Model ID."""

import numpy as np
import pytest
import soundfile as sf

from voxscribe import transcriber as t
from voxscribe.transcription import pipeline
from voxscribe.transcription.backend import FallbackRequested
from voxscribe.transcription.registry import BackendRegistry


class FakeBackend:
    name = "Fake"
    skips_alignment = True  # so the test needs no alignment model

    def __init__(self, language="de", error=None):
        self.requests = []
        self.language = language
        self.error = error

    def describe(self, model_id, device):
        return f"Device: fake for {model_id} on {device}"

    def transcribe(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        request.progress(0.1, "working")
        audio = request.audio.load()
        return {"segments": [{"start": 0.0, "end": len(audio) / 16000, "text": "hi"}],
                "word_segments": [], "language": self.language}


@pytest.fixture
def wav(tmp_path):
    path = str(tmp_path / "one_second.wav")
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    return path


@pytest.fixture
def registry(monkeypatch):
    reg = BackendRegistry()
    monkeypatch.setattr(pipeline, "default_registry", reg)
    return reg


def test_transcribe_uses_the_registered_backend(registry, wav):
    backend = FakeBackend()
    registry.register("fake:", lambda: backend)
    progress = []

    result = t.transcribe(wav, language="de", model_size="fake:1", diarize=False, device="cpu",
                          batch_size=4, on_progress=lambda pct, msg: progress.append(pct))

    (request,) = backend.requests
    assert (request.audio_path, request.language, request.model_id) == (wav, "de", "fake:1")
    assert (request.batch_size, request.device) == (4, "cpu")
    assert result["segments"][0]["text"] == "hi"
    assert result["language"] == "de"
    assert result["duration"] == 1.0
    assert progress[0] == 0.1 and progress[-1] == 1.0
    assert 0.1 in progress and 0.4 in progress and 0.65 in progress


def test_provider_failure_falls_back_to_the_local_model(registry, wav, monkeypatch):
    failing = FakeBackend(error=FallbackRequested("server down"))
    local = FakeBackend(language="en")
    registry.register("fake:", lambda: failing)
    registry.register("local:", lambda: local)
    monkeypatch.setattr(pipeline, "default_local_model_size", lambda: "local:best")

    result = t.transcribe(wav, language=None, model_size="fake:1", diarize=False, device="cpu")

    assert result["language"] == "en"
    assert [r.model_id for r in failing.requests] == ["fake:1"]
    assert [r.model_id for r in local.requests] == ["local:best"]


def test_other_backend_errors_propagate(registry, wav):
    registry.register("fake:", lambda: FakeBackend(error=ValueError("boom")))

    with pytest.raises(ValueError, match="boom"):
        t.transcribe(wav, model_size="fake:1", diarize=False, device="cpu")


class StreamingBackend(FakeBackend):
    """Streams two segments as it goes, and needs the alignment step."""
    skips_alignment = False

    def transcribe(self, request):
        segments = [{"start": 0.0, "end": 0.4, "text": "hello"},
                    {"start": 0.5, "end": 0.9, "text": "there"}]
        for seg in segments:
            request.on_segments([dict(seg)])
        return {"segments": segments, "language": "en"}


@pytest.fixture
def fake_models(monkeypatch):
    """Alignment and speaker identification without any model."""
    def align(result, audio, device, model_dir, cache_only, progress_callback=None):
        return dict(result, segments=[dict(s, words=[]) for s in result["segments"]])

    def diarize(result, audio, device, hf_token, min_speakers, max_speakers, prog):
        segments = [dict(s, speaker=f"SPEAKER_0{i}") for i, s in enumerate(result["segments"])]
        return dict(result, segments=segments), {}

    monkeypatch.setattr(pipeline, "align_by_language", align)
    monkeypatch.setattr(pipeline, "run_diarization", diarize)
    monkeypatch.setattr(pipeline, "split_by_track", lambda result, source: set())
    monkeypatch.setattr(pipeline, "label_known_speakers", lambda result, emb, split: None)


def test_events_arrive_in_step_order(registry, wav, fake_models):
    registry.register("fake:", StreamingBackend)
    events = []

    result = t.transcribe(wav, model_size="fake:1", hf_token="hf", device="cpu",
                          on_event=lambda kind, data: events.append((kind, data)))

    assert [kind if kind != "step" else data for kind, data in events] == [
        "transcription", "segments", "segments",
        "alignment", "aligned",
        "speaker_identification", "speaker_turns",
    ]
    assert [seg["text"] for _, segs in events[1:3] for seg in segs] == ["hello", "there"]
    assert all("words" in seg for seg in events[4][1])
    assert [seg["speaker"] for seg in events[6][1]] == ["SPEAKER_00", "SPEAKER_01"]
    assert set(result["step_times"]) == {"transcription", "alignment", "speaker_identification"}
    assert all(seconds >= 0 for seconds in result["step_times"].values())
    assert result["total_time"] >= sum(result["step_times"].values())


def test_skipped_steps_are_not_announced_or_timed(registry, wav):
    registry.register("fake:", FakeBackend)  # word timestamps already there, no streaming
    events = []

    result = t.transcribe(wav, model_size="fake:1", diarize=False, device="cpu",
                          on_event=lambda kind, data: events.append((kind, data)))

    assert [kind for kind, _ in events] == ["step", "aligned"]
    assert events[0][1] == "transcription"
    assert [seg["text"] for seg in events[1][1]] == ["hi"]
    assert set(result["step_times"]) == {"transcription"}


# --- why there are no speakers (the Timeline's note, #49) --------------------

def test_no_token_says_why_speakers_are_missing(registry, wav, fake_models):
    registry.register("fake:", FakeBackend)

    result = t.transcribe(wav, model_size="fake:1", hf_token=None, device="cpu")

    assert "token" in result["speakers_missing"].lower()


def test_disabled_speaker_identification_says_so(registry, wav):
    registry.register("fake:", FakeBackend)

    result = t.transcribe(wav, model_size="fake:1", diarize=False, device="cpu")

    assert "off" in result["speakers_missing"].lower()


def test_failed_speaker_identification_keeps_the_transcript(registry, wav, fake_models,
                                                           monkeypatch):
    registry.register("fake:", FakeBackend)

    def broken(*args):
        raise RuntimeError("license not accepted")

    monkeypatch.setattr(pipeline, "run_diarization", broken)
    events = []

    result = t.transcribe(wav, model_size="fake:1", hf_token="hf", device="cpu",
                          on_event=lambda kind, data: events.append(kind))

    assert [seg["text"] for seg in result["segments"]] == ["hi"]
    assert "license not accepted" in result["speakers_missing"]
    assert "speaker_turns" not in events


def test_identified_speakers_leave_no_reason(registry, wav, fake_models):
    registry.register("fake:", FakeBackend)

    result = t.transcribe(wav, model_size="fake:1", hf_token="hf", device="cpu")

    assert "speakers_missing" not in result
