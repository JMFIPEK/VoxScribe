"""Stray speaker merge: a Diarization ID with under 3 s of speech is given to
the neighbouring speaker, through transcribe() with a fake Backend and fake
diarization (no models)."""

import numpy as np
import pytest
import soundfile as sf

from voxscribe import tracks, transcriber as t
from voxscribe.speaker_profiles import SpeakerMatches
from voxscribe.transcription import pipeline
from voxscribe.transcription.registry import BackendRegistry


def _seg(start, end, speaker, text="x"):
    return {"start": start, "end": end, "text": text, "speaker": speaker,
            "words": [{"start": start, "end": end, "word": text, "speaker": speaker}]}


class FakeBackend:
    name = "Fake"
    skips_alignment = True

    def __init__(self, segments):
        self.segments = segments

    def describe(self, model_id, device):
        return "Device: fake"

    word_segments = None

    def transcribe(self, request):
        words = self.word_segments
        if words is None:
            words = [w for s in self.segments for w in s["words"]]
        return {"segments": [dict(s) for s in self.segments], "language": "de",
                "word_segments": words}


@pytest.fixture
def run(tmp_path, monkeypatch):
    """transcribe() on a silent 40 s file whose diarization returns `segments`;
    also returns the Diarization IDs voice-print matching was asked about."""
    path = str(tmp_path / "meeting.wav")
    sf.write(path, np.zeros(40 * 16000, dtype=np.float32), 16000)
    asked = []

    def match_speakers(embeddings):
        asked.extend(embeddings)
        return SpeakerMatches(certain={}, tentative={})

    monkeypatch.setattr("voxscribe.speaker_profiles.match_speakers", match_speakers)

    def go(segments, extra_embeddings=(), word_segments=None):
        reg = BackendRegistry()
        backend = FakeBackend(segments)
        if word_segments is not None:
            backend.word_segments = word_segments
        reg.register("fake:", lambda: backend)
        monkeypatch.setattr(pipeline, "default_registry", reg)
        embeddings = {s["speaker"]: [0.1] for s in segments}
        embeddings.update({spk: [0.2] for spk in extra_embeddings})
        monkeypatch.setattr(pipeline, "run_diarization", lambda result, *a: (
            dict(result, speaker_embeddings=embeddings), embeddings))
        return t.transcribe(path, model_size="fake:1", hf_token="hf_x", device="cpu"), asked

    go.path = path
    return go


def test_a_short_speaker_between_two_turns_goes_to_the_nearer_neighbour(run):
    result, _ = run([
        _seg(0.0, 10.0, "SPEAKER_00"),
        _seg(10.5, 12.0, "SPEAKER_01", "ja"),  # 1.5 s, 0.5 s after SPEAKER_00
        _seg(14.0, 20.0, "SPEAKER_02"),        # 2 s later
    ])

    assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_02"]
    assert result["segments"][1]["text"] == "ja"  # nothing deleted
    assert result["segments"][1]["words"][0]["speaker"] == "SPEAKER_00"
    assert [w["speaker"] for w in result["word_segments"]] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_02"]
    assert set(result["speaker_id_map"]) == {"SPEAKER_00", "SPEAKER_02"}
    assert set(result["speaker_embeddings"]) == {"SPEAKER_00", "SPEAKER_02"}


def test_a_speaker_with_more_than_3_seconds_stays(run):
    result, _ = run([
        _seg(0.0, 10.0, "SPEAKER_00"),
        _seg(10.5, 14.0, "SPEAKER_01"),  # 3.5 s
        _seg(14.5, 20.0, "SPEAKER_00"),
    ])

    assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert set(result["speaker_id_map"]) == {"SPEAKER_00", "SPEAKER_01"}


def test_a_stray_speaker_is_never_voice_print_matched(run):
    _, asked = run([
        _seg(0.0, 10.0, "SPEAKER_00"),
        _seg(10.5, 11.0, "SPEAKER_01"),
        _seg(11.5, 20.0, "SPEAKER_00"),
    ])

    assert asked == ["SPEAKER_00"]


def test_track_split_speakers_are_unaffected(run):
    sr = tracks.SAMPLE_RATE

    def signal(loud, seed):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal(40 * sr) * 0.001
        a[int(loud[0] * sr):int(loud[1] * sr)] = rng.standard_normal(int((loud[1] - loud[0]) * sr)) * 0.5
        return a * 32768

    # Mic is loud 0-20 s, system 20-40 s: SPEAKER_00's 20-30 s turn is split off.
    tracks.save_tracks(run.path, signal((0, 20), 1), signal((20, 40), 2))

    result, _ = run([
        _seg(0.0, 20.0, "SPEAKER_00"),
        _seg(20.0, 30.0, "SPEAKER_00"),
        _seg(30.0, 40.0, "SPEAKER_01"),
    ])

    assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_00", "SPEAKER_02", "SPEAKER_01"]
    assert set(result["speaker_id_map"]) == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}


def test_a_speaker_with_an_embedding_but_no_segment_is_merged_too(run):
    result, asked = run([
        _seg(0.0, 10.0, "SPEAKER_00"),
        _seg(10.5, 20.0, "SPEAKER_01"),
    ], extra_embeddings=["SPEAKER_02"])

    assert asked == ["SPEAKER_00", "SPEAKER_01"]
    assert set(result["speaker_id_map"]) == {"SPEAKER_00", "SPEAKER_01"}
    assert set(result["speaker_embeddings"]) == {"SPEAKER_00", "SPEAKER_01"}


def test_a_speaker_only_on_words_is_merged_and_its_words_move(run):
    segments = [_seg(0.0, 10.0, "SPEAKER_00"), _seg(12.0, 20.0, "SPEAKER_01")]
    words = [w for s in segments for w in s["words"]]
    words.insert(1, {"start": 10.2, "end": 10.6, "word": "hm", "speaker": "SPEAKER_02"})

    result, asked = run(segments, extra_embeddings=["SPEAKER_02"], word_segments=words)

    assert asked == ["SPEAKER_00", "SPEAKER_01"]
    assert [w["speaker"] for w in result["word_segments"]] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
    assert set(result["speaker_id_map"]) == {"SPEAKER_00", "SPEAKER_01"}
