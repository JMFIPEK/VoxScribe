"""Alignment, diarization and speaker-handling steps, with whisperx and the
voice-print store replaced by fakes (no models)."""

import sys
import types

import numpy as np
import pytest

from voxscribe import tracks
from voxscribe.transcription import align, diarize, speakers
from voxscribe.transcription.audio import AudioSource

AUDIO = np.zeros(16000, dtype=np.float32)


def _fake_whisperx(monkeypatch, failing_languages=()):
    import whisperx

    loaded = []

    def load_align_model(language_code, device, model_dir=None, model_cache_only=False):
        if language_code in failing_languages:
            raise RuntimeError(f"no model for {language_code}")
        loaded.append(language_code)
        return f"model-{language_code}", {"language": language_code}

    def fake_align(segs, model, metadata, audio, device, return_char_alignments, progress_callback):
        progress_callback(100.0)
        words = [{"word": s["text"], "start": s["start"], "end": s["end"]} for s in segs]
        return {"segments": [dict(s, words=[w]) for s, w in zip(segs, words)], "word_segments": words}

    monkeypatch.setattr(whisperx, "load_align_model", load_align_model, raising=False)
    monkeypatch.setattr(whisperx, "align", fake_align, raising=False)
    return loaded


# --- alignment ---------------------------------------------------------------

def test_align_by_language_aligns_each_language_with_its_own_model(monkeypatch):
    loaded = _fake_whisperx(monkeypatch)
    result = {"language": "de", "segments": [
        {"start": 0.0, "end": 1.0, "text": "eins", "language": "de"},
        {"start": 1.0, "end": 2.0, "text": "two", "language": "en"},
        {"start": 2.0, "end": 3.0, "text": "drei"},  # no tag: main language
    ]}

    out = align.align_by_language(result, AUDIO, "cpu", None, False)

    assert loaded == ["de", "en"]  # main language first
    assert [s["text"] for s in out["segments"]] == ["eins", "two", "drei"]  # time order
    assert [s["language"] for s in out["segments"]] == ["de", "en", "de"]
    assert [w["word"] for w in out["word_segments"]] == ["eins", "two", "drei"]
    assert out["language"] == "de"


def test_align_by_language_falls_back_to_the_main_model_for_a_minor_language(monkeypatch):
    loaded = _fake_whisperx(monkeypatch, failing_languages=("en",))
    result = {"language": "de", "segments": [
        {"start": 0.0, "end": 1.0, "text": "eins", "language": "de"},
        {"start": 1.0, "end": 2.0, "text": "two", "language": "en"},
    ]}

    out = align.align_by_language(result, AUDIO, "cpu", None, False)

    assert loaded == ["de", "de"]
    assert [s["language"] for s in out["segments"]] == ["de", "en"]


def test_align_by_language_main_language_failure_is_an_error(monkeypatch):
    _fake_whisperx(monkeypatch, failing_languages=("de",))

    with pytest.raises(RuntimeError, match="no model for de"):
        align.align_by_language({"language": "de", "segments": [
            {"start": 0.0, "end": 1.0, "text": "x"}]}, AUDIO, "cpu", None, False)


# --- diarization -------------------------------------------------------------

def test_run_diarization_labels_words_and_reports_progress(monkeypatch):
    import whisperx

    created = {}

    class FakePipeline:
        def __init__(self, token, device, cache_dir):
            created.update(token=token, device=device)

        def __call__(self, audio, min_speakers, max_speakers, return_embeddings, progress_callback):
            created.update(min=min_speakers, max=max_speakers)
            progress_callback(50.0)
            return "diarization", {"SPEAKER_00": [0.1]}

    fake_module = types.ModuleType("whisperx.diarize")
    fake_module.DiarizationPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_module)
    monkeypatch.setattr(whisperx, "assign_word_speakers",
                        lambda d, r, speaker_embeddings: dict(r, assigned=d), raising=False)
    events = []

    result, embeddings = diarize.run_diarization(
        {"segments": []}, AUDIO, "cpu", "hf_x", 1, 3, lambda pct, msg: events.append((round(pct, 3), msg)))

    assert created == dict(token="hf_x", device="cpu", min=1, max=3)
    assert result == {"segments": [], "assigned": "diarization"}
    assert embeddings == {"SPEAKER_00": [0.1]}
    assert events == [(0.7, "Loading diarization model..."), (0.8, "Running speaker diarization..."),
                      (0.86, "Running speaker diarization... 50%"), (0.92, "Assigning speakers...")]


# --- speakers ----------------------------------------------------------------

def test_label_known_speakers_renames_recognized_speakers(monkeypatch):
    monkeypatch.setattr("voxscribe.speaker_profiles.match_speakers",
                        lambda embeddings: {"SPEAKER_01": "Alice"})
    result = {
        "segments": [{"speaker": "SPEAKER_00"}, {"speaker": "SPEAKER_01"}],
        "word_segments": [{"speaker": "SPEAKER_01"}],
    }

    speakers.label_known_speakers(
        result, {"SPEAKER_00": [0.1], "SPEAKER_01": [0.2]}, split_speakers={"SPEAKER_02": "SPEAKER_00"})

    assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_00", "Alice"]
    assert result["word_segments"][0]["speaker"] == "Alice"
    assert result["speaker_id_map"] == {
        "SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "Alice", "SPEAKER_02": "SPEAKER_02"}


def test_label_known_speakers_survives_a_broken_profile_store(monkeypatch):
    def boom(embeddings):
        raise OSError("profiles unreadable")

    monkeypatch.setattr("voxscribe.speaker_profiles.match_speakers", boom)
    result = {"segments": [{"speaker": "SPEAKER_00"}]}

    speakers.label_known_speakers(result, {"SPEAKER_00": [0.1]}, split_speakers={})

    assert result["speaker_id_map"] == {"SPEAKER_00": "SPEAKER_00"}


def test_split_by_track_without_sidecar_changes_nothing(tmp_path):
    result = {"segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]}

    assert speakers.split_by_track(result, AudioSource(str(tmp_path / "meeting.wav"))) == {}


def test_split_by_track_uses_the_sidecar(tmp_path):
    sr = tracks.SAMPLE_RATE
    rng = np.random.default_rng(0)

    def signal(loud, seed):
        a = np.random.default_rng(seed).standard_normal(40 * sr) * 0.001
        a[int(loud[0] * sr):int(loud[1] * sr)] = rng.standard_normal(int((loud[1] - loud[0]) * sr)) * 0.5
        return a * 32768

    audio_path = str(tmp_path / "meeting.wav")
    tracks.save_tracks(audio_path, signal((0, 20), 1), signal((20, 40), 2))
    result = {"segments": [
        {"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"},
        {"start": 20.0, "end": 30.0, "speaker": "SPEAKER_00"},
    ]}

    assert speakers.split_by_track(result, AudioSource(audio_path)) == {"SPEAKER_01": "SPEAKER_00"}
    assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_00", "SPEAKER_01"]


def test_split_by_track_failure_is_not_fatal(tmp_path, monkeypatch):
    def boom(path):
        raise ValueError("corrupt sidecar")

    monkeypatch.setattr("voxscribe.tracks.load_tracks", boom)

    assert speakers.split_by_track({"segments": []}, AudioSource(str(tmp_path / "x.wav"))) == {}
