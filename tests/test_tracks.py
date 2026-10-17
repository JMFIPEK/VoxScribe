"""Characterization tests for voxscribe.tracks (numpy/soundfile only)."""

import numpy as np

from voxscribe import tracks

SR = tracks.SAMPLE_RATE


def _signal(loud_ranges, seconds=40.0, seed=0):
    """Noise at 0.5 amplitude inside loud_ranges (seconds), near-silence elsewhere."""
    rng = np.random.default_rng(seed)
    audio = rng.standard_normal(int(seconds * SR)).astype(np.float32) * 0.001
    for a, b in loud_ranges:
        audio[int(a * SR):int(b * SR)] = rng.standard_normal(int((b - a) * SR)) * 0.5
    return audio


def _tracks():
    # Mic is loud 0-20s, system is loud 20-40s.
    return _signal([(0, 20)], seed=1), _signal([(20, 40)], seed=2)


def _seg(start, end, speaker):
    return {"start": start, "end": end, "text": "x", "speaker": speaker,
            "words": [{"start": start, "end": end, "word": "x", "speaker": speaker}]}


def test_tracks_path_for_swaps_extension():
    assert tracks.tracks_path_for("recordings/x.wav") == "recordings/x.tracks.flac"


def test_load_tracks_returns_none_without_sidecar(tmp_path):
    assert tracks.load_tracks(str(tmp_path / "missing.wav")) is None


def test_save_then_load_roundtrip(tmp_path):
    audio_path = str(tmp_path / "meeting.wav")
    mic = np.array([0, 1000, -1000, 5], dtype=np.float32)
    system = np.array([7, 7], dtype=np.float32)  # shorter track is zero-padded

    saved = tracks.save_tracks(audio_path, mic, system)
    loaded_mic, loaded_sys = tracks.load_tracks(audio_path)

    assert saved == tracks.tracks_path_for(audio_path)
    assert len(loaded_mic) == len(loaded_sys) == 4
    np.testing.assert_allclose(loaded_mic * 32768, mic, atol=1)
    np.testing.assert_allclose(loaded_sys * 32768, [7, 7, 0, 0], atol=1)


def test_no_segments_means_no_split():
    mic, system = _tracks()
    assert tracks.split_speakers_by_track({"segments": []}, mic, system) == {}


def test_speaker_on_both_sides_is_split():
    mic, system = _tracks()
    result = {
        "segments": [
            _seg(0, 10, "SPEAKER_00"),    # mic side
            _seg(10, 20, "SPEAKER_00"),   # mic side
            _seg(20, 30, "SPEAKER_00"),   # system side: 10s >= 8s and >= 10%
            _seg(30, 40, "SPEAKER_01"),   # system side only
        ],
        "word_segments": [
            {"start": 20.5, "end": 21.0, "word": "x", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 5.5, "word": "x", "speaker": "SPEAKER_00"},
        ],
    }

    split = tracks.split_speakers_by_track(result, mic, system)

    assert split == {"SPEAKER_02": "SPEAKER_00"}
    assert [s["speaker"] for s in result["segments"]] == [
        "SPEAKER_00", "SPEAKER_00", "SPEAKER_02", "SPEAKER_01"]
    assert result["segments"][2]["words"][0]["speaker"] == "SPEAKER_02"
    assert [w["speaker"] for w in result["word_segments"]] == ["SPEAKER_02", "SPEAKER_00"]


def test_small_minority_share_is_not_split():
    mic, system = _tracks()
    result = {"segments": [
        _seg(0, 10, "SPEAKER_00"),
        _seg(10, 20, "SPEAKER_00"),
        _seg(20, 25, "SPEAKER_00"),   # only 5s on the other side (< 8s)
    ]}

    assert tracks.split_speakers_by_track(result, mic, system) == {}
    assert {s["speaker"] for s in result["segments"]} == {"SPEAKER_00"}


def test_save_tracks_stores_setup_and_load_setup_reads_it(tmp_path):
    audio_path = str(tmp_path / "x.wav")
    mic, system = _tracks()
    tracks.save_tracks(audio_path, mic * 1000, system * 1000, setup="loudspeaker")
    assert tracks.load_setup(audio_path) == "loudspeaker"
    assert tracks.load_tracks(audio_path) is not None


def test_load_setup_is_none_for_untagged_or_missing_sidecar(tmp_path):
    audio_path = str(tmp_path / "x.wav")
    assert tracks.load_setup(audio_path) is None
    mic, system = _tracks()
    tracks.save_tracks(audio_path, mic, system)
    assert tracks.load_setup(audio_path) is None
