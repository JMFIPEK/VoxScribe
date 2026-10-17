"""Characterization tests for the pure logic in voxscribe.transcriber.

These record current behavior (refactor step 2); they do not assert it is ideal.
"""

import numpy as np
import pytest

from voxscribe import transcriber as t

SR = t.SAMPLE_RATE


# --- Model ID prefix helpers -------------------------------------------------

def test_model_id_builders():
    assert t.remote_model_id("my-provider") == "server:my-provider"
    assert t.openvino_model_id("GPU", "medium") == "openvino:GPU:medium"
    assert t.PARAKEET_MODEL == "parakeet:tdt-0.6b-v3"
    assert t.APPLE_SPEECHANALYZER_MODEL == "apple:speechanalyzer"


def test_openvino_model_id_roundtrip():
    assert t._parse_openvino_model("openvino:GPU:medium") == ("GPU", "medium")


@pytest.mark.parametrize("model_id, remote, openvino, parakeet, apple", [
    ("server:abc", True, False, False, False),
    ("openvino:GPU:medium", False, True, False, False),
    ("parakeet:tdt-0.6b-v3", False, False, True, False),
    ("apple:speechanalyzer", False, False, False, True),
    ("large-v3", False, False, False, False),
])
def test_prefix_predicates(model_id, remote, openvino, parakeet, apple):
    assert t.is_remote_model(model_id) is remote
    assert t.is_openvino_model(model_id) is openvino
    assert t.is_parakeet_model(model_id) is parakeet
    assert t.is_apple_model(model_id) is apple


# --- Language helpers --------------------------------------------------------

@pytest.mark.parametrize("raw, fallback, expected", [
    ("german", None, "de"),
    ("German", None, "de"),
    (" EN ", None, "en"),
    ("de", None, "de"),
    (None, "en", "en"),
    ("", "fr", "fr"),
    ("klingon", "en", "en"),
    ("klingon", None, None),
])
def test_normalize_language_code(raw, fallback, expected):
    assert t._normalize_language_code(raw, fallback) == expected


def test_guess_language_german():
    text = "ich habe das nicht gesehen und wir sind jetzt auch noch hier"
    assert t._guess_language_from_text(text) == "de"


def test_guess_language_english():
    text = "we have the plan and it is not what you want but we can do this"
    assert t._guess_language_from_text(text) == "en"


def test_guess_language_needs_enough_words():
    assert t._guess_language_from_text("ich bin das") is None
    assert t._guess_language_from_text("") is None
    assert t._guess_language_from_text(None) is None


def test_guess_language_without_stopwords_is_none():
    assert t._guess_language_from_text("alpha beta gamma delta epsilon zeta eta theta iota") is None


# --- _find_quiet_split -------------------------------------------------------

def test_find_quiet_split_picks_quietest_frame():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(20 * SR) * 0.5).astype(np.float32)
    audio[8 * SR:8 * SR + SR // 10] = 0.0  # silent 100ms frame at 8.0-8.1s

    # Search window starts at 10s - 4s = 6s; the silent frame is index 20 in it.
    assert t._find_quiet_split(audio, 0, 10 * SR) == 6 * SR + 20 * (SR // 10) + (SR // 20)


def test_find_quiet_split_past_end_returns_length():
    audio = np.zeros(5 * SR, dtype=np.float32)
    assert t._find_quiet_split(audio, 0, 10 * SR) == len(audio)


def test_find_quiet_split_window_too_small_returns_nominal_end():
    audio = np.ones(5 * SR, dtype=np.float32)
    assert t._find_quiet_split(audio, 0, SR // 10) == SR // 10


# --- Output formats ----------------------------------------------------------

@pytest.mark.parametrize("seconds, expected", [
    (0, "00:00"), (59.9, "00:59"), (65, "01:05"), (3661, "01:01:01"),
])
def test_format_time(seconds, expected):
    assert t._format_time(seconds) == expected


@pytest.mark.parametrize("seconds, expected", [
    (0.0, "00:00:00,000"), (3661.5, "01:01:01,500"), (1.25, "00:00:01,250"),
])
def test_format_time_srt(seconds, expected):
    assert t._format_time_srt(seconds) == expected


RESULT = {"segments": [
    {"start": 0.0, "end": 1.5, "text": " Hello ", "speaker": "SPEAKER_00"},
    {"start": 65.0, "end": 70.0, "text": "No speaker"},
]}


def test_format_transcript_with_speakers():
    assert t.format_transcript(RESULT) == (
        "[00:00 - 00:01] SPEAKER_00: Hello\n"
        "[01:05 - 01:10] No speaker"
    )


def test_format_transcript_without_speakers():
    assert t.format_transcript(RESULT, include_speakers=False) == (
        "[00:00 - 00:01] Hello\n"
        "[01:05 - 01:10] No speaker"
    )


def test_to_srt():
    assert t._to_srt(RESULT) == (
        "1\n00:00:00,000 --> 00:00:01,500\n[SPEAKER_00] Hello\n"
        "\n"
        "2\n00:01:05,000 --> 00:01:10,000\nNo speaker\n"
    )


# --- transcribe_multi speaker-label merge ------------------------------------

def _file_result(duration, language="de"):
    return {
        "language": language,
        "duration": duration,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "a", "speaker": "SPEAKER_00",
             "words": [{"start": 0.0, "end": 1.0, "word": "a", "speaker": "SPEAKER_00"}]},
            {"start": 2.0, "end": 4.0, "text": "b", "speaker": "Alice"},
        ],
        "word_segments": [{"start": 0.0, "end": 1.0, "word": "a", "speaker": "SPEAKER_00"}],
        "speaker_id_map": {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "Alice"},
        "speaker_embeddings": {"SPEAKER_00": [0.1], "SPEAKER_01": [0.2]},
    }


@pytest.fixture
def fake_transcribe(monkeypatch):
    calls = []
    results = iter([_file_result(10.0), _file_result(20.0)])

    def fake(**kwargs):
        calls.append(kwargs)
        return next(results)

    monkeypatch.setattr(t, "transcribe", fake)
    return calls


def test_transcribe_multi_requires_files():
    with pytest.raises(ValueError):
        t.transcribe_multi([])


def test_transcribe_multi_prefixes_unrecognized_speakers_and_shifts_time(fake_transcribe):
    combined = t.transcribe_multi(["/x/one.wav", "/x/two.wav"], language=None)

    segs = combined["segments"]
    assert [s["speaker"] for s in segs] == [
        "File 1: SPEAKER_00", "Alice", "File 2: SPEAKER_00", "Alice"]
    assert [(s["start"], s["end"]) for s in segs] == [
        (0.0, 2.0), (2.0, 4.0), (10.0, 12.0), (12.0, 14.0)]
    assert segs[2]["words"][0]["start"] == 10.0
    # Current behavior (possible bug, not fixed here): the per-word speaker
    # inside a segment is shifted in time but NOT renamed, unlike
    # seg["speaker"] and the top-level word_segments.
    assert segs[2]["words"][0]["speaker"] == "SPEAKER_00"
    assert [(w["start"], w["speaker"]) for w in combined["word_segments"]] == [
        (0.0, "File 1: SPEAKER_00"), (10.0, "File 2: SPEAKER_00")]
    assert combined["duration"] == 30.0
    assert combined["language"] == "de"
    assert combined["speaker_id_map"] == {
        "file0:SPEAKER_00": "File 1: SPEAKER_00",
        "file1:SPEAKER_00": "File 2: SPEAKER_00",
        "Alice": "Alice",
    }
    assert set(combined["speaker_embeddings"]) == {
        "file0:SPEAKER_00", "file1:SPEAKER_00", "Alice"}


def test_transcribe_multi_locks_in_first_detected_language(fake_transcribe):
    t.transcribe_multi(["one.wav", "two.wav"], language=None)

    assert [c["language"] for c in fake_transcribe] == [None, "de"]


def test_transcribe_multi_explicit_language_is_kept(fake_transcribe):
    t.transcribe_multi(["one.wav", "two.wav"], language="en")

    assert [c["language"] for c in fake_transcribe] == ["en", "en"]


def test_transcribe_multi_single_file_keeps_labels(monkeypatch):
    monkeypatch.setattr(t, "transcribe", lambda **kw: _file_result(10.0))

    combined = t.transcribe_multi(["one.wav"], language="de")

    assert [s["speaker"] for s in combined["segments"]] == ["SPEAKER_00", "Alice"]


def test_transcribe_multi_result_key_order_is_stable(fake_transcribe):
    combined = t.transcribe_multi(["one.wav", "two.wav"], language=None)

    assert list(combined) == [
        "segments", "word_segments", "language", "duration",
        "speaker_id_map", "speaker_embeddings"]


def test_transcribe_multi_without_diarization_has_no_speaker_maps(monkeypatch):
    monkeypatch.setattr(t, "transcribe", lambda **kw: _file_result(10.0))

    combined = t.transcribe_multi(["one.wav"], language="de", diarize=False)

    assert list(combined) == ["segments", "word_segments", "language", "duration"]
