"""Second opinion from per-sentence voice prints (#62), with fake voice prints:
the embedding of a stretch of audio is the voice of whoever really speaks
there, so the tests say who speaks when and what pyannote made of it."""

import numpy as np
import pytest

from voxscribe.transcription import pipeline, sentence_voices
from voxscribe.transcription.sentence_voices import second_opinion

VOICES = {"anna": [1.0, 0.0, 0.0], "berta": [0.0, 1.0, 0.0], "clara": [0.0, 0.0, 1.0]}


def _sentence(start, end, pyannote, text="Ein Satz."):
    words = text.split()
    step = (end - start) / len(words)
    return {"start": start, "end": end, "text": text, "speaker": pyannote,
            "words": [{"start": start + i * step, "end": start + (i + 1) * step,
                       "word": w, "speaker": pyannote} for i, w in enumerate(words)]}


def _run(turns, embeddings=None):
    """turns: (start, end, real voice, pyannote's label). Returns (result, new embeddings)."""
    result = {"segments": [_sentence(a, b, spk) for a, b, _, spk in turns]}
    rng = np.random.default_rng(0)

    def embed(start, end):
        voice = next(v for a, b, v, _ in turns if a <= (start + end) / 2 <= b)
        return np.array(VOICES[voice]) + rng.normal(0, 0.05, 3)

    labels = {spk for *_, spk in turns}
    return result, second_opinion(result, embeddings or {s: [0.1, 0.1, 0.1] for s in labels}, embed)


def _dialog(voice_a, voice_b, label_a, label_b, n=8, length=3.0):
    """n alternating sentences of `length` s each."""
    return [(i * length, (i + 1) * length - 0.2,
             voice_a if i % 2 == 0 else voice_b, label_a if i % 2 == 0 else label_b) for i in range(n)]


def test_two_voices_pyannote_took_for_one_are_split():
    result, embeddings = _run(_dialog("anna", "berta", "SPEAKER_00", "SPEAKER_00"))

    speakers = [s["speaker"] for s in result["segments"]]
    assert speakers == ["SPEAKER_00", "SPEAKER_01"] * 4 or speakers == ["SPEAKER_01", "SPEAKER_00"] * 4
    assert all(w["speaker"] == s["speaker"] for s in result["segments"] for w in s["words"])
    assert set(embeddings) == {"SPEAKER_00", "SPEAKER_01"}


def test_one_voice_pyannote_split_in_two_is_merged():
    result, embeddings = _run(_dialog("anna", "anna", "SPEAKER_00", "SPEAKER_01"))

    assert {s["speaker"] for s in result["segments"]} == {"SPEAKER_00"}
    assert set(embeddings) == {"SPEAKER_00"}


def test_new_speaker_ids_never_reuse_an_existing_one():
    turns = _dialog("anna", "berta", "SPEAKER_00", "SPEAKER_00")
    result, embeddings = _run(turns, embeddings={"SPEAKER_00": [0.1] * 3, "SPEAKER_01": [0.2] * 3})

    assert {s["speaker"] for s in result["segments"]} == {"SPEAKER_00", "SPEAKER_02"}
    assert set(embeddings) == {"SPEAKER_00", "SPEAKER_02"}


@pytest.mark.parametrize("turns", [
    _dialog("anna", "berta", "SPEAKER_00", "SPEAKER_01"),   # agree on two
    _dialog("anna", "anna", "SPEAKER_00", "SPEAKER_00"),    # agree on one
])
def test_pyannote_stays_when_the_sentences_agree(turns):
    result, embeddings = _run(turns)

    assert embeddings is None
    assert [s["speaker"] for s in result["segments"]] == [spk for *_, spk in turns]


def test_a_few_seconds_of_another_voice_dont_count_as_a_contradiction():
    # 24 s of Anna, one 2.8 s sentence of Clara: not substantial, nothing changes.
    turns = _dialog("anna", "anna", "SPEAKER_00", "SPEAKER_00") + [(30.0, 32.8, "clara", "SPEAKER_00")]
    result, embeddings = _run(turns)

    assert embeddings is None


def test_a_small_group_joins_the_closest_substantial_one():
    # Two people for 12 s each, plus 2 s of a third voice closer to Berta's.
    turns = _dialog("anna", "berta", "SPEAKER_00", "SPEAKER_00") + [(30.0, 32.0, "clara", "SPEAKER_00")]
    VOICES["clara"] = [0.0, 0.85, 0.53]
    try:
        result, embeddings = _run(turns)
    finally:
        VOICES["clara"] = [0.0, 0.0, 1.0]

    assert len(embeddings) == 2
    assert result["segments"][-1]["speaker"] == result["segments"][1]["speaker"]  # Berta's


def test_words_without_a_voice_print_follow_the_nearest_sentence():
    turns = _dialog("anna", "berta", "SPEAKER_00", "SPEAKER_00")
    result = {"segments": [_sentence(a, b, spk) for a, b, _, spk in turns]}
    # A 0.4 s "Ja." right after Berta's second sentence (ends 8.8 s), too short for a voice print.
    result["segments"].insert(4, _sentence(8.85, 9.25, "SPEAKER_00", "Ja."))
    result["word_segments"] = [dict(w) for s in result["segments"] for w in s["words"]]

    def embed(start, end):
        voice = next(v for a, b, v, _ in turns if a <= (start + end) / 2 <= b)
        return np.array(VOICES[voice])

    second_opinion(result, {"SPEAKER_00": [0.1] * 3}, embed)

    berta = result["segments"][3]["speaker"]
    assert result["segments"][4]["speaker"] == berta
    assert result["word_segments"][8]["speaker"] == berta  # the "Ja." in word_segments


def test_a_sentence_holding_both_voices_is_cut_at_the_change():
    turns = _dialog("anna", "berta", "SPEAKER_00", "SPEAKER_00")
    result = {"segments": [_sentence(a, b, spk) for a, b, _, spk in turns]}
    # pyannote's turns put two sentences in one segment: merge segments 0 and 1.
    first, second = result["segments"][:2]
    result["segments"][:2] = [dict(first, end=second["end"], words=first["words"] + second["words"])]

    def embed(start, end):
        return np.array(VOICES[next(v for a, b, v, _ in turns if a <= (start + end) / 2 <= b)])

    second_opinion(result, {"SPEAKER_00": [0.1] * 3}, embed)

    assert len(result["segments"]) == 8
    assert result["segments"][0]["speaker"] != result["segments"][1]["speaker"]


def test_sentences_end_at_punctuation_and_long_pauses():
    seg = {"words": [{"start": 0.0, "end": 0.5, "word": "Hallo."},
                     {"start": 0.6, "end": 1.0, "word": "Wie"},
                     {"start": 1.1, "end": 1.4, "word": "geht's"},
                     {"start": 3.0, "end": 3.5, "word": "dir?"},
                     {"start": 3.6, "word": "untimed"}]}

    assert [[w["word"] for w in s] for s in sentence_voices.sentences({"segments": [seg]})] == [
        ["Hallo."], ["Wie", "geht's"], ["dir?"]]


# --- when the pipeline asks for it --------------------------------------------

class _Source:
    def __init__(self, tracks):
        self._tracks = tracks

    def tracks(self):
        if isinstance(self._tracks, Exception):
            raise self._tracks
        return self._tracks


@pytest.mark.parametrize("tracks, min_spk, max_spk, wanted", [
    (None, None, None, True),               # In room / Playback only
    ((np.zeros(1), np.zeros(1)), None, None, False),  # a Call: its tracks decide
    (None, 2, None, False),                 # the user gave the speaker count
    (None, None, 3, False),
    (OSError("broken sidecar"), None, None, False),
])
def test_second_opinion_only_without_tracks_and_speaker_count(tracks, min_spk, max_spk, wanted):
    assert pipeline._wants_second_opinion(_Source(tracks), min_spk, max_spk) is wanted


def test_a_failing_second_opinion_keeps_pyannotes_labels(monkeypatch):
    from voxscribe.transcription import diarize

    class Broken:
        class model:
            @staticmethod
            def _embedding(x):
                raise RuntimeError("no MPS")

    result = {"segments": [_sentence(0, 3, "SPEAKER_00"), _sentence(3, 6, "SPEAKER_00")]}
    embeddings = diarize._second_opinion(result, {"SPEAKER_00": [0.1]}, np.zeros(16000 * 6, np.float32),
                                         Broken, lambda pct, msg: None)

    assert embeddings == {"SPEAKER_00": [0.1]}
    assert {s["speaker"] for s in result["segments"]} == {"SPEAKER_00"}
