"""Voice-print matching: certain and Tentative matches (synthetic embeddings)."""

import math

from voxscribe.speaker_profiles import match_speakers


def _voice(similarity: float) -> list[float]:
    """An embedding with exactly `similarity` to the profile voice [1, 0]."""
    return [similarity, math.sqrt(1 - similarity ** 2)]


PROFILES = {"Jonas": {"embedding": [1.0, 0.0], "samples": 1}}


def test_a_close_voice_is_a_tentative_match_and_a_distant_one_none():
    matches = match_speakers({"SPEAKER_00": _voice(0.65)}, PROFILES)
    assert matches.certain == {}
    assert matches.tentative == {"SPEAKER_00": "Jonas"}

    matches = match_speakers({"SPEAKER_00": _voice(0.55)}, PROFILES)
    assert matches.certain == {}
    assert matches.tentative == {}


def test_a_name_matched_for_certain_is_not_also_offered_as_tentative():
    matches = match_speakers({"SPEAKER_00": _voice(0.9), "SPEAKER_01": _voice(0.7)}, PROFILES)

    assert matches.certain == {"SPEAKER_00": "Jonas"}
    assert matches.tentative == {}


def test_two_speakers_close_to_the_same_name_only_the_closer_gets_it():
    matches = match_speakers({"SPEAKER_00": _voice(0.62), "SPEAKER_01": _voice(0.7)}, PROFILES)

    assert matches.certain == {}
    assert matches.tentative == {"SPEAKER_01": "Jonas"}


def test_certain_matches_pick_the_best_speaker_per_name():
    profiles = dict(PROFILES, Alice={"embedding": [0.0, 1.0], "samples": 1})
    matches = match_speakers(
        {"SPEAKER_00": _voice(0.8), "SPEAKER_01": _voice(0.95), "SPEAKER_02": [0.0, 1.0]}, profiles)

    assert matches.certain == {"SPEAKER_01": "Jonas", "SPEAKER_02": "Alice"}


def test_each_match_logs_its_similarity(capsys):
    match_speakers({"SPEAKER_00": _voice(0.65)}, PROFILES)

    assert "SPEAKER_00 -> Jonas? (similarity 0.65)" in capsys.readouterr().out


def _voice2(jonas: float, anna: float) -> list[float]:
    """An embedding with these similarities to Jonas [1, 0, 0] and Anna [0, 1, 0]."""
    return [jonas, anna, math.sqrt(1 - jonas ** 2 - anna ** 2)]


TWO_PROFILES = {"Jonas": {"embedding": [1.0, 0.0, 0.0], "samples": 1},
                "Anna": {"embedding": [0.0, 1.0, 0.0], "samples": 1}}


def test_a_speaker_whose_best_name_went_to_a_closer_speaker_gets_no_tentative_match():
    matches = match_speakers(
        {"SPEAKER_00": _voice2(0.7, 0.65), "SPEAKER_01": _voice2(0.72, 0.0)}, TWO_PROFILES)

    assert matches.tentative == {"SPEAKER_01": "Jonas"}


def test_a_speaker_whose_best_name_is_matched_for_certain_gets_no_tentative_match():
    matches = match_speakers(
        {"SPEAKER_00": _voice2(0.7, 0.65), "SPEAKER_01": _voice2(0.9, 0.0)}, TWO_PROFILES)

    assert matches.certain == {"SPEAKER_01": "Jonas"}
    assert matches.tentative == {}


def test_a_speaker_beaten_to_a_certain_name_gets_no_tentative_match_either():
    matches = match_speakers(
        {"SPEAKER_00": _voice2(0.78, 0.6), "SPEAKER_01": _voice2(0.95, 0.0)}, TWO_PROFILES)

    assert matches.certain == {"SPEAKER_01": "Jonas"}
    assert matches.tentative == {}


def test_candidates_that_are_not_assigned_are_logged_too(capsys):
    match_speakers({"SPEAKER_00": _voice2(0.7, 0.65), "SPEAKER_01": _voice2(0.72, 0.0)}, TWO_PROFILES)

    out = capsys.readouterr().out
    assert "SPEAKER_01 -> Jonas? (similarity 0.72)" in out
    assert "SPEAKER_00 -> Jonas (similarity 0.70, not assigned)" in out
    assert "SPEAKER_00 -> Anna (similarity 0.65, not assigned)" in out
