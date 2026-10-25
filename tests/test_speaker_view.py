"""Tests for qt_app.speaker_view: stable speaker colours, ranks, shares and
the top 5 / Others split (pure, no Qt), on synthetic result dicts."""

from qt_app.speaker_view import SpeakerView

PALETTE = ["red", "blue", "green", "yellow", "purple", "teal", "orange"]


def _seg(start, end, speaker):
    return {"start": start, "end": end, "text": "x", "speaker": speaker,
            "words": [{"start": start, "end": end, "word": "x", "speaker": speaker}]}


def _result(seconds_by_id, embeddings=None):
    """One segment per Diarization ID, back to back, labelled with the ID."""
    segments, t = [], 0.0
    for spk, seconds in seconds_by_id.items():
        segments.append(_seg(t, t + seconds, spk))
        t += seconds
    return {
        "segments": segments,
        "word_segments": [w for s in segments for w in s["words"]],
        "speaker_id_map": {spk: spk for spk in seconds_by_id},
        "speaker_embeddings": embeddings or {},
    }


def test_colours_follow_speaking_time():
    view = SpeakerView(_result({"SPEAKER_00": 5, "SPEAKER_01": 30, "SPEAKER_02": 12}), PALETTE)

    assert view.color_of("SPEAKER_01") == "red"
    assert view.color_of("SPEAKER_02") == "blue"
    assert view.color_of("SPEAKER_00") == "green"


def test_renames_do_not_change_colours():
    result = _result({"SPEAKER_00": 5, "SPEAKER_01": 30, "SPEAKER_02": 12})
    view = SpeakerView(result, PALETTE)

    view.rename("SPEAKER_00", "Anna")
    view.rename("SPEAKER_01", "Ben")

    assert view.color_of("Anna") == "green"
    assert view.color_of("Ben") == "red"
    assert view.color_of("SPEAKER_02") == "blue"
    assert [s["speaker"] for s in result["segments"]] == ["Anna", "Ben", "SPEAKER_02"]
    assert [w["speaker"] for w in result["word_segments"]] == ["Anna", "Ben", "SPEAKER_02"]
    assert result["speaker_id_map"]["SPEAKER_00"] == "Anna"


def test_merge_keeps_the_larger_speakers_colour():
    result = _result({"SPEAKER_00": 5, "SPEAKER_01": 30, "SPEAKER_02": 12})
    view = SpeakerView(result, PALETTE)
    view.rename("SPEAKER_02", "Anna")

    # Anna (12s, blue) absorbs SPEAKER_01 (30s, red): the merged speaker is red.
    view.rename("SPEAKER_01", "Anna")

    assert view.color_of("Anna") == "red"
    assert [(s.label, s.color, s.seconds) for s in view.speakers()] == [
        ("Anna", "red", 42), ("SPEAKER_00", "green", 5)]
    assert set(s["speaker"] for s in result["segments"]) == {"Anna", "SPEAKER_00"}


def test_speakers_ranked_by_share():
    view = SpeakerView(_result({"SPEAKER_00": 10, "SPEAKER_01": 60, "SPEAKER_02": 30}), PALETTE)

    speakers = view.speakers()

    assert [s.label for s in speakers] == ["SPEAKER_01", "SPEAKER_02", "SPEAKER_00"]
    assert [s.rank for s in speakers] == [0, 1, 2]
    assert [round(s.share, 2) for s in speakers] == [0.6, 0.3, 0.1]


def test_top_five_get_lanes_and_the_rest_share_others():
    seconds = {f"SPEAKER_{i:02d}": s for i, s in enumerate([3, 50, 8, 40, 30, 20, 10])}
    view = SpeakerView(_result(seconds), PALETTE)

    lanes, others = view.lanes()

    assert [s.label for s in lanes] == ["SPEAKER_01", "SPEAKER_03", "SPEAKER_04", "SPEAKER_05", "SPEAKER_06"]
    assert [s.label for s in others] == ["SPEAKER_02", "SPEAKER_00"]
    # In the Others lane every speaker keeps their own colour.
    assert [s.color for s in others] == ["teal", "orange"]


def test_five_speakers_or_fewer_need_no_others_lane():
    view = SpeakerView(_result({f"SPEAKER_{i:02d}": 10 + i for i in range(5)}), PALETTE)

    lanes, others = view.lanes()

    assert len(lanes) == 5
    assert others == []


def test_voice_prints_only_for_speakers_with_ten_seconds_and_an_embedding():
    result = _result({"SPEAKER_00": 9.5, "SPEAKER_01": 12, "SPEAKER_02": 20},
                     embeddings={"SPEAKER_00": [0.1], "SPEAKER_01": [0.2]})
    view = SpeakerView(result, PALETTE)

    assert view.voice_prints("SPEAKER_01") == [[0.2]]
    assert view.enroll_hint("SPEAKER_01") is None
    assert view.voice_prints("SPEAKER_00") == []
    assert "10 s" in view.enroll_hint("SPEAKER_00")
    assert view.voice_prints("SPEAKER_02") == []
    assert "voice print" in view.enroll_hint("SPEAKER_02")


def test_a_merged_speaker_enrolls_only_its_long_enough_parts():
    result = _result({"SPEAKER_00": 4, "SPEAKER_01": 30},
                     embeddings={"SPEAKER_00": [0.1], "SPEAKER_01": [0.2]})
    view = SpeakerView(result, PALETTE)
    view.rename("SPEAKER_00", "Anna")
    view.rename("SPEAKER_01", "Anna")

    assert view.voice_prints("Anna") == [[0.2]]
