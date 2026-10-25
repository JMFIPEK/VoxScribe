"""Tests for the "x realtime" line under the Timeline (spec #40, #49)."""

from qt_app.step_timing import realtime_line


def test_realtime_line_lists_each_step_and_the_total():
    result = {"duration": 120.0, "total_time": 40.0,
              "step_times": {"transcription": 10.0, "alignment": 2.5,
                             "speaker_identification": 25.0}}

    assert realtime_line(result) == (
        "Transcription 12× · Alignment 48× · Speaker identification 4.8× · Total 3.0× realtime")


def test_realtime_line_leaves_out_steps_that_did_not_run():
    result = {"duration": 60.0, "total_time": 20.0, "step_times": {"transcription": 20.0}}

    assert realtime_line(result) == "Transcription 3.0× · Total 3.0× realtime"


def test_realtime_line_is_empty_without_timings():
    assert realtime_line({"segments": []}) == ""
    assert realtime_line({"duration": 0.0, "total_time": 1.0, "step_times": {}}) == ""
