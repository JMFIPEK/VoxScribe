"""Tests for the Timeline as the loading bar (spec #40, #49): what it shows
while the pipeline's step events arrive (pure, qt_app.timeline_progress),
and the "x realtime" line under it."""

from qt_app.timeline_progress import TimelineProgress, realtime_line

GREY = "#888888"
PALETTE = ["#ff0000", "#0000ff", "#00ff00"]


def _seg(start, end, words=None, speaker=None):
    seg = {"start": start, "end": end, "text": "x"}
    if words is not None:
        seg["words"] = [{"word": "x", "start": s, "end": e} for s, e in words]
    if speaker:
        seg["speaker"] = speaker
    return seg


def _spans(lane):
    return [(run.start, run.end) for run in lane.runs]


def test_non_streaming_transcription_is_a_plain_sweep_with_the_percentage():
    progress = TimelineProgress(duration=100.0, speech_color=GREY)
    progress.step("transcription")
    progress.progress(0.2)  # transcription covers 0-40% of the pipeline

    timeline = progress.timeline()

    (lane,) = timeline.lanes
    assert lane.runs == ()
    assert timeline.sweep == 0.5
    assert timeline.percent == 20
    assert timeline.duration == 100.0


def test_streamed_segments_fill_the_speech_lane_as_soft_blocks():
    progress = TimelineProgress(duration=100.0, speech_color=GREY)
    progress.step("transcription")
    progress.segments([_seg(0.0, 10.0)])
    progress.segments([_seg(12.0, 20.0)])

    timeline = progress.timeline()

    (lane,) = timeline.lanes
    assert _spans(lane) == [(0.0, 10.0), (12.0, 20.0)]
    assert {run.color for run in lane.runs} == {GREY}
    assert timeline.sweep is None  # the segments are the fill
    assert timeline.soft


def test_without_a_known_duration_the_axis_grows_with_the_segments():
    progress = TimelineProgress(duration=None, speech_color=GREY)
    progress.step("transcription")
    progress.segments([_seg(0.0, 10.0), _seg(12.0, 20.0)])

    assert progress.timeline().duration == 20.0


def test_the_recording_length_can_arrive_after_the_start():
    progress = TimelineProgress(duration=None, speech_color=GREY)
    progress.set_duration(90.0)
    progress.step("transcription")
    progress.segments([_seg(0.0, 10.0)])

    assert progress.timeline().duration == 90.0


def test_a_non_streaming_backend_keeps_its_full_sweep_until_aligned_segments_arrive():
    progress = TimelineProgress(duration=100.0, speech_color=GREY)
    progress.step("transcription")
    progress.progress(0.4)
    progress.step("alignment")
    progress.progress(0.5)

    timeline = progress.timeline()

    assert timeline.sweep == 1.0
    assert timeline.percent == 50


def test_alignment_sharpens_the_lane_to_word_runs():
    progress = TimelineProgress(duration=100.0, speech_color=GREY)
    progress.step("transcription")
    progress.segments([_seg(0.0, 10.0)])
    progress.step("alignment")
    # Words close together form one run; a pause splits it. A segment
    # without word timestamps keeps its block.
    progress.aligned([_seg(0.0, 10.0, words=[(0.5, 1.0), (1.1, 2.0), (6.0, 7.0)]),
                      _seg(12.0, 15.0, words=[])])

    timeline = progress.timeline()

    (lane,) = timeline.lanes
    assert _spans(lane) == [(0.5, 2.0), (6.0, 7.0), (12.0, 15.0)]
    assert not timeline.soft
    assert timeline.sweep is None


def test_speaker_turns_split_the_lane_into_coloured_speaker_lanes():
    progress = TimelineProgress(duration=30.0, speech_color=GREY, palette=PALETTE)
    progress.step("transcription")
    progress.step("alignment")
    progress.aligned([_seg(0.0, 10.0), _seg(10.0, 30.0)])
    progress.step("speaker_identification")
    progress.speaker_turns([_seg(0.0, 10.0, speaker="SPEAKER_01"),
                            _seg(10.0, 30.0, speaker="SPEAKER_00")])

    timeline = progress.timeline()

    # Most speaking time first, each in its own colour.
    assert [(lane.label, lane.color) for lane in timeline.lanes] == [
        ("SPEAKER_00", "#ff0000"), ("SPEAKER_01", "#0000ff")]
    assert _spans(timeline.lanes[0]) == [(10.0, 30.0)]
    assert not timeline.soft


def test_a_provider_fallback_starts_the_lane_over():
    progress = TimelineProgress(duration=100.0, speech_color=GREY)
    progress.step("transcription")
    progress.segments([_seg(0.0, 10.0)])
    progress.step("transcription")  # Provider failed, the local model starts
    progress.progress(0.1)

    timeline = progress.timeline()

    assert timeline.lanes[0].runs == ()
    assert timeline.sweep == 0.25


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
