"""Tests for the Timeline (spec #40, #46): which lanes and coloured runs a
result gives (pure, qt_app.timeline_model) and that the widget paints them
(offscreen)."""

import os

import pytest

from qt_app.speaker_view import SpeakerView
from qt_app.timeline_model import timeline_from_result

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PALETTE = ["#ff0000", "#0000ff", "#00ff00", "#ffff00", "#ff00ff", "#00ffff",
           "#ff8800", "#8800ff"]


def _seg(start, end, speaker):
    return {"start": start, "end": end, "text": "x", "speaker": speaker}


def _result(seconds_by_id):
    """One segment per Diarization ID, back to back, labelled with the ID."""
    segments, t = [], 0.0
    for spk, seconds in seconds_by_id.items():
        segments.append(_seg(t, t + seconds, spk))
        t += seconds
    return {"segments": segments, "speaker_id_map": {spk: spk for spk in seconds_by_id},
            "duration": t}


EIGHT = {f"SPEAKER_0{i}": 10 + i for i in range(8)}  # SPEAKER_07 talks most


def test_eight_speakers_give_five_lanes_and_others():
    result = _result(EIGHT)
    timeline = timeline_from_result(result, SpeakerView(result, PALETTE))

    assert [lane.label for lane in timeline.lanes] == [
        "SPEAKER_07", "SPEAKER_06", "SPEAKER_05", "SPEAKER_04", "SPEAKER_03", "Others"]
    assert timeline.duration == sum(EIGHT.values())


def test_others_lane_keeps_each_speakers_colour():
    result = _result(EIGHT)
    view = SpeakerView(result, PALETTE)
    others = timeline_from_result(result, view).lanes[-1]

    # SPEAKER_00 starts at 0 s, SPEAKER_01 at 10 s, SPEAKER_02 at 21 s.
    assert [(run.start, run.end, run.color) for run in others.runs] == [
        (0.0, 10.0, view.color_of("SPEAKER_00")),
        (10.0, 21.0, view.color_of("SPEAKER_01")),
        (21.0, 33.0, view.color_of("SPEAKER_02")),
    ]


def test_few_speakers_have_no_others_lane():
    result = _result({"SPEAKER_00": 5, "SPEAKER_01": 8})
    timeline = timeline_from_result(result, SpeakerView(result, PALETTE))

    assert [lane.label for lane in timeline.lanes] == ["SPEAKER_01", "SPEAKER_00"]
    assert [lane.color for lane in timeline.lanes] == ["#ff0000", "#0000ff"]


def test_rename_updates_lanes_and_keeps_colours():
    result = _result({"SPEAKER_00": 5, "SPEAKER_01": 8, "SPEAKER_02": 3})
    view = SpeakerView(result, PALETTE)
    before = timeline_from_result(result, view)

    view.rename("SPEAKER_00", "Anna")
    after = timeline_from_result(result, view)

    assert [lane.label for lane in after.lanes] == ["SPEAKER_01", "Anna", "SPEAKER_02"]
    assert [lane.color for lane in after.lanes] == [lane.color for lane in before.lanes]
    assert after.lanes[1].runs == before.lanes[1].runs


def test_merge_by_rename_joins_the_lanes():
    result = _result({"SPEAKER_00": 5, "SPEAKER_01": 8, "SPEAKER_02": 3})
    view = SpeakerView(result, PALETTE)

    view.rename("SPEAKER_02", "SPEAKER_01")
    timeline = timeline_from_result(result, view)

    assert [lane.label for lane in timeline.lanes] == ["SPEAKER_01", "SPEAKER_00"]
    assert [(r.start, r.end) for r in timeline.lanes[0].runs] == [(5.0, 13.0), (13.0, 16.0)]
    assert {r.color for r in timeline.lanes[0].runs} == {"#ff0000"}


def test_without_speakers_one_grey_speech_lane_with_note():
    result = {"segments": [{"start": 1.0, "end": 4.0, "text": "hi"}], "duration": 10.0}
    timeline = timeline_from_result(result, SpeakerView(result, PALETTE), speech_color="#888888",
                                    note="Speaker identification was skipped")

    assert len(timeline.lanes) == 1
    assert [(r.start, r.end, r.color) for r in timeline.lanes[0].runs] == [(1.0, 4.0, "#888888")]
    assert timeline.note == "Speaker identification was skipped"


def test_duration_falls_back_to_the_last_segment_end():
    result = _result({"SPEAKER_00": 5})
    del result["duration"]

    assert timeline_from_result(result, SpeakerView(result, PALETTE)).duration == 5.0


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_widget_paints_each_run_in_its_lane(app):
    from PySide6.QtGui import QColor
    from qt_app.widgets.timeline import Timeline

    result = _result({"SPEAKER_00": 10, "SPEAKER_01": 30})  # 01: 10..40 s, lane 0
    widget = Timeline()
    widget.resize(800, widget.height())
    widget.set_timeline(timeline_from_result(result, SpeakerView(result, PALETTE)))
    image = widget.grab().toImage()

    lane0, lane1 = widget.lane_rect(0), widget.lane_rect(1)
    x_at = widget.x_of
    assert QColor(image.pixel(int(x_at(25)), int(lane0.center().y()))) == QColor("#ff0000")
    assert QColor(image.pixel(int(x_at(5)), int(lane1.center().y()))) == QColor("#0000ff")
    assert QColor(image.pixel(int(x_at(5)), int(lane0.center().y()))) != QColor("#ff0000")
    assert widget.seconds_at(x_at(25)) == pytest.approx(25, abs=0.1)


def test_widget_paints_the_plain_sweep_up_to_its_fraction(app):
    from PySide6.QtGui import QColor
    from qt_app.timeline_model import SPEECH, Lane, TimelineModel
    from qt_app.widgets.timeline import Timeline

    widget = Timeline()
    widget.resize(800, widget.height())
    widget.set_timeline(TimelineModel(100.0, (Lane(SPEECH, "#888888", ()),), sweep=0.5, percent=20))
    image = widget.grab().toImage()

    y = int(widget.lane_rect(0).center().y())
    swept, ahead = QColor(image.pixel(int(widget.x_of(20)), y)), QColor(image.pixel(int(widget.x_of(80)), y))
    assert swept != ahead


def test_widget_draws_the_playhead_and_a_click_asks_to_seek_there(app):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QColor
    from PySide6.QtTest import QTest
    from qt_app import theme
    from qt_app.widgets.timeline import Timeline

    result = _result({"SPEAKER_00": 10, "SPEAKER_01": 30})
    widget = Timeline()
    widget.resize(800, widget.height())
    widget.set_timeline(timeline_from_result(result, SpeakerView(result, PALETTE)))
    seeks = []
    widget.seekRequested.connect(seeks.append)

    QTest.mouseClick(widget, Qt.LeftButton, pos=QPointF(widget.x_of(25), 10).toPoint())
    assert seeks == []  # not before the result is in (set_seekable)

    widget.set_seekable(True)
    widget.set_playhead(25)
    image = widget.grab().toImage()
    assert QColor(image.pixel(int(widget.x_of(25)), int(widget.lane_rect(0).center().y()))) == QColor(theme.TEXT)

    QTest.mouseClick(widget, Qt.LeftButton, pos=QPointF(widget.x_of(30), 10).toPoint())
    assert seeks == [pytest.approx(30, abs=0.1)]
