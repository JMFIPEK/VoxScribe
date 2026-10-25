"""Tests for qt_app.transcript_view: the transcript as one block per speaker
turn (spec #40, #46), on synthetic result dicts."""

from qt_app.speaker_view import SpeakerView
from qt_app.transcript_view import (
    longest_turn,
    row_of_turn,
    speaker_turns,
    transcript_html,
    transcript_text,
    turn_at,
    turn_of_row,
)

PALETTE = ["#ff0000", "#0000ff", "#00ff00"]


def _seg(start, end, speaker, text):
    return {"start": start, "end": end, "text": f" {text} ", "speaker": speaker}


def _result():
    segments = [
        _seg(0, 2, "SPEAKER_00", "Hello."),
        _seg(2, 5, "SPEAKER_00", "How are you?"),
        _seg(5, 9, "SPEAKER_01", "Fine, thanks."),
        _seg(9, 10, "SPEAKER_00", "Good."),
    ]
    return {"segments": segments, "speaker_id_map": {"SPEAKER_00": "SPEAKER_00",
                                                      "SPEAKER_01": "SPEAKER_01"}}


def test_consecutive_segments_of_one_speaker_are_one_turn():
    turns = speaker_turns(_result()["segments"])

    assert [(t.speaker, t.start, t.end, t.text) for t in turns] == [
        ("SPEAKER_00", 0, 5, "Hello. How are you?"),
        ("SPEAKER_01", 5, 9, "Fine, thanks."),
        ("SPEAKER_00", 9, 10, "Good."),
    ]


def test_a_merge_by_rename_joins_neighbouring_turns():
    result = _result()
    view = SpeakerView(result, PALETTE)
    view.rename("SPEAKER_01", "SPEAKER_00")

    assert [t.text for t in speaker_turns(result["segments"])] == [
        "Hello. How are you? Fine, thanks. Good."]


def test_html_shows_name_in_its_colour_start_time_and_escaped_text():
    result = _result()
    result["segments"][2]["text"] = "a <b> & c"
    view = SpeakerView(result, PALETTE)
    view.rename("SPEAKER_01", "Anna")

    html = transcript_html(result, view)

    assert "Anna" in html and view.color_of("Anna") in html
    assert "00:05" in html
    assert "a &lt;b&gt; &amp; c" in html


def test_segments_without_speaker_still_show_their_text():
    result = {"segments": [{"start": 0, "end": 1, "text": "Just text"}]}

    html = transcript_html(result, SpeakerView(result, PALETTE))

    assert "Just text" in html


def test_plain_text_has_one_paragraph_per_turn():
    assert transcript_text(_result()) == (
        "[00:00] SPEAKER_00: Hello. How are you?\n\n"
        "[00:05] SPEAKER_01: Fine, thanks.\n\n"
        "[00:09] SPEAKER_00: Good.")

def test_the_turn_at_a_time_is_the_one_being_spoken():
    turns = speaker_turns(_result()["segments"])

    assert turn_at(turns, 0) == 0
    assert turn_at(turns, 4.9) == 0
    assert turn_at(turns, 5) == 1
    assert turn_at(turns, 9.5) == 2


def test_a_pause_keeps_the_turn_before_it_and_before_the_first_there_is_none():
    turns = speaker_turns([_seg(2, 4, "SPEAKER_00", "Hi."), _seg(6, 8, "SPEAKER_01", "Hey.")])

    assert turn_at(turns, 1) is None
    assert turn_at(turns, 5) == 0
    assert turn_at(turns, 30) == 1


def test_a_speakers_longest_turn_is_the_one_to_play():
    turns = speaker_turns(_result()["segments"])

    assert (longest_turn(turns, "SPEAKER_00").start, longest_turn(turns, "SPEAKER_00").end) == (0, 5)
    assert longest_turn(turns, "SPEAKER_01").start == 5
    assert longest_turn(turns, "Nobody") is None


def test_each_turn_has_its_own_table_row_in_the_rendered_document():
    from PySide6.QtGui import QTextDocument
    from PySide6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([])
    result = _result()
    document = QTextDocument()
    document.setHtml(transcript_html(result, SpeakerView(result, PALETTE)))
    table = document.rootFrame().childFrames()[0]

    def text_of_row(row):
        cell = table.cellAt(row, 2)
        cursor = cell.firstCursorPosition()
        cursor.setPosition(cell.lastCursorPosition().position(), cursor.MoveMode.KeepAnchor)
        return cursor.selectedText()

    assert "Fine, thanks." in text_of_row(row_of_turn(1))
    assert turn_of_row(row_of_turn(2)) == 2
    assert turn_of_row(row_of_turn(2) + 1) == 2  # the gap below a turn belongs to it
