"""The transcript as one block per speaker turn (spec #40, #46): coloured
bar, name, start time, text. Pure Python, no Qt: builds the HTML that the
Transcription tab's read-only QTextEdit shows, rendered completely from
the result on every change (one table row per turn, not one widget, so
1,000+ turns stay fast)."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from html import escape

from qt_app import theme
from qt_app.constants import format_time_short
from qt_app.speaker_view import SpeakerView


@dataclass(frozen=True)
class Turn:
    """Consecutive segments of the same Speaker label."""
    speaker: str             # "" if diarization gave none
    start: float
    end: float
    text: str


def speaker_turns(segments: list[dict]) -> list[Turn]:
    turns: list[Turn] = []
    for seg in segments:
        speaker = seg.get("speaker") or ""
        text = (seg.get("text") or "").strip()
        start, end = seg.get("start", 0), seg.get("end", 0)
        if turns and turns[-1].speaker == speaker:
            last = turns[-1]
            turns[-1] = Turn(speaker, last.start, end, f"{last.text} {text}".strip())
        else:
            turns.append(Turn(speaker, start, end, text))
    return turns


def turn_at(turns: list[Turn], seconds: float) -> int | None:
    """Index of the turn being spoken at `seconds` (playback highlighting).
    In a pause it is the turn before it, so the highlight doesn't flicker
    off between turns; None before the first turn. Turns are in time order."""
    starts = [turn.start for turn in turns]
    index = bisect_right(starts, seconds) - 1
    return index if index >= 0 else None


def longest_turn(turns: list[Turn], speaker: str) -> Turn | None:
    """The turn the speaker list's play button plays: the speaker's longest,
    the earliest of equally long ones. None if the speaker never talks."""
    own = [turn for turn in turns if turn.speaker == speaker]
    return max(own, key=lambda turn: turn.end - turn.start, default=None)


# transcript_html() gives each turn two table rows: the block, then the gap
# below it. Clicking and highlighting map between turns and rows with these.
_ROWS_PER_TURN = 2


def row_of_turn(index: int) -> int:
    return index * _ROWS_PER_TURN


def turn_of_row(row: int) -> int:
    return row // _ROWS_PER_TURN


def transcript_html(result: dict, view: SpeakerView) -> str:
    rows = []
    for turn in speaker_turns(result.get("segments", [])):
        color = view.color_of(turn.speaker) if turn.speaker else None
        bar = color or theme.TEXT_MUTED
        name = (f'<span style="color:{color or theme.TEXT}; font-weight:600;">'
                f'{escape(turn.speaker)}</span>&nbsp;&nbsp;') if turn.speaker else ""
        rows.append(
            f'<tr><td width="4" bgcolor="{bar}"></td><td width="8"></td>'
            f'<td>{name}<span style="color:{theme.TEXT_MUTED};">{format_time_short(turn.start)}</span>'
            f'<br><span style="color:{theme.TEXT};">{escape(turn.text)}</span></td></tr>'
            f'<tr><td colspan="3" height="8"></td></tr>')
    return f'<table width="100%" cellspacing="0" cellpadding="0">{"".join(rows)}</table>'


def transcript_text(result: dict) -> str:
    """The same turns as plain text, for Copy."""
    return "\n\n".join(
        f"[{format_time_short(t.start)}] " + (f"{t.speaker}: " if t.speaker else "") + t.text
        for t in speaker_turns(result.get("segments", [])))