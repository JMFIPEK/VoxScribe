"""The transcript as one block per speaker turn (spec #40, #46): coloured
bar, name, start time, text. Pure Python, no Qt: builds the HTML that the
Transcription tab's read-only QTextEdit shows, rendered completely from
the result on every change (one table row per turn, not one widget, so
1,000+ turns stay fast)."""

from __future__ import annotations

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