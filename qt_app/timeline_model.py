"""What the Timeline shows (spec #40, #46): its Speaker lanes and the
coloured runs in each, as plain data. Pure Python, no Qt, so the lane
mapping is tested without a display (tests/test_timeline.py) and the
widget (qt_app/widgets/timeline.py) only paints what it is given - a
finished result now, the pipeline's progress later."""

from __future__ import annotations

from dataclasses import dataclass

from qt_app.speaker_view import SpeakerView

OTHERS = "Others"
SPEECH = "Speech"


@dataclass(frozen=True)
class Run:
    """A stretch of speech drawn in one colour."""
    start: float
    end: float
    color: str


@dataclass(frozen=True)
class Lane:
    label: str               # Speaker label, "Others" or "Speech"
    color: str | None        # the speaker's colour; None for Others
    runs: tuple[Run, ...]


@dataclass(frozen=True)
class TimelineModel:
    duration: float          # x axis: the recording length in seconds
    lanes: tuple[Lane, ...]
    note: str = ""           # e.g. why there are no Speaker lanes
    # While the pipeline runs (qt_app/timeline_progress.py); a finished
    # result leaves them at their defaults:
    sweep: float | None = None   # plain sweep 0..1 (Backends that don't stream)
    percent: int | None = None   # overall pipeline percentage
    soft: bool = False           # segment blocks, before alignment sharpens them


def timeline_from_result(result: dict, view: SpeakerView, speech_color: str = "#8e9aaf",
                         note: str = "") -> TimelineModel:
    """The finished result's Timeline: one lane each for the top speakers
    (SpeakerView.lanes()), then an Others lane in which everyone keeps their
    own colour. A result without speakers gives one grey Speech lane."""
    segments = sorted(result.get("segments", []), key=lambda s: float(s.get("start", 0.0)))
    top, others = view.lanes()

    if not top:
        runs = tuple(Run(*_span(seg), speech_color) for seg in segments)
        return TimelineModel(_duration(result, segments), (Lane(SPEECH, speech_color, runs),), note)

    def runs_of(labels):
        return tuple(Run(*_span(seg), view.color_of(seg["speaker"]))
                     for seg in segments if seg.get("speaker") in labels)

    lanes = [Lane(s.label, s.color, runs_of({s.label})) for s in top]
    if others:
        lanes.append(Lane(OTHERS, None, runs_of({s.label for s in others})))
    return TimelineModel(_duration(result, segments), tuple(lanes), note)


def _span(seg: dict) -> tuple[float, float]:
    return float(seg.get("start", 0.0)), float(seg.get("end", 0.0))


def _duration(result: dict, segments: list[dict]) -> float:
    if result.get("duration"):
        return float(result["duration"])
    return max((float(s.get("end", 0.0)) for s in segments), default=0.0)
