"""The Timeline as the loading bar (spec #40, #49): turns the pipeline's step
events (transcribe()'s on_event, re-emitted by TranscribeController) into the
TimelineModel to draw while it runs, plus the "x realtime" line under the
Timeline. Pure Python, no Qt, so it is tested without a display
(tests/test_timeline_progress.py)."""

from __future__ import annotations

from dataclasses import replace

from qt_app.speaker_view import SpeakerView
from qt_app.timeline_model import SPEECH, Lane, Run, TimelineModel, timeline_from_result

# transcribe()'s progress ranges: transcription covers 0-40% of on_progress.
TRANSCRIPTION_SHARE = 0.40
# Words closer than this (seconds) are drawn as one run after alignment.
WORD_GAP = 0.3

STEP_NAMES = {
    "transcription": "Transcription",
    "alignment": "Alignment",
    "speaker_identification": "Speaker identification",
}


class TimelineProgress:
    """One run of the pipeline, fed its events in order; timeline() is what
    to draw right now."""

    def __init__(self, duration: float | None, speech_color: str, palette: list[str] = ()):
        self._duration = duration or 0.0
        self._speech_color = speech_color
        self._palette = list(palette) or [speech_color]
        self._step = None
        self._start_over()

    def _start_over(self):
        self._fraction = 0.0
        self._spans: list[tuple[float, float]] = []
        self._aligned = False
        self._turns: list[dict] | None = None

    def set_duration(self, seconds: float):
        """The recording length, once known (TranscribeController.durationKnown)."""
        self._duration = seconds or 0.0

    def step(self, name: str):
        """A step starts. A second "transcription" means the Provider failed
        and the local fallback starts over: what it streamed is dropped."""
        if name == "transcription" and self._step is not None:
            self._start_over()
        self._step = name

    def progress(self, fraction: float):
        self._fraction = fraction

    def segments(self, segments: list[dict]):
        """Text segments streamed by the Backend (Providers, Parakeet)."""
        self._spans.extend(_span(seg) for seg in segments)

    def aligned(self, segments: list[dict]):
        """The authoritative segments with word timestamps: the lane goes
        from segment blocks to word runs."""
        self._spans = [span for seg in segments for span in _word_runs(seg)]
        self._aligned = True

    def speaker_turns(self, segments: list[dict]):
        """The final segments with "speaker": the grey lane splits into the
        coloured Speaker lanes."""
        self._turns = segments

    def timeline(self) -> TimelineModel:
        percent = round(self._fraction * 100)
        if self._turns:
            result = {"segments": self._turns, "duration": self._duration}
            view = SpeakerView(result, self._palette)
            if view.speakers():
                return replace(timeline_from_result(result, view, self._speech_color),
                               percent=percent)
        sweep = None
        if not self._spans and not self._aligned:
            sweep = min(1.0, self._fraction / TRANSCRIPTION_SHARE)
        runs = tuple(Run(start, end, self._speech_color) for start, end in sorted(self._spans))
        duration = self._duration or max((end for _, end in self._spans), default=0.0)
        return TimelineModel(duration, (Lane(SPEECH, self._speech_color, runs),),
                             sweep=sweep, percent=percent, soft=not self._aligned)


def _span(seg: dict) -> tuple[float, float]:
    return float(seg.get("start", 0.0)), float(seg.get("end", 0.0))


def _word_runs(seg: dict) -> list[tuple[float, float]]:
    """A segment's words as runs: words less than WORD_GAP apart join one
    run. Without word timestamps the segment stays one block."""
    words = [w for w in seg.get("words", []) if "start" in w and "end" in w]
    if not words:
        return [_span(seg)]
    runs: list[tuple[float, float]] = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if runs and start - runs[-1][1] < WORD_GAP:
            runs[-1] = (runs[-1][0], max(runs[-1][1], end))
        else:
            runs.append((start, end))
    return runs


def realtime_line(result: dict) -> str:
    """ "Transcription 12× · Alignment 48× · Total 3.0× realtime": audio
    duration / wall time, per step that ran and in total. Empty without
    timings."""
    duration = float(result.get("duration") or 0.0)
    step_times = result.get("step_times") or {}
    if duration <= 0 or not result.get("total_time"):
        return ""
    parts = [f"{STEP_NAMES[step]} {_factor(duration, step_times[step])}"
             for step in STEP_NAMES if step_times.get(step)]
    parts.append(f"Total {_factor(duration, result['total_time'])} realtime")
    return " · ".join(parts)


def _factor(duration: float, seconds: float) -> str:
    factor = duration / seconds
    return f"{factor:.1f}×" if factor < 10 else f"{factor:.0f}×"
