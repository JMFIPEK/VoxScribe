"""Step names and the "x realtime" line under the Timeline (spec #40, #49).
Pure Python, no Qt (tests/test_step_timing.py). The run itself shows a
plain progress bar with the running step's name; an earlier version drew
the progress into the Timeline and was dropped after trying it in the GUI."""

from __future__ import annotations

from voxscribe.transcription.backend import (
    STEP_ALIGNMENT, STEP_SPEAKER_IDENTIFICATION, STEP_TRANSCRIPTION)

STEP_NAMES = {
    STEP_TRANSCRIPTION: "Transcription",
    STEP_ALIGNMENT: "Alignment",
    STEP_SPEAKER_IDENTIFICATION: "Speaker identification",
}


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
