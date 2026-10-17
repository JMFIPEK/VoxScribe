"""Transcript output formats (TXT, SRT, JSON)."""

import json
import os


def format_transcript(result: dict, include_speakers: bool = True) -> str:
    """Formats the transcript as readable text with timestamps."""
    lines = []
    for seg in result.get("segments", []):
        start = _format_time(seg.get("start", 0))
        end = _format_time(seg.get("end", 0))
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")

        if include_speakers and speaker:
            lines.append(f"[{start} - {end}] {speaker}: {text}")
        else:
            lines.append(f"[{start} - {end}] {text}")

    return "\n".join(lines)


def save_transcript(result: dict, output_path: str,
                    formats: list[str] | None = None):
    """Saves the transcript in various formats.

    Args:
        result: WhisperX result dict
        output_path: Base path without extension (e.g. "recordings/meeting")
        formats: List of formats ("txt", "srt", "json"). Default: ["txt"]
    """
    if formats is None:
        formats = ["txt"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    saved = []

    for fmt in formats:
        path = f"{output_path}.{fmt}"
        if fmt == "txt":
            content = format_transcript(result)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        elif fmt == "srt":
            content = _to_srt(result)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        else:
            print(f"  Unknown format: {fmt}")
            continue
        saved.append(path)
        print(f"  Saved: {path}")

    return saved


def _format_time(seconds: float) -> str:
    """Formats seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_time_srt(seconds: float) -> str:
    """Formats seconds in SRT format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _to_srt(result: dict) -> str:
    """Converts a WhisperX result to SRT format."""
    lines = []
    for i, seg in enumerate(result.get("segments", []), 1):
        start = _format_time_srt(seg.get("start", 0))
        end = _format_time_srt(seg.get("end", 0))
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)
