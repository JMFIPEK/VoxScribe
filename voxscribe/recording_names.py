"""File names of recordings: `YYMMDD_HHMM_<Recording title>.wav` (see GLOSSARY.md).

Stdlib only - imported by the GUI's Record page and the CLI.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

from voxscribe.recording_modes import MODE_CALL, MODE_PLAYBACK, MODE_ROOM

_DEFAULT_TITLE_BY_MODE = {MODE_PLAYBACK: "Playback", MODE_CALL: "Meeting", MODE_ROOM: "Room"}
# Not allowed in a Windows file name; control characters are dropped too.
_FORBIDDEN = set('\\/:*?"<>|')
MAX_TITLE_LENGTH = 60
# The recording and its Call sidecar (tracks.TRACKS_SUFFIX - tracks.py needs
# numpy, so it isn't imported here).
_EXTENSIONS = (".wav", ".tracks.flac")


def default_title(mode: str) -> str:
    """The Recording title of a recording without one: Playback, Meeting
    (a Call) or Room."""
    return _DEFAULT_TITLE_BY_MODE[mode]


def clean_title(title: str | None, mode: str) -> str:
    """The title as it goes into a file name; nothing left -> the mode's
    default."""
    # Whitespace control characters (tab, newline) count as spaces.
    kept = "".join(c for c in title or ""
                   if c not in _FORBIDDEN and (c.isspace() or c.isprintable()))
    cleaned = re.sub(r"\s+", "_", kept.strip())
    # Windows drops a trailing dot from a file name itself.
    cleaned = cleaned[:MAX_TITLE_LENGTH].rstrip("_.")
    return cleaned or default_title(mode)


def recording_stem(title: str | None, mode: str, started: datetime) -> str:
    """File name without extension: the minute Start was pressed, then the
    Recording title."""
    return f"{started:%y%m%d_%H%M}_{clean_title(title, mode)}"


def recording_path(directory: str | os.PathLike, title: str | None, mode: str,
                   started: datetime) -> str:
    """Path of the `.wav` to save the recording to. A stem that is taken by a
    `.wav` or a Call's `.tracks.flac` gets `_2`, `_3`, ..., so a Call's two
    files always share one stem and a recording in the same minute never
    overwrites another."""
    stem = recording_stem(title, mode, started)
    candidate, number = stem, 1
    while any(os.path.exists(os.path.join(directory, candidate + ext)) for ext in _EXTENSIONS):
        number += 1
        candidate = f"{stem}_{number}"
    return os.path.join(directory, candidate + ".wav")
