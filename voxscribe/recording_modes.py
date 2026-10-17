"""Recording modes and Call setups (see GLOSSARY.md), plus the device-name
heuristics that pick a sensible Setup and microphone for a playback device.

Stdlib only - imported by the GUI's Record page, the CLI and recorder.py.

- Recording mode: what a recording captures - "playback" (Playback audio
  only), "call" (Playback + Microphone audio as separate tracks) or "room"
  (Microphone audio only).
- Setup (Call only): "headset" (headphones + their own mic, clean tracks) or
  "loudspeaker" (speakers + a room mic/mic array/speakerphone, so remote
  voices also reach the mic).
"""

from __future__ import annotations

import re

MODE_PLAYBACK = "playback"
MODE_CALL = "call"
MODE_ROOM = "room"
MODES = (MODE_PLAYBACK, MODE_CALL, MODE_ROOM)

SETUP_HEADSET = "headset"
SETUP_LOUDSPEAKER = "loudspeaker"
SETUPS = (SETUP_HEADSET, SETUP_LOUDSPEAKER)

# AudioRecorder's internal source names, which predate the modes.
_SOURCE_BY_MODE = {MODE_PLAYBACK: "system", MODE_CALL: "both", MODE_ROOM: "mic"}
_MODE_BY_SOURCE = {source: mode for mode, source in _SOURCE_BY_MODE.items()}

# A device whose physical name contains one of these is a speakerphone or
# loudspeaker even if mic and output share one physical device (e.g.
# "Jabra Speak 510" - which the same-device rule would take for a headset).
_LOUDSPEAKER_DEVICE_HINTS = ("speak", "lautsprecher")
# A microphone named like this sits in the room, not at the user's mouth
# ("Mikrofonarray (Intel Smart Sound ...)", "Microphone Array (Realtek ...)").
_ROOM_MIC_HINTS = ("array",)


def source_for_mode(mode: str) -> str:
    """'call' -> 'both' (the source name AudioRecorder expects)."""
    return _SOURCE_BY_MODE[mode]


def mode_for_source(source: str) -> str:
    """'both' -> 'call' - for the CLI's deprecated --source alias."""
    return _MODE_BY_SOURCE[source]


def normalize_device_name(name: str) -> str:
    return " ".join(name.casefold().split())


def physical_device_key(name: str) -> str:
    """'Mikrofon (Razer BlackShark V2 Pro 2.4)' and
    'Lautsprecher (Razer BlackShark V2 Pro 2.4) [Loopback]' -> 'razer blackshark v2 pro 2.4'.

    Windows names a headset's input and output endpoints "<role> (<device>)";
    the part in parentheses identifies the physical device (minus the
    "2- " instance prefix Windows adds for repeated devices)."""
    name = name.replace("[Loopback]", "").strip()
    start, end = name.rfind("("), name.rfind(")")
    if 0 <= start < end:
        inner = name[start + 1:end]
    elif start >= 0:
        inner = name[start + 1:]  # MME-truncated before the closing ")"
    else:
        inner = name
    inner = re.sub(r"^\d+-\s*", "", inner.strip())
    return normalize_device_name(inner)


def suggest_mic_for_loopback(microphones: list[dict], loopback_name: str | None) -> dict | None:
    """For a Call: the microphone belonging to the same physical device as
    the playback device (e.g. a headset's own mic), or None if there's no
    such match.

    Windows' default *recording* device is independent of its default
    *playback* device - with two headsets connected it's easy to listen on
    one while the other one's (unworn) mic is the default input, which then
    records only silence for your own voice."""
    for mic in microphones:
        if mic_matches_playback(mic["name"], loopback_name):
            return mic
    return None


def mic_matches_playback(mic_name: str, loopback_name: str | None) -> bool:
    """True if the microphone is part of the same physical device as the
    playback device."""
    if not loopback_name:
        return False
    key = physical_device_key(loopback_name)
    mic_key = physical_device_key(mic_name)
    # MME truncates names to 31 chars, hence prefix matching.
    return bool(key and mic_key and (mic_key == key or (len(mic_key) >= 8 and key.startswith(mic_key))))


def guess_setup(microphones: list[dict], loopback_name: str | None) -> tuple[str, dict | None]:
    """Preselects the Setup for a Call and the microphone to go with it.

    Headset if a microphone belongs to the same physical device as the
    playback device - unless that device is a speakerphone/loudspeaker or the
    mic is a room mic array. Otherwise Loudspeaker, with a room mic array if
    there is one (else None: the caller keeps the default input, which may
    well be an unrelated headset's mic)."""
    match = suggest_mic_for_loopback(microphones, loopback_name)
    if match is None:
        room_mic = next((m for m in microphones if _is_room_mic(m["name"])), None)
        return SETUP_LOUDSPEAKER, room_mic
    device_key = physical_device_key(loopback_name)
    if any(hint in device_key for hint in _LOUDSPEAKER_DEVICE_HINTS):
        return SETUP_LOUDSPEAKER, match
    if _is_room_mic(match["name"]):
        return SETUP_LOUDSPEAKER, match
    return SETUP_HEADSET, match


def _is_room_mic(name: str) -> bool:
    return any(hint in normalize_device_name(name) for hint in _ROOM_MIC_HINTS)
