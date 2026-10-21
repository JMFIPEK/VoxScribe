"""Separate microphone/system tracks for "microphone + system" recordings.

The recorder still saves the usual mono mix (`<name>.wav`, which everything
else consumes unchanged), plus a 2-channel sidecar `<name>.tracks.flac`
(channel 0 = microphone, channel 1 = system audio, 16kHz). transcribe() uses
the sidecar - if present - as a hard constraint for speaker diarization: a
person speaks either into the local microphone or comes out of the speakers,
never both. Diarization clusters that clearly span both sides (pyannote
merged a local and a remote voice) are split apart again.

The sidecar's FLAC comment tag records the Call's Setup ("mode=call;
setup=headset" or "...setup=loudspeaker", see recording_modes.py); sidecars
recorded before the tag existed have no Setup (load_setup() returns None).
With a Loudspeaker Setup the microphone track also holds the Playback audio
as Echo, so load_tracks() returns it cleaned (echo.py, docs/adr/0003); the
file itself stays untouched.

Deliberately numpy/soundfile-only (no pyaudio, no torch), so both
recorder.py and transcriber.py can import it cheaply.
"""

import os
import time

import numpy as np
import soundfile as sf

from voxscribe import echo
from voxscribe.recording_modes import SETUP_LOUDSPEAKER

SAMPLE_RATE = 16000
TRACKS_SUFFIX = ".tracks.flac"

MIC = "mic"
SYSTEM = "system"
# How the two tracks are named to the user (GLOSSARY: Microphone, Playback audio).
TRACK_NAMES = {MIC: "Microphone", SYSTEM: "Playback"}

# Energy analysis frame length (seconds).
_FRAME_SECONDS = 0.02
# A segment counts as clearly mic- or system-side if the normalized level
# difference between both tracks exceeds this many dB.
_SIDE_MARGIN_DB = 6.0
# A diarization speaker is only split if its "wrong side" share is
# substantial - otherwise single misclassified segments (double-talk,
# loudspeaker echo, laughter) would create spurious extra speakers.
_SPLIT_MIN_SECONDS = 8.0
_SPLIT_MIN_FRACTION = 0.10
# Inside one segment, a run of words on one side shorter than this is not a
# speaker change (a cough, a short echo, an ambiguous word or two).
_MIN_RUN_SECONDS = 1.0


def tracks_path_for(audio_path: str) -> str:
    """'recordings/x.wav' -> 'recordings/x.tracks.flac'."""
    return os.path.splitext(audio_path)[0] + TRACKS_SUFFIX


def save_tracks(audio_path: str, mic_audio: np.ndarray, sys_audio: np.ndarray,
                setup: str | None = None) -> str:
    """Saves both tracks (int16 value range as float, same sample rate
    SAMPLE_RATE, as produced by track_assembler) next to the mix,
    tagged with the Call's Setup if given."""
    n = max(len(mic_audio), len(sys_audio))
    stereo = np.zeros((n, 2), dtype=np.int16)
    stereo[:len(mic_audio), 0] = np.clip(mic_audio, -32768, 32767).astype(np.int16)
    stereo[:len(sys_audio), 1] = np.clip(sys_audio, -32768, 32767).astype(np.int16)
    path = tracks_path_for(audio_path)
    with sf.SoundFile(path, "w", SAMPLE_RATE, 2, "PCM_16", format="FLAC") as f:
        f.comment = f"mode=call;setup={setup}" if setup else "mode=call"
        f.write(stereo)
    return path


def mix(mic: np.ndarray, system: np.ndarray, full_scale: float) -> np.ndarray:
    """The mono mix of a Call: both tracks at 0.65, the shorter one padded
    with silence, scaled down if the peak would exceed full_scale (32767
    for int16 values, 32767/32768 for -1..1). Shared by the recorder's
    saved mix and the cleaned mix transcribed for Loudspeaker Calls, so
    the two stay the same mix. Returns float64."""
    n = max(len(mic), len(system))
    mixed = np.zeros(n, dtype=np.float64)
    mixed[:len(mic)] += 0.65 * np.asarray(mic, dtype=np.float64)
    mixed[:len(system)] += 0.65 * np.asarray(system, dtype=np.float64)
    peak = float(np.max(np.abs(mixed))) if n else 0.0
    if peak > full_scale:
        mixed *= full_scale / peak
    return mixed


# Balancing the two tracks before they're mixed for transcription (#39): a
# noise-suppressed or low-gain Microphone can sit 20+ dB below the Playback,
# and the local speaker then all but vanishes from the transcript.
_BALANCE_MIN_DB = 12.0   # closer than this: mix as recorded (normal Calls differ by ~10 dB)
_BALANCE_MAX_GAIN_DB = 20.0  # never boost more, or a near-silent track becomes noise
_SILENT_DB = -80.0       # a track whose speech level is below this has no speech


_SPEECH_ABOVE_FLOOR_DB = 15.0  # a frame this far above the track's noise floor is speech
_MIN_SPEECH_FRAMES = 25        # 0.5 s of speech frames, else the track counts as speechless


def speech_level_db(audio: np.ndarray) -> float:
    """The track's typical speech level in dBFS (-1..1 samples): the median
    of its speech frames, i.e. 20 ms frames at least _SPEECH_ABOVE_FLOOR_DB
    above its noise floor (10th percentile). -200 if it has less than
    _MIN_SPEECH_FRAMES of them - silence, steady noise or a steady signal
    like music, none of which may be boosted. A percentile of all frames
    would measure the noise of a rarely speaking person instead."""
    hop = int(_FRAME_SECONDS * SAMPLE_RATE)
    n = len(audio) // hop
    if n == 0:
        return -200.0
    frames = np.asarray(audio[:n * hop], dtype=np.float64).reshape(n, hop)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)
    speech = db[db >= np.percentile(db, 10) + _SPEECH_ABOVE_FLOOR_DB]
    return float(np.median(speech)) if len(speech) >= _MIN_SPEECH_FRAMES else -200.0


def balance(mic: np.ndarray, system: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], str, float] | None:
    """(mic, system) with the quieter track boosted towards the louder one's
    speech level, or None if they're already within _BALANCE_MIN_DB (or one
    has no speech at all). The boost is capped at _BALANCE_MAX_GAIN_DB.
    Returns ((mic, system), boosted_track_name, gain_db)."""
    mic_db, sys_db = speech_level_db(mic), speech_level_db(system)
    if min(mic_db, sys_db) < _SILENT_DB or abs(mic_db - sys_db) < _BALANCE_MIN_DB:
        return None
    gain_db = min(abs(mic_db - sys_db), _BALANCE_MAX_GAIN_DB)
    gain = 10 ** (gain_db / 20)

    def boost(track):
        # Clipped at full scale: one boosted click would otherwise make mix()
        # scale the whole mix down and undo most of the balancing.
        return np.clip(np.asarray(track, dtype=np.float64) * gain, -1.0, 1.0)

    if mic_db < sys_db:
        return (boost(mic), system), MIC, gain_db
    return (mic, boost(system)), SYSTEM, gain_db


def load_setup(audio_path: str) -> str | None:
    """The Setup stored in the recording's sidecar, or None if there's no
    sidecar or it predates the tag."""
    path = tracks_path_for(audio_path)
    if not os.path.isfile(path):
        return None
    with sf.SoundFile(path) as f:
        comment = f.comment or ""
    fields = dict(part.split("=", 1) for part in comment.split(";") if "=" in part)
    return fields.get("setup") or None


def load_tracks(audio_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (mic, system) as float32 arrays, or None if there's no sidecar.
    For a Loudspeaker Setup the mic track comes with the Echo removed."""
    path = tracks_path_for(audio_path)
    if not os.path.isfile(path):
        return None
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE or data.shape[1] != 2:
        return None
    mic, system = data[:, 0], data[:, 1]
    if load_setup(audio_path) == SETUP_LOUDSPEAKER:
        print("     Loudspeaker Setup: removing the Echo from the microphone track...")
        t0 = time.monotonic()
        mic = echo.cancel_echo(mic, system)
        print(f"     Echo removal took {time.monotonic() - t0:.1f}s "
              f"for {len(mic) / SAMPLE_RATE:.0f}s of audio")
    return mic, system


def _frame_db(audio: np.ndarray) -> np.ndarray:
    """Per-frame RMS level in dB, normalized so 0 dB = the track's typical
    speech level (95th percentile) and clamped at its noise floor (10th
    percentile) - makes both tracks comparable regardless of mic gain or
    playback volume."""
    hop = int(_FRAME_SECONDS * SAMPLE_RATE)
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = audio[:n * hop].reshape(n, hop).astype(np.float64)
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    floor = np.percentile(db, 10)
    ref = np.percentile(db, 95)
    if ref - floor < 1e-3:
        # Silent track (e.g. muted mic for the entire meeting).
        return np.full(n, -100.0)
    return np.maximum(db, floor) - ref


def _segment_side(mic_db: np.ndarray, sys_db: np.ndarray, start: float, end: float) -> str | None:
    a = int(start / _FRAME_SECONDS)
    b = max(a + 1, int(end / _FRAME_SECONDS))
    m, s = mic_db[a:b], sys_db[a:b]
    n = min(len(m), len(s))
    if n == 0:
        return None
    diff = float(np.mean(m[:n] - s[:n]))
    if diff > _SIDE_MARGIN_DB:
        return MIC
    if diff < -_SIDE_MARGIN_DB:
        return SYSTEM
    return None


def _word_runs(seg: dict, mic_db: np.ndarray, sys_db: np.ndarray) -> list[tuple[list[dict], str]]:
    """The segment's words grouped into runs on one side each, [(words,
    side)], or [] if the words don't show a side change.

    Long segments (Apple SpeechAnalyzer returns 10-20 s) can hold both a
    local and a remote speaker; classifying the whole segment then gives
    "ambiguous" and keeps the remote words with the local speaker (#36).
    Ambiguous words and words without timestamps join the run before them
    (or after, at the start); runs shorter than _MIN_RUN_SECONDS merge into
    their neighbour."""
    words = seg.get("words") or []
    sides = [_segment_side(mic_db, sys_db, w["start"], w.get("end", w["start"]))
             if "start" in w else None for w in words]
    known = [side for side in sides if side]
    if not known:
        return []
    current = known[0]
    runs: list[list] = []  # [words, side]
    for w, side in zip(words, sides):
        current = side or current
        if runs and runs[-1][1] == current:
            runs[-1][0].append(w)
        else:
            runs.append([[w], current])

    def duration(run_words):
        timed = [w for w in run_words if "start" in w]
        return timed[-1].get("end", timed[-1]["start"]) - timed[0]["start"] if timed else 0.0

    while len(runs) > 1:
        short = next((i for i, (ws, _) in enumerate(runs) if duration(ws) < _MIN_RUN_SECONDS), None)
        if short is None:
            break
        into = short - 1 if short > 0 else 1
        if into < short:
            runs[into][0].extend(runs[short][0])
        else:
            runs[into][0][:0] = runs[short][0]
        del runs[short]
        merged = [runs[0]]  # neighbours that now share a side
        for run in runs[1:]:
            if run[1] == merged[-1][1]:
                merged[-1][0].extend(run[0])
            else:
                merged.append(run)
        runs = merged
    return [(ws, side) for ws, side in runs] if len(runs) > 1 else []


def _piece(seg: dict, words: list[dict], first: bool, last: bool) -> dict:
    """The part of seg holding words (a cut at a side change)."""
    timed = [w for w in words if "start" in w]
    piece = dict(seg)
    piece["start"] = seg.get("start", 0.0) if first else timed[0]["start"]
    piece["end"] = seg.get("end", 0.0) if last else timed[-1].get("end", timed[-1]["start"])
    piece["text"] = " ".join(w.get("word", "").strip() for w in words)
    piece["words"] = words
    return piece


def split_speakers_by_track(result: dict, mic: np.ndarray, system: np.ndarray) -> dict[str, str]:
    """Splits diarization speakers that appear substantially on both the
    microphone and the system side into two speakers (in place).

    Operates on raw diarization labels (call before voice-print relabeling).
    Each speaker keeps its label on the side where it speaks most; its
    segments on the other side move to a new SPEAKER_NN label. A segment
    whose words change side is cut at the change, if one of its parts
    moves; other segments stay as they are. Returns {new_label:
    original_label} for every split performed.
    """
    segments = result.get("segments", [])
    if not segments:
        return {}
    mic_db, sys_db = _frame_db(mic), _frame_db(system)

    # Per segment: [(piece, side)]; a single entry with the segment itself
    # unless its words change side.
    parts: list[list[tuple[dict, str | None]]] = []
    totals: dict[str, dict[str, float]] = {}
    for seg in segments:
        spk = seg.get("speaker")
        runs = _word_runs(seg, mic_db, sys_db) if spk else []
        if runs:
            seg_parts = [(_piece(seg, ws, i == 0, i == len(runs) - 1), side)
                         for i, (ws, side) in enumerate(runs)]
        else:
            side = _segment_side(mic_db, sys_db, seg.get("start", 0.0), seg.get("end", 0.0)) if spk else None
            seg_parts = [(seg, side)]
        parts.append(seg_parts)
        for piece, side in seg_parts:
            if spk and side:
                dur = max(0.0, piece.get("end", 0.0) - piece.get("start", 0.0))
                totals.setdefault(spk, {MIC: 0.0, SYSTEM: 0.0})[side] += dur

    existing = {seg.get("speaker") for seg in segments if seg.get("speaker")}
    next_idx = 0

    def _new_label() -> str:
        nonlocal next_idx
        while f"SPEAKER_{next_idx:02d}" in existing:
            next_idx += 1
        label = f"SPEAKER_{next_idx:02d}"
        existing.add(label)
        return label

    move_side: dict[str, tuple[str, str]] = {}  # speaker -> (minority side, new label)
    for spk, t in totals.items():
        home = MIC if t[MIC] >= t[SYSTEM] else SYSTEM
        other = SYSTEM if home == MIC else MIC
        total = t[MIC] + t[SYSTEM]
        if t[other] >= _SPLIT_MIN_SECONDS and t[other] >= _SPLIT_MIN_FRACTION * total:
            move_side[spk] = (other, _new_label())

    if not move_side:
        return {}

    new_segments = []
    moved_windows: dict[str, list[tuple[float, float, str]]] = {}
    for seg, seg_parts in zip(segments, parts):
        spk = seg.get("speaker")
        moves = spk in move_side and any(side == move_side[spk][0] for _, side in seg_parts)
        if not moves:
            new_segments.append(seg)
            continue
        for piece, side in seg_parts:
            new_segments.append(piece)
            if side != move_side[spk][0]:
                continue
            new = move_side[spk][1]
            piece["speaker"] = new
            for w in piece.get("words", []) or []:
                if w.get("speaker") == spk:
                    w["speaker"] = new
            moved_windows.setdefault(spk, []).append((piece.get("start", 0.0), piece.get("end", 0.0), new))
    result["segments"] = new_segments

    # word_segments may hold separate dict copies of the same words.
    for w in result.get("word_segments", []) or []:
        spk = w.get("speaker")
        if spk not in moved_windows or "start" not in w:
            continue
        for start, end, new in moved_windows[spk]:
            if start <= w["start"] < end:
                w["speaker"] = new
                break

    return {new: spk for spk, (_, new) in move_side.items()}
