"""Separate microphone/system tracks for "microphone + system" recordings.

The recorder still saves the usual mono mix (`<name>.wav`, which everything
else consumes unchanged), plus a 2-channel sidecar `<name>.tracks.flac`
(channel 0 = microphone, channel 1 = system audio, 16kHz). transcribe() uses
the sidecar - if present - as a hard constraint for speaker diarization: a
person speaks either into the local microphone or comes out of the speakers,
never both. Diarization clusters that clearly span both sides (pyannote
merged a local and a remote voice) are split apart again.

Deliberately numpy/soundfile-only (no pyaudio, no torch), so both
recorder.py and transcriber.py can import it cheaply.
"""

import os

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
TRACKS_SUFFIX = ".tracks.flac"

MIC = "mic"
SYSTEM = "system"

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


def tracks_path_for(audio_path: str) -> str:
    """'recordings/x.wav' -> 'recordings/x.tracks.flac'."""
    return os.path.splitext(audio_path)[0] + TRACKS_SUFFIX


def save_tracks(audio_path: str, mic_audio: np.ndarray, sys_audio: np.ndarray) -> str:
    """Saves both tracks (int16 value range as float, same sample rate
    SAMPLE_RATE, as produced by recorder._record_both*) next to the mix."""
    n = max(len(mic_audio), len(sys_audio))
    stereo = np.zeros((n, 2), dtype=np.int16)
    stereo[:len(mic_audio), 0] = np.clip(mic_audio, -32768, 32767).astype(np.int16)
    stereo[:len(sys_audio), 1] = np.clip(sys_audio, -32768, 32767).astype(np.int16)
    path = tracks_path_for(audio_path)
    sf.write(path, stereo, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return path


def load_tracks(audio_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (mic, system) as float32 arrays, or None if there's no sidecar."""
    path = tracks_path_for(audio_path)
    if not os.path.isfile(path):
        return None
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE or data.shape[1] != 2:
        return None
    return data[:, 0], data[:, 1]


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


def split_speakers_by_track(result: dict, mic: np.ndarray, system: np.ndarray) -> dict[str, str]:
    """Splits diarization speakers that appear substantially on both the
    microphone and the system side into two speakers (in place).

    Operates on raw diarization labels (call before voice-print relabeling).
    Each speaker keeps its label on the side where it speaks most; its
    segments on the other side move to a new SPEAKER_NN label. Returns
    {new_label: original_label} for every split performed.
    """
    segments = result.get("segments", [])
    if not segments:
        return {}
    mic_db, sys_db = _frame_db(mic), _frame_db(system)

    sides = []
    totals: dict[str, dict[str, float]] = {}
    for seg in segments:
        spk = seg.get("speaker")
        side = _segment_side(mic_db, sys_db, seg.get("start", 0.0), seg.get("end", 0.0)) if spk else None
        sides.append(side)
        if spk and side:
            dur = max(0.0, seg.get("end", 0.0) - seg.get("start", 0.0))
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

    moved_windows: dict[str, list[tuple[float, float, str]]] = {}
    for seg, side in zip(segments, sides):
        spk = seg.get("speaker")
        if spk not in move_side or side != move_side[spk][0]:
            continue
        new = move_side[spk][1]
        seg["speaker"] = new
        for w in seg.get("words", []) or []:
            if w.get("speaker") == spk:
                w["speaker"] = new
        moved_windows.setdefault(spk, []).append((seg.get("start", 0.0), seg.get("end", 0.0), new))

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
