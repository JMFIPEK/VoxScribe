"""Track assembler: turns a Call's captured Microphone audio and Playback
audio into the files a Call recording consists of.

Both Call recording paths (recorder._record_both() on Windows,
recorder._record_both_macos() on macOS) feed it each source's raw chunks as
they arrive, at the source's native sample rate and channel count. After
stop, save() downmixes each source to mono, resamples it to 16kHz, pads the
shorter track, mixes both and writes the mono mix (`<name>.wav`) plus the
separate-tracks sidecar (`<name>.tracks.flac`, see tracks.py).

No device code (no pyaudio, no sounddevice): it only sees bytes, so it can be
tested with synthetic chunks, and a sync fix applies to both platforms.
"""

from __future__ import annotations

import os
import wave

import numpy as np

from voxscribe.tracks import MIC, SYSTEM, mix, save_tracks

OUT_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

# How far (seconds) Playback may fall behind the Microphone before the gap
# counts as "nothing arrived" and is padded. Above the jitter between the
# two readers (each polls every 10-20 ms); a reader that falls further
# behind reports its backlog (buffered_frames), so a late read is never
# mistaken for a gap.
MAX_PLAYBACK_LAG = 0.05


# A Microphone that delivered this much (seconds) less than the wall clock
# says has lost samples. Far above read jitter and the drift between the
# device clock and the wall clock (tens of ms per hour).
MAX_UNREPORTED_DROP = 0.2


class DropDetector:
    """Finds Microphone samples lost without a report (PyAudio on Windows
    drops input overflows silently): compares the samples received with
    the wall clock since the first read. Only a drop detector - the
    Microphone's sample count stays the Call's time base."""

    def __init__(self, rate: int, threshold: float = MAX_UNREPORTED_DROP):
        self._rate = rate
        self._threshold = threshold
        self._start: float | None = None

    def missing(self, received: int, now: float) -> int:
        """Frames lost so far, or 0. received counts every frame added to
        the track, including silence added for earlier drops; now is
        time.monotonic() right after a read that emptied the buffer."""
        if self._start is None:
            self._start = now - received / self._rate
            return 0
        deficit = int(round((now - self._start) * self._rate)) - received
        return deficit if deficit > self._threshold * self._rate else 0


class TrackAssembler:
    """Collects one Call's per-source chunks (int16 PCM bytes, interleaved
    if multi-channel) and assembles the 16kHz tracks and mix from them.

    Sources are tracks.MIC (Microphone audio) and tracks.SYSTEM (Playback
    audio). add_chunk() may be called from each source's own capture
    thread; save() only once both have stopped.
    """

    def __init__(self, mic_rate: int, mic_channels: int,
                 playback_rate: int, playback_channels: int,
                 playback_is_continuous: bool = False):
        """playback_is_continuous: the Playback source streams without gaps
        from its first sample on (macOS' system-audio helper). Microphone
        audio from before that sample is dropped, so both tracks start
        together, and Playback is never padded (a slow read of the helper's
        pipe is a backlog, not a gap). Otherwise (WASAPI loopback, which
        sends nothing while nothing plays) Playback starts with the
        Microphone and every gap, including a late first packet, is padded."""
        self._format = {MIC: (mic_rate, mic_channels),
                        SYSTEM: (playback_rate, playback_channels)}
        self._chunks: dict[str, list[bytes]] = {MIC: [], SYSTEM: []}
        self._frames = {MIC: 0, SYSTEM: 0}  # per channel, real + padded
        self._playback_is_continuous = playback_is_continuous
        self._mic_trim = 0  # Microphone frames before the Playback start
        self._playback_started = False

    def add_chunk(self, source: str, data: bytes, buffered_frames: int = 0):
        """Appends audio captured from source (MIC or SYSTEM), as soon as it
        was read; buffered_frames is how much more the driver already holds
        behind it.

        The Microphone is the time base: it delivers audio continuously.
        Playback that falls behind it by more than MAX_PLAYBACK_LAG (WASAPI
        loopback sends nothing while nothing plays) is padded with silence
        first, so the chunk ends where the Microphone track is now, minus
        the backlog still buffered behind it."""
        frames = self._frames_in(source, data)
        if source == SYSTEM and self._playback_is_continuous and not self._playback_started:
            # This chunk's first sample is the Playback start.
            chunk_in_mic_frames = int(frames * self._format[MIC][0] / self._format[SYSTEM][0])
            self._mic_trim = max(0, self._frames[MIC] - chunk_in_mic_frames)
        if source == SYSTEM and not self._playback_is_continuous:
            lag = self._mic_position(SYSTEM) - buffered_frames - (self._frames[SYSTEM] + frames)
            if lag > MAX_PLAYBACK_LAG * self._format[SYSTEM][0]:
                self._append_silence(SYSTEM, lag)
        self._chunks[source].append(data)
        self._frames[source] += frames
        if source == SYSTEM:
            self._playback_started = True

    def add_dropped(self, source: str, frames: int):
        """Records that source lost `frames` frames (at its native rate),
        e.g. an input overflow: they become silence of the same length, so
        the rest of the track stays in place, and the drop is logged."""
        if frames <= 0:
            return
        rate = self._format[source][0]
        name = "Microphone" if source == MIC else "Playback audio"
        print(f"[WARNING] {name} dropped {frames / rate * 1000:.0f} ms "
              f"at {self._frames[source] / rate:.1f}s; filled with silence.")
        self._append_silence(source, frames)

    def _frames_in(self, source: str, data: bytes) -> int:
        return len(data) // (SAMPLE_WIDTH * self._format[source][1])

    def _mic_position(self, source: str) -> int:
        """The Microphone track's current length, in source's frames."""
        mic_rate = self._format[MIC][0]
        return int((self._frames[MIC] - self._mic_trim) * self._format[source][0] / mic_rate)

    def _append_silence(self, source: str, frames: int):
        channels = self._format[source][1]
        self._chunks[source].append(bytes(frames * channels * SAMPLE_WIDTH))
        self._frames[source] += frames

    def has_audio(self, source: str) -> bool:
        return bool(self._chunks[source])

    def save(self, output_path: str, setup: str | None = None) -> np.ndarray:
        """Writes the mono mix to output_path and the tracks sidecar next to
        it (tagged with the Call's Setup, if given). Returns the mix as
        16kHz int16 samples."""
        mic_audio = self._track(MIC)
        sys_audio = self._track(SYSTEM)
        mixed = mix(mic_audio, sys_audio, full_scale=32767)
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        _save_wav(output_path, mixed)
        _save_tracks_quietly(output_path, mic_audio, sys_audio, setup)
        return mixed

    def _track(self, source: str) -> np.ndarray:
        rate, channels = self._format[source]
        audio = _frames_to_mono_float(self._chunks[source], channels)
        if source == MIC:
            audio = audio[self._mic_trim:]
        return _resample_to_rate(audio, rate, OUT_RATE)


def _frames_to_mono_float(frames: list[bytes], channels: int) -> np.ndarray:
    if not frames:
        return np.array([], dtype=np.float64)

    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float64)
    if channels > 1:
        usable_len = len(audio) - (len(audio) % channels)
        audio = audio[:usable_len].reshape(-1, channels).mean(axis=1)
    return audio


def _resample_to_rate(audio: np.ndarray, source_rate: int,
                      target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(audio) == 0:
        return audio

    from math import gcd
    from scipy.signal import resample_poly

    divisor = gcd(target_rate, source_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor)


def _save_wav(output_path: str, mixed: np.ndarray):
    """Saves the 16kHz mono int16 mix as a WAV file."""
    data = mixed.tobytes()
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(OUT_RATE)
        wf.writeframes(data)

    duration = len(data) / (OUT_RATE * SAMPLE_WIDTH)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({duration:.1f}s, {size_mb:.1f} MB)")


def _save_tracks_quietly(output_path: str, mic_audio: np.ndarray, sys_audio: np.ndarray,
                         setup: str | None = None):
    """Saves the separate mic/system tracks next to the mix (see
    voxscribe/tracks.py). Never fails the recording itself - the mix is
    already saved and fully usable without the sidecar."""
    try:
        path = save_tracks(output_path, mic_audio, sys_audio, setup=setup)
        print(f"Saved: {path} (separate microphone/system tracks)")
    except Exception as e:
        print(f"[WARNING] Could not save separate tracks: {e}")
