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

from voxscribe.tracks import MIC, SYSTEM, save_tracks

OUT_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

# How far (seconds) Playback may fall behind the Microphone before the gap
# counts as "nothing arrived" and is padded. Above the normal read jitter
# (64 ms chunks plus 20 ms polling), so a late read never inserts silence
# into continuous Playback, which would push the rest of it too late.
MAX_PLAYBACK_LAG = 0.2


class TrackAssembler:
    """Collects one Call's per-source chunks (int16 PCM bytes, interleaved
    if multi-channel) and assembles the 16kHz tracks and mix from them.

    Sources are tracks.MIC (Microphone audio) and tracks.SYSTEM (Playback
    audio). add_chunk() may be called from each source's own capture
    thread; save() only once both have stopped.
    """

    def __init__(self, mic_rate: int, mic_channels: int,
                 playback_rate: int, playback_channels: int,
                 trim_mic_lead: bool = False):
        """trim_mic_lead: the Playback source streams continuously from its
        first sample on (macOS' system-audio helper), so Microphone audio
        from before that sample is dropped and both tracks start together.
        Without it (WASAPI loopback, which is silent until something plays),
        Playback starts with the Microphone and its late first packet is
        padded like any other gap."""
        self._format = {MIC: (mic_rate, mic_channels),
                        SYSTEM: (playback_rate, playback_channels)}
        self._chunks: dict[str, list[bytes]] = {MIC: [], SYSTEM: []}
        self._frames = {MIC: 0, SYSTEM: 0}  # per channel, real + padded
        self._trim_mic_lead = trim_mic_lead
        self._mic_trim = 0  # Microphone frames before the Playback start
        self._playback_started = False

    def add_chunk(self, source: str, data: bytes):
        """Appends audio captured from source (MIC or SYSTEM), at the moment
        its last sample was captured.

        The Microphone is the time base: it delivers audio continuously.
        Playback that falls behind it by more than MAX_PLAYBACK_LAG (WASAPI
        loopback sends nothing while nothing plays) is padded with silence
        first, so the chunk ends where the Microphone track is now."""
        frames = self._frames_in(source, data)
        if source == SYSTEM and self._trim_mic_lead and not self._playback_started:
            # This chunk's first sample is the Playback start.
            chunk_in_mic_frames = int(frames * self._format[MIC][0] / self._format[SYSTEM][0])
            self._mic_trim = max(0, self._frames[MIC] - chunk_in_mic_frames)
        if source == SYSTEM:
            lag = self._mic_position(SYSTEM) - (self._frames[SYSTEM] + frames)
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
        mixed = _mix_sources(mic_audio, sys_audio)
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


def _pad_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)), mode="constant")


def _mix_sources(mic_audio: np.ndarray, sys_audio: np.ndarray) -> np.ndarray:
    """Mixes two mono signals already resampled to the same sample rate
    (microphone + playback, int16 value range as float). Returns the int16
    value range (neither normalized to -1..1 nor cast to int16)."""
    out_len = max(len(mic_audio), len(sys_audio))
    mic_audio = _pad_to_length(mic_audio, out_len)
    sys_audio = _pad_to_length(sys_audio, out_len)
    mixed = (mic_audio * 0.65) + (sys_audio * 0.65)
    peak = np.max(np.abs(mixed)) if len(mixed) else 0
    if peak > 32767:
        mixed = mixed * (32767 / peak)
    return mixed


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
