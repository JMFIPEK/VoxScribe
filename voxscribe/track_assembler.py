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


class TrackAssembler:
    """Collects one Call's per-source chunks (int16 PCM bytes, interleaved
    if multi-channel) and assembles the 16kHz tracks and mix from them.

    Sources are tracks.MIC (Microphone audio) and tracks.SYSTEM (Playback
    audio). add_chunk() may be called from each source's own capture
    thread; save() only once both have stopped.
    """

    def __init__(self, mic_rate: int, mic_channels: int,
                 playback_rate: int, playback_channels: int):
        self._format = {MIC: (mic_rate, mic_channels),
                        SYSTEM: (playback_rate, playback_channels)}
        self._chunks: dict[str, list[bytes]] = {MIC: [], SYSTEM: []}

    def add_chunk(self, source: str, data: bytes):
        """Appends audio captured from source (MIC or SYSTEM)."""
        self._chunks[source].append(data)

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
