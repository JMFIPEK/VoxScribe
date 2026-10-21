"""Echo cancellation on synthetic Loudspeaker Calls.

The Microphone track is local "speech" plus a delayed, quieter, slightly
filtered copy of the Playback track (the Echo); assertions are on levels of
the cleaned Microphone track, not on how the canceller works inside.
"""

import functools

import numpy as np
import pytest

from voxscribe import echo

SR = 16000
SECONDS = 30.0

# Timeline (seconds): remote talks alone, local talks alone, both talk
# (double-talk), local talks alone again.
FAR_ONLY = (0.0, 14.0)
LOCAL_ONLY = (14.0, 20.0)
DOUBLE_TALK = (20.0, 26.0)
LOCAL_ONLY_2 = (26.0, 30.0)
# The canceller needs a moment to adapt; measure Echo after that.
WARMUP = 2.0


def _speech(ranges, f0, seed, seconds=SECONDS):
    """Speech-like signal inside ranges: a harmonic series with a wandering
    pitch, syllable-rate amplitude modulation and short noise bursts
    ("consonants"), silent elsewhere."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    pitch = f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6)))
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    voiced = sum(rng.uniform(0.2, 1.0) / k * np.sin(k * phase) for k in range(1, 16))
    syllables = np.clip(np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6)), 0, None)
    noise = np.convolve(rng.standard_normal(n), np.hanning(9), mode="same")
    bursts = (rng.uniform(size=n // 1600 + 1) < 0.3).repeat(1600)[:n]
    sig = 0.12 * voiced * syllables + 0.04 * noise * bursts
    gate = np.zeros(n)
    for a, b in ranges:
        gate[int(a * SR):int(b * SR)] = 1.0
    gate = np.convolve(gate, np.hanning(321) / np.hanning(321).sum(), mode="same")
    return (sig * gate).astype(np.float32)


def _room(playback, delay_s, gain=0.3):
    """What the microphone hears of the speakers: delayed (delay_s may be an
    array per sample, for drifting clocks), low-passed, with a short
    reverberant tail, attenuated."""
    n = len(playback)
    idx = np.arange(n) - np.asarray(delay_s) * SR
    delayed = np.interp(idx, np.arange(n), playback, left=0.0, right=0.0)
    rng = np.random.default_rng(99)
    taps = np.array([1.0, 0.6, 0.25])
    tail = rng.standard_normal(800) * np.exp(-np.arange(800) / 150) * 0.05
    ir = np.concatenate([taps, tail])
    return (gain * np.convolve(delayed, ir)[:n]).astype(np.float32)


@functools.lru_cache(maxsize=None)
def _call(drift: bool):
    """(local speech, Echo, Microphone track, Playback track, cleaned Microphone track)."""
    far = _speech([FAR_ONLY, DOUBLE_TALK], f0=180.0, seed=1)
    local = _speech([LOCAL_ONLY, DOUBLE_TALK, LOCAL_ONLY_2], f0=120.0, seed=2)
    n = len(far)
    delay = np.linspace(0.300, 0.330, n) if drift else 0.300
    echo_part = _room(far, delay)
    mic_noise = np.random.default_rng(3).standard_normal(n).astype(np.float32) * 1e-4
    mic = local + echo_part + mic_noise
    cleaned = echo.cancel_echo(mic, far)
    return local, echo_part, mic, far, cleaned


def _span(x, span):
    a, b = span
    return x[int(a * SR):int(b * SR)]


def _db(x):
    return 10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20)


def _echo_reduction_db(drift):
    _, _, mic, _, cleaned = _call(drift)
    span = (FAR_ONLY[0] + WARMUP, FAR_ONLY[1])
    return _db(_span(mic, span)) - _db(_span(cleaned, span))


def test_cleaned_track_has_same_length_and_dtype():
    _, _, mic, _, cleaned = _call(False)
    assert cleaned.shape == mic.shape
    assert cleaned.dtype == np.float32


def test_echo_drops_by_at_least_15_db_while_only_playback_is_heard():
    assert _echo_reduction_db(drift=False) >= 15.0


def test_drifting_delay_is_still_cancelled():
    assert _echo_reduction_db(drift=True) >= 15.0


def _frame_levels_db(x, span):
    """Level of every 20 ms frame in span."""
    frames = _span(x, span).astype(np.float64)
    frames = frames[:len(frames) // 320 * 320].reshape(-1, 320)
    return 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-20)


def _speech_dips_db(cleaned, local, span):
    """How far the cleaned track drops below the local speech in the worst
    10% of the 20 ms frames where the local speaker talks (within 30 dB of
    their loudest frame) - catches muted syllables that an overall level
    would hide. The canceller may rotate phases (WebRTC's built-in high-pass
    filter), so this compares levels, not waveforms."""
    ref = _frame_levels_db(local, span)
    talking = ref > ref.max() - 30
    return float(np.percentile((_frame_levels_db(cleaned, span) - ref)[talking], 10))


@pytest.mark.parametrize("span", [LOCAL_ONLY, LOCAL_ONLY_2])
def test_local_speech_is_kept(span):
    local, _, _, _, cleaned = _call(False)
    # Measured: -0.5 dB overall, worst 10% of frames -1.6 dB.
    assert abs(_db(_span(cleaned, span)) - _db(_span(local, span))) <= 1.0
    assert _speech_dips_db(cleaned, local, span) >= -3.0


@pytest.mark.parametrize("drift", [False, True])
def test_double_talk_keeps_local_speech(drift):
    # The local speaker talking over a remote one must not be muted.
    # Measured: -0.8 dB overall, worst 10% of frames -3.4 dB (drift: -4.0 dB);
    # WebRTC AEC3 alone dropped them by 38 dB.
    local, _, _, _, cleaned = _call(drift)
    assert abs(_db(_span(cleaned, DOUBLE_TALK)) - _db(_span(local, DOUBLE_TALK))) <= 1.5
    assert _speech_dips_db(cleaned, local, DOUBLE_TALK) >= -6.0


def test_aec3_failure_keeps_the_linear_stage(monkeypatch, capsys):
    import sys

    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("AEC exploded")

    monkeypatch.setitem(sys.modules, "livekit", Broken())
    monkeypatch.setitem(sys.modules, "livekit.rtc", Broken())
    _, _, mic, far, _ = _call(False)
    cleaned = echo.cancel_echo(mic, far)
    # The linear stage alone already removes most of the Echo, so a broken
    # livekit costs the residual suppression, not the whole cleaning.
    span = (FAR_ONLY[0] + WARMUP, FAR_ONLY[1])
    assert _db(_span(mic, span)) - _db(_span(cleaned, span)) >= 15
    assert "[WARNING]" in capsys.readouterr().out


def test_failure_before_aec3_returns_the_uncleaned_track(monkeypatch, capsys):
    def broken(*args):
        raise RuntimeError("linear stage exploded")

    monkeypatch.setattr(echo, "_cancel_linear", broken)
    mic = np.random.default_rng(0).standard_normal(SR * 2).astype(np.float32) * 0.1
    playback = np.random.default_rng(1).standard_normal(SR * 2).astype(np.float32) * 0.1
    out = echo.cancel_echo(mic, playback)
    np.testing.assert_array_equal(out, mic)
    assert "[WARNING]" in capsys.readouterr().out
