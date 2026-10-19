"""Echo cancellation for Loudspeaker Calls (see docs/adr/0003).

With a Loudspeaker Setup the microphone picks up the Playback audio again
(Echo), a few hundred milliseconds late. cancel_echo() removes it from the
Microphone track offline, using the Playback track as the reference:

1. The delay between both tracks is estimated block by block (cross-
   correlation), because it drifts slowly with the two device clocks.
2. The Playback track is lined up accordingly, so only a small delay is
   left for the cancellers.
3. A linear canceller subtracts the bulk of the Echo: per frequency, the
   echo path is fitted by least squares over a few seconds around each
   moment. Local speech doesn't correlate with the Playback track, so it
   doesn't disturb the fit - which is what keeps it intact when both sides
   talk at once (double-talk).
4. WebRTC's echo canceller (AEC3, via livekit's AudioProcessingModule)
   removes what's left. On its own it reaches more cancellation but
   suppresses whole stretches of local speech during double-talk (measured
   on synthetic Calls: 10% of 20 ms frames down by 38 dB); after step 3 the
   remaining Echo is too weak for that.

livekit is imported only inside cancel_echo(), never at module import: the
tracks module (and through it the transcriber facade) imports this one.
"""

import numpy as np

SAMPLE_RATE = 16000

# livekit's AudioProcessingModule takes exactly 10 ms frames.
_FRAME = SAMPLE_RATE // 100
# Delay estimate: block length / hop and the searched delay range (seconds).
_BLOCK_SECONDS = 4.0
_HOP_SECONDS = 2.0
_MAX_DELAY_SECONDS = 1.0
# Cross-correlation weighting: only the speech band counts (the room and the
# speakers drop the rest, so there it's only noise), and the spectrum is
# partially whitened (PHAT-beta) for a sharp peak despite the room coloring.
_BAND_HZ = (150.0, 4000.0)
_PHAT_BETA = 0.7
# A block's correlation peak must stand out this far (in standard deviations
# of the correlation) to count - blocks without playback or with too much
# local speech give no reliable estimate and are skipped.
_MIN_PEAK_SCORE = 6.0
# Smoothing of the block estimates: a median over this many neighbors drops
# single wrong peaks, a line fit over this many neighbors evens out jitter.
_MEDIAN_BLOCKS = 5
_LINE_FIT_BLOCKS = 11
# Delay deliberately left after lining up: the reference must reach the
# canceller slightly before its echo, even if the estimate is a bit late.
_RESIDUAL_DELAY_MS = 20
# Linear canceller: STFT with sqrt-Hann windows (perfect reconstruction at
# 50% overlap), an echo path of this many frames per frequency (4 x 16 ms
# covers the residual delay plus the strong part of a room's reverb), fitted
# over a window this long, refitted every step.
_STFT_SIZE = 512
_STFT_HOP = 256
_PATH_FRAMES = 4
_FIT_SECONDS = 5.0
_FIT_STEP_SECONDS = 1.0
_FIT_REGULARIZATION = 1e-3
# livekit's AudioProcessingModule hands each frame back 128 samples (8 ms)
# late (measured with livekit 1.1.21); the cleaned track is shifted back by
# that much so it stays in sync with the Playback track and the mix.
_APM_LATENCY = 128


def cancel_echo(mic: np.ndarray, playback: np.ndarray) -> np.ndarray:
    """Removes the Echo of `playback` from `mic` (both float32 mono at
    SAMPLE_RATE, value range -1..1, as tracks.load_tracks() returns them).
    Returns the cleaned microphone track, same length as `mic`; on any error
    a warning is printed and `mic` is returned unchanged."""
    try:
        reference = _line_up(mic, playback)
        return _run_aec3(_cancel_linear(mic, reference), reference)
    except Exception as e:
        print(f"     [WARNING] Echo cancellation failed, using the uncleaned microphone track: {e}")
        return mic


def _estimate_delays(mic: np.ndarray, playback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(block centers, delays) in samples for every block with a reliable
    estimate - the lag at which `playback` best matches `mic`, searched
    over 0.._MAX_DELAY_SECONDS."""
    block = int(_BLOCK_SECONDS * SAMPLE_RATE)
    hop = int(_HOP_SECONDS * SAMPLE_RATE)
    max_lag = int(_MAX_DELAY_SECONDS * SAMPLE_RATE)
    nfft = 1 << int(np.ceil(np.log2(block + max_lag)))
    freqs = np.fft.rfftfreq(nfft, 1 / SAMPLE_RATE)
    band = (freqs >= _BAND_HZ[0]) & (freqs <= _BAND_HZ[1])
    centers, delays = [], []
    for start in range(0, max(1, len(mic) - block + 1), hop):
        m = mic[start:start + block].astype(np.float64)
        # Reference window starts max_lag earlier, so lag k = max_lag - index.
        r_start = start - max_lag
        r = playback[max(0, r_start):start + block].astype(np.float64)
        if r_start < 0:
            r = np.concatenate([np.zeros(-r_start), r])
        if np.mean(m ** 2) < 1e-10 or np.mean(r ** 2) < 1e-10:
            continue
        spec = np.conj(np.fft.rfft(m, nfft)) * np.fft.rfft(r, nfft)
        weighted = np.where(band, spec / (np.abs(spec) ** _PHAT_BETA + 1e-12), 0)
        corr = np.fft.irfft(weighted, nfft)[:max_lag + 1]
        peak = int(np.argmax(corr))
        score = (corr[peak] - np.mean(corr)) / (np.std(corr) + 1e-12)
        if score >= _MIN_PEAK_SCORE:
            # Parabola through the peak and its neighbors: sub-sample delay.
            if 0 < peak < max_lag:
                left, mid, right = corr[peak - 1], corr[peak], corr[peak + 1]
                curve = left - 2 * mid + right
                offset = 0.5 * (left - right) / curve if curve < 0 else 0.0
            else:
                offset = 0.0
            centers.append(start + len(m) / 2)
            delays.append(max_lag - peak - offset)
    if not delays:
        return np.zeros(0), np.zeros(0)
    delays = np.asarray(delays, dtype=np.float64)
    half = _MEDIAN_BLOCKS // 2
    medians = np.array([np.median(delays[max(0, i - half):i + half + 1])
                        for i in range(len(delays))])
    # Then a straight line through the neighborhood of each block: evens out
    # the jitter of single estimates, which would otherwise make the
    # canceller re-adapt, without lagging behind a steady clock drift (also
    # at both ends of the recording, where a plain mean would be biased).
    centers = np.asarray(centers)
    k = _LINE_FIT_BLOCKS // 2
    smoothed = np.empty_like(medians)
    for i in range(len(medians)):
        a, b = max(0, i - k), min(len(medians), i + k + 1)
        if b - a < 3:
            smoothed[i] = medians[i]
            continue
        slope, offset = np.polyfit(centers[a:b], medians[a:b], 1)
        smoothed[i] = slope * centers[i] + offset
    return centers, smoothed


def _line_up(mic: np.ndarray, playback: np.ndarray) -> np.ndarray:
    """The Playback track shifted per sample by the estimated (drifting)
    delay minus _RESIDUAL_DELAY_MS, same length as `mic`. Without any
    reliable estimate it's returned as is, and the cancellers have to cope."""
    n = len(mic)
    playback = np.asarray(playback, dtype=np.float32)
    if len(playback) < n:
        playback = np.concatenate([playback, np.zeros(n - len(playback), np.float32)])
    centers, delays = _estimate_delays(mic, playback)
    if len(delays) == 0:
        return playback[:n]
    shift = np.interp(np.arange(n), centers, delays) - _RESIDUAL_DELAY_MS * SAMPLE_RATE / 1000
    idx = np.arange(n) - shift
    return np.interp(idx, np.arange(len(playback)), playback, left=0.0, right=0.0).astype(np.float32)


def _stft_frames(padded: np.ndarray, first: int, last: int) -> np.ndarray:
    """Spectra of STFT frames first..last-1 (frames x bins) of a signal padded
    by _STFT_SIZE at the front; frames before the signal are silent."""
    window = np.sqrt(np.hanning(_STFT_SIZE + 1)[:_STFT_SIZE])
    out = np.zeros((last - first, _STFT_SIZE // 2 + 1), np.complex128)
    lo = max(first, 0)
    if lo < last:
        idx = np.arange(lo, last)[:, None] * _STFT_HOP + np.arange(_STFT_SIZE)[None, :]
        out[lo - first:] = np.fft.rfft(padded[idx] * window, axis=1)
    return out


def _cancel_linear(mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Subtracts the part of `mic` that's a linear function of the (lined up)
    `reference`: per frequency bin, mic(t) ~ sum_l H_l * reference(t - l)
    with H fitted by least squares over _FIT_SECONDS around each
    _FIT_STEP_SECONDS chunk, so a slowly changing echo path is followed.
    Works chunk by chunk, so memory stays small for hour-long recordings."""
    n = len(mic)
    frames = (n + _STFT_SIZE) // _STFT_HOP + 1
    pad = (_STFT_SIZE, frames * _STFT_HOP + _STFT_SIZE - n)
    mic_p = np.pad(mic.astype(np.float64), pad)
    ref_p = np.pad(reference[:n].astype(np.float64), pad)
    window = np.sqrt(np.hanning(_STFT_SIZE + 1)[:_STFT_SIZE])
    out = np.zeros(len(mic_p))
    fit = int(_FIT_SECONDS * SAMPLE_RATE / _STFT_HOP)
    step = int(_FIT_STEP_SECONDS * SAMPLE_RATE / _STFT_HOP)
    taps = np.arange(_PATH_FRAMES)
    for c in range(0, frames, step):
        c_end = min(frames, c + step)
        b = min(frames, c + (step + fit) // 2)
        a = max(0, b - fit)
        b = max(b, c_end)
        m = _stft_frames(mic_p, a, b)
        r = _stft_frames(ref_p, a - _PATH_FRAMES + 1, b)
        # stacked[t, k, l] = reference spectrum l frames before frame a + t.
        stacked = np.stack([r[_PATH_FRAMES - 1 - l:len(r) - l] for l in taps], axis=2)
        gram = np.einsum("tkl,tkm->klm", stacked.conj(), stacked)
        cross = np.einsum("tkl,tk->kl", stacked.conj(), m)
        load = _FIT_REGULARIZATION * np.trace(gram, axis1=1, axis2=2).real / _PATH_FRAMES + 1e-12
        path = np.linalg.solve(gram + load[:, None, None] * np.eye(_PATH_FRAMES), cross[..., None])[..., 0]
        residual = m[c - a:c_end - a] - np.einsum("tkl,kl->tk", stacked[c - a:c_end - a], path)
        chunk = np.fft.irfft(residual, _STFT_SIZE, axis=1) * window
        for i, frame in enumerate(chunk):
            start = (c + i) * _STFT_HOP
            out[start:start + _STFT_SIZE] += frame
    return out[_STFT_SIZE:_STFT_SIZE + n].astype(np.float32)


def _to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def _run_aec3(mic: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """WebRTC AEC3 over both tracks in 10 ms lockstep: the reference as the
    reverse (far-end) stream, the microphone as the forward stream. Noise
    suppression, gain control and the high-pass filter are left off - only
    the Echo should go. (AEC3 still filters below ~100 Hz and rotates the
    phase of low frequencies; levels and spectra are kept, which is what
    transcription and the track split look at.)"""
    from livekit import rtc

    n = len(mic)
    frames = -(-(n + _APM_LATENCY) // _FRAME)
    padded = frames * _FRAME
    mic16 = np.zeros(padded, np.int16)
    mic16[:n] = _to_int16(mic)
    ref16 = np.zeros(padded, np.int16)
    ref16[:n] = _to_int16(reference[:n])

    apm = rtc.AudioProcessingModule(echo_cancellation=True)
    out = np.empty(padded, np.int16)
    for i in range(frames):
        a = i * _FRAME
        far = rtc.AudioFrame(bytearray(ref16[a:a + _FRAME].tobytes()), SAMPLE_RATE, 1, _FRAME)
        apm.process_reverse_stream(far)
        near = rtc.AudioFrame(bytearray(mic16[a:a + _FRAME].tobytes()), SAMPLE_RATE, 1, _FRAME)
        apm.set_stream_delay_ms(_RESIDUAL_DELAY_MS)
        apm.process_stream(near)
        out[a:a + _FRAME] = np.frombuffer(near.data, dtype=np.int16)
    return out[_APM_LATENCY:_APM_LATENCY + n].astype(np.float32) / 32768.0
