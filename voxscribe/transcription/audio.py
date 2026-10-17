"""Audio loading: any audio/video file to 16kHz mono float32, without ffmpeg."""

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def load_audio_without_ffmpeg(audio_path: str) -> np.ndarray:
    """Loads an audio file and converts it to 16kHz mono float32 (no ffmpeg)."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SAMPLE_RATE, sr)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return data


def _load_audio_via_container(audio_path: str) -> np.ndarray:
    """Extracts the audio track from an arbitrary container (MKV, MP4, MOV, ...) via PyAV.

    Used as a fallback when soundfile can't read the format directly (e.g.
    video containers like MKV, where only the audio track is needed).
    """
    import av

    container = av.open(audio_path)
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        container.close()
        raise ValueError(f"No audio track found in: {audio_path}")

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray())
    for resampled in resampler.resample(None):
        chunks.append(resampled.to_ndarray())
    container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    data = np.concatenate(chunks, axis=1).reshape(-1)
    return (data.astype(np.float32) / 32768.0)


def load_audio_universal(audio_path: str) -> np.ndarray:
    """Loads an audio or video file as a 16kHz mono float32 array.

    Tries soundfile first (fast path for WAV/FLAC/OGG etc.). If that fails
    (e.g. video containers like MKV/MP4, or compressed formats libsndfile
    doesn't know), falls back to decoding just the audio track via PyAV.
    """
    try:
        return load_audio_without_ffmpeg(audio_path)
    except Exception:
        return _load_audio_via_container(audio_path)
