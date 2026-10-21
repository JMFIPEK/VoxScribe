"""Audio loading: any audio/video file to 16kHz mono float32, without ffmpeg."""

import os
import tempfile
from contextlib import contextmanager

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


class AudioSource:
    """An audio file that is decoded at most once, however many pipeline
    stages (Backend, alignment, diarization, track split) ask for the samples.

    For a Loudspeaker Call the samples are the mix of the cleaned Microphone
    track and the Playback track (the saved mix holds the Echo, which
    garbles the transcript); the saved files stay untouched. The tracks are
    loaded once too, so the echo canceller runs once per file."""

    def __init__(self, path: str):
        self.path = path
        self._samples: np.ndarray | None = None
        self._tracks = None
        self._tracks_loaded = False
        self._cleaned = False  # load() returned the cleaned mix, not the file

    def load(self) -> np.ndarray:
        if self._samples is None:
            self._samples = self._cleaned_mix()
            self._cleaned = self._samples is not None
            if self._samples is None:
                self._samples = load_audio_universal(self.path)
        return self._samples

    @contextmanager
    def as_file(self):
        """A path to the samples load() returns, for Backends that read a
        file (Apple SpeechAnalyzer): the recording itself, or for a
        Loudspeaker Call a temporary WAV of the cleaned mix, removed again
        afterwards (also when the Backend fails)."""
        from voxscribe.recording_modes import SETUP_LOUDSPEAKER
        from voxscribe.tracks import load_setup
        if load_setup(self.path) != SETUP_LOUDSPEAKER:
            yield self.path
            return
        samples = self.load()
        if not self._cleaned:
            yield self.path
            return
        fd, tmp_path = tempfile.mkstemp(prefix="voxscribe-cleaned-", suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp_path, samples, SAMPLE_RATE, subtype="PCM_16")
            yield tmp_path
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def tracks(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(mic, system) from the Call's sidecar, the mic without Echo for a
        Loudspeaker Setup; None without a sidecar (see tracks.load_tracks())."""
        if not self._tracks_loaded:
            from voxscribe.tracks import load_tracks
            self._tracks_loaded = True
            self._tracks = load_tracks(self.path)
        return self._tracks

    def _cleaned_mix(self) -> np.ndarray | None:
        from voxscribe.recording_modes import SETUP_LOUDSPEAKER
        from voxscribe.tracks import load_setup
        try:
            if load_setup(self.path) != SETUP_LOUDSPEAKER or self.tracks() is None:
                return None
        except Exception as e:
            print(f"     [WARNING] Could not load the Call's tracks ({e}) - "
                  "transcribing the saved mix, Echo included.")
            return None
        mic, system = self._tracks
        print("     Echo removal ran: transcribing the mix of the cleaned "
              "microphone track and the playback track.")
        return _mix_tracks(mic, system)


def _mix_tracks(mic: np.ndarray, system: np.ndarray) -> np.ndarray:
    """Both tracks at 0.65, scaled down if they'd clip - the same mix the
    recorder saves (track_assembler), in the -1..1 range."""
    mixed = 0.65 * (mic.astype(np.float64) + system.astype(np.float64))
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 32767 / 32768:
        mixed *= (32767 / 32768) / peak
    return mixed.astype(np.float32)
