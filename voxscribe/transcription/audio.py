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


def probe_duration(audio_path: str) -> float | None:
    """The file's length in seconds from its header, without decoding it
    (for the Timeline's x axis while the pipeline runs). None if unknown."""
    try:
        return float(sf.info(audio_path).duration)
    except Exception:
        pass
    try:
        import av

        with av.open(audio_path) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is not None and stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            if container.duration:
                return container.duration / av.time_base
    except Exception:
        pass
    return None


class AudioSource:
    """An audio file that is decoded at most once, however many pipeline
    stages (Backend, alignment, diarization, track split) ask for the samples.

    For a Call the samples can be a new mix of its two tracks instead of the
    saved mix: for a Loudspeaker Setup with the Echo removed from the
    Microphone track (#23), and for any Call whose tracks' speech levels are
    far apart with the quieter one brought up (#39). The saved files stay
    untouched. The tracks are loaded once too, so the echo canceller runs
    once per file."""

    def __init__(self, path: str):
        self.path = path
        self._samples: np.ndarray | None = None
        self._tracks = None
        self._tracks_loaded = False
        self._remixed = False  # load() returned a new mix, not the file

    def load(self) -> np.ndarray:
        if self._samples is None:
            self._samples = self._remix()
            self._remixed = self._samples is not None
            if self._samples is None:
                self._samples = load_audio_universal(self.path)
        return self._samples

    @contextmanager
    def as_file(self):
        """A path to the samples load() returns, for Backends that read a
        file (Apple SpeechAnalyzer): the recording itself, or a temporary
        WAV of the new mix (see the class), removed again afterwards (also
        when the Backend fails)."""
        samples = self.load()
        if not self._remixed:
            yield self.path
            return
        fd, tmp_path = tempfile.mkstemp(prefix="voxscribe-mix-", suffix=".wav")
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
            self._tracks = load_tracks(self.path)
            self._tracks_loaded = True
        return self._tracks

    def _remix(self) -> np.ndarray | None:
        """The new mix of the Call's tracks, or None to use the saved file."""
        from voxscribe.recording_modes import SETUP_LOUDSPEAKER
        from voxscribe.tracks import TRACK_NAMES, balance, load_setup, mix
        try:
            if self.tracks() is None:
                return None
            loudspeaker = load_setup(self.path) == SETUP_LOUDSPEAKER
        except Exception as e:
            print(f"     [WARNING] Could not load the Call's tracks ({e}) - "
                  "transcribing the saved mix.")
            return None
        mic, system = self._tracks
        balanced = balance(mic, system)
        if not loudspeaker and balanced is None:
            return None
        if loudspeaker:
            print("     Echo removal ran: transcribing the mix of the cleaned "
                  "Microphone track and the Playback track.")
        if balanced is not None:
            (mic, system), boosted, gain_db = balanced
            print(f"     {TRACK_NAMES[boosted]} track boosted by "
                  f"{gain_db:.0f} dB for transcription (its speech was much quieter).")
        return mix(mic, system, full_scale=32767 / 32768).astype(np.float32)
