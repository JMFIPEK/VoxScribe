"""Characterization tests for voxscribe.track_assembler.

They pin what a Call recording produced before the track assembler existed
(the after-recording code of recorder._record_both()/_record_both_macos()):
the golden hashes below were computed with that code on the same inputs.
"""

import hashlib

import numpy as np
import soundfile as sf

from voxscribe import tracks
from voxscribe.track_assembler import TrackAssembler


def _chunks(seed, rate, channels, seconds, frames_per_chunk):
    """Random int16 PCM as the recorder receives it: a list of byte chunks,
    interleaved if stereo."""
    rng = np.random.default_rng(seed)
    pcm = rng.integers(-12000, 12000, size=(int(rate * seconds), channels), dtype=np.int16)
    data = pcm.tobytes()
    step = frames_per_chunk * channels * 2
    return [data[i:i + step] for i in range(0, len(data), step)]


def _record(assembler, mic_chunks, playback_chunks):
    for chunk in mic_chunks:
        assembler.add_chunk(tracks.MIC, chunk)
    for chunk in playback_chunks:
        assembler.add_chunk(tracks.SYSTEM, chunk)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sidecar(audio_path):
    data, sr = sf.read(tracks.tracks_path_for(audio_path), dtype="int16", always_2d=True)
    assert sr == 16000
    return np.ascontiguousarray(data)


def _wav_samples(audio_path):
    data, sr = sf.read(audio_path, dtype="int16")
    assert sr == 16000
    return data


def test_windows_like_call_matches_todays_files(tmp_path):
    # 48kHz stereo microphone, 44.1kHz stereo playback that delivered only
    # half as long - both resampled to 16kHz, playback padded with silence.
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=48000, mic_channels=2,
                               playback_rate=44100, playback_channels=2)
    _record(assembler, _chunks(1, 48000, 2, 1.0, 3072), _chunks(2, 44100, 2, 0.5, 2822))

    mix = assembler.save(out, setup="headset")

    assert mix.dtype == np.int16 and len(mix) == 16000
    with open(out, "rb") as f:
        assert _sha(f.read()) == "bcce6ff3db526dfd356d4856dc006a5c1b85373501711b8ac7fa30d9829b0c94"
    sidecar = _sidecar(out)
    assert sidecar.shape == (16000, 2)
    assert not sidecar[8000:, 1].any()  # shorter playback track padded with silence
    assert _sha(sidecar.tobytes()) == "a6d93b3f4b9c6620a0f816720df1fbe50fd1043b0e9b2bb5229d6b7f2f280cb4"
    assert tracks.load_setup(out) == "headset"


def test_macos_like_call_matches_todays_files(tmp_path):
    # 44.1kHz mono microphone shorter than the 16kHz mono playback
    # (SystemAudioCapture's fixed format), no Setup given.
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=44100, mic_channels=1,
                               playback_rate=16000, playback_channels=1)
    _record(assembler, _chunks(3, 44100, 1, 0.5, 1024), _chunks(4, 16000, 1, 1.0, 1024))

    mix = assembler.save(out)

    assert len(mix) == 16000
    with open(out, "rb") as f:
        assert _sha(f.read()) == "1df878075ad30de5a6e6c12de93a1b2a0c2ad6dc6d1aa81e214e543b4b481bda"
    sidecar = _sidecar(out)
    assert not sidecar[8000:, 0].any()  # shorter microphone track padded
    assert _sha(sidecar.tobytes()) == "55d057019250e4b6f862867d41f18f9d441c4e573bfd061d5bb0dc4d446a97d4"
    assert tracks.load_setup(out) is None


def test_mix_is_both_tracks_at_065_padded_to_the_longer_one(tmp_path):
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=16000, mic_channels=1,
                               playback_rate=16000, playback_channels=1)
    assembler.add_chunk(tracks.MIC, np.array([1000, -1000], dtype=np.int16).tobytes())
    assembler.add_chunk(tracks.MIC, np.array([500], dtype=np.int16).tobytes())
    assembler.add_chunk(tracks.SYSTEM, np.array([2000], dtype=np.int16).tobytes())

    mix = assembler.save(out, setup="loudspeaker")

    assert mix.tolist() == [1950, -650, 325]
    assert _wav_samples(out).tolist() == [1950, -650, 325]
    assert _sidecar(out).tolist() == [[1000, 2000], [-1000, 0], [500, 0]]
    assert tracks.load_setup(out) == "loudspeaker"


def test_mix_peak_is_scaled_down_to_int16_range(tmp_path):
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=16000, mic_channels=1,
                               playback_rate=16000, playback_channels=1)
    assembler.add_chunk(tracks.MIC, np.array([30000, 100], dtype=np.int16).tobytes())
    assembler.add_chunk(tracks.SYSTEM, np.array([30000, -100], dtype=np.int16).tobytes())

    assert assembler.save(out).tolist() == [32767, 0]


def test_stereo_chunks_are_downmixed_to_the_channel_mean(tmp_path):
    out = str(tmp_path / "call.wav")
    assembler = TrackAssembler(mic_rate=16000, mic_channels=2,
                               playback_rate=16000, playback_channels=1)
    assembler.add_chunk(tracks.MIC, np.array([100, 300, -50, -150], dtype=np.int16).tobytes())
    assembler.add_chunk(tracks.SYSTEM, np.array([0, 0], dtype=np.int16).tobytes())

    assembler.save(out)

    assert _sidecar(out)[:, 0].tolist() == [200, -100]


def test_has_audio_reports_which_sources_delivered_chunks():
    assembler = TrackAssembler(mic_rate=48000, mic_channels=1,
                               playback_rate=48000, playback_channels=2)
    assembler.add_chunk(tracks.MIC, bytes(4))

    assert assembler.has_audio(tracks.MIC)
    assert not assembler.has_audio(tracks.SYSTEM)
