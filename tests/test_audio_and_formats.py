"""Tests for voxscribe.transcription.audio / .formats (cheap modules, no torch)."""

import json

import numpy as np
import soundfile as sf

from voxscribe.transcription import audio, formats


def test_loader_downmixes_and_resamples_to_16k_mono(tmp_path):
    path = str(tmp_path / "stereo48k.wav")
    sf.write(path, np.zeros((48000, 2), dtype=np.float32), 48000)

    data = audio.load_audio_universal(path)

    assert data.dtype == np.float32
    assert data.ndim == 1
    assert len(data) == audio.SAMPLE_RATE  # 1s at 48kHz -> 1s at 16kHz


def test_loader_keeps_16k_mono_samples(tmp_path):
    path = str(tmp_path / "mono16k.wav")
    samples = np.linspace(-0.5, 0.5, 160, dtype=np.float32)
    sf.write(path, samples, 16000, subtype="FLOAT")

    np.testing.assert_allclose(audio.load_audio_universal(path), samples, atol=1e-6)


def test_save_transcript_writes_requested_formats(tmp_path):
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": "Hi", "speaker": "SPEAKER_00"}]}
    base = str(tmp_path / "out" / "meeting")

    saved = formats.save_transcript(result, base, formats=["txt", "srt", "json", "bogus"])

    assert saved == [f"{base}.txt", f"{base}.srt", f"{base}.json"]
    assert open(f"{base}.txt", encoding="utf-8").read() == "[00:00 - 00:01] SPEAKER_00: Hi"
    assert open(f"{base}.srt", encoding="utf-8").read().startswith("1\n00:00:00,000 --> 00:00:01,000\n")
    assert json.load(open(f"{base}.json", encoding="utf-8")) == result


def test_save_transcript_defaults_to_txt(tmp_path):
    base = str(tmp_path / "x")
    assert formats.save_transcript({"segments": []}, base) == [f"{base}.txt"]


def test_audio_source_decodes_only_once(tmp_path, monkeypatch):
    path = str(tmp_path / "a.wav")
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000)
    calls = []
    real = audio.load_audio_universal
    monkeypatch.setattr(audio, "load_audio_universal", lambda p: calls.append(p) or real(p))

    source = audio.AudioSource(path)
    first, second = source.load(), source.load()

    assert first is second
    assert calls == [path]


# --- Loudspeaker Calls: transcribe the mix without the Echo (#23) ----------

import hashlib  # noqa: E402

import pytest  # noqa: E402

from voxscribe import echo, tracks  # noqa: E402

SR = audio.SAMPLE_RATE


def _loudspeaker_recording(tmp_path, setup="loudspeaker", seconds=20.0):
    """A Call as the recorder saves it: the mono mix plus the sidecar. The
    microphone holds only the Playback's Echo (300 ms late, a third of the
    level). Returns (path, playback in -1..1)."""
    rng = np.random.default_rng(5)
    n = int(seconds * SR)
    bursts = (rng.uniform(size=n // 4000 + 1) < 0.7).repeat(4000)[:n]
    playback = np.convolve(rng.standard_normal(n), np.hanning(7), mode="same") * bursts * 3000
    delay = int(0.3 * SR)
    mic = np.convolve(np.concatenate([np.zeros(delay), playback[:-delay]]), [0.3, 0.15, 0.05])[:n]
    path = str(tmp_path / "call.wav")
    sf.write(path, np.clip(0.65 * (mic + playback), -32768, 32767).astype(np.int16), SR)
    tracks.save_tracks(path, mic, playback, setup=setup)
    return path, playback / 32768


def _level_db(x):
    x = np.asarray(x[2 * SR:], dtype=np.float64)  # after the canceller has adapted
    return 10 * np.log10(np.mean(x ** 2) + 1e-20)


def _hashes(path):
    return [hashlib.sha256(open(p, "rb").read()).hexdigest()
            for p in (path, tracks.tracks_path_for(path))]


def test_loudspeaker_recording_is_transcribed_without_the_echo(tmp_path, capsys):
    path, playback = _loudspeaker_recording(tmp_path)
    before = _hashes(path)
    raw_echo = audio.load_audio_universal(path) - 0.65 * playback

    mix = audio.AudioSource(path).load()

    assert mix.dtype == np.float32 and len(mix) == len(playback)
    assert _level_db(mix - 0.65 * playback) <= _level_db(raw_echo) - 15.0
    assert _hashes(path) == before  # the saved files are never written to
    assert "Echo removal" in capsys.readouterr().out


@pytest.mark.parametrize("setup", ["headset", None])
def test_headset_and_pre_setup_recordings_load_the_saved_mix(tmp_path, setup):
    path, _ = _loudspeaker_recording(tmp_path, setup=setup)

    np.testing.assert_array_equal(audio.AudioSource(path).load(), audio.load_audio_universal(path))


def test_the_echo_canceller_runs_once_per_file(tmp_path, monkeypatch):
    path, _ = _loudspeaker_recording(tmp_path, seconds=5.0)
    calls = []
    real = echo.cancel_echo
    monkeypatch.setattr(echo, "cancel_echo", lambda m, p: calls.append(1) or real(m, p))

    source = audio.AudioSource(path)
    source.load()
    source.tracks()

    assert len(calls) == 1
