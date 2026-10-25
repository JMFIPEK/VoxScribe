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


def test_probe_duration_reads_the_length_without_decoding(tmp_path, monkeypatch):
    path = str(tmp_path / "a.wav")
    sf.write(path, np.zeros(24000, dtype=np.float32), 48000)
    monkeypatch.setattr(audio, "load_audio_universal", lambda p: pytest.fail("decoded"))

    assert audio.probe_duration(path) == 0.5


def test_probe_duration_of_an_unreadable_file_is_none(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not audio")

    assert audio.probe_duration(str(path)) is None
    assert audio.probe_duration(str(tmp_path / "missing.wav")) is None


# --- Loudspeaker Calls: transcribe the mix without the Echo (#23) ----------

import hashlib  # noqa: E402
import os  # noqa: E402

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


# --- Backends that read a file (Apple SpeechAnalyzer, #24) -----------------

def test_a_loudspeaker_recording_is_handed_out_as_a_cleaned_temporary_file(tmp_path):
    path, playback = _loudspeaker_recording(tmp_path)
    source = audio.AudioSource(path)

    with source.as_file() as file_path:
        assert file_path != path and os.path.isfile(file_path)
        cleaned, sr = sf.read(file_path, dtype="float32")
        assert sr == SR
        assert _level_db(cleaned - 0.65 * playback) <= _level_db(
            audio.load_audio_universal(path) - 0.65 * playback) - 15.0

    assert not os.path.exists(file_path)


def test_the_temporary_file_is_removed_when_the_backend_fails(tmp_path):
    path, _ = _loudspeaker_recording(tmp_path, seconds=5.0)

    with pytest.raises(RuntimeError):
        with audio.AudioSource(path).as_file() as file_path:
            raise RuntimeError("helper crashed")

    assert not os.path.exists(file_path)


@pytest.mark.parametrize("setup", ["headset", None])
def test_other_recordings_are_handed_out_as_the_original_file(tmp_path, setup):
    path, _ = _loudspeaker_recording(tmp_path, setup=setup, seconds=5.0)

    with audio.AudioSource(path).as_file() as file_path:
        assert file_path == path
    assert os.path.isfile(path)


# --- A quiet Microphone track in the transcribed mix (#39) -----------------

def _call_with_levels(tmp_path, mic_db, playback_db, setup="headset", seconds=20.0):
    """A Call whose local speech (on the Microphone track, 2-8 s and
    12-18 s) and remote speech (Playback track, 0-2, 8-12, 18-20 s) have the
    given speech levels in dBFS. Returns (path, mic_spans, playback_spans)."""
    rng = np.random.default_rng(7)
    n = int(seconds * SR)
    mic_spans, pb_spans = [(2, 8), (12, 18)], [(0, 2), (8, 12), (18, 20)]

    def track(spans, level_db):
        x = np.zeros(n)
        for a, b in spans:
            x[int(a * SR):int(b * SR)] = rng.standard_normal(int((b - a) * SR))
        return x * 32768 * 10 ** (level_db / 20)

    mic, playback = track(mic_spans, mic_db), track(pb_spans, playback_db)
    path = str(tmp_path / "call.wav")
    sf.write(path, np.clip(0.65 * (mic + playback), -32768, 32767).astype(np.int16), SR)
    tracks.save_tracks(path, mic, playback, setup=setup)
    return path, mic_spans, pb_spans


def _span_db(audio_samples, spans):
    x = np.concatenate([audio_samples[int(a * SR) + 800:int(b * SR) - 800] for a, b in spans])
    return 10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20)


def test_a_quiet_microphone_is_brought_up_to_the_playback_level(tmp_path, capsys):
    path, mic_spans, pb_spans = _call_with_levels(tmp_path, mic_db=-44, playback_db=-22)
    saved = audio.load_audio_universal(path)
    assert _span_db(saved, pb_spans) - _span_db(saved, mic_spans) > 20  # the problem

    mix = audio.AudioSource(path).load()

    assert abs(_span_db(mix, pb_spans) - _span_db(mix, mic_spans)) <= 3
    assert "Microphone" in capsys.readouterr().out  # the log says it was boosted


def test_tracks_at_similar_levels_load_the_saved_mix_unchanged(tmp_path):
    path, _, _ = _call_with_levels(tmp_path, mic_db=-25, playback_db=-22)

    np.testing.assert_array_equal(audio.AudioSource(path).load(), audio.load_audio_universal(path))


def test_a_silent_microphone_track_is_not_boosted_into_noise(tmp_path):
    path, _, _ = _call_with_levels(tmp_path, mic_db=-200, playback_db=-22)

    np.testing.assert_array_equal(audio.AudioSource(path).load(), audio.load_audio_universal(path))


def test_the_boost_is_limited(tmp_path):
    path, mic_spans, pb_spans = _call_with_levels(tmp_path, mic_db=-65, playback_db=-20)

    mix = audio.AudioSource(path).load()

    gap = _span_db(mix, pb_spans) - _span_db(mix, mic_spans)
    assert 23 <= gap <= 27  # 45 dB apart, boosted by at most 20 dB


def test_rare_local_speech_is_measured_as_speech_not_as_noise(tmp_path):
    # The local person says one sentence (1 s) in 60 s; the rest of the
    # Microphone track is room noise 30 dB below it.
    rng = np.random.default_rng(3)
    n = 60 * SR
    mic = rng.standard_normal(n) * 32768 * 10 ** (-74 / 20)
    mic[30 * SR:31 * SR] = rng.standard_normal(SR) * 32768 * 10 ** (-44 / 20)
    playback = np.zeros(n)
    playback[: 50 * SR] = rng.standard_normal(50 * SR) * 32768 * 10 ** (-40 / 20)
    path = str(tmp_path / "call.wav")
    sf.write(path, np.clip(0.65 * (mic + playback), -32768, 32767).astype(np.int16), SR)
    tracks.save_tracks(path, mic, playback, setup="headset")

    # The speech levels are only 4 dB apart, so nothing is boosted - measured
    # on the noise, the Microphone would look 34 dB quieter.
    np.testing.assert_array_equal(audio.AudioSource(path).load(), audio.load_audio_universal(path))


def test_a_microphone_with_only_room_noise_is_not_boosted(tmp_path):
    path, _, _ = _call_with_levels(tmp_path, mic_db=-200, playback_db=-22)
    d, _ = sf.read(tracks.tracks_path_for(path), dtype="float64", always_2d=True)
    noise = np.random.default_rng(4).standard_normal(len(d)) * 10 ** (-60 / 20) * 32768
    tracks.save_tracks(path, noise, d[:, 1] * 32768, setup="headset")
    sf.write(path, np.clip(0.65 * (noise + d[:, 1] * 32768), -32768, 32767).astype(np.int16), SR)

    np.testing.assert_array_equal(audio.AudioSource(path).load(), audio.load_audio_universal(path))
