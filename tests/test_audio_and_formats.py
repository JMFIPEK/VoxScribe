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
