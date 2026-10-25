"""`python main.py transcribe`: one or several input files (#43).

The Backends are replaced by a fake `transcribe()` per file; everything from
there on (multi-file merge, saving the transcript) runs for real.
"""

import json
import sys

import pytest

from voxscribe import cli, gpu_setup
from voxscribe import transcriber as t
from voxscribe.transcription import pipeline


def _file_result(text, duration):
    return {
        "language": "de",
        "duration": duration,
        "segments": [{"start": 0.0, "end": 2.0, "text": text, "speaker": "SPEAKER_00"}],
        "speaker_id_map": {"SPEAKER_00": "SPEAKER_00"},
        "speaker_embeddings": {"SPEAKER_00": [0.1]},
    }


@pytest.fixture
def transcribed(monkeypatch, tmp_path):
    """Audio files in tmp_path, each "transcribed" to its own text."""
    calls = []
    texts = {}

    def fake(**kwargs):
        calls.append(kwargs["audio_path"])
        return _file_result(texts[kwargs["audio_path"]], duration=10.0)

    monkeypatch.setattr(pipeline, "transcribe", fake)
    monkeypatch.setattr(t, "transcribe", fake)
    monkeypatch.setattr(gpu_setup, "ensure_correct_torch_backend", lambda: None)

    def add(name, text):
        path = tmp_path / name
        path.write_bytes(b"")
        texts[str(path)] = text
        return str(path)

    return add, calls


def run_cli(*argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    cli.main()


def test_two_inputs_give_one_merged_transcript(transcribed, tmp_path, monkeypatch):
    add, calls = transcribed
    one, two = add("one.wav", "hello"), add("two.wav", "world")
    out = str(tmp_path / "merged")

    run_cli("transcribe", "-i", one, "-i", two, "-o", out, "-f", "json", monkeypatch=monkeypatch)

    assert calls == [one, two]
    merged = json.loads((tmp_path / "merged.json").read_text(encoding="utf-8"))
    assert [(s["start"], s["text"], s["speaker"]) for s in merged["segments"]] == [
        (0.0, "hello", "File 1: SPEAKER_00"), (10.0, "world", "File 2: SPEAKER_00")]


def test_one_input_is_transcribed_as_before(transcribed, tmp_path, monkeypatch):
    add, calls = transcribed
    one = add("one.wav", "hello")

    run_cli("transcribe", "-i", one, monkeypatch=monkeypatch)

    assert calls == [one]
    assert (tmp_path / "one.txt").read_text(encoding="utf-8").strip().endswith("SPEAKER_00: hello")


def test_a_missing_input_stops_before_transcribing(transcribed, tmp_path, monkeypatch):
    add, calls = transcribed
    one = add("one.wav", "hello")

    with pytest.raises(SystemExit):
        run_cli("transcribe", "-i", one, "-i", str(tmp_path / "missing.wav"),
                monkeypatch=monkeypatch)
    assert calls == []
