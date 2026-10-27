"""`python main.py transcribe`: one or several input files (#43); `record`
and `run`: the recording's file name (#51).

The Backends are replaced by a fake `transcribe()` per file; everything from
there on (multi-file merge, saving the transcript) runs for real. The
recorder is replaced by one that only writes an empty file.
"""

import json
import re
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


# --- record/run: Recording title (#51) ---

@pytest.fixture
def recorded(monkeypatch, tmp_path):
    """The recorder replaced by one that writes an empty file and notes its
    path; run from tmp_path, so recordings/ lands there."""
    from voxscribe import recorder
    paths = []

    def fake(output, **_kwargs):
        paths.append(output)
        open(output, "wb").close()
        return output

    for name in ("record_microphone", "record_system_audio", "record_microphone_and_system"):
        monkeypatch.setattr(recorder, name, fake)
    monkeypatch.chdir(tmp_path)
    return paths


def test_a_call_with_a_name_is_saved_under_that_title(recorded, monkeypatch):
    run_cli("record", "--mode", "call", "--setup", "headset", "--name", "ASAM WG",
            monkeypatch=monkeypatch)

    [path] = recorded
    assert re.fullmatch(r"recordings[\\/]\d{6}_\d{4}_ASAM_WG\.wav", path)


def test_without_a_name_the_recording_is_named_after_its_mode(recorded, monkeypatch):
    run_cli("record", "--mode", "room", monkeypatch=monkeypatch)
    run_cli("record", "--mode", "room", monkeypatch=monkeypatch)

    assert re.fullmatch(r"recordings[\\/]\d{6}_\d{4}_Room\.wav", recorded[0])
    # Same minute (or the next one, should the clock just turn over).
    assert recorded[1] != recorded[0]


def test_an_explicit_output_wins_over_the_name(recorded, monkeypatch, capsys):
    run_cli("record", "--mode", "playback", "--name", "ASAM WG", "-o", "mine.wav",
            monkeypatch=monkeypatch)

    assert recorded == ["mine.wav"]
    assert "--name 'ASAM WG' is ignored" in capsys.readouterr().out


def test_run_records_under_the_name_and_transcribes_next_to_it(recorded, transcribed, tmp_path, monkeypatch):
    _add, calls = transcribed
    # The recording isn't known to `transcribed` in advance.
    monkeypatch.setattr(t, "transcribe",
                        lambda **kw: calls.append(kw["audio_path"]) or _file_result("hi", 2.0))

    run_cli("run", "--mode", "room", "--name", "Standup", monkeypatch=monkeypatch)

    [path] = recorded
    assert calls == [path]
    assert (tmp_path / path).with_suffix(".txt").is_file()
