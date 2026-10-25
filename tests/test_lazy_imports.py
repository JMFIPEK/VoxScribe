"""Importing modules the GUI loads eagerly must not pull in torch/whisperx
(a cold import takes minutes) or livekit (only the echo canceller needs it).
Each check runs in a fresh interpreter, because sys.modules is shared across
tests."""

import subprocess
import sys
from pathlib import Path

import pytest

# -I keeps the user's environment out, but also drops the current directory
# from sys.path; put this checkout first, so the code under test is imported
# rather than whichever checkout the venv was installed from.
_ROOT = str(Path(__file__).resolve().parent.parent)


def _heavy_modules_loaded_by(module: str) -> set[str]:
    code = (
        f"import importlib, sys; sys.path.insert(0, {_ROOT!r}); "
        f"importlib.import_module({module!r}); "
        "print(' '.join(m for m in ('torch', 'whisperx', 'livekit') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True,
                         text=True, check=True).stdout
    return set(out.split())


@pytest.mark.parametrize("module", [
    "voxscribe.paths",
    "voxscribe.providers",
    "voxscribe.tracks",
    "voxscribe.echo",
    "voxscribe.speaker_profiles",
    "voxscribe.transcription",
    "voxscribe.transcription.align",
    "voxscribe.transcription.audio",
    "voxscribe.transcription.diarize",
    "voxscribe.transcription.speakers",
    "voxscribe.transcription.formats",
    "voxscribe.transcription.language",
    "voxscribe.transcription.model_ids",
    "voxscribe.transcription.pipeline",
    "voxscribe.transcription.registry",
    "voxscribe.transcription.backends",
    "voxscribe.transcription.backends.apple",
    "voxscribe.transcription.backends.openvino",
    "voxscribe.transcription.backends.parakeet",
    "voxscribe.transcription.backends.remote",
    "voxscribe.transcription.backends.whisperx_local",
    "voxscribe.transcription.devices",
    "qt_app.constants",
    "qt_app.speaker_view",
    "qt_app.timeline_model",
    "qt_app.timeline_progress",
    "qt_app.transcript_view",
])
def test_cheap_module_stays_light(module):
    assert _heavy_modules_loaded_by(module) == set()


def test_transcriber_facade_stays_light():
    assert _heavy_modules_loaded_by("voxscribe.transcriber") == set()
