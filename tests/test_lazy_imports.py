"""Importing modules the GUI loads eagerly must not pull in torch/whisperx
(a cold import takes minutes). Each check runs in a fresh interpreter,
because sys.modules is shared across tests."""

import subprocess
import sys

import pytest


def _heavy_modules_loaded_by(module: str) -> set[str]:
    code = (
        "import importlib, sys; "
        f"importlib.import_module({module!r}); "
        "print(' '.join(m for m in ('torch', 'whisperx') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True,
                         text=True, check=True).stdout
    return set(out.split())


@pytest.mark.parametrize("module", [
    "voxscribe.paths",
    "voxscribe.providers",
    "voxscribe.tracks",
    "voxscribe.speaker_profiles",
    "voxscribe.transcription",
    "voxscribe.transcription.align",
    "voxscribe.transcription.audio",
    "voxscribe.transcription.diarize",
    "voxscribe.transcription.speakers",
    "voxscribe.transcription.formats",
    "voxscribe.transcription.language",
    "voxscribe.transcription.model_ids",
    "voxscribe.transcription.registry",
    "voxscribe.transcription.backends",
    "voxscribe.transcription.backends.apple",
    "voxscribe.transcription.backends.openvino",
    "voxscribe.transcription.backends.parakeet",
    "voxscribe.transcription.backends.remote",
    "voxscribe.transcription.backends.whisperx_local",
    "voxscribe.transcription.devices",
    "qt_app.constants",
])
def test_cheap_module_stays_light(module):
    assert _heavy_modules_loaded_by(module) == set()


@pytest.mark.xfail(strict=True, reason=(
    "The facade still holds all code and imports torch/whisperx at load. "
    "Becomes lazy once the backends move out (refactor steps 4-9); "
    "then remove this marker."))
def test_transcriber_facade_stays_light():
    assert _heavy_modules_loaded_by("voxscribe.transcriber") == set()
