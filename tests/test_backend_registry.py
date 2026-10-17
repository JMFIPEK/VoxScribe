"""The registry maps Model ID prefixes to Backends (docs/adr/0001)."""

import pytest

from voxscribe.transcription.registry import BackendRegistry


class _Stub:
    def __init__(self, name):
        self.name = name

    def transcribe(self, audio, language, model_id, on_progress=None):
        return {"segments": [], "language": self.name}


def test_resolves_by_prefix():
    reg = BackendRegistry()
    reg.register("server:", lambda: _Stub("remote"))
    reg.register("parakeet:", lambda: _Stub("parakeet"))

    assert reg.resolve("server:my-provider").name == "remote"
    assert reg.resolve("parakeet:tdt-0.6b-v3").name == "parakeet"


def test_unprefixed_model_id_uses_the_default_backend():
    reg = BackendRegistry()
    reg.register("", lambda: _Stub("whisperx"))
    reg.register("server:", lambda: _Stub("remote"))

    assert reg.resolve("large-v3").name == "whisperx"


def test_longest_prefix_wins():
    reg = BackendRegistry()
    reg.register("", lambda: _Stub("default"))
    reg.register("openvino:", lambda: _Stub("openvino"))
    reg.register("openvino:NPU:", lambda: _Stub("npu"))

    assert reg.resolve("openvino:GPU:medium").name == "openvino"
    assert reg.resolve("openvino:NPU:small").name == "npu"


def test_loader_runs_lazily_and_only_once():
    calls = []

    def loader():
        calls.append(1)
        return _Stub("remote")

    reg = BackendRegistry()
    reg.register("server:", loader)
    assert calls == []  # registering must not import the backend

    first = reg.resolve("server:a")
    second = reg.resolve("server:b")

    assert calls == [1]
    assert first is second


def test_no_match_raises_lookup_error():
    reg = BackendRegistry()
    reg.register("server:", lambda: _Stub("remote"))

    with pytest.raises(LookupError, match="large-v3"):
        reg.resolve("large-v3")


def test_duplicate_prefix_is_rejected():
    reg = BackendRegistry()
    reg.register("server:", lambda: _Stub("a"))

    with pytest.raises(ValueError):
        reg.register("server:", lambda: _Stub("b"))
