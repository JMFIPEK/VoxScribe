"""Maps Model ID prefixes to Backends.

Backends are registered as loaders and imported on first use, so importing
the registry never pulls in torch or whisperx.
"""

from typing import Callable

from voxscribe.transcription.backend import Backend


class BackendRegistry:
    def __init__(self):
        self._loaders: dict[str, Callable[[], Backend]] = {}
        self._loaded: dict[str, Backend] = {}

    def register(self, prefix: str, loader: Callable[[], Backend]) -> None:
        """Registers `loader` for Model IDs starting with `prefix`.
        The empty prefix is the default Backend for IDs without one."""
        if prefix in self._loaders:
            raise ValueError(f"Backend prefix already registered: {prefix!r}")
        self._loaders[prefix] = loader

    def resolve(self, model_id: str) -> Backend:
        """Returns the Backend for `model_id` (longest matching prefix wins)."""
        matches = [p for p in self._loaders if model_id.startswith(p)]
        if not matches:
            raise LookupError(f"No Backend for Model ID: {model_id!r}")
        prefix = max(matches, key=len)
        if prefix not in self._loaded:
            self._loaded[prefix] = self._loaders[prefix]()
        return self._loaded[prefix]


default_registry = BackendRegistry()
