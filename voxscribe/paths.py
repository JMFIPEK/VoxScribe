"""Application/bundled-model paths - stdlib-only, so cheap callers (e.g. the
GUI's HardwareInfoController) don't have to import transcriber.py (whisperx/
torch) just to find a folder."""

import os
import sys


def get_base_dir() -> str:
    """Returns the application's base directory (PyInstaller-compatible)."""
    if getattr(sys, "frozen", False):
        # PyInstaller --onedir: the exe lives in dist/VoxScribe/
        return os.path.dirname(sys.executable)
    # Non-frozen: this file lives in voxscribe/, the project root (where
    # bundled_models/ and macos/ actually live) is one level up.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_bundled_models_dir() -> str | None:
    """Returns the path to the bundled_models folder, if present."""
    models_dir = os.path.join(get_base_dir(), "bundled_models")
    if os.path.isdir(models_dir):
        return models_dir
    return None
