"""VoxScribe — builds the standalone .exe with PyInstaller.

Thin entry point - the actual logic lives in voxscribe/build_exe.py.

Usage:
    python build_exe.py

The result is in dist/VoxScribe/VoxScribe.exe
"""

from voxscribe.build_exe import build

if __name__ == "__main__":
    build()
