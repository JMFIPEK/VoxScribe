"""VoxScribe — pre-download models for offline/bundled operation.

Thin entry point - the actual logic lives in voxscribe/download_models.py.

Usage:
    python download_models.py
"""

from voxscribe.download_models import main

if __name__ == "__main__":
    main()
