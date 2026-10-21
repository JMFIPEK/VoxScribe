"""VoxScribe — cuts a Release (maintainer tooling).

Thin entry point - the actual logic lives in voxscribe/release.py.

Usage (on a clean main checkout):
    python release.py 1.1.0 --dry-run   # check everything, print every step
    python release.py 1.1.0             # bump, commit, tag v1.1.0, push to origin + public
"""

import sys

from voxscribe.release import main

if __name__ == "__main__":
    sys.exit(main())
