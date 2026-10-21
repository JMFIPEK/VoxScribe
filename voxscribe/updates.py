"""Installed Release detection - stdlib-only (see docs/adr/0002-*).

An install is on a Release when it is a clean checkout sitting exactly on a
Release tag (vMAJOR.MINOR.PATCH, detached HEAD). Everything else - a branch,
local edits, commits past the tag, no Git - is "development".
"""

import re
import subprocess
import sys

from voxscribe.paths import get_base_dir

DEVELOPMENT = "development"

VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# Each call is a local, fast Git operation; the timeout only guards against
# a hung git (e.g. a stale lock on a network drive).
_GIT_TIMEOUT_SECONDS = 3


def _git(repo, *args) -> str | None:
    """Runs git in repo, returns stdout or None on any failure (no git,
    not a repository, timeout, non-zero exit)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECONDS,
            # No console window flashing up from the windowless pythonw.exe
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def parse_version(text: str) -> tuple[int, int, int] | None:
    """"1.10.0" -> (1, 10, 0), None for anything that isn't MAJOR.MINOR.PATCH.
    Tuples compare numerically, so v1.10.0 sorts above v1.9.0."""
    match = VERSION_RE.match(text)
    return tuple(int(n) for n in match.groups()) if match else None


def parse_release_tag(tag: str) -> tuple[int, int, int] | None:
    """"v1.10.0" -> (1, 10, 0), None if the tag isn't a Release tag."""
    return parse_version(tag[1:]) if tag.startswith("v") else None


def installed_version(repo=None) -> str:
    """Returns the installed Release ("v1.2.3") or "development".

    Runs two local git calls - cheap, but not free on Windows (process
    start), so the GUI calls it off the GUI thread."""
    if getattr(sys, "frozen", False):
        # The PyInstaller .exe has no checkout and gets no Updates (out of
        # scope, see docs/adr/0002-*); it can be built from any commit.
        return DEVELOPMENT

    repo = repo if repo is not None else get_base_dir()

    # Untracked files (.env, recordings/, ...) aren't local edits - they
    # survive a checkout and are mostly gitignored anyway.
    status = _git(repo, "status", "--porcelain=v2", "--branch", "--untracked-files=no")
    if status is None:
        return DEVELOPMENT
    lines = status.splitlines()
    if any(not line.startswith("#") for line in lines):
        return DEVELOPMENT
    # A branch checkout is development even when it sits on the tag (e.g.
    # main right after a Release) - only a detached HEAD gets Updates.
    if "# branch.head (detached)" not in lines:
        return DEVELOPMENT

    tags = _git(repo, "tag", "--points-at", "HEAD")
    if tags is None:
        return DEVELOPMENT
    releases = [t for t in tags.split() if parse_release_tag(t)]
    if not releases:
        return DEVELOPMENT
    return max(releases, key=parse_release_tag)
