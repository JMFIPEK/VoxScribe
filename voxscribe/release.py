"""Cuts a Release: bumps the version, commits, tags vX.Y.Z and pushes branch
and tag to every remote users install from (see docs/adr/0002-*).

Maintainer tooling, not shipped behaviour - run via the thin root release.py.
"""

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

from voxscribe import updates
from voxscribe.gpu_setup import find_uv
from voxscribe.paths import get_base_dir

# The private GitHub mirror ("github") is deliberately not listed: it's a
# personal backup and must never receive Releases.
REMOTES = ("origin", "public")
BRANCH = "main"


# pyproject.toml's [project] name, also the package's entry in uv.lock
PROJECT_NAME = "whisperx-recorder"


class ReleaseError(Exception):
    """A precondition failed or a step went wrong; the message says what."""


def parse_version(text: str) -> tuple[int, int, int]:
    """"1.10.0" -> (1, 10, 0); tuples compare numerically. Raises ValueError
    for anything that isn't MAJOR.MINOR.PATCH."""
    parsed = updates.parse_version(text)
    if parsed is None:
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {text!r}")
    return parsed


# Both files are edited textually (not re-serialized), so the commit diff is
# exactly one line each and comments/formatting stay untouched. \r?\n:
# checkouts with core.autocrlf have CRLF line endings on Windows.
_PYPROJECT_VERSION_RE = re.compile(
    r'(^\[project\]\r?\n(?:(?!\[).*\n)*?version = ")([^"]*)(")', re.MULTILINE)
_LOCK_VERSION_RE = re.compile(
    r'(^\[\[package\]\]\r?\nname = "' + re.escape(PROJECT_NAME) + r'"\r?\nversion = ")([^"]*)(")',
    re.MULTILINE)


def _replace_once(pattern, text: str, new_version: str, what: str) -> str:
    result, count = pattern.subn(lambda m: m.group(1) + new_version + m.group(3), text)
    if count != 1:
        raise ReleaseError(f"could not find the project version in {what}")
    return result


def read_version(pyproject_text: str) -> str:
    """The [project] version from pyproject.toml's text."""
    match = _PYPROJECT_VERSION_RE.search(pyproject_text)
    if not match:
        raise ReleaseError("could not find the project version in pyproject.toml")
    return match.group(2)


def bump_pyproject(text: str, new_version: str) -> str:
    return _replace_once(_PYPROJECT_VERSION_RE, text, new_version, "pyproject.toml")


def bump_uv_lock(text: str, new_version: str) -> str:
    """uv.lock records the project's own version in its editable package
    entry; a stale one makes `uv sync` rewrite uv.lock in every install."""
    return _replace_once(_LOCK_VERSION_RE, text, new_version, "uv.lock")


def _git(repo: Path, *args) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise ReleaseError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_ok(repo: Path, *args) -> bool:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True).returncode == 0


def _check_preconditions(repo: Path, new_version: str) -> str:
    """Everything that can refuse the Release - read-only, so the dry-run
    runs it too. Returns the current version."""
    try:
        new = parse_version(new_version)
    except ValueError as e:
        raise ReleaseError(str(e)) from None

    if not _git_ok(repo, "rev-parse", "--git-dir"):
        raise ReleaseError(f"{repo} is not a Git checkout")
    branch = subprocess.run(["git", "-C", str(repo), "symbolic-ref", "--quiet", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    if branch != BRANCH:
        raise ReleaseError(f"releases are cut from {BRANCH}, but the current branch is "
                           f"{branch or '(detached HEAD)'}")
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ReleaseError("the working tree has uncommitted changes - commit or stash them first")

    current = read_version((repo / "pyproject.toml").read_text(encoding="utf-8"))
    if new <= parse_version(current):
        raise ReleaseError(f"{new_version} is not greater than the current version {current}")
    if _git_ok(repo, "rev-parse", "--quiet", "--verify", f"refs/tags/v{new_version}"):
        raise ReleaseError(f"tag v{new_version} already exists")

    remotes = _git(repo, "remote").split()
    for remote in REMOTES:
        if remote not in remotes:
            raise ReleaseError(f"remote {remote!r} is not configured")
        # Uses the last-fetched state, no network: catches the common case
        # of a stale local main before anything is committed or tagged.
        tracking = f"refs/remotes/{remote}/{BRANCH}"
        if (_git_ok(repo, "rev-parse", "--quiet", "--verify", tracking)
                and not _git_ok(repo, "merge-base", "--is-ancestor", tracking, "HEAD")):
            raise ReleaseError(f"{remote}/{BRANCH} has commits that local {BRANCH} lacks - "
                               f"pull/merge them first")

    if (repo / "uv.lock").exists() and find_uv() is None:
        raise ReleaseError("uv.lock needs a version bump verified by uv, but uv was not found")
    return current


def release(repo, new_version: str, dry_run: bool = False, log=print) -> None:
    """Bumps pyproject.toml (and uv.lock) to new_version, commits, tags
    v<new_version> and pushes main + tag to every remote in REMOTES. With
    dry_run, checks everything and prints the steps without changing
    anything. Raises ReleaseError (message ready for the user) on refusal
    or failure."""
    repo = Path(repo)
    current = _check_preconditions(repo, new_version)
    tag = f"v{new_version}"
    prefix = "[dry-run] " if dry_run else ""

    def run(*cmd):
        log(f"{prefix}$ {shlex.join(cmd)}")
        if dry_run:
            return
        result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        if result.returncode != 0:
            raise ReleaseError(f"{shlex.join(cmd)} failed: {(result.stderr or result.stdout).strip()}")

    files = ["pyproject.toml"] + (["uv.lock"] if (repo / "uv.lock").exists() else [])
    log(f"{prefix}Release {current} -> {new_version} ({tag}) from {repo}")
    log(f"{prefix}Set version {current} -> {new_version} in {', '.join(files)}")
    if not dry_run:
        originals = {}
        for name in files:
            path = repo / name
            # newline="" keeps the file's line endings exactly as they are
            with open(path, encoding="utf-8", newline="") as f:
                originals[name] = f.read()
            bump = bump_pyproject if name == "pyproject.toml" else bump_uv_lock
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(bump(originals[name], new_version))
    if "uv.lock" in files:
        try:
            # Only checks; never re-resolves or syncs the environment.
            run(find_uv() or "uv", "lock", "--check")
        except ReleaseError:
            for name, text in originals.items():
                with open(repo / name, "w", encoding="utf-8", newline="") as f:
                    f.write(text)
            raise

    run("git", "add", *files)
    run("git", "commit", "-q", "-m", f"Release {tag}")
    run("git", "tag", "-a", tag, "-m", f"Release {tag}")

    pushed = []
    for remote in REMOTES:
        try:
            # --atomic: branch and tag land together or not at all, per remote
            run("git", "push", "--atomic", remote, BRANCH, tag)
        except ReleaseError as e:
            not_pushed = [r for r in REMOTES if r not in pushed]
            raise ReleaseError(
                f"{e}\n\nCommit and tag {tag} exist locally.\n"
                f"Pushed to: {', '.join(pushed) or 'none'}\n"
                f"NOT pushed to: {', '.join(not_pushed)}\n"
                f"Fix the problem, then finish with:\n"
                + "\n".join(f"  git push --atomic {r} {BRANCH} {tag}" for r in not_pushed)
            ) from None
        pushed.append(remote)

    log(f"{prefix}Done: {tag} " + ("would be pushed" if dry_run else "pushed")
        + f" to {', '.join(REMOTES)}.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Cut a Release: bump the version, commit, tag vX.Y.Z and push "
                    f"{BRANCH} + tag to {' and '.join(REMOTES)}.")
    parser.add_argument("version", help="the new version, MAJOR.MINOR.PATCH (e.g. 1.1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="check everything and print every step without changing anything")
    args = parser.parse_args(argv)
    try:
        release(get_base_dir(), args.version, dry_run=args.dry_run)
    except ReleaseError as e:
        print(f"Release refused/failed: {e}", file=sys.stderr)
        return 1
    return 0
