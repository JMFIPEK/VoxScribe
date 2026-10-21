"""Updates to new Releases - stdlib-only (see docs/adr/0002-*, spec #15).

An install is on a Release when it is a clean checkout sitting exactly on a
Release tag (vMAJOR.MINOR.PATCH, detached HEAD). Everything else - a branch,
local edits, commits past the tag, no Git - is "development" and never gets
Updates.

check() asks the remote the install was cloned from for a newer Release;
the app runs it in the background and offers the Update. Everything that
touches Git lives here; the app and the launchers only call this module.
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass

from voxscribe.paths import get_base_dir

DEVELOPMENT = "development"

VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# Local Git calls are fast; the timeout only guards against a hung git (e.g.
# a stale lock on a network drive).
_GIT_TIMEOUT_SECONDS = 3
# Asking the remote goes over the network (GitLab outside the VPN can just
# hang), so it gets longer - the check runs in the background anyway.
_REMOTE_TIMEOUT_SECONDS = 20

# Never ask for credentials: an expired token must fail the check quietly,
# not hang it or pop up a login dialog (spec #15, stories 9-10).
_NO_PROMPT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}


def _run_git(repo, *args, timeout=_GIT_TIMEOUT_SECONDS):
    """Runs git in repo; returns the CompletedProcess, or None if git
    couldn't run at all (not installed, timeout)."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, **_NO_PROMPT_ENV},
            # No console window flashing up from the windowless pythonw.exe
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _git(repo, *args, timeout=_GIT_TIMEOUT_SECONDS) -> str | None:
    """Runs git in repo, returns stdout or None on any failure (no git,
    not a repository, timeout, non-zero exit)."""
    result = _run_git(repo, *args, timeout=timeout)
    if result is None or result.returncode != 0:
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


@dataclass
class CheckResult:
    """status: "available" (latest is newer than installed), "up_to_date",
    "not_eligible" (development checkout, never updated) or "failed"
    (remote unreachable etc.; error says why - only logged, never shown)."""
    status: str
    installed: str
    latest: str | None = None
    error: str | None = None


def _base(repo):
    return repo if repo is not None else get_base_dir()


def update_remote(repo=None) -> str:
    """The remote the install was cloned from: the upstream of its default
    branch (main/master, else any branch with an upstream), else "origin".
    No remote name or URL is hardcoded (docs/adr/0002-*)."""
    out = _git(_base(repo), "for-each-ref", "--format=%(refname:short) %(upstream:remotename)",
               "refs/heads") or ""
    upstreams = dict(line.split(" ", 1) for line in out.splitlines() if " " in line.strip())
    for branch in ("main", "master", *upstreams):
        if upstreams.get(branch):
            return upstreams[branch]
    return "origin"


def remote_releases(repo=None) -> list[str] | None:
    """The Release tags on the update remote, without fetching; None if the
    remote couldn't be asked."""
    repo = _base(repo)
    result = _run_git(repo, "ls-remote", "--tags", "--refs", update_remote(repo),
                      timeout=_REMOTE_TIMEOUT_SECONDS)
    if result is None or result.returncode != 0:
        return None
    tags = [line.rsplit("refs/tags/", 1)[-1] for line in result.stdout.splitlines()]
    return [t for t in tags if parse_release_tag(t)]


def check(repo=None) -> CheckResult:
    """Is a newer Release available for this install? Never raises."""
    installed = installed_version(repo)
    if installed == DEVELOPMENT:
        return CheckResult("not_eligible", installed)
    releases = remote_releases(repo)
    if releases is None:
        return CheckResult("failed", installed,
                           error=f"Could not list Releases on '{update_remote(repo)}'")
    latest = max(releases, key=parse_release_tag, default=installed)
    if parse_release_tag(latest) > parse_release_tag(installed):
        return CheckResult("available", installed, latest)
    return CheckResult("up_to_date", installed, installed)


# --- Installing an Update (the launchers, #34/#35) -------------------------
#
# The app never replaces its own files: clicking the notice leaves a marker
# naming the target Release and quits; the launcher runs apply() before it
# starts the app again, and the app shows the result marker once.

PENDING_FILE = "update-pending.txt"
RESULT_FILE = "update-result.json"
# A checkout rewrites the working tree; generous for slow disks.
_CHECKOUT_TIMEOUT_SECONDS = 120


def _user_dir(user_dir):
    return user_dir if user_dir is not None else os.path.join(os.path.expanduser("~"), ".voxscribe")


def request_update(tag: str, user_dir=None) -> str:
    """Leaves the marker for the launcher; returns its path."""
    directory = _user_dir(user_dir)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, PENDING_FILE)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tag + "\n")
    return path


def pending_update(user_dir=None) -> str | None:
    """The Release a marker asks for, or None."""
    try:
        with open(os.path.join(_user_dir(user_dir), PENDING_FILE), encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def take_update_result(user_dir=None) -> dict | None:
    """The last apply()'s result ({"ok", "target", "installed", "error"}),
    removed on reading so the app shows it once; None if there is none."""
    import json
    path = os.path.join(_user_dir(user_dir), RESULT_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, ValueError):
        return None
    try:
        os.remove(path)
    except OSError:
        pass
    return result


@dataclass
class ApplyOutcome:
    ok: bool
    target: str
    installed: str  # the Release the install is on afterwards
    error: str | None = None


LOG_FILE = "voxscribe.log"  # the app's log (gui_qt.py), so details end up in one place


def _log(message: str, user_dir=None):
    """Prints and appends to the app's log: the launcher's console window
    closes right after an Update, so its output would be lost otherwise."""
    print(message)
    try:
        directory = _user_dir(user_dir)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, LOG_FILE), "a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
    except OSError:
        pass


def uv_sync(uv: str = "uv", args=(), user_dir=None):
    """The environment sync apply() runs by default: `uv sync --locked
    [args]` in the install (the Windows launcher passes its Intel Arc
    arguments). --locked: a Release whose uv.lock doesn't match fails and
    is rolled back, instead of uv rewriting uv.lock - a dirty checkout
    would never get an Update again."""
    def sync(repo) -> bool:
        command = [uv, "sync", "--locked", *args]
        _log("Update: " + " ".join(command), user_dir)
        # The install's own .venv, whatever environment launched us.
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        try:
            result = subprocess.run(command, cwd=str(repo), env=env,
                                    capture_output=True, text=True, errors="replace")
        except OSError as e:
            _log(f"Update: could not run uv: {e}", user_dir)
            return False
        output = (result.stdout + result.stderr).strip()
        if output:
            _log(output, user_dir)
        return result.returncode == 0
    return sync


def apply(target: str, repo=None, sync=None, user_dir=None) -> ApplyOutcome:
    """Moves the install to Release target: fetch the tag, check it out,
    sync the environment. On any failure the previous Release is checked out
    and synced again. Writes the result marker and consumes the pending
    marker either way. Never raises."""
    repo = _base(repo)
    sync = sync or uv_sync()
    try:
        os.remove(os.path.join(_user_dir(user_dir), PENDING_FILE))  # never retried in a loop
    except OSError:
        pass

    previous = installed_version(repo)
    outcome = _apply(target, repo, sync, previous)
    _log(f"Update to {target}: {'done' if outcome.ok else 'failed - ' + str(outcome.error)}"
         f" (installed: {outcome.installed})", user_dir)
    _write_result(outcome, user_dir)
    return outcome


def _apply(target, repo, sync, previous) -> ApplyOutcome:
    if previous == DEVELOPMENT:
        return ApplyOutcome(False, target, previous, "the install is not on a Release")
    if not parse_release_tag(target):
        return ApplyOutcome(False, target, previous, f"'{target}' is not a Release tag")

    fetch = _run_git(repo, "fetch", "--quiet", "--no-tags", update_remote(repo),
                     f"refs/tags/{target}:refs/tags/{target}", timeout=_REMOTE_TIMEOUT_SECONDS)
    if fetch is None or fetch.returncode != 0:
        return ApplyOutcome(False, target, previous, "could not fetch the Release: "
                            + (fetch.stderr.strip() if fetch else "git did not run"))

    checkout = _run_git(repo, "checkout", "--quiet", "--detach", target,
                        timeout=_CHECKOUT_TIMEOUT_SECONDS)
    if checkout is not None and checkout.returncode == 0 and _safe_sync(sync, repo):
        return ApplyOutcome(True, target, target)

    error = ("checkout failed: " + (checkout.stderr.strip() if checkout else "git did not run")
             if checkout is None or checkout.returncode != 0 else "the environment sync failed")
    back = _run_git(repo, "checkout", "--quiet", "--detach", previous, timeout=_CHECKOUT_TIMEOUT_SECONDS)
    if back is None or back.returncode != 0 or not _safe_sync(sync, repo):
        error += f"; going back to {previous} failed too"
    return ApplyOutcome(False, target, installed_version(repo), error)


def _safe_sync(sync, repo) -> bool:
    try:
        return bool(sync(repo))
    except Exception as e:  # noqa: BLE001 - any failure means roll back
        _log(f"Update: sync raised {e!r}")
        return False


def _write_result(outcome: ApplyOutcome, user_dir):
    import json
    directory = _user_dir(user_dir)
    try:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, RESULT_FILE), "w", encoding="utf-8") as f:
            json.dump({"ok": outcome.ok, "target": outcome.target,
                       "installed": outcome.installed, "error": outcome.error}, f)
    except OSError as e:
        print(f"Update: could not write the result marker: {e}")


def main(argv=None) -> int:
    """`python -m voxscribe.updates apply-pending [--uv UV] [-- SYNC_ARGS...]`:
    the launchers' entry point. Applies the Release a marker asks for, if
    any. Always returns 0 - the launcher starts the app either way, and the
    app reports the result."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "apply-pending":
        print(__doc__)
        return 0
    rest = argv[1:]
    sync_args = rest[rest.index("--") + 1:] if "--" in rest else []
    rest = rest[:rest.index("--")] if "--" in rest else rest
    uv = rest[rest.index("--uv") + 1] if "--uv" in rest and rest.index("--uv") + 1 < len(rest) else "uv"
    target = pending_update()
    if target:
        _log(f"Installing Update {target}...")
        apply(target, sync=uv_sync(uv, sync_args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
