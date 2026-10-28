"""Installed-version detection (voxscribe.updates) against real Git repos in tmp_path."""

import os
import shutil
import subprocess
import sys

import pytest

from voxscribe import updates

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def install(tmp_path):
    """A clean checkout sitting detached on Release tag v1.2.3."""
    repo = tmp_path / "install"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("print('hi')\n")
    git(repo, "add", "app.py")
    git(repo, "commit", "-q", "-m", "initial")
    git(repo, "tag", "-a", "v1.2.3", "-m", "Release v1.2.3")
    git(repo, "checkout", "-q", "--detach", "v1.2.3")
    return repo


def test_clean_checkout_on_release_tag_shows_the_release(install):
    assert updates.installed_version(install) == "v1.2.3"


def test_frozen_build_is_development(install, monkeypatch):
    # Frozen builds are out of scope for Updates (#15); an .exe can be built
    # from any commit, so it never claims to be a Release.
    monkeypatch.setattr(updates.sys, "frozen", True, raising=False)
    assert updates.installed_version(install) == "development"


def test_local_edits_mean_development(install):
    (install / "app.py").write_text("print('tweaked')\n")
    assert updates.installed_version(install) == "development"


def test_branch_on_the_tag_means_development(install):
    git(install, "checkout", "-q", "main")
    assert updates.installed_version(install) == "development"


def test_commits_past_the_tag_mean_development(install):
    (install / "app.py").write_text("print('next')\n")
    git(install, "commit", "-q", "-am", "work after the release")
    assert updates.installed_version(install) == "development"


def test_highest_release_wins_when_several_tags_point_at_head(install):
    git(install, "tag", "v1.10.0")
    git(install, "tag", "v1.9.0")
    git(install, "tag", "not-a-release")
    assert updates.installed_version(install) == "v1.10.0"


def test_directory_outside_any_repository_means_development(tmp_path, monkeypatch):
    # Don't let Git walk up into whatever repository tmp_path might live in.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert updates.installed_version(tmp_path) == "development"


def test_no_git_installed_means_development(install, monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert updates.installed_version(install) == "development"


# --- Update check and apply (#33, #34) -------------------------------------
#
# A bare repository plays the remote, a clone of it the install, as in
# spec #15: Releases are commits with a tag, pushed to the bare repository.


@pytest.fixture
def remote_and_install(tmp_path):
    """(remote, install, release): a bare remote with Release v1.0.0 and a
    clean install sitting detached on it. release(tag) publishes a new
    Release on the remote."""
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    work = tmp_path / "maintainer"
    git(tmp_path, "clone", "-q", str(remote), str(work))
    git(work, "config", "user.email", "test@example.com")
    git(work, "config", "user.name", "Test")

    def release(tag):
        (work / "version.txt").write_text(tag + "\n")
        git(work, "add", "version.txt")
        git(work, "commit", "-q", "-m", f"Release {tag}")
        git(work, "tag", "-a", tag, "-m", f"Release {tag}")
        git(work, "push", "-q", "origin", "main", tag)

    release("v1.0.0")
    install = tmp_path / "install"
    git(tmp_path, "clone", "-q", str(remote), str(install))
    git(install, "checkout", "-q", "--detach", "v1.0.0")
    return remote, install, release


def test_a_newer_release_on_the_remote_is_offered(remote_and_install):
    _, install, release = remote_and_install
    release("v1.1.0")

    result = updates.check(install)

    assert (result.status, result.installed, result.latest) == ("available", "v1.0.0", "v1.1.0")


def test_the_highest_release_is_offered_compared_numerically(remote_and_install):
    _, install, release = remote_and_install
    for tag in ("v1.9.0", "v1.10.0", "v1.2.0"):
        release(tag)

    assert updates.check(install).latest == "v1.10.0"


def test_nothing_is_offered_without_a_newer_release(remote_and_install):
    _, install, _ = remote_and_install

    result = updates.check(install)

    assert (result.status, result.latest) == ("up_to_date", "v1.0.0")


@pytest.mark.parametrize("change", ["branch", "local edit", "commit past the tag"])
def test_development_checkouts_are_not_eligible(remote_and_install, change):
    _, install, release = remote_and_install
    release("v1.1.0")
    if change == "branch":
        git(install, "checkout", "-q", "main")
    elif change == "local edit":
        (install / "version.txt").write_text("tweaked\n")
    else:
        git(install, "config", "user.email", "test@example.com")
        git(install, "config", "user.name", "Test")
        (install / "version.txt").write_text("next\n")
        git(install, "commit", "-q", "-am", "local work")

    assert updates.check(install).status == "not_eligible"


def test_an_unreachable_remote_fails_the_check_quietly(remote_and_install, tmp_path):
    _, install, _ = remote_and_install
    git(install, "remote", "set-url", "origin", str(tmp_path / "gone.git"))

    result = updates.check(install)

    assert result.status == "failed" and result.error


def head_tag(repo):
    return subprocess.run(["git", "-C", str(repo), "describe", "--tags", "--exact-match"],
                          capture_output=True, text=True).stdout.strip()


def test_requesting_an_update_leaves_a_marker_for_the_launcher(tmp_path):
    user_dir = tmp_path / "home"

    updates.request_update("v1.1.0", user_dir)

    assert updates.pending_update(user_dir) == "v1.1.0"


def test_apply_moves_the_install_to_the_new_release(remote_and_install, tmp_path):
    _, install, release = remote_and_install
    release("v1.1.0")
    user_dir = tmp_path / "home"
    updates.request_update("v1.1.0", user_dir)
    synced = []

    outcome = updates.apply("v1.1.0", install, sync=lambda repo: synced.append(head_tag(repo)) or True,
                            user_dir=user_dir)

    assert outcome.ok and head_tag(install) == "v1.1.0"
    assert synced == ["v1.1.0"]                      # environment synced for the new Release
    assert updates.installed_version(install) == "v1.1.0"
    assert updates.pending_update(user_dir) is None  # marker consumed
    assert updates.take_update_result(user_dir) == {"ok": True, "target": "v1.1.0",
                                                    "installed": "v1.1.0", "error": None}
    assert updates.take_update_result(user_dir) is None  # shown once


def test_a_failing_sync_returns_to_the_previous_release(remote_and_install, tmp_path):
    _, install, release = remote_and_install
    release("v1.1.0")
    user_dir = tmp_path / "home"
    synced = []

    def sync(repo):
        synced.append(head_tag(repo))
        return head_tag(repo) != "v1.1.0"  # the new Release's environment can't be built

    outcome = updates.apply("v1.1.0", install, sync=sync, user_dir=user_dir)

    assert not outcome.ok and head_tag(install) == "v1.0.0"
    assert synced == ["v1.1.0", "v1.0.0"]  # previous environment restored
    result = updates.take_update_result(user_dir)
    assert (result["ok"], result["target"], result["installed"]) == (False, "v1.1.0", "v1.0.0")
    assert result["error"]


def test_a_release_missing_on_the_remote_leaves_the_install_alone(remote_and_install, tmp_path):
    _, install, _ = remote_and_install

    outcome = updates.apply("v9.9.9", install, sync=lambda repo: True, user_dir=tmp_path / "home")

    assert not outcome.ok and head_tag(install) == "v1.0.0"


def test_apply_never_touches_a_development_checkout(remote_and_install, tmp_path):
    _, install, release = remote_and_install
    release("v1.1.0")
    git(install, "checkout", "-q", "main")

    outcome = updates.apply("v1.1.0", install, sync=lambda repo: True, user_dir=tmp_path / "home")

    assert not outcome.ok
    assert updates.installed_version(install) == "development"


def test_the_launcher_entry_removes_even_an_unusable_marker(tmp_path):
    # The launcher restarts itself while a marker exists, so a marker that
    # names no Release must not survive apply-pending.
    user_dir = tmp_path / "home"
    user_dir.mkdir()
    (user_dir / updates.PENDING_FILE).write_text("  \n")

    assert updates.main(["apply-pending"], user_dir=user_dir) == 0

    assert not (user_dir / updates.PENDING_FILE).exists()


class FakePopen:
    def __init__(self):
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))


@pytest.mark.skipif(sys.platform != "win32", reason="VoxScribe.bat is the Windows launcher")
@pytest.mark.parametrize("relaunch", [True, False])
def test_the_app_hands_the_update_to_the_launcher(tmp_path, relaunch):
    (tmp_path / "VoxScribe.bat").write_text("@echo off\n")
    popen = FakePopen()

    assert updates.start_launcher(relaunch, [11, 22], repo=tmp_path, popen=popen)

    [(args, kwargs)] = popen.calls
    assert args == ["cmd.exe", "/c", str(tmp_path / "VoxScribe.bat")]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["env"]["VOXSCRIBE_WAIT_PIDS"] == "11 22"
    assert ("VOXSCRIBE_NO_START" in kwargs["env"]) is not relaunch
    # Its own console shows the install's progress; the app is gone by then.
    assert kwargs["creationflags"] & subprocess.CREATE_NEW_CONSOLE


def test_without_a_launcher_the_update_waits_for_the_next_start(tmp_path):
    popen = FakePopen()

    assert not updates.start_launcher(True, [11], repo=tmp_path, popen=popen)
    assert popen.calls == []


@pytest.mark.skipif(sys.platform != "win32", reason="VoxScribe.bat is the Windows launcher")
def test_an_app_started_after_an_update_can_still_hand_over_the_next_one(tmp_path, monkeypatch):
    # The launcher sets these for itself; the app it started inherits them.
    monkeypatch.setenv("VOXSCRIBE_UPDATE_RAN", "1")
    monkeypatch.setenv("VOXSCRIBE_NO_START", "1")
    (tmp_path / "VoxScribe.bat").write_text("@echo off\n")
    popen = FakePopen()

    updates.start_launcher(True, [11], repo=tmp_path, popen=popen)

    [(_args, kwargs)] = popen.calls
    assert "VOXSCRIBE_UPDATE_RAN" not in kwargs["env"]
    assert "VOXSCRIBE_NO_START" not in kwargs["env"]


def sleeper(seconds):
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])


def test_the_launcher_installs_only_after_the_app_has_exited(tmp_path, monkeypatch):
    # The quitting app still holds files in .venv that the sync replaces.
    user_dir = tmp_path / "home"
    updates.request_update("v1.1.1", user_dir=user_dir)
    app = sleeper(1.5)
    monkeypatch.setenv("VOXSCRIBE_WAIT_PIDS", str(app.pid))
    applied_while_app_ran = []
    monkeypatch.setattr(updates, "apply",
                        lambda target, **kw: applied_while_app_ran.append(app.poll() is None))

    assert updates.main(["apply-pending"], user_dir=user_dir) == 0

    assert applied_while_app_ran == [False]


def test_the_launcher_does_not_wait_forever_for_a_hanging_app(tmp_path, monkeypatch):
    user_dir = tmp_path / "home"
    updates.request_update("v1.1.1", user_dir=user_dir)
    app = sleeper(60)
    monkeypatch.setenv("VOXSCRIBE_WAIT_PIDS", str(app.pid))
    monkeypatch.setattr(updates, "WAIT_FOR_APP_SECONDS", 0.5)
    applied = []
    monkeypatch.setattr(updates, "apply", lambda target, **kw: applied.append(target))
    try:
        updates.main(["apply-pending"], user_dir=user_dir)
    finally:
        app.kill()

    # Goes ahead; a sync that still finds locked files rolls back as usual.
    assert applied == ["v1.1.1"]


@pytest.mark.skipif(sys.platform != "darwin", reason="VoxScribe.app is the macOS launcher")
@pytest.mark.parametrize("relaunch", [True, False])
def test_on_macos_the_app_hands_the_update_to_its_app_bundle(tmp_path, monkeypatch, relaunch):
    # Set by the launcher script inside the bundle (macos/launcher.sh).
    bundle = tmp_path / "VoxScribe.app"
    bundle.mkdir()
    monkeypatch.setenv("VOXSCRIBE_APP_BUNDLE", str(bundle))
    popen = FakePopen()

    assert updates.start_launcher(relaunch, [11, 22], repo=tmp_path, popen=popen)

    # Through LaunchServices, so the new launcher is VoxScribe again (Dock
    # name, microphone / screen-recording permissions), not a child of the
    # quitting app; -n because the old instance is still shutting down.
    [(args, _kwargs)] = popen.calls
    expected = ["/usr/bin/open", "-n", str(bundle), "--env", "VOXSCRIBE_WAIT_PIDS=11 22"]
    if not relaunch:
        expected += ["--env", "VOXSCRIBE_NO_START=1"]
    assert args == expected


@pytest.mark.skipif(sys.platform != "darwin", reason="VoxScribe.app is the macOS launcher")
def test_on_macos_an_app_not_started_from_its_bundle_has_no_launcher(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXSCRIBE_APP_BUNDLE", raising=False)
    popen = FakePopen()

    assert not updates.start_launcher(True, [11], repo=tmp_path, popen=popen)
    assert popen.calls == []


def test_the_app_waits_for_a_venv_shim_only_on_windows(monkeypatch):
    # pythonw.exe in a Windows venv is a shim process around the app; on
    # macOS .venv/bin/python is a symlink, and the parent is launchd.
    monkeypatch.setattr(sys, "prefix", "/venv")
    monkeypatch.setattr(sys, "base_prefix", "/python")

    pids = updates.app_pids()

    assert pids == ([os.getpid(), os.getppid()] if sys.platform == "win32" else [os.getpid()])
