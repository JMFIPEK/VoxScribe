"""Installed-version detection (voxscribe.updates) against real Git repos in tmp_path."""

import shutil
import subprocess

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
