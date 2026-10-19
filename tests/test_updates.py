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
