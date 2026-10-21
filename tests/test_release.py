"""Release script (voxscribe.release): version handling, and its refusals,
dry-run and pushes against temporary Git repos - never real remotes."""

import shutil
import subprocess
from pathlib import Path

import pytest

from voxscribe import release

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("text, expected", [("1.1.0", (1, 1, 0)), ("10.20.30", (10, 20, 30))])
def test_parse_version_accepts_major_minor_patch(text, expected):
    assert release.parse_version(text) == expected


@pytest.mark.parametrize("text", ["1.1", "1.1.0.1", "v1.1.0", "1.1.0-rc1", "", "a.b.c"])
def test_parse_version_rejects_anything_else(text):
    with pytest.raises(ValueError):
        release.parse_version(text)


def test_versions_compare_numerically_not_as_strings():
    assert release.parse_version("1.10.0") > release.parse_version("1.9.0")


def _changed_lines(before: str, after: str):
    a, b = before.splitlines(), after.splitlines()
    assert len(a) == len(b)
    return [(x, y) for x, y in zip(a, b) if x != y]


def test_bump_pyproject_changes_only_the_project_version():
    before = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    current = release.read_version(before)
    after = release.bump_pyproject(before, "9.8.7")
    assert release.read_version(after) == "9.8.7"
    assert _changed_lines(before, after) == [(f'version = "{current}"', 'version = "9.8.7"')]


def test_bump_uv_lock_changes_only_the_projects_own_entry():
    # uv.lock records the project's own version too; leaving it stale makes
    # the launcher's `uv sync` rewrite uv.lock and dirty every install.
    before = (ROOT / "uv.lock").read_text(encoding="utf-8")
    current = release.read_version((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    after = release.bump_uv_lock(before, "9.8.7")
    assert _changed_lines(before, after) == [(f'version = "{current}"', 'version = "9.8.7"')]
    block = after[after.index(f'name = "{release.PROJECT_NAME}"'):]
    assert block.splitlines()[1] == 'version = "9.8.7"'


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")

PYPROJECT = '[project]\nname = "whisperx-recorder"\nversion = "1.0.0"\n\n[tool.x]\nversion = "0.1"\n'


def git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def maintainer(tmp_path):
    """A clean main checkout at 1.0.0 whose origin/public/github remotes are
    temporary bare repos already holding main."""
    repo = tmp_path / "checkout"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "core.autocrlf", "false")
    (repo / "pyproject.toml").write_text(PYPROJECT, newline="")
    git(repo, "add", "pyproject.toml")
    git(repo, "commit", "-q", "-m", "initial")
    git(repo, "tag", "-a", "v1.0.0", "-m", "Release v1.0.0")
    for name in ("origin", "public", "github"):
        bare = tmp_path / f"{name}.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
        git(repo, "remote", "add", name, str(bare))
        git(repo, "push", "-q", name, "main")
    return repo


def remote_refs(tmp_path, name) -> str:
    return git(tmp_path / f"{name}.git", "for-each-ref", "--format=%(refname) %(objectname)")


@needs_git
def test_refuses_a_branch_other_than_main(maintainer):
    git(maintainer, "checkout", "-q", "-b", "feature")
    with pytest.raises(release.ReleaseError, match="main"):
        release.release(maintainer, "1.1.0", log=lambda _: None)


@needs_git
def test_refuses_a_dirty_tree(maintainer):
    (maintainer / "pyproject.toml").write_text(PYPROJECT + "# local edit\n")
    with pytest.raises(release.ReleaseError, match="uncommitted"):
        release.release(maintainer, "1.1.0", log=lambda _: None)


@needs_git
@pytest.mark.parametrize("version", ["1.0.0", "0.9.9", "1.1"])
def test_refuses_a_version_not_greater_than_the_current_one(maintainer, version):
    with pytest.raises(release.ReleaseError):
        release.release(maintainer, version, log=lambda _: None)


def snapshot(tmp_path, repo):
    return (git(repo, "rev-parse", "HEAD"), git(repo, "tag"), git(repo, "status", "--porcelain"),
            (repo / "pyproject.toml").read_bytes(),
            [remote_refs(tmp_path, name) for name in ("origin", "public", "github")])


@needs_git
def test_dry_run_prints_every_step_and_changes_nothing(tmp_path, maintainer):
    before = snapshot(tmp_path, maintainer)
    lines = []
    release.release(maintainer, "1.1.0", dry_run=True, log=lines.append)
    assert snapshot(tmp_path, maintainer) == before
    output = "\n".join(lines)
    for step in ("1.0.0 -> 1.1.0", "commit", "tag -a v1.1.0",
                 "push --atomic origin main v1.1.0", "push --atomic public main v1.1.0"):
        assert step in output
    assert "github" not in output


@needs_git
def test_release_pushes_branch_and_tag_to_origin_and_public_only(tmp_path, maintainer):
    release.release(maintainer, "1.1.0", log=lambda _: None)

    assert 'version = "1.1.0"' in (maintainer / "pyproject.toml").read_text()
    assert git(maintainer, "status", "--porcelain") == ""
    head = git(maintainer, "rev-parse", "HEAD")
    assert git(maintainer, "rev-parse", "v1.1.0^{commit}") == head
    for name in ("origin", "public"):
        refs = remote_refs(tmp_path, name)
        assert f"refs/heads/main {head}" in refs
        assert "refs/tags/v1.1.0" in refs
    assert "v1.1.0" not in remote_refs(tmp_path, "github")
    assert head not in remote_refs(tmp_path, "github")


@needs_git
def test_failed_push_reports_what_was_and_wasnt_pushed(tmp_path, maintainer):
    git(maintainer, "remote", "set-url", "public", str(tmp_path / "unreachable.git"))
    with pytest.raises(release.ReleaseError) as excinfo:
        release.release(maintainer, "1.1.0", log=lambda _: None)
    message = str(excinfo.value)
    assert "Pushed to: origin" in message
    assert "NOT pushed to: public" in message
    assert "git push --atomic public main v1.1.0" in message
    assert "refs/tags/v1.1.0" in remote_refs(tmp_path, "origin")


def test_bumps_keep_crlf_line_endings():
    # Windows checkouts with core.autocrlf have CRLF files.
    text = PYPROJECT.replace("\n", "\r\n")
    assert release.bump_pyproject(text, "1.1.0") == text.replace('"1.0.0"', '"1.1.0"')
    lock = '[[package]]\r\nname = "whisperx-recorder"\r\nversion = "1.0.0"\r\n'
    assert release.bump_uv_lock(lock, "1.1.0") == lock.replace("1.0.0", "1.1.0")
