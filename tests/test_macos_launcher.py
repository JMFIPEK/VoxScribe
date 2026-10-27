"""The VoxScribe.app launcher script (macos/launcher.sh, rendered by
macos/make-app.sh), run with bash against a fake repo: a fake .venv python,
uv and build.sh that only log how they were called."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                                reason="the macOS launcher is a bash script")

TEMPLATE = Path(__file__).resolve().parent.parent / "macos" / "launcher.sh"

LOGGER = '#!/bin/bash\necho "{name} $*" >> "{log}"\n'


def fake_tool(path: Path, name: str, log: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(LOGGER.format(name=name, log=log))
    path.chmod(0o755)


@pytest.fixture
def launcher(tmp_path):
    """Renders the launcher for a fake repo whose venv is in sync and whose
    Swift helpers are built; returns run(env=None) -> list of logged calls."""
    repo, home, log = tmp_path / "repo", tmp_path / "home", tmp_path / "calls.log"
    home.mkdir()
    fake_tool(repo / ".venv/bin/python", "python", log)
    fake_tool(repo / "macos/build.sh", "build", log)
    for helper in ("SystemAudioCapture", "SpeechAnalyzerTranscribe"):
        (repo / f"macos/{helper}.swift").write_text("// source\n")
        fake_tool(repo / f"macos/{helper}", helper, log)
        # Built after the source was last changed
        now = time.time()
        os.utime(repo / f"macos/{helper}.swift", (now - 60, now - 60))
        os.utime(repo / f"macos/{helper}", (now - 30, now - 30))
    (repo / "uv.lock").write_text("lock\n")
    (repo / ".venv/.voxscribe-synced-uv.lock").write_text("lock\n")
    uv = tmp_path / "bin/uv"
    fake_tool(uv, "uv", log)

    bundle = tmp_path / "VoxScribe.app"
    script = bundle / "Contents/MacOS/VoxScribe"
    script.parent.mkdir(parents=True)
    script.write_text(TEMPLATE.read_text().replace("@REPO@", str(repo)).replace("@UV@", str(uv)))
    script.chmod(0o755)

    def run(env=None):
        log.unlink(missing_ok=True)
        subprocess.run([str(script)], check=True, timeout=30,
                       env={"PATH": os.environ["PATH"], "HOME": str(home), **(env or {})})
        return log.read_text().splitlines() if log.exists() else []

    run.repo, run.home, run.uv, run.bundle = repo, home, uv, bundle
    return run


def leave_marker(home: Path, tag="v1.1.1"):
    (home / ".voxscribe").mkdir(exist_ok=True)
    (home / ".voxscribe/update-pending.txt").write_text(tag + "\n")


def test_without_a_marker_the_launcher_only_starts_the_app(launcher):
    assert launcher() == ["python gui_qt.py"]


def test_a_pending_update_is_installed_before_sync_and_start(launcher):
    leave_marker(launcher.home)
    # The checkout changes uv.lock, so the launcher's own sync runs after it.
    (launcher.repo / "uv.lock").write_text("lock of the new release\n")

    calls = launcher()

    assert calls == [f"python -m voxscribe.updates apply-pending --uv {launcher.uv}",
                     "uv sync",
                     "python gui_qt.py"]


def test_the_marker_is_removed_even_if_python_never_gets_to_it(launcher):
    leave_marker(launcher.home)

    launcher()

    assert not (launcher.home / ".voxscribe/update-pending.txt").exists()


def test_install_and_quit_installs_without_starting_the_app(launcher):
    leave_marker(launcher.home)

    calls = launcher({"VOXSCRIBE_NO_START": "1"})

    assert calls == [f"python -m voxscribe.updates apply-pending --uv {launcher.uv}"]


def test_the_app_learns_its_bundle_so_it_can_hand_the_next_update_over(launcher):
    (launcher.repo / ".venv/bin/python").write_text(
        f'#!/bin/bash\necho "$VOXSCRIBE_APP_BUNDLE" > "{launcher.repo / "env.log"}"\n')

    launcher()

    assert (launcher.repo / "env.log").read_text().strip() == str(launcher.bundle)


def test_swift_helpers_are_rebuilt_when_an_update_changed_their_source(launcher):
    # A checkout gives changed files a fresh mtime; the built helpers are
    # gitignored and would stay on the old source otherwise.
    (launcher.repo / "macos/SystemAudioCapture.swift").write_text("// new source\n")

    assert launcher() == ["build ", "python gui_qt.py"]
