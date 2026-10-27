"""The Update notice offers Install and quit / Install and relaunch (#55)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def notice():
    from PySide6.QtWidgets import QApplication
    from qt_app.widgets.update_notice import UpdateNotice

    QApplication.instance() or QApplication([])
    notice = UpdateNotice()
    notice.offer("v1.1.1")
    return notice


def requests_from(notice):
    requests = []
    notice.installRequested.connect(lambda release, relaunch: requests.append((release, relaunch)))
    return requests


def test_install_and_quit_asks_for_the_update_without_relaunch(notice):
    requests = requests_from(notice)

    notice.quit_button.click()

    assert notice.quit_button.text() == "Install and quit"
    assert requests == [("v1.1.1", False)]


def test_install_and_relaunch_asks_for_the_update_with_relaunch(notice):
    requests = requests_from(notice)

    notice.relaunch_button.click()

    assert notice.relaunch_button.text() == "Install and relaunch"
    assert requests == [("v1.1.1", True)]


def test_neither_button_works_while_recording_or_transcribing(notice):
    notice.set_busy(True)

    assert not notice.quit_button.isEnabled()
    assert not notice.relaunch_button.isEnabled()

    notice.set_busy(False)

    assert notice.quit_button.isEnabled()
    assert notice.relaunch_button.isEnabled()
