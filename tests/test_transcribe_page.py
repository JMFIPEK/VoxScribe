"""Transcription page's model dropdown (spec #59): the pick is remembered
and preselected, a gone pick falls back to the default Provider, then the
local default; macOS gets the dropdown with only the Mac's models -
offscreen, with a fake transcribe controller, fake Providers and an
ini-backed QSettings."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

KIT = {"id": "kit-toolbox", "name": "KIT ToolBox"}
OTHER = {"id": "other", "name": "Other"}


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def providers(monkeypatch):
    """The configured Providers: edit `state["list"]`/`state["default"]`."""
    import voxscribe.providers as providers_module

    state = {"list": [], "default": None}
    monkeypatch.setattr(providers_module, "list_providers", lambda: list(state["list"]))
    monkeypatch.setattr(providers_module, "get_default_provider", lambda: state["default"])
    return state


@pytest.fixture
def make_page(app, tmp_path, monkeypatch):
    """Builds a Transcription page; every page shares one ini file, so a
    second page is the app after a restart."""
    from PySide6.QtCore import QObject, QSettings, Signal

    from qt_app.pages import transcribe_page

    class FakeTranscribeController(QObject):
        progressUpdated = Signal(float, str)
        finished = Signal(object, object)
        stepStarted = Signal(str)

        def __init__(self):
            super().__init__()
            self.started = []

        def start(self, **kwargs):
            self.started.append(kwargs)

    class FakeWindow:
        settings = {"output_dir": str(tmp_path)}

        def __init__(self):
            self.transcribe_controller = FakeTranscribeController()

    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(transcribe_page, "QSettings", lambda *a: QSettings(ini, QSettings.IniFormat))

    pages = []

    def make():
        page = transcribe_page.TranscribePage(FakeWindow())
        pages.append(page)
        return page

    yield make
    for page in pages:
        page.close()


def ids(page):
    combo = page.model_combo
    return [combo.itemData(i) for i in range(combo.count())]


def pick(page, model_id):
    page.model_combo.setCurrentIndex(ids(page).index(model_id))


def test_a_pick_survives_rebuilding_the_page(make_page, providers, monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    providers["list"], providers["default"] = [KIT, OTHER], KIT
    pick(make_page(), "medium")

    page = make_page()
    assert page.model_combo.currentData() == "medium"
    page.on_page_shown()
    assert page.model_combo.currentData() == "medium"


def test_no_pick_preselects_the_default_provider(make_page, providers, monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    providers["list"], providers["default"] = [OTHER, KIT], KIT
    page = make_page()
    assert page.model_combo.currentData() == "server:kit-toolbox"
    assert page.pipeline_info.text().startswith("Transcription: remote provider")


def test_a_removed_provider_falls_back_to_the_default_provider(make_page, providers, monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    providers["list"], providers["default"] = [KIT, OTHER], KIT
    pick(make_page(), "server:other")

    providers["list"] = [KIT]
    assert make_page().model_combo.currentData() == "server:kit-toolbox"


def test_without_provider_the_local_default_is_preselected(make_page, providers, monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    page = make_page()
    page.apply_hardware_info({"recommended_model": "medium"})
    assert page.model_combo.currentData() == "medium"


def test_windows_lists_the_whisper_models_and_the_providers(make_page, providers, monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    providers["list"], providers["default"] = [KIT], KIT
    assert ids(make_page()) == [
        "large-v3", "large-v2", "medium", "base", "openvino:GPU:medium",
        "parakeet:tdt-0.6b-v3", "server:kit-toolbox",
    ]


def test_mac_dropdown_has_the_mac_models_and_the_providers(make_page, providers, monkeypatch):
    from voxscribe.transcription.model_ids import default_local_model_size

    monkeypatch.setattr("sys.platform", "darwin")
    providers["list"], providers["default"] = [KIT, OTHER], KIT
    page = make_page()
    page.apply_hardware_info({"recommended_model": "large-v3"})  # a Whisper size

    assert ids(page) == [
        default_local_model_size(), "apple:speechanalyzer",
        "server:kit-toolbox", "server:other",
    ]
    # The Mac app now uses the default Provider, not always the local model.
    assert page.model_combo.currentData() == "server:kit-toolbox"
    assert page.recommended_hint.text() == ""


def test_mac_without_provider_transcribes_with_the_local_default(make_page, providers, monkeypatch, tmp_path):
    from voxscribe.transcription.model_ids import PARAKEET_MODEL

    monkeypatch.setattr("sys.platform", "darwin")
    page = make_page()
    assert page.model_combo.currentData() == PARAKEET_MODEL
    assert "Parakeet" in page.pipeline_info.text()

    pick(page, "apple:speechanalyzer")
    assert "SpeechAnalyzer" in page.pipeline_info.text()

    audio = tmp_path / "call.wav"
    audio.write_bytes(b"")
    page.set_audio_file(str(audio))
    page.start_transcription()
    [kwargs] = page.window_.transcribe_controller.started
    assert kwargs["audio_path"] == str(audio)
    assert kwargs["model_size"] == "apple:speechanalyzer"


def test_mac_lists_a_new_local_default_even_if_unknown(make_page, providers, monkeypatch):
    import voxscribe.transcription.model_ids as model_ids

    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr(model_ids, "default_local_model_size", lambda: "canary:1b-v2")
    page = make_page()
    assert ids(page) == ["canary:1b-v2", "parakeet:tdt-0.6b-v3", "apple:speechanalyzer"]
    assert page.model_combo.currentText() == "Canary 1B v2 (CPU)"
    assert page.pipeline_info.text().startswith("Transcription: Canary 1B v2 (CPU)")


def test_local_models_adds_an_unlisted_default_first():
    from qt_app.constants import local_models

    assert local_models("darwin", "x:new") == ["x:new", "parakeet:tdt-0.6b-v3", "apple:speechanalyzer"]
    assert local_models("darwin", "apple:speechanalyzer")[0] == "apple:speechanalyzer"
    assert "large-v3" not in local_models("darwin")
