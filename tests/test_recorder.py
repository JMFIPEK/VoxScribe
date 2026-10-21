"""Characterization tests for the recorder (spec #17, #27): every Recording
mode on Windows and macOS, recorded through AudioRecorder's public interface
with fake capture sources (tests/fake_capture.py). They pin the files a
recording writes, so the clean-ups of #28/#29 can show nothing changed.

The fakes deliver audio in real time, so each test records for about a
second, and lengths are checked with a tolerance.
"""

import os
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from voxscribe import tracks
from voxscribe.recorder import AudioRecorder, CaptureSources
from tests.fake_capture import FakeHelper, FakePyAudio, FakeSounddevice, FakeSource

SECONDS = 1.0


@pytest.fixture
def helper_binary(tmp_path):
    path = tmp_path / "SystemAudioCapture"
    path.write_bytes(b"")  # the recorder only checks that the helper exists
    return str(path)


def windows_sources(mic=None, loopback=None):
    return CaptureSources(platform="win32", pyaudio=FakePyAudio(
        mic or FakeSource(rate=48000, channels=2, freq=440, name="Mikrofon (Fake Headset)"),
        loopback or FakeSource(rate=48000, channels=2, freq=880, name="Kopfhörer (Fake Headset) [Loopback]")))


def macos_sources(helper_binary, mic=None, helper=None):
    return CaptureSources(platform="darwin",
                          sounddevice=FakeSounddevice(mic or FakeSource(rate=48000, channels=1, freq=440)),
                          popen=helper or FakeHelper(freq=880), helper_path=lambda: helper_binary)


def record(sources, tmp_path, source, setup=None, seconds=SECONDS, **start_kwargs):
    """Records for `seconds` and returns what on_done reported plus the
    level callbacks seen: (path, duration, error, levels)."""
    done, result, levels = threading.Event(), {}, []
    recorder = AudioRecorder(sources)
    recorder.start(str(tmp_path / "rec.wav"), source=source, setup=setup,
                   on_level=lambda ch, v: levels.append(ch),
                   on_done=lambda p, d, e: (result.update(path=p, duration=d, error=e), done.set()),
                   **start_kwargs)
    time.sleep(seconds)
    recorder.stop()
    assert done.wait(10), "on_done never came"
    return result["path"], result["duration"], result["error"], levels


def wav_info(path):
    info = sf.info(path)
    return info.samplerate, info.channels, info.frames / info.samplerate


def assert_call_files(path, duration, setup):
    """A Call writes a 16 kHz mono mix plus a 2-channel sidecar (Microphone,
    Playback) tagged with the Setup, and the mix is both tracks at 0.65."""
    rate, channels, length = wav_info(path)
    assert (rate, channels) == (16000, 1)
    assert abs(length - duration) < 0.01
    sidecar, sr = sf.read(tracks.tracks_path_for(path), dtype="int16", always_2d=True)
    assert sr == 16000 and sidecar.shape[1] == 2 and len(sidecar) == round(length * 16000)
    assert tracks.load_setup(path) == setup
    mix, _ = sf.read(path, dtype="int16")
    expected = tracks.mix(sidecar[:, 0].astype(float), sidecar[:, 1].astype(float), full_scale=32767)
    # The recorder truncates tracks and mix to int16 separately, so the mix
    # recomputed from the stored tracks is off by up to ~2.3 LSB.
    assert np.max(np.abs(mix.astype(float) - expected)) <= 3
    return sidecar


def assert_recorded_about(duration):
    assert SECONDS - 0.4 <= duration <= SECONDS + 0.5, duration


# --- Windows -------------------------------------------------------------------

def test_windows_in_room_keeps_the_microphones_native_format(tmp_path):
    path, duration, error, levels = record(windows_sources(), tmp_path, "mic")

    assert error is None
    rate, channels, length = wav_info(path)
    assert (rate, channels) == (48000, 2)  # native rate and channels, not 16 kHz
    assert abs(length - duration) < 0.01
    assert_recorded_about(duration)
    assert set(levels) == {"mic"}
    assert not os.path.exists(tracks.tracks_path_for(path))


def test_windows_playback_only_records_the_loopback(tmp_path):
    path, duration, error, levels = record(windows_sources(), tmp_path, "system")

    assert error is None
    rate, channels, _ = wav_info(path)
    assert (rate, channels) == (48000, 2)
    assert_recorded_about(duration)
    assert set(levels) == {"system"}


def test_windows_call_writes_the_mix_and_the_tagged_tracks(tmp_path):
    # Playback only during 0.2-0.6 s: WASAPI sends nothing outside that.
    sources = windows_sources(loopback=FakeSource(rate=44100, channels=2, freq=880,
                                                  active=[(0.2, 0.6)], name="Fake [Loopback]"))
    path, duration, error, levels = record(sources, tmp_path, "both", setup="headset",
                                           device_index=1, loopback_index=2)

    assert error is None
    assert_recorded_about(duration)
    sidecar = assert_call_files(path, duration, "headset")
    playback = np.abs(sidecar[:, 1].astype(int))
    assert playback[:int(0.1 * 16000)].max() == 0  # silent before it played
    assert playback.max() > 5000
    assert {"mic", "system"} <= set(levels)


def test_windows_call_where_nothing_plays_saves_a_silent_playback_track(tmp_path):
    sources = windows_sources(loopback=FakeSource(rate=48000, channels=2, active=[], name="Fake [Loopback]"))
    path, duration, error, _ = record(sources, tmp_path, "both", setup="loudspeaker")

    assert error is None
    sidecar = assert_call_files(path, duration, "loudspeaker")
    assert not sidecar[:, 1].any()


# --- macOS ---------------------------------------------------------------------

def test_macos_in_room_records_the_microphone(tmp_path, helper_binary):
    path, duration, error, levels = record(macos_sources(helper_binary), tmp_path, "mic")

    assert error is None
    rate, channels, length = wav_info(path)
    assert (rate, channels) == (48000, 1)
    assert abs(length - duration) < 0.01
    assert_recorded_about(duration)
    assert set(levels) == {"mic"}


def test_macos_playback_only_records_the_helper_and_stops_it(tmp_path, helper_binary):
    helper = FakeHelper()
    path, duration, error, levels = record(macos_sources(helper_binary, helper=helper), tmp_path, "system")

    assert error is None
    assert wav_info(path)[:2] == (16000, 1)
    assert_recorded_about(duration)
    assert set(levels) == {"system"}
    assert [p.returncode for p in helper.launched] == [-15]  # terminated on stop


def test_macos_call_writes_the_mix_and_the_tagged_tracks(tmp_path, helper_binary):
    helper = FakeHelper()
    path, duration, error, levels = record(macos_sources(helper_binary, helper=helper), tmp_path,
                                           "both", setup="loudspeaker")

    assert error is None
    assert_recorded_about(duration)
    sidecar = assert_call_files(path, duration, "loudspeaker")
    assert np.abs(sidecar[:, 0].astype(int)).max() > 5000 and np.abs(sidecar[:, 1].astype(int)).max() > 5000
    assert {"mic", "system"} <= set(levels)
    assert [p.returncode for p in helper.launched] == [-15]


def test_macos_without_the_helper_reports_how_to_build_it(tmp_path):
    sources = macos_sources(str(tmp_path / "missing"))
    path, duration, error, _ = record(sources, tmp_path, "system", seconds=0.1)

    assert path is None and "build.sh" in error


def test_linux_offers_only_the_microphone(tmp_path):
    sources = CaptureSources(platform="linux", sounddevice=FakeSounddevice(FakeSource()))
    done, result = threading.Event(), {}
    AudioRecorder(sources).start(str(tmp_path / "rec.wav"), source="system",
                                 on_done=lambda p, d, e: (result.update(error=e), done.set()))

    assert done.wait(2) and "isn't supported" in result["error"]


# --- Command-line recording uses the same recorder (#28) ---------------------

from voxscribe import recorder as recorder_module  # noqa: E402


def stop_after(seconds):
    """Stands in for "press ENTER": returns after `seconds`."""
    return lambda: time.sleep(seconds)


def test_command_line_microphone_writes_what_the_recorder_writes(tmp_path):
    out = str(tmp_path / "cli.wav")

    returned = recorder_module.record_microphone(out, sources=windows_sources(),
                                                 wait_for_stop=stop_after(SECONDS))

    assert returned == out
    rate, channels, length = wav_info(out)
    assert (rate, channels) == (48000, 2)  # as test_windows_in_room_...
    assert_recorded_about(length)


def test_command_line_playback_writes_what_the_recorder_writes(tmp_path):
    out = str(tmp_path / "cli.wav")

    recorder_module.record_system_audio(out, sources=windows_sources(), wait_for_stop=stop_after(SECONDS))

    assert wav_info(out)[:2] == (48000, 2)
    assert_recorded_about(wav_info(out)[2])


def test_command_line_call_writes_what_the_recorder_writes(tmp_path):
    out = str(tmp_path / "cli.wav")

    recorder_module.record_microphone_and_system(out, setup="headset", sources=windows_sources(),
                                                 wait_for_stop=stop_after(SECONDS))

    _, _, length = wav_info(out)
    assert_recorded_about(length)
    assert_call_files(out, length, "headset")


def test_command_line_playback_stops_although_nothing_plays(tmp_path):
    # The old command-line loop blocked in stream.read() while WASAPI sent
    # nothing, so ENTER never stopped it.
    silent = windows_sources(loopback=FakeSource(rate=48000, channels=2, active=[], name="L"))
    out = str(tmp_path / "cli.wav")
    finished = threading.Event()

    threading.Thread(target=lambda: (recorder_module.record_system_audio(
        out, sources=silent, wait_for_stop=stop_after(0.5)), finished.set()), daemon=True).start()

    assert finished.wait(5), "stopping blocked"


def test_command_line_macos_paths_use_the_recorder_too(tmp_path, helper_binary):
    out = str(tmp_path / "cli.wav")

    recorder_module.record_system_audio(out, sources=macos_sources(helper_binary),
                                        wait_for_stop=stop_after(SECONDS))

    assert wav_info(out)[:2] == (16000, 1)


# --- The macOS helper's failure messages (#29) --------------------------------

def test_macos_helper_crash_reports_its_error_output(tmp_path, helper_binary):
    helper = FakeHelper(exit_code=1, stderr=b"Stream error (possibly missing Screen Recording permission)")
    _, _, error, _ = record(macos_sources(helper_binary, helper=helper), tmp_path, "system", seconds=0.2)

    assert error == "Stream error (possibly missing Screen Recording permission)"


def test_macos_helper_without_audio_asks_for_the_permission(tmp_path, helper_binary):
    helper = FakeHelper(exit_code=0)
    _, _, error, _ = record(macos_sources(helper_binary, helper=helper), tmp_path, "system", seconds=0.2)

    assert "Screen & System Audio Recording" in error


def test_macos_call_reports_a_helper_crash(tmp_path, helper_binary):
    helper = FakeHelper(exit_code=1, stderr=b"")
    _, _, error, _ = record(macos_sources(helper_binary, helper=helper), tmp_path, "both", seconds=0.2)

    assert error == "SystemAudioCapture exited with code 1"


def test_macos_call_without_the_helper_reports_how_to_build_it(tmp_path):
    _, _, error, _ = record(macos_sources(str(tmp_path / "missing")), tmp_path, "both", seconds=0.1)

    assert "build.sh" in error
