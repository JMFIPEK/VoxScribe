"""Fake capture sources for AudioRecorder (spec #17): scripted audio instead
of real devices, delivered at the real-time pace a device would deliver it.

- FakePyAudio: Windows (PyAudioWPatch) microphones and WASAPI loopback.
  Like WASAPI, a loopback stream delivers nothing outside its `active`
  stretches (nothing playing).
- FakeSounddevice: macOS/Linux microphones (callback streams).
- FakeHelper: the macOS system-audio helper (16 kHz mono int16 on stdout).

Each source plays a sine of its own frequency, so tests can tell the
tracks apart.
"""

import threading
import time
from types import SimpleNamespace

import numpy as np


class FakeSource:
    """Audio a device delivers: a sine at `freq`, `rate` Hz, `channels`
    channels; `active` lists (start, end) seconds after the stream opened in
    which it delivers anything at all (None = always)."""

    def __init__(self, rate=48000, channels=1, freq=440.0, active=None, name="Fake device"):
        self.rate, self.channels, self.freq = rate, channels, freq
        self.active, self.name = active, name

    def produced(self, elapsed: float) -> int:
        """Frames delivered after `elapsed` seconds of real time."""
        if self.active is None:
            return int(elapsed * self.rate)
        return int(sum(max(0.0, min(elapsed, b) - a) for a, b in self.active) * self.rate)

    def frames(self, first: int, count: int) -> bytes:
        t = (first + np.arange(count)) / self.rate
        mono = (0.3 * 32767 * np.sin(2 * np.pi * self.freq * t)).astype(np.int16)
        return np.repeat(mono[:, None], self.channels, axis=1).tobytes()


# --- PyAudioWPatch -----------------------------------------------------------

class _FakePaStream:
    def __init__(self, source: FakeSource):
        self.source, self.opened, self.read_frames = source, time.monotonic(), 0
        self.active = True

    def get_read_available(self) -> int:
        return self.source.produced(time.monotonic() - self.opened) - self.read_frames

    def read(self, n, exception_on_overflow=True):
        while self.get_read_available() < n:  # a real blocking read waits too
            time.sleep(0.002)
        data = self.source.frames(self.read_frames, n)
        self.read_frames += n
        return data

    def is_active(self):
        return self.active

    def stop_stream(self):
        self.active = False

    def close(self):
        self.active = False


class FakePyAudio:
    """Stands in for the pyaudiowpatch module: device 1 is the microphone,
    device 2 the WASAPI loopback (also the default one)."""
    paInt16 = 8

    def __init__(self, mic: FakeSource, loopback: FakeSource):
        self.devices = {1: mic, 2: loopback}

    def PyAudio(self):  # noqa: N802 - mirrors the module's API
        return _FakePaInstance(self.devices)


class _FakePaInstance:
    def __init__(self, devices):
        self.devices = devices

    def _info(self, index):
        src = self.devices[index]
        return {"index": index, "name": src.name, "maxInputChannels": src.channels,
                "defaultSampleRate": float(src.rate), "hostApi": 0}

    def get_device_info_by_index(self, index):
        return self._info(index)

    def get_default_input_device_info(self):
        return self._info(1)

    def get_default_wasapi_loopback(self):
        return self._info(2)

    def get_loopback_device_info_generator(self):
        yield self._info(2)

    def get_host_api_info_by_index(self, index):
        return {"name": "Windows WASAPI"}

    def open(self, format, channels, rate, input, input_device_index, frames_per_buffer):
        return _FakePaStream(self.devices[input_device_index])

    def terminate(self):
        pass


# --- sounddevice -------------------------------------------------------------

class FakeSounddevice:
    """Stands in for the sounddevice module (microphone only)."""

    def __init__(self, mic: FakeSource):
        self.mic = mic

    def query_devices(self, device=None, kind=None):
        return {"name": self.mic.name, "default_samplerate": float(self.mic.rate),
                "max_input_channels": self.mic.channels}

    def InputStream(self, samplerate, channels, dtype, device, blocksize, callback):  # noqa: N802
        return _FakeSdStream(self.mic, channels, blocksize or 512, callback)


class _FakeSdStream:
    def __init__(self, source, channels, blocksize, callback):
        self.source, self.channels, self.blocksize, self.callback = source, channels, blocksize, callback
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        opened, delivered = time.monotonic(), 0
        while not self._stop.is_set():
            if self.source.produced(time.monotonic() - opened) - delivered < self.blocksize:
                time.sleep(0.002)
                continue
            mono = np.frombuffer(self.source.frames(delivered, self.blocksize), dtype=np.int16)
            block = mono.reshape(-1, self.source.channels)[:, :1].repeat(self.channels, axis=1)
            self.callback(block, self.blocksize,
                          SimpleNamespace(inputBufferAdcTime=opened + delivered / self.source.rate),
                          SimpleNamespace(input_overflow=False))
            delivered += self.blocksize

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def close(self):
        self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


# --- macOS system-audio helper ------------------------------------------------

class FakeHelper:
    """Stands in for subprocess.Popen launching macos/SystemAudioCapture:
    streams 16 kHz mono int16 on stdout until terminated (-15, like SIGTERM)."""

    def __init__(self, freq=880.0, exit_code=None, stderr=b""):
        """exit_code: the helper exits right away with this code and no
        audio (missing permission, a crash); None = it streams until stopped."""
        self.source = FakeSource(rate=16000, channels=1, freq=freq, name="helper")
        self.exit_code, self.stderr = exit_code, stderr
        self.launched = []

    def __call__(self, args, stdout=None, stderr=None):
        proc = _FakeProc(self.source, self.exit_code, self.stderr)
        self.launched.append(proc)
        return proc


class _FakeProc:
    def __init__(self, source, exit_code=None, stderr=b""):
        self.source, self.returncode = source, None
        self._terminated = threading.Event()
        self.stdout = SimpleNamespace(read=self._read)
        self.stderr = SimpleNamespace(read=lambda: stderr)
        self._opened, self._read_frames = time.monotonic(), 0
        if exit_code is not None:
            self.returncode = exit_code
            self._terminated.set()

    def _read(self, n_bytes):
        frames = n_bytes // 2
        while not self._terminated.is_set():
            if self.source.produced(time.monotonic() - self._opened) - self._read_frames >= frames:
                data = self.source.frames(self._read_frames, frames)
                self._read_frames += frames
                return data
            time.sleep(0.002)
        return b""

    def terminate(self):
        if self.returncode is None:
            self.returncode = -15
        self._terminated.set()

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode
