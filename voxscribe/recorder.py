"""Audio recording from microphone (all platforms) and system audio (Windows +
experimental macOS).

- Windows: PyAudioWPatch for microphone + WASAPI loopback (system audio/
  meeting recording, including combined "microphone + system" recording).
- macOS: `sounddevice` for the microphone; system-audio recording runs via a
  small native helper (macos/SystemAudioCapture, ScreenCaptureKit) launched
  as a subprocess - see macos/README.md. Combined "microphone + system"
  recording (_record_both_macos()) mixes both sources like on Windows.
  EXPERIMENTAL: the system-audio path is tested on real Apple Silicon
  hardware, the combined microphone+system variant is not yet.
- Linux: microphone only via `sounddevice`, no system-audio path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import wave
import threading
import time
import numpy as np

from voxscribe.recording_modes import normalize_device_name as _normalize_device_name
from voxscribe.recording_modes import suggest_mic_for_loopback  # noqa: F401 (re-export)
from voxscribe.track_assembler import DropDetector, TrackAssembler
from voxscribe.tracks import MIC, SYSTEM

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"

if IS_WINDOWS:
    import pyaudiowpatch as pyaudio
else:
    import sounddevice as sd

CHUNK = 1024
FORMAT = pyaudio.paInt16 if IS_WINDOWS else None
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
OUT_RATE = 16000

_GENERIC_INPUT_ALIASES = (
    "microsoft soundmapper - input",
    "microsoft sound mapper - input",
    "primaerer soundaufnahmetreiber",
    "primarer soundaufnahmetreiber",
    "primärer soundaufnahmetreiber",
    "primary sound capture driver",
)


def _host_api_name(p: pyaudio.PyAudio, info: dict) -> str:
    try:
        host_api = p.get_host_api_info_by_index(info["hostApi"])
        return host_api.get("name", "")
    except Exception:
        return ""


def _device_score(device: dict) -> tuple[int, int, int]:
    host_api = device.get("host_api", "").lower()
    if "wasapi" in host_api:
        api_score = 4
    elif "directsound" in host_api:
        api_score = 3
    elif "wdm" in host_api:
        api_score = 2
    elif "mme" in host_api:
        api_score = 1
    else:
        api_score = 0

    default_score = 1 if device.get("is_default") else 0
    channel_score = min(int(device.get("channels", 0)), 2)
    return (api_score, default_score, channel_score)


def _is_generic_input_alias(name: str) -> bool:
    normalized = _normalize_device_name(name)
    return any(alias in normalized for alias in _GENERIC_INPUT_ALIASES)


_MME_NAME_LIMIT = 31  # MME cuts device names to 31 characters


def _looks_like_truncated_alias(candidate: str, existing: str) -> bool:
    if len(candidate) >= 36 or not existing.startswith(candidate):
        return False
    # Exactly at MME's limit, any longer name with the same prefix is the
    # same device (e.g. "Kopfhörermikrofon (2- Jabra Lin" vs. "... Link 390)").
    if len(candidate) == _MME_NAME_LIMIT:
        return len(existing) > len(candidate)
    return len(existing) > len(candidate) + 6


def _dedupe_audio_devices(devices: list[dict]) -> list[dict]:
    """Collapse Windows host-API aliases while keeping a real device index."""
    filtered = [d for d in devices if not _is_generic_input_alias(d["name"])]
    if not filtered:
        filtered = devices

    by_name = {}
    for device in filtered:
        key = _normalize_device_name(device["name"])
        previous = by_name.get(key)
        if previous is None or _device_score(device) > _device_score(previous):
            by_name[key] = device

    chosen = sorted(by_name.values(), key=_device_score, reverse=True)
    result = []
    for device in chosen:
        name = _normalize_device_name(device["name"])
        if any(
            _looks_like_truncated_alias(name, _normalize_device_name(existing["name"]))
            or _looks_like_truncated_alias(_normalize_device_name(existing["name"]), name)
            for existing in result
        ):
            continue
        result.append(device)

    return sorted(result, key=lambda d: (not d.get("is_default"), d["name"].casefold()))


def _chunk_size_for_rate(sample_rate: int) -> int:
    return max(256, int(round(CHUNK * sample_rate / OUT_RATE)))


def _mic_channels(info: dict) -> int:
    """Channel count to open a microphone with: its native count (max 2).

    Never open a multi-channel WASAPI mic as mono. In shared mode PortAudio
    then has to convert from the engine's mix format itself, and for some
    devices (seen with a Jabra Link 390 Bluetooth headset, 48kHz/2ch) that
    path delivers mostly digital silence with garbled fragments of speech,
    while a native 2-channel stream of the same device is clean. Callers
    downmix to mono themselves (e.g. the TrackAssembler for a Call).
    """
    return max(1, min(2, int(info.get("maxInputChannels", 1) or 1)))


def _macos_system_audio_binary_path() -> str:
    """Path to the compiled ScreenCaptureKit helper (see macos/README.md)."""
    # This file lives in voxscribe/, macos/ is one level up at the project root.
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "macos", "SystemAudioCapture")


def _get_loopback_device(p: pyaudio.PyAudio, device_index: int | None = None):
    if device_index is None:
        return p.get_default_wasapi_loopback()

    try:
        for device in p.get_loopback_device_info_generator():
            if int(device["index"]) == int(device_index):
                return device
    except OSError as e:
        raise RuntimeError("No WASAPI loopback devices found.") from e

    raise RuntimeError(
        f"Device [{device_index}] is not a WASAPI loopback device."
    )


def get_devices():
    """Returns all available audio devices as structured lists."""
    if not IS_WINDOWS:
        return _get_devices_sounddevice()

    p = pyaudio.PyAudio()
    microphones = []
    loopback = []
    # PortAudio's own default input is the MME alias, which deduplication
    # below drops in favor of the WASAPI entry of the same device - so mark
    # the WASAPI default input as default too, or the flag gets lost.
    default_input_indices = set()
    try:
        default_input_indices.add(int(p.get_default_input_device_info()["index"]))
    except OSError:
        pass
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        if wasapi.get("defaultInputDevice", -1) >= 0:
            default_input_indices.add(int(wasapi["defaultInputDevice"]))
    except OSError:
        pass

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and "loopback" not in info["name"].lower():
            microphones.append({
                "index": i,
                "name": info["name"],
                "host_api": _host_api_name(p, info),
                "channels": info["maxInputChannels"],
                "sample_rate": int(info["defaultSampleRate"]),
                "is_default": i in default_input_indices,
            })

    try:
        for device in p.get_loopback_device_info_generator():
            loopback.append({
                "index": device["index"],
                "name": device["name"],
                "host_api": _host_api_name(p, device),
                "channels": device["maxInputChannels"],
                "sample_rate": int(device["defaultSampleRate"]),
            })
    except OSError:
        pass

    default_loopback = None
    try:
        dl = p.get_default_wasapi_loopback()
        default_loopback = dl["index"]
    except OSError:
        pass

    p.terminate()
    return {"microphones": _dedupe_audio_devices(microphones),
            "loopback": _dedupe_audio_devices(loopback),
            "default_loopback": default_loopback}


def _get_devices_sounddevice():
    """get_devices() equivalent for macOS/Linux via sounddevice.

    Only returns microphones - system audio (loopback) has no WASAPI
    equivalent on these platforms and is deliberately not supported here.
    """
    microphones = []
    default_input_index = None
    try:
        default_input_index = sd.default.device[0]
        if default_input_index is not None and default_input_index < 0:
            default_input_index = None
    except Exception:
        pass

    try:
        hostapis = sd.query_hostapis()
    except Exception:
        hostapis = []

    for i, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) > 0:
            host_api_name = ""
            try:
                host_api_name = hostapis[info["hostapi"]]["name"]
            except Exception:
                pass
            microphones.append({
                "index": i,
                "name": info["name"],
                "host_api": host_api_name,
                "channels": info["max_input_channels"],
                "sample_rate": int(info["default_samplerate"]),
                "is_default": i == default_input_index,
            })

    return {"microphones": _dedupe_audio_devices(microphones),
            "loopback": [],
            "default_loopback": None}


def compute_rms(data: bytes) -> float:
    """Computes the RMS level of an audio chunk."""
    samples = np.frombuffer(data, dtype=np.int16)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


class AudioRecorder:
    """Reusable audio recorder with callback support for the GUI."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._frames = []
        self._thread = None
        self._channels = 1
        self._sample_rate = 16000
        self._on_level = None
        self._on_done = None

    def start(self, output_path: str, source: str = "mic",
              device_index: int | None = None,
              on_level=None, on_done=None,
              loopback_index: int | None = None, setup: str | None = None):
        """Starts recording in a background thread.

        For source "both" (a Call), device_index is the microphone,
        loopback_index the playback device (None = Windows default; ignored
        on macOS, which captures all playback), and setup ("headset" /
        "loudspeaker", see recording_modes) is stored in the tracks sidecar."""
        if not IS_WINDOWS and source != "mic":
            if IS_MACOS and source == "system":
                pass  # experimentally supported, see _record_system_macos()
            elif IS_MACOS and source == "both":
                pass  # experimentally supported, see _record_both_macos()
            else:
                # System audio (WASAPI loopback) otherwise only exists on Windows.
                if on_done:
                    on_done(None, 0,
                             "System-audio recording isn't supported on this OS yet "
                             "(Windows/WASAPI only, macOS experimentally). "
                             "Please choose the 'Microphone' source.")
                return

        self._stop_event.clear()
        self._frames = []
        self._output_path = output_path
        self._on_level = on_level
        self._on_done = on_done
        self._source = source
        self._device_index = device_index
        self._loopback_index = loopback_index
        self._setup = setup
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops recording."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _record(self):
        if IS_MACOS and self._source == "system":
            self._record_system_macos()
        elif IS_MACOS and self._source == "both":
            self._record_both_macos()
        elif not IS_WINDOWS:
            # start() already guarantees source == "mic" on non-Windows
            # (except the macOS system-audio branch above).
            self._record_single_sounddevice()
        elif self._source == "both":
            self._record_both()
        else:
            self._record_single()

    def _record_single_sounddevice(self):
        """Microphone recording via sounddevice (macOS/Linux)."""
        level_channel = "mic"
        try:
            if self._device_index is not None:
                info = sd.query_devices(self._device_index)
            else:
                info = sd.query_devices(kind="input")
            self._channels = 1
            self._sample_rate = int(info["default_samplerate"])

            def callback(indata, frames, time_info, status):
                data = indata.tobytes()
                self._frames.append(data)
                if self._on_level:
                    self._on_level(level_channel, compute_rms(data))

            with sd.InputStream(samplerate=self._sample_rate, channels=self._channels,
                                 dtype="int16", device=self._device_index,
                                 blocksize=CHUNK, callback=callback):
                while not self._stop_event.is_set():
                    self._stop_event.wait(0.05)
        except Exception as e:
            if self._on_done:
                self._on_done(None, 0, str(e))
            return

        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
        _save_wav(self._output_path, self._frames, self._channels, self._sample_rate)
        duration = (len(b"".join(self._frames))
                    / (self._sample_rate * self._channels * SAMPLE_WIDTH))
        if self._on_done:
            self._on_done(self._output_path, duration, None)

    def _record_system_macos(self):
        """System-audio recording on macOS via a ScreenCaptureKit subprocess.

        Tested and debugged on real Apple Silicon hardware (see
        macos/README.md) - the helper (macos/SystemAudioCapture) streams raw
        16kHz mono int16 PCM bytes to stdout, read here in a blocking loop. A
        separate watcher thread terminates the subprocess as soon as stop()
        is called - this closes stdout (EOF), so the blocking read() returns
        immediately instead of closing a resource from another thread while a
        read is still active (the original WASAPI bug this pattern avoids).
        """
        level_channel = "system"
        binary = _macos_system_audio_binary_path()
        if not os.path.isfile(binary):
            if self._on_done:
                self._on_done(
                    None, 0,
                    "System-audio helper not found. Run './build.sh' in "
                    "macos/ first (see macos/README.md)."
                )
            return

        self._channels = 1
        self._sample_rate = OUT_RATE  # SystemAudioCapture is fixed at 16kHz/mono
        chunk_bytes = CHUNK * SAMPLE_WIDTH

        proc = None
        try:
            proc = subprocess.Popen(
                [binary], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            def _watch_stop():
                self._stop_event.wait()
                try:
                    proc.terminate()
                except Exception:
                    pass

            watcher = threading.Thread(target=_watch_stop, daemon=True)
            watcher.start()

            while True:
                data = proc.stdout.read(chunk_bytes)
                if not data:
                    break
                self._frames.append(data)
                if self._on_level:
                    self._on_level(level_channel, compute_rms(data))

            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

            # -15/-2 = terminated cleanly via SIGTERM/SIGINT (the expected stop path)
            if proc.returncode not in (0, None, -15, -2):
                stderr_output = (proc.stderr.read() or b"").decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    stderr_output or f"SystemAudioCapture exited with code {proc.returncode}")

            if not self._frames:
                stderr_output = (proc.stderr.read() or b"").decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    stderr_output or
                    "No audio data received. Has the 'Screen & System Audio "
                    "Recording' permission been granted? "
                    "(System Settings > Privacy & Security)")
        except Exception as e:
            if proc is not None and proc.poll() is None:
                proc.kill()
            if self._on_done:
                self._on_done(None, 0, str(e))
            return

        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
        _save_wav(self._output_path, self._frames, self._channels, self._sample_rate)
        duration = (len(b"".join(self._frames))
                    / (self._sample_rate * self._channels * SAMPLE_WIDTH))
        if self._on_done:
            self._on_done(self._output_path, duration, None)

    def _record_both_macos(self):
        """Records the microphone (sounddevice) and system audio
        (ScreenCaptureKit subprocess, see _record_system_macos())
        simultaneously and mixes them - macOS equivalent of _record_both() on
        Windows.

        The microphone runs on sounddevice's own callback thread, the
        system-audio side is read in a blocking loop on the current thread
        (identical to _record_system_macos()); both feed a TrackAssembler,
        which builds the tracks and the mix after stop.
        """
        binary = _macos_system_audio_binary_path()
        if not os.path.isfile(binary):
            if self._on_done:
                self._on_done(
                    None, 0,
                    "System-audio helper not found. Run './build.sh' in "
                    "macos/ first (see macos/README.md)."
                )
            return

        try:
            if self._device_index is not None:
                mic_info = sd.query_devices(self._device_index)
            else:
                mic_info = sd.query_devices(kind="input")
        except Exception as e:
            if self._on_done:
                self._on_done(None, 0, str(e))
            return

        mic_ch = 1
        mic_rate = int(mic_info["default_samplerate"])
        sys_ch = 1
        sys_rate = OUT_RATE  # SystemAudioCapture is fixed at 16kHz/mono
        chunk_bytes = CHUNK * SAMPLE_WIDTH
        # The helper streams continuously from its first sample on, so the
        # Microphone's lead while it launches is trimmed (#20) and a slow
        # pipe read isn't padded as a gap.
        assembler = TrackAssembler(mic_rate, mic_ch, sys_rate, sys_ch,
                                   playback_is_continuous=True)
        next_adc_time = [0.0]  # where the next mic block should start

        def mic_callback(indata, frames, time_info, status):
            adc_time = time_info.inputBufferAdcTime
            if status.input_overflow:
                if next_adc_time[0] and adc_time:
                    assembler.add_dropped(MIC, round((adc_time - next_adc_time[0]) * mic_rate))
                else:
                    print("[WARNING] Microphone input overflow (length unknown).")
            next_adc_time[0] = adc_time + frames / mic_rate if adc_time else 0.0
            data = indata.tobytes()
            assembler.add_chunk(MIC, data)
            if self._on_level:
                self._on_level("mic", compute_rms(data))

        proc = None
        mic_stream = None
        try:
            # ~10 ms blocks: the Microphone's sample count is the Call's time
            # base, see TrackAssembler.
            mic_stream = sd.InputStream(
                samplerate=mic_rate, channels=mic_ch, dtype="int16",
                device=self._device_index, blocksize=max(1, mic_rate // 100),
                callback=mic_callback)
            mic_stream.start()

            proc = subprocess.Popen(
                [binary], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            def _watch_stop():
                self._stop_event.wait()
                try:
                    proc.terminate()
                except Exception:
                    pass

            watcher = threading.Thread(target=_watch_stop, daemon=True)
            watcher.start()

            while True:
                data = proc.stdout.read(chunk_bytes)
                if not data:
                    break
                assembler.add_chunk(SYSTEM, data)
                if self._on_level:
                    self._on_level("system", compute_rms(data))

            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

            # -15/-2 = terminated cleanly via SIGTERM/SIGINT (the expected stop path)
            if proc.returncode not in (0, None, -15, -2):
                stderr_output = (proc.stderr.read() or b"").decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    stderr_output or f"SystemAudioCapture exited with code {proc.returncode}")
        except Exception as e:
            if proc is not None and proc.poll() is None:
                proc.kill()
            if self._on_done:
                self._on_done(None, 0, str(e))
            return
        finally:
            if mic_stream is not None:
                try:
                    mic_stream.stop()
                    mic_stream.close()
                except Exception:
                    pass

        self._finish_call(assembler)

    def _record_single(self):
        p = pyaudio.PyAudio()
        level_channel = "system" if self._source == "system" else "mic"
        try:
            if self._source == "system":
                loopback = _get_loopback_device(p, self._device_index)
                self._channels = loopback["maxInputChannels"]
                self._sample_rate = int(loopback["defaultSampleRate"])
                device_idx = loopback["index"]
            else:
                if self._device_index is not None:
                    info = p.get_device_info_by_index(self._device_index)
                else:
                    info = p.get_default_input_device_info()
                device_idx = info["index"]
                self._channels = _mic_channels(info)
                self._sample_rate = int(info["defaultSampleRate"])

            stream = p.open(
                format=FORMAT,
                channels=self._channels,
                rate=self._sample_rate,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=CHUNK,
            )

            while not self._stop_event.is_set():
                if self._source == "system" and stream.get_read_available() < CHUNK:
                    # WASAPI loopback delivers no packets while the system is
                    # silent - without this polling, read() would block
                    # indefinitely and "stop" wouldn't respond.
                    self._stop_event.wait(0.02)
                    continue
                data = stream.read(CHUNK, exception_on_overflow=False)
                self._frames.append(data)
                if self._on_level:
                    self._on_level(level_channel, compute_rms(data))

            stream.stop_stream()
            stream.close()
        except Exception as e:
            if self._on_done:
                self._on_done(None, 0, str(e))
            return
        finally:
            p.terminate()

        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
        _save_wav(self._output_path, self._frames, self._channels,
                  self._sample_rate)
        duration = (len(b"".join(self._frames))
                    / (self._sample_rate * self._channels * SAMPLE_WIDTH))
        if self._on_done:
            self._on_done(self._output_path, duration, None)

    def _record_both(self):
        """Records microphone + system audio simultaneously and mixes them."""
        p = pyaudio.PyAudio()
        mic_stream = None
        sys_stream = None
        mic_thread = None
        sys_thread = None
        read_errors = []

        try:
            # Set up microphone
            if self._device_index is not None:
                mic_info = p.get_device_info_by_index(self._device_index)
            else:
                mic_info = p.get_default_input_device_info()
            mic_idx = mic_info["index"]
            mic_rate = int(mic_info["defaultSampleRate"])
            mic_ch = _mic_channels(mic_info)
            mic_chunk = max(1, mic_rate // 100)  # read from ~10 ms on, see read_loop

            # Set up system audio (loopback)
            loopback = _get_loopback_device(p, self._loopback_index)
            sys_idx = loopback["index"]
            sys_rate = int(loopback["defaultSampleRate"])
            sys_ch = loopback["maxInputChannels"]
            sys_chunk = _chunk_size_for_rate(sys_rate)
            assembler = TrackAssembler(mic_rate, mic_ch, sys_rate, sys_ch)

            mic_stream = p.open(
                format=FORMAT, channels=mic_ch, rate=mic_rate,
                input=True, input_device_index=mic_idx,
                # A 64 ms buffer, although it's read from 10 ms on: a 10 ms
                # one overflowed while the GUI kept the reader thread from
                # the GIL (15 % of the audio lost under load).
                frames_per_buffer=_chunk_size_for_rate(mic_rate))

            sys_stream = p.open(
                format=FORMAT, channels=sys_ch, rate=sys_rate,
                input=True, input_device_index=sys_idx,
                frames_per_buffer=sys_chunk)

            def read_loop(stream, source, chunk_size, read_all=False):
                # WASAPI loopback streams deliver no packets while nothing is
                # playing on the system - a blocking read() can therefore
                # hang indefinitely. If you stop and close the stream from
                # another thread while read() is still blocked in that state,
                # the process crashes natively (no Python traceback, no
                # output). So poll via get_read_available() and only call
                # read() once enough data is actually available - this keeps
                # the loop responsive to stop_event at all times.
                #
                # The silent stretches this leaves in the Playback track are
                # padded by the TrackAssembler, which follows the
                # Microphone's sample count. read_all keeps that count fresh:
                # the Microphone is read as soon as ~10 ms are available,
                # all of it at once. Playback reports what's still buffered
                # behind each chunk, so a late read isn't taken for a gap.
                #
                # PyAudio drops input overflows silently, and a lost
                # Microphone stretch would shift the whole Call; the
                # DropDetector finds it against the wall clock and it
                # becomes silence of the same length (#21).
                drops = DropDetector(mic_rate) if read_all else None
                received = 0
                last_level = time.monotonic()
                while not self._stop_event.is_set():
                    try:
                        available = stream.get_read_available()
                    except Exception as e:
                        read_errors.append(e)
                        self._stop_event.set()
                        break

                    if available < chunk_size:
                        # Silent loopback sends nothing, which would freeze
                        # its level meter at the last value: report 0.
                        now = time.monotonic()
                        if self._on_level and now - last_level >= 0.05:
                            self._on_level(source, 0.0)
                            last_level = now
                        self._stop_event.wait(0.01 if read_all else 0.02)
                        continue

                    frames = available if read_all else chunk_size
                    try:
                        data = stream.read(frames, exception_on_overflow=False)
                    except Exception as e:
                        read_errors.append(e)
                        self._stop_event.set()
                        break

                    assembler.add_chunk(source, data, buffered_frames=available - frames)
                    if drops is not None:
                        received += frames
                        lost = drops.missing(received, time.monotonic())
                        if lost:
                            assembler.add_dropped(source, lost)
                            received += lost
                    if self._on_level:
                        self._on_level(source, compute_rms(data))
                        last_level = time.monotonic()

            mic_thread = threading.Thread(
                target=read_loop, args=(mic_stream, MIC, mic_chunk),
                kwargs={"read_all": True}, daemon=True)
            sys_thread = threading.Thread(
                target=read_loop, args=(sys_stream, SYSTEM, sys_chunk),
                daemon=True)
            mic_thread.start()
            sys_thread.start()

            while not self._stop_event.is_set():
                self._stop_event.wait(0.05)

            mic_thread.join(timeout=2)
            sys_thread.join(timeout=2)

            if read_errors:
                raise read_errors[0]
        except Exception as e:
            if self._on_done:
                self._on_done(None, 0, str(e))
            return
        finally:
            # Defensive: never close a stream while its read thread could
            # theoretically still be stuck in stream.read() (see comment in
            # read_loop) - that's a native crash, not a catchable Python error.
            for stream, thread in ((mic_stream, mic_thread), (sys_stream, sys_thread)):
                if stream is None:
                    continue
                if thread is not None and thread.is_alive():
                    continue
                try:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if (mic_thread is None or not mic_thread.is_alive()) and \
               (sys_thread is None or not sys_thread.is_alive()):
                p.terminate()

        self._finish_call(assembler, require_playback=False)

    def _finish_call(self, assembler: TrackAssembler, require_playback: bool = True):
        """After-recording work shared by both Call paths (_record_both(),
        _record_both_macos()): saves the mix and the tracks sidecar and
        reports the result via on_done.

        require_playback=False: Playback that never delivered is a silent
        track, not an error (WASAPI loopback sends nothing while nothing
        plays; the macOS helper sending nothing means missing permission)."""
        if not assembler.has_audio(MIC):
            if self._on_done:
                self._on_done(None, 0, "Microphone produced no audio data.")
            return
        if require_playback and not assembler.has_audio(SYSTEM):
            if self._on_done:
                self._on_done(None, 0, "Playback audio produced no audio data.")
            return

        mixed = assembler.save(self._output_path, self._setup)
        self._channels = 1
        self._sample_rate = OUT_RATE
        self._frames = [mixed.tobytes()]

        duration = len(mixed) / OUT_RATE
        if self._on_done:
            self._on_done(self._output_path, duration, None)


def list_devices():
    """Shows all available audio devices (microphones + WASAPI loopback)."""
    devices = get_devices()
    print("\n=== Available audio devices ===\n")

    print("--- Microphones (input devices) ---")
    if devices["microphones"]:
        for device in devices["microphones"]:
            host_api = f", API: {device['host_api']}" if device.get("host_api") else ""
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Channels: {device['channels']}, "
                  f"Sample rate: {device['sample_rate']} Hz{host_api}")
    else:
        print("  No microphones found.")

    print("\n--- System audio (WASAPI loopback) ---")
    if IS_MACOS:
        print("  Experimental via ScreenCaptureKit (see macos/README.md) - "
              "no device selection, captures the entire system playback.")
    elif not IS_WINDOWS:
        print("  Not supported on this OS (Windows/WASAPI only).")
    elif devices["loopback"]:
        for device in devices["loopback"]:
            host_api = f", API: {device['host_api']}" if device.get("host_api") else ""
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Channels: {device['channels']}, "
                  f"Sample rate: {device['sample_rate']} Hz{host_api}")
    else:
        print("  No WASAPI loopback devices found.")

    default_loopback = devices.get("default_loopback")
    if default_loopback is not None:
        print(f"\n  Default loopback: [{default_loopback}]")

    print()


def _level_bar(data: bytes, channels: int) -> str:
    """Builds a simple volume bar for the console."""
    samples = np.frombuffer(data, dtype=np.int16)
    if len(samples) == 0:
        return ""
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    # Normalize to 0-50 characters
    level = min(int(rms / 500), 50)
    bar = "█" * level + "░" * (50 - level)
    return f"\r  Level: |{bar}| {rms:6.0f}"


def record_microphone(output_path: str, device_index: int | None = None,
                      sample_rate: int | None = None, channels: int | None = None):
    """Records audio from the microphone and saves it as a WAV file.

    Stops on Enter.
    """
    if IS_WINDOWS:
        return _record_microphone_pyaudio(output_path, device_index, sample_rate, channels)
    return _record_microphone_sounddevice(output_path, device_index, sample_rate, channels or 1)


def _record_microphone_sounddevice(output_path: str, device_index: int | None = None,
                                   sample_rate: int | None = None, channels: int = 1):
    """Microphone recording via sounddevice (macOS/Linux). Stops on ENTER."""
    if device_index is not None:
        info = sd.query_devices(device_index)
        print(f"Microphone: [{device_index}] {info['name']}")
    else:
        info = sd.query_devices(kind="input")
        device_index = None
        print(f"Default microphone: {info['name']}")

    if sample_rate is None:
        sample_rate = int(info["default_samplerate"])
    print(f"  Channels: {channels}, Sample rate: {sample_rate} Hz")

    frames = []
    stop_event = threading.Event()

    def callback(indata, frame_count, time_info, status):
        data = indata.tobytes()
        frames.append(data)
        print(_level_bar(data, channels), end="", flush=True)

    def wait_for_enter():
        input()
        stop_event.set()

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    print("Recording... Press ENTER to stop.\n")

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="int16",
                        device=device_index, blocksize=CHUNK, callback=callback):
        try:
            while not stop_event.is_set():
                stop_event.wait(0.05)
        except KeyboardInterrupt:
            pass

    print("\n\nRecording stopped.")

    _save_wav(output_path, frames, channels, sample_rate)
    return output_path


def _record_microphone_pyaudio(output_path: str, device_index: int | None = None,
                               sample_rate: int | None = None, channels: int | None = None):
    p = pyaudio.PyAudio()

    if device_index is not None:
        info = p.get_device_info_by_index(device_index)
        print(f"Microphone: [{device_index}] {info['name']}")
    else:
        info = p.get_default_input_device_info()
        device_index = info["index"]
        print(f"Default microphone: [{device_index}] {info['name']}")

    if sample_rate is None:
        sample_rate = int(info["defaultSampleRate"])
    if channels is None:
        channels = _mic_channels(info)
    print(f"  Channels: {channels}, Sample rate: {sample_rate} Hz")

    stream = p.open(
        format=FORMAT,
        channels=channels,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK,
    )

    frames = []
    stop_event = threading.Event()

    def wait_for_enter():
        input()
        stop_event.set()

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    print("Recording... Press ENTER to stop.\n")

    try:
        while not stop_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            print(_level_bar(data, channels), end="", flush=True)
    except KeyboardInterrupt:
        pass

    print("\n\nRecording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    _save_wav(output_path, frames, channels, sample_rate)
    return output_path


def record_system_audio(output_path: str, device_index: int | None = None):
    """Records system audio and saves it as a WAV file. Stops on Enter.

    Windows: WASAPI loopback. macOS: experimental via ScreenCaptureKit (see
    macos/README.md) - device_index is ignored there, there's no device
    selection. Linux: not supported.
    """
    if IS_MACOS:
        return _record_system_audio_macos_cli(output_path)

    if not IS_WINDOWS:
        raise RuntimeError(
            "System-audio recording isn't supported on this OS yet "
            "(Windows only, macOS experimentally). Use --source mic."
        )
    p = pyaudio.PyAudio()

    try:
        loopback = _get_loopback_device(p, device_index)
    except (OSError, RuntimeError) as e:
        p.terminate()
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(
            "No WASAPI loopback device found. "
            "Make sure an audio output device is active."
        ) from e

    channels = loopback["maxInputChannels"]
    sample_rate = int(loopback["defaultSampleRate"])

    print(f"System audio (loopback): [{loopback['index']}] {loopback['name']}")
    print(f"  Channels: {channels}, Sample rate: {sample_rate} Hz")

    stream = p.open(
        format=FORMAT,
        channels=channels,
        rate=sample_rate,
        input=True,
        input_device_index=loopback["index"],
        frames_per_buffer=CHUNK,
    )

    frames = []
    stop_event = threading.Event()

    def wait_for_enter():
        input()
        stop_event.set()

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    print("Recording... Press ENTER to stop.\n")

    try:
        while not stop_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            print(_level_bar(data, channels), end="", flush=True)
    except KeyboardInterrupt:
        pass

    print("\n\nRecording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    _save_wav(output_path, frames, channels, sample_rate)
    return output_path


def _record_system_audio_macos_cli(output_path: str):
    """CLI variant of the experimental macOS system-audio recording (ENTER stops).

    Uses AudioRecorder._record_system_macos() under the hood (subprocess-based
    ScreenCaptureKit helper) - see macos/README.md for status.
    """
    audio_recorder = AudioRecorder()
    done_event = threading.Event()
    stop_requested = threading.Event()
    result = {"path": None, "duration": 0, "error": None}

    def on_done(path, duration, error):
        result["path"] = path
        result["duration"] = duration
        result["error"] = error
        done_event.set()

    def wait_for_enter():
        input()
        stop_requested.set()

    print("System audio (macOS, experimental via ScreenCaptureKit)")
    audio_recorder.start(output_path=output_path, source="system", on_done=on_done)

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    print("Recording... Press ENTER to stop.\n")
    try:
        while not stop_requested.is_set() and not done_event.is_set():
            done_event.wait(0.1)
    except KeyboardInterrupt:
        pass

    if not done_event.is_set():
        print("\n\nRecording stopped. Saving...")
        audio_recorder.stop()
        if not done_event.wait(timeout=120):
            raise RuntimeError("Recording could not be completed.")

    if result["error"]:
        raise RuntimeError(result["error"])

    return result["path"] or output_path


def record_microphone_and_system(output_path: str,
                                 device_index: int | None = None,
                                 loopback_index: int | None = None,
                                 setup: str | None = None):
    """Records a Call (microphone + playback audio) and saves the mix as a
    WAV file, plus the separate tracks tagged with the Setup."""
    recorder = AudioRecorder()
    done_event = threading.Event()
    stop_requested = threading.Event()
    result = {"path": None, "duration": 0, "error": None}

    def on_done(path, duration, error):
        result["path"] = path
        result["duration"] = duration
        result["error"] = error
        done_event.set()

    def wait_for_enter():
        input()
        stop_requested.set()

    print("Microphone + system audio")
    recorder.start(
        output_path=output_path,
        source="both",
        device_index=device_index,
        loopback_index=loopback_index,
        setup=setup,
        on_done=on_done,
    )

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    print("Recording... Press ENTER to stop.\n")
    try:
        while not stop_requested.is_set() and not done_event.is_set():
            done_event.wait(0.1)
    except KeyboardInterrupt:
        pass

    if not done_event.is_set():
        print("\n\nRecording stopped. Saving...")
        recorder.stop()
        if not done_event.wait(timeout=120):
            raise RuntimeError("Recording could not be completed.")

    if result["error"]:
        raise RuntimeError(result["error"])

    return result["path"] or output_path


def _save_wav(output_path: str, frames: list[bytes], channels: int,
              sample_rate: int):
    """Saves recorded frames as a WAV file."""
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    duration = len(b"".join(frames)) / (sample_rate * channels * SAMPLE_WIDTH)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({duration:.1f}s, {size_mb:.1f} MB)")
