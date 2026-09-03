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
import numpy as np

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


def _normalize_device_name(name: str) -> str:
    return " ".join(name.casefold().split())


def _is_generic_input_alias(name: str) -> bool:
    normalized = _normalize_device_name(name)
    return any(alias in normalized for alias in _GENERIC_INPUT_ALIASES)


def _looks_like_truncated_alias(candidate: str, existing: str) -> bool:
    if len(candidate) >= 36:
        return False
    return existing.startswith(candidate) and len(existing) > len(candidate) + 6


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


def _frames_to_mono_float(frames: list[bytes], channels: int) -> np.ndarray:
    if not frames:
        return np.array([], dtype=np.float64)

    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float64)
    if channels > 1:
        usable_len = len(audio) - (len(audio) % channels)
        audio = audio[:usable_len].reshape(-1, channels).mean(axis=1)
    return audio


def _resample_to_rate(audio: np.ndarray, source_rate: int,
                      target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(audio) == 0:
        return audio

    from math import gcd
    from scipy.signal import resample_poly

    divisor = gcd(target_rate, source_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor)


def _pad_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)), mode="constant")


def _tail_audio_from_frames(frames: list[bytes], channels: int, sample_rate: int,
                            seconds: float) -> np.ndarray:
    """Converts frame chunks to a mono float64 array (int16 value range) and
    trims it to the last `seconds` seconds - building block for
    AudioRecorder.snapshot_recent_audio()."""
    audio = _frames_to_mono_float(frames, channels)
    max_samples = int(seconds * sample_rate)
    if len(audio) > max_samples:
        audio = audio[-max_samples:]
    return audio


def _mix_sources(mic_audio: np.ndarray, sys_audio: np.ndarray) -> np.ndarray:
    """Mixes two mono signals already resampled to the same sample rate
    (microphone + system loopback, int16 value range as float) - shared basis
    for both the final mix after stop (_record_both()/_record_both_macos())
    and a live snapshot while recording is still in progress (see
    AudioRecorder.snapshot_recent_audio()). Returns the int16 value range
    (neither normalized to -1..1 nor cast to int16) - callers decide based on
    their purpose (WAV saving vs. STT upload)."""
    out_len = max(len(mic_audio), len(sys_audio))
    mic_audio = _pad_to_length(mic_audio, out_len)
    sys_audio = _pad_to_length(sys_audio, out_len)
    mixed = (mic_audio * 0.65) + (sys_audio * 0.65)
    peak = np.max(np.abs(mixed)) if len(mixed) else 0
    if peak > 32767:
        mixed = mixed * (32767 / peak)
    return mixed


def _macos_system_audio_binary_path() -> str:
    """Path to the compiled ScreenCaptureKit helper (see macos/README.md)."""
    base = os.path.dirname(os.path.abspath(__file__))
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
    default_input_index = None
    try:
        default_input_index = int(p.get_default_input_device_info()["index"])
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
                "is_default": i == default_input_index,
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
        # Only populated when source == "both" (see _record_both()/
        # _record_both_macos()) - separate microphone/system buffers, so
        # snapshot_recent_audio() can mix both live while recording is still
        # in progress, not just once at the end (see there).
        self._mic_frames = None
        self._sys_frames = None
        self._mic_rate = None
        self._mic_channels = None
        self._sys_rate = None
        self._sys_channels = None
        self._on_level = None
        self._on_done = None

    def start(self, output_path: str, source: str = "mic",
              device_index: int | None = None,
              on_level=None, on_done=None):
        """Starts recording in a background thread."""
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

    def snapshot_recent_audio(self, seconds: float = 180.0):
        """Returns the last `seconds` seconds of audio recorded so far as a
        normalized mono float32 array (range -1..1, 16kHz) - WITHOUT
        interrupting the ongoing recording.

        Supports all three sources, including "both" (microphone + system):
        for that, _record_both()/_record_both_macos() also expose their
        otherwise-local microphone/system frame lists as instance attributes
        (`self._mic_frames`/`self._sys_frames`), which are trimmed and mixed
        live here (_mix_sources() - the same mix formula as the final mix
        after stop).

        Thread-safe because it's read-only: the frame lists are only ever
        extended by the recording thread via `.append()` (GIL-atomic),
        `list(...)` copies the current state immediately without blocking
        the recording thread.
        """
        if not self.is_recording:
            return None

        if self._source == "both":
            if not self._mic_frames or not self._sys_frames:
                return None
            mic_audio = _tail_audio_from_frames(
                list(self._mic_frames), self._mic_channels, self._mic_rate, seconds)
            sys_audio = _tail_audio_from_frames(
                list(self._sys_frames), self._sys_channels, self._sys_rate, seconds)
            mic_audio = _resample_to_rate(mic_audio, self._mic_rate, OUT_RATE)
            sys_audio = _resample_to_rate(sys_audio, self._sys_rate, OUT_RATE)
            mixed = _mix_sources(mic_audio, sys_audio)
            if len(mixed) == 0:
                return None
            return (mixed / 32768.0).astype(np.float32)

        if not self._frames:
            return None

        audio = _tail_audio_from_frames(
            list(self._frames), self._channels, self._sample_rate, seconds)
        if len(audio) == 0:
            return None
        audio = _resample_to_rate(audio, self._sample_rate, OUT_RATE)

        # _frames_to_mono_float returns the int16 value range as float64 (for
        # the internal mix/resample pipeline) - callers processing this for
        # STT typically expect values normalized to -1..1, like
        # load_audio_universal() returns.
        return (audio / 32768.0).astype(np.float32)

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
        (identical to _record_system_macos()); both frame lists are only
        mixed after stop.
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

        mic_frames = []
        sys_frames = []

        # Exposed as instance attributes for snapshot_recent_audio() - same
        # list objects, mic_callback()/the read loop below keep extending
        # them via `.append()`.
        self._mic_frames = mic_frames
        self._mic_rate = mic_rate
        self._mic_channels = mic_ch
        self._sys_frames = sys_frames
        self._sys_rate = sys_rate
        self._sys_channels = sys_ch

        def mic_callback(indata, frames, time_info, status):
            data = indata.tobytes()
            mic_frames.append(data)
            if self._on_level:
                self._on_level("mic", compute_rms(data))

        proc = None
        mic_stream = None
        try:
            mic_stream = sd.InputStream(
                samplerate=mic_rate, channels=mic_ch, dtype="int16",
                device=self._device_index, blocksize=CHUNK, callback=mic_callback)
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
                sys_frames.append(data)
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

        if not mic_frames:
            if self._on_done:
                self._on_done(None, 0, "Microphone produced no audio data.")
            return
        if not sys_frames:
            if self._on_done:
                self._on_done(None, 0, "System audio produced no audio data.")
            return

        # Mix both streams down to mono 16kHz (identical to _record_both())
        mic_audio = _frames_to_mono_float(mic_frames, mic_ch)
        sys_audio = _frames_to_mono_float(sys_frames, sys_ch)
        mic_audio = _resample_to_rate(mic_audio, mic_rate, OUT_RATE)
        sys_audio = _resample_to_rate(sys_audio, sys_rate, OUT_RATE)
        mixed = _mix_sources(mic_audio, sys_audio)
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
        self._channels = 1
        self._sample_rate = OUT_RATE
        self._frames = [mixed.tobytes()]
        _save_wav(self._output_path, self._frames, 1, OUT_RATE)

        duration = len(mixed) / OUT_RATE
        if self._on_done:
            self._on_done(self._output_path, duration, None)

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
                self._channels = 1
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
        mic_frames = []
        sys_frames = []
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
            mic_ch = 1
            mic_chunk = _chunk_size_for_rate(mic_rate)

            # Set up system audio (loopback)
            loopback = p.get_default_wasapi_loopback()
            sys_idx = loopback["index"]
            sys_rate = int(loopback["defaultSampleRate"])
            sys_ch = loopback["maxInputChannels"]
            sys_chunk = _chunk_size_for_rate(sys_rate)

            # Exposed as instance attributes for snapshot_recent_audio() -
            # same list objects, `read_loop()` keeps extending them via
            # `.append()`.
            self._mic_frames = mic_frames
            self._mic_rate = mic_rate
            self._mic_channels = mic_ch
            self._sys_frames = sys_frames
            self._sys_rate = sys_rate
            self._sys_channels = sys_ch

            mic_stream = p.open(
                format=FORMAT, channels=mic_ch, rate=mic_rate,
                input=True, input_device_index=mic_idx,
                frames_per_buffer=mic_chunk)

            sys_stream = p.open(
                format=FORMAT, channels=sys_ch, rate=sys_rate,
                input=True, input_device_index=sys_idx,
                frames_per_buffer=sys_chunk)

            def read_loop(stream, frames, chunk_size, level_channel):
                # WASAPI loopback streams deliver no packets while nothing is
                # playing on the system - a blocking read() can therefore
                # hang indefinitely. If you stop and close the stream from
                # another thread while read() is still blocked in that state,
                # the process crashes natively (no Python traceback, no
                # output). So poll via get_read_available() and only call
                # read() once enough data is actually available - this keeps
                # the loop responsive to stop_event at all times.
                while not self._stop_event.is_set():
                    try:
                        available = stream.get_read_available()
                    except Exception as e:
                        read_errors.append(e)
                        self._stop_event.set()
                        break

                    if available < chunk_size:
                        self._stop_event.wait(0.02)
                        continue

                    try:
                        data = stream.read(chunk_size, exception_on_overflow=False)
                    except Exception as e:
                        read_errors.append(e)
                        self._stop_event.set()
                        break

                    frames.append(data)
                    if self._on_level:
                        self._on_level(level_channel, compute_rms(data))

            mic_thread = threading.Thread(
                target=read_loop, args=(mic_stream, mic_frames, mic_chunk, "mic"),
                daemon=True)
            sys_thread = threading.Thread(
                target=read_loop, args=(sys_stream, sys_frames, sys_chunk, "system"),
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

        if not mic_frames:
            if self._on_done:
                self._on_done(None, 0, "Microphone produced no audio data.")
            return
        if not sys_frames:
            if self._on_done:
                self._on_done(None, 0, "System audio produced no audio data.")
            return

        # Mix both streams down to mono 16kHz
        mic_audio = _frames_to_mono_float(mic_frames, mic_ch)
        sys_audio = _frames_to_mono_float(sys_frames, sys_ch)
        mic_audio = _resample_to_rate(mic_audio, mic_rate, OUT_RATE)
        sys_audio = _resample_to_rate(sys_audio, sys_rate, OUT_RATE)
        mixed = _mix_sources(mic_audio, sys_audio)
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
        self._channels = 1
        self._sample_rate = OUT_RATE
        self._frames = [mixed.tobytes()]
        _save_wav(self._output_path, self._frames, 1, OUT_RATE)

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
                      sample_rate: int | None = None, channels: int = 1):
    """Records audio from the microphone and saves it as a WAV file.

    Stops on Enter.
    """
    if IS_WINDOWS:
        return _record_microphone_pyaudio(output_path, device_index, sample_rate, channels)
    return _record_microphone_sounddevice(output_path, device_index, sample_rate, channels)


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
                               sample_rate: int | None = None, channels: int = 1):
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
                                 device_index: int | None = None):
    """Records microphone + system audio and saves the mix as a WAV file."""
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
