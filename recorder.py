"""Audio-Aufnahme von Mikrofon und System-Audio (WASAPI Loopback)."""

import os
import wave
import threading
import numpy as np

import pyaudiowpatch as pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes


def get_devices():
    """Gibt alle verfuegbaren Audio-Geraete als strukturierte Listen zurueck."""
    p = pyaudio.PyAudio()
    microphones = []
    loopback = []

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and "loopback" not in info["name"].lower():
            microphones.append({
                "index": i,
                "name": info["name"],
                "channels": info["maxInputChannels"],
                "sample_rate": int(info["defaultSampleRate"]),
            })

    try:
        for device in p.get_loopback_device_info_generator():
            loopback.append({
                "index": device["index"],
                "name": device["name"],
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
    return {"microphones": microphones, "loopback": loopback,
            "default_loopback": default_loopback}


def compute_rms(data: bytes) -> float:
    """Berechnet den RMS-Pegel eines Audio-Chunks."""
    samples = np.frombuffer(data, dtype=np.int16)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


class AudioRecorder:
    """Wiederverwendbarer Audio-Recorder mit Callback-Support fuer GUI."""

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
              on_level=None, on_done=None):
        """Startet die Aufnahme in einem Background-Thread."""
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
        """Stoppt die Aufnahme."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _record(self):
        if self._source == "both":
            self._record_both()
        else:
            self._record_single()

    def _record_single(self):
        p = pyaudio.PyAudio()
        try:
            if self._source == "system":
                loopback = p.get_default_wasapi_loopback()
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
                self._sample_rate = 16000

            stream = p.open(
                format=FORMAT,
                channels=self._channels,
                rate=self._sample_rate,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=CHUNK,
            )

            while not self._stop_event.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                self._frames.append(data)
                if self._on_level:
                    self._on_level(compute_rms(data))

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
        """Nimmt gleichzeitig Mikrofon + System-Audio auf und mischt beides."""
        p = pyaudio.PyAudio()
        mic_frames = []
        sys_frames = []
        out_rate = 16000

        try:
            # Mikrofon einrichten
            if self._device_index is not None:
                mic_info = p.get_device_info_by_index(self._device_index)
            else:
                mic_info = p.get_default_input_device_info()
            mic_idx = mic_info["index"]
            mic_rate = 16000
            mic_ch = 1

            # System-Audio (Loopback) einrichten
            loopback = p.get_default_wasapi_loopback()
            sys_idx = loopback["index"]
            sys_rate = int(loopback["defaultSampleRate"])
            sys_ch = loopback["maxInputChannels"]

            mic_stream = p.open(
                format=FORMAT, channels=mic_ch, rate=mic_rate,
                input=True, input_device_index=mic_idx,
                frames_per_buffer=CHUNK)

            sys_stream = p.open(
                format=FORMAT, channels=sys_ch, rate=sys_rate,
                input=True, input_device_index=sys_idx,
                frames_per_buffer=CHUNK)

            while not self._stop_event.is_set():
                mic_data = mic_stream.read(CHUNK, exception_on_overflow=False)
                sys_data = sys_stream.read(CHUNK, exception_on_overflow=False)
                mic_frames.append(mic_data)
                sys_frames.append(sys_data)

                if self._on_level:
                    self._on_level(max(compute_rms(mic_data),
                                      compute_rms(sys_data)))

            mic_stream.stop_stream()
            mic_stream.close()
            sys_stream.stop_stream()
            sys_stream.close()
        except Exception as e:
            if self._on_done:
                self._on_done(None, 0, str(e))
            return
        finally:
            p.terminate()

        # Beide Streams zu Mono 16kHz mischen
        mic_audio = np.frombuffer(b"".join(mic_frames), dtype=np.int16
                                  ).astype(np.float64)
        sys_audio = np.frombuffer(b"".join(sys_frames), dtype=np.int16
                                  ).astype(np.float64)

        # System-Audio: Multi-Channel → Mono
        if sys_ch > 1:
            sys_audio = sys_audio.reshape(-1, sys_ch).mean(axis=1)

        # System-Audio: Resample auf out_rate
        if sys_rate != out_rate:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(out_rate, sys_rate)
            sys_audio = resample_poly(sys_audio, out_rate // g, sys_rate // g)

        # Auf gleiche Laenge bringen
        min_len = min(len(mic_audio), len(sys_audio))
        mic_audio = mic_audio[:min_len]
        sys_audio = sys_audio[:min_len]

        # Mischen (beide gleich gewichtet) und clippen
        mixed = mic_audio + sys_audio
        mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
        self._channels = 1
        self._sample_rate = out_rate
        self._frames = [mixed.tobytes()]
        _save_wav(self._output_path, self._frames, 1, out_rate)

        duration = len(mixed) / out_rate
        if self._on_done:
            self._on_done(self._output_path, duration, None)


def list_devices():
    """Zeigt alle verfuegbaren Audio-Geraete (Mikrofone + WASAPI Loopback)."""
    p = pyaudio.PyAudio()
    print("\n=== Verfuegbare Audio-Geraete ===\n")

    print("--- Mikrofone (Eingabegeraete) ---")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and "loopback" not in info["name"].lower():
            print(f"  [{i}] {info['name']}")
            print(f"      Kanaele: {info['maxInputChannels']}, "
                  f"Samplerate: {int(info['defaultSampleRate'])} Hz")

    print("\n--- System-Audio (WASAPI Loopback) ---")
    try:
        for device in p.get_loopback_device_info_generator():
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Kanaele: {device['maxInputChannels']}, "
                  f"Samplerate: {int(device['defaultSampleRate'])} Hz")
    except OSError:
        print("  Keine WASAPI Loopback-Geraete gefunden.")

    try:
        default_lb = p.get_default_wasapi_loopback()
        print(f"\n  Standard-Loopback: [{default_lb['index']}] {default_lb['name']}")
    except OSError:
        pass

    print()
    p.terminate()


def _level_bar(data: bytes, channels: int) -> str:
    """Erzeugt einen einfachen Lautstaerke-Balken fuer die Konsole."""
    samples = np.frombuffer(data, dtype=np.int16)
    if len(samples) == 0:
        return ""
    # RMS berechnen
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    # Normalisieren auf 0-50 Zeichen
    level = min(int(rms / 500), 50)
    bar = "█" * level + "░" * (50 - level)
    return f"\r  Pegel: |{bar}| {rms:6.0f}"


def record_microphone(output_path: str, device_index: int | None = None,
                      sample_rate: int = 16000, channels: int = 1):
    """Nimmt Audio vom Mikrofon auf und speichert es als WAV.

    Stoppt bei Enter-Tastendruck.
    """
    p = pyaudio.PyAudio()

    if device_index is not None:
        info = p.get_device_info_by_index(device_index)
        print(f"Mikrofon: [{device_index}] {info['name']}")
    else:
        info = p.get_default_input_device_info()
        device_index = info["index"]
        print(f"Standard-Mikrofon: [{device_index}] {info['name']}")

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

    print("Aufnahme laeuft... Druecke ENTER zum Stoppen.\n")

    try:
        while not stop_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            print(_level_bar(data, channels), end="", flush=True)
    except KeyboardInterrupt:
        pass

    print("\n\nAufnahme beendet.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    _save_wav(output_path, frames, channels, sample_rate)
    return output_path


def record_system_audio(output_path: str):
    """Nimmt System-Audio (WASAPI Loopback) auf und speichert es als WAV.

    Ideal fuer Teams/Zoom-Aufnahmen.
    Stoppt bei Enter-Tastendruck.
    """
    p = pyaudio.PyAudio()

    try:
        loopback = p.get_default_wasapi_loopback()
    except OSError as e:
        p.terminate()
        raise RuntimeError(
            "Kein WASAPI Loopback-Geraet gefunden. "
            "Stelle sicher, dass ein Audio-Ausgabegeraet aktiv ist."
        ) from e

    channels = loopback["maxInputChannels"]
    sample_rate = int(loopback["defaultSampleRate"])

    print(f"System-Audio (Loopback): [{loopback['index']}] {loopback['name']}")
    print(f"  Kanaele: {channels}, Samplerate: {sample_rate} Hz")

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

    print("Aufnahme laeuft... Druecke ENTER zum Stoppen.\n")

    try:
        while not stop_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            print(_level_bar(data, channels), end="", flush=True)
    except KeyboardInterrupt:
        pass

    print("\n\nAufnahme beendet.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    _save_wav(output_path, frames, channels, sample_rate)
    return output_path


def _save_wav(output_path: str, frames: list[bytes], channels: int,
              sample_rate: int):
    """Speichert aufgenommene Frames als WAV-Datei."""
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    duration = len(b"".join(frames)) / (sample_rate * channels * SAMPLE_WIDTH)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Gespeichert: {output_path} ({duration:.1f}s, {size_mb:.1f} MB)")
