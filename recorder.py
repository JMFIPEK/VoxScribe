"""Audio-Aufnahme von Mikrofon und System-Audio (WASAPI Loopback)."""

import wave
import threading
import numpy as np

import pyaudiowpatch as pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes


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
