"""WhisperX-basierte Transkription mit Alignment und Speaker Diarization."""

import gc
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

SAMPLE_RATE = 16000


def load_audio_without_ffmpeg(audio_path: str) -> np.ndarray:
    """Laedt Audio-Datei und konvertiert zu 16kHz Mono float32 (ohne ffmpeg)."""
    data, sr = sf.read(audio_path, dtype="float32")
    # Zu Mono konvertieren
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Auf 16kHz resamplen
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SAMPLE_RATE, sr)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return data


def transcribe(
    audio_path: str,
    language: str = "de",
    model_size: str = "large-v2",
    diarize: bool = True,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    batch_size: int = 16,
    device: str | None = None,
    on_progress=None,
) -> dict:
    """Fuehrt die vollstaendige WhisperX-Pipeline aus.

    1. Transkription (batched inference)
    2. Forced Alignment (word-level timestamps)
    3. Speaker Diarization (optional)

    Args:
        audio_path: Pfad zur Audio-Datei (WAV, MP3, etc.)
        language: Sprache des Audios (z.B. "de", "en")
        model_size: Whisper-Modellgroesse ("large-v2", "large-v3", "medium", "base")
        diarize: Speaker Diarization aktivieren
        hf_token: HuggingFace Token fuer pyannote (nur beim ersten Download noetig)
        min_speakers: Minimale Anzahl Sprecher (optional)
        max_speakers: Maximale Anzahl Sprecher (optional)
        batch_size: Batch-Groesse fuer Inference (kleiner = weniger VRAM)
        device: "cuda" oder "cpu" (auto-detect wenn None)

    Returns:
        Dict mit "segments", "language" und ggf. Speaker-Labels
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Device: {device} ({compute_type})")
    print(f"Modell: {model_size}")
    print(f"Sprache: {language}")
    print(f"Audio: {audio_path}")
    print()

    # Fortschritts-Bereiche: Transkription 0-40%, Alignment 40-65%, Diarization 65-100%
    def _prog(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    def _transcribe_progress(pct_within):
        """pct_within: 0-100 vom whisperx callback → mapped auf 0.15-0.40"""
        mapped = 0.15 + (pct_within / 100) * 0.25
        _prog(mapped, f"Transkription... {pct_within:.0f}%")

    def _align_progress(pct_within):
        """pct_within: 0-100 vom whisperx callback → mapped auf 0.50-0.63"""
        mapped = 0.50 + (pct_within / 100) * 0.13
        _prog(mapped, f"Alignment... {pct_within:.0f}%")

    # --- 1. Transkription ---
    print("1/3  Transkription laeuft...")
    _prog(0.0, "Whisper-Modell laden...")
    t0 = time.time()

    model = whisperx.load_model(
        model_size, device, compute_type=compute_type, language=language
    )

    _prog(0.10, "Audio laden...")
    audio = load_audio_without_ffmpeg(audio_path)

    _prog(0.15, "Transkription läuft...")
    result = model.transcribe(audio, batch_size=batch_size,
                              progress_callback=_transcribe_progress)

    t1 = time.time()
    n_segs = len(result['segments'])
    print(f"     Transkription abgeschlossen ({t1 - t0:.1f}s)")
    print(f"     {n_segs} Segmente erkannt")
    _prog(0.40, f"Transkription fertig — {n_segs} Segmente ({t1 - t0:.0f}s)")

    # Modell entladen
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- 2. Alignment ---
    print("2/3  Alignment laeuft...")
    _prog(0.45, "Alignment-Modell laden...")
    t2 = time.time()

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )

    _prog(0.50, "Wort-Alignment läuft...")
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False,
        progress_callback=_align_progress,
    )

    t3 = time.time()
    print(f"     Alignment abgeschlossen ({t3 - t2:.1f}s)")
    _prog(0.65, f"Alignment fertig ({t3 - t2:.0f}s)")

    # Modell entladen
    del model_a
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- 3. Speaker Diarization ---
    if diarize:
        if not hf_token:
            print("3/3  Diarization uebersprungen (kein HF_TOKEN gesetzt)")
            print("     Setze HF_TOKEN in .env fuer Speaker-Erkennung")
            _prog(1.0, "Fertig (Diarization übersprungen — kein Token)")
        else:
            print("3/3  Speaker Diarization laeuft...")
            _prog(0.70, "Diarization-Modell laden...")
            t4 = time.time()

            diarize_model = DiarizationPipeline(token=hf_token, device=device)

            _prog(0.80, "Speaker Diarization läuft...")
            diarize_segments = diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            _prog(0.92, "Sprecher zuordnen...")
            result = whisperx.assign_word_speakers(diarize_segments, result)

            t5 = time.time()
            print(f"     Diarization abgeschlossen ({t5 - t4:.1f}s)")

            # Sprecher zaehlen
            speakers = set()
            for seg in result.get("segments", []):
                if "speaker" in seg:
                    speakers.add(seg["speaker"])
            print(f"     {len(speakers)} Sprecher erkannt: {', '.join(sorted(speakers))}")
            _prog(0.98, f"Diarization fertig — {len(speakers)} Sprecher ({t5 - t4:.0f}s)")

            del diarize_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    else:
        print("3/3  Diarization deaktiviert")

    total = time.time() - t0
    _prog(1.0, f"Fertig! (Gesamt: {total:.0f}s)")
    print(f"\nGesamt: {total:.1f}s")

    return result


def format_transcript(result: dict, include_speakers: bool = True) -> str:
    """Formatiert das Transkript als lesbaren Text mit Zeitstempeln."""
    lines = []
    for seg in result.get("segments", []):
        start = _format_time(seg.get("start", 0))
        end = _format_time(seg.get("end", 0))
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")

        if include_speakers and speaker:
            lines.append(f"[{start} - {end}] {speaker}: {text}")
        else:
            lines.append(f"[{start} - {end}] {text}")

    return "\n".join(lines)


def save_transcript(result: dict, output_path: str,
                    formats: list[str] | None = None):
    """Speichert das Transkript in verschiedenen Formaten.

    Args:
        result: WhisperX-Ergebnis-Dict
        output_path: Basis-Pfad ohne Endung (z.B. "recordings/meeting")
        formats: Liste von Formaten ("txt", "srt", "json"). Default: ["txt"]
    """
    if formats is None:
        formats = ["txt"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    saved = []

    for fmt in formats:
        path = f"{output_path}.{fmt}"
        if fmt == "txt":
            content = format_transcript(result)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        elif fmt == "srt":
            content = _to_srt(result)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        else:
            print(f"  Unbekanntes Format: {fmt}")
            continue
        saved.append(path)
        print(f"  Gespeichert: {path}")

    return saved


def _format_time(seconds: float) -> str:
    """Formatiert Sekunden als MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_time_srt(seconds: float) -> str:
    """Formatiert Sekunden im SRT-Format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _to_srt(result: dict) -> str:
    """Konvertiert WhisperX-Ergebnis in SRT-Format."""
    lines = []
    for i, seg in enumerate(result.get("segments", []), 1):
        start = _format_time_srt(seg.get("start", 0))
        end = _format_time_srt(seg.get("end", 0))
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)
