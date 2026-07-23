"""WhisperX-basierte Transkription mit Alignment und Speaker Diarization."""

import gc
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import LANGUAGES as _WHISPER_LANGUAGE_NAMES

# Muss vor der ersten MPS-Operation gesetzt sein: pyannote/wav2vec2 nutzen
# vereinzelt Ops, die (noch) nicht fuer Apple's MPS-Backend implementiert sind -
# ohne Fallback wuerde das mit einem NotImplementedError abstuerzen, statt
# transparent auf die CPU auszuweichen.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SAMPLE_RATE = 16000

# Volle Sprachnamen (z.B. "german", wie von OpenAI-kompatiblen APIs zurueckgegeben)
# auf ISO-639-1-Codes (z.B. "de") abbilden, wie sie whisperx fuer Alignment erwartet.
_LANGUAGE_NAME_TO_CODE = {name.lower(): code for code, name in _WHISPER_LANGUAGE_NAMES.items()}


def _normalize_language_code(raw: str | None, fallback: str | None = None) -> str | None:
    """Normalisiert einen von einer Server-API zurueckgegebenen Sprachbezeichner (Name oder Code)."""
    if not raw:
        return fallback
    raw = raw.strip().lower()
    if raw in _WHISPER_LANGUAGE_NAMES:
        return raw
    return _LANGUAGE_NAME_TO_CODE.get(raw, fallback)


def _get_base_dir() -> str:
    """Gibt das Basisverzeichnis der Anwendung zurueck (PyInstaller-kompatibel)."""
    if getattr(sys, "frozen", False):
        # PyInstaller --onedir: exe liegt in dist/WhisperX/
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _get_bundled_models_dir() -> str | None:
    """Gibt den Pfad zum bundled_models Ordner zurueck, falls vorhanden."""
    base = _get_base_dir()
    models_dir = os.path.join(base, "bundled_models")
    if os.path.isdir(models_dir):
        return models_dir
    return None


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


def _load_audio_via_container(audio_path: str) -> np.ndarray:
    """Extrahiert die Audiospur aus einem beliebigen Container (MKV, MP4, MOV, ...) via PyAV.

    Wird als Fallback genutzt, wenn soundfile das Format nicht direkt lesen kann
    (z.B. Video-Container wie MKV, bei denen nur die Audiospur benoetigt wird).
    """
    import av

    container = av.open(audio_path)
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        container.close()
        raise ValueError(f"Keine Audiospur gefunden in: {audio_path}")

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray())
    for resampled in resampler.resample(None):
        chunks.append(resampled.to_ndarray())
    container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    data = np.concatenate(chunks, axis=1).reshape(-1)
    return (data.astype(np.float32) / 32768.0)


def load_audio_universal(audio_path: str) -> np.ndarray:
    """Laedt eine Audio- oder Video-Datei als 16kHz Mono float32 Array.

    Versucht zuerst soundfile (schnell, fuer WAV/FLAC/OGG etc.). Schlaegt das
    fehl (z.B. bei Video-Containern wie MKV/MP4 oder komprimierten Formaten,
    die libsndfile nicht kennt), wird per PyAV nur die Audiospur dekodiert.
    """
    try:
        return load_audio_without_ffmpeg(audio_path)
    except Exception:
        return _load_audio_via_container(audio_path)


DEFAULT_API_BASE_URL = "https://ki-toolbox.scc.kit.edu/api/v1"
REMOTE_MODEL_PREFIX = "server:"


def is_remote_model(model_size: str) -> bool:
    """Prueft, ob es sich um ein serverseitig gehostetes Modell handelt."""
    return model_size.startswith(REMOTE_MODEL_PREFIX)


class _PayloadTooLarge(Exception):
    """Interner Marker: Server hat den Upload mit 413 abgelehnt."""


REMOTE_CHUNK_SECONDS = 180.0
REMOTE_MIN_CHUNK_SECONDS = 5.0


def _post_audio_chunk(chunk: np.ndarray, language: str | None, model_name: str,
                       api_key: str, base_url: str) -> tuple[list[dict], str | None]:
    """Schickt einen einzelnen Audio-Chunk an den Server.

    Gibt (segmente, roher_sprachname_aus_der_antwort) zurueck.
    """
    import io
    import requests

    # FLAC statt WAV: verlustfrei, aber deutlich kleinere Uploads (~50-60%),
    # damit auch lange Aufnahmen nicht an Server-Upload-Limits scheitern.
    buf = io.BytesIO()
    sf.write(buf, chunk, SAMPLE_RATE, format="FLAC")
    buf.seek(0)

    url = base_url.rstrip("/") + "/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model_name, "response_format": "verbose_json"}
    if language:
        data["language"] = language
    files = {"file": ("audio.flac", buf, "audio/flac")}

    resp = requests.post(url, headers=headers, data=data, files=files, timeout=1800)
    if resp.status_code == 413:
        raise _PayloadTooLarge()
    resp.raise_for_status()
    payload = resp.json()

    segments = []
    for seg in payload.get("segments", []):
        segments.append({
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", ""),
        })
    if not segments and payload.get("text"):
        duration = len(chunk) / SAMPLE_RATE
        segments = [{"start": 0.0, "end": duration, "text": payload["text"]}]
    return segments, payload.get("language")


def _transcribe_chunk_with_backoff(chunk: np.ndarray, language: str | None, model_name: str,
                                    api_key: str, base_url: str) -> tuple[list[dict], str | None]:
    """Sendet einen Chunk; wird er mit 413 abgelehnt, wird er rekursiv halbiert."""
    try:
        return _post_audio_chunk(chunk, language, model_name, api_key, base_url)
    except _PayloadTooLarge:
        duration = len(chunk) / SAMPLE_RATE
        if duration <= REMOTE_MIN_CHUNK_SECONDS:
            raise ValueError(
                "Server lehnt Uploads mit 413 (Request Entity Too Large) ab, "
                "selbst fuer sehr kurze Audio-Chunks. Server-Konfiguration pruefen."
            )
        mid = len(chunk) // 2
        left, lang_left = _transcribe_chunk_with_backoff(chunk[:mid], language, model_name, api_key, base_url)
        right, lang_right = _transcribe_chunk_with_backoff(chunk[mid:], language, model_name, api_key, base_url)
        offset = mid / SAMPLE_RATE
        for seg in right:
            seg["start"] += offset
            seg["end"] += offset
        return left + right, (lang_left or lang_right)


def transcribe_remote(
    audio: np.ndarray,
    language: str | None,
    model_name: str,
    api_key: str,
    base_url: str = DEFAULT_API_BASE_URL,
    chunk_seconds: float = REMOTE_CHUNK_SECONDS,
    on_progress=None,
) -> dict:
    """Schickt Audio in Chunks an einen OpenAI-kompatiblen Transkriptions-Endpoint (z.B. KIT ToolBox).

    Lange Aufnahmen werden in Stuecke von `chunk_seconds` zerlegt (und bei einem
    413-Fehler des Servers automatisch weiter halbiert), da Server-Endpoints
    typischerweise ein Upload-Groessenlimit haben.

    Ist `language` None/leer, wird keine Sprache mitgeschickt (Server erkennt sie
    selbst). Die vom ersten Chunk erkannte Sprache wird anschliessend fuer alle
    weiteren Chunks fest verwendet, damit die Spracherkennung nicht pro Chunk
    hin- und herspringt (kurze/stille Chunks erkennen sonst leicht die falsche
    Sprache).

    Gibt ein Dict im WhisperX-Format zurueck ({"segments": [...], "language": ...}),
    das anschliessend fuer Alignment und Diarization weiterverwendet werden kann.
    """
    if not api_key:
        raise ValueError(
            "Kein API-Key fuer das Server-Modell gesetzt "
            "(KIT_TOOLBOX_API_KEY in .env oder in den Einstellungen eintragen)."
        )

    total_samples = len(audio)
    chunk_samples = max(1, int(chunk_seconds * SAMPLE_RATE))
    n_chunks = max(1, -(-total_samples // chunk_samples))  # ceil division

    all_segments = []
    detected_language_code = None
    offset = 0
    chunk_idx = 0
    request_language = language
    while offset < total_samples:
        chunk = audio[offset: offset + chunk_samples]
        chunk_start_time = offset / SAMPLE_RATE

        chunk_segments, raw_language = _transcribe_chunk_with_backoff(
            chunk, request_language, model_name, api_key, base_url)
        for seg in chunk_segments:
            seg["start"] += chunk_start_time
            seg["end"] += chunk_start_time
            all_segments.append(seg)

        if detected_language_code is None:
            detected_language_code = _normalize_language_code(raw_language, fallback=language)
            if not request_language and detected_language_code:
                # Sprache fuer die restlichen Chunks fixieren (siehe Docstring)
                request_language = detected_language_code

        chunk_idx += 1
        if on_progress:
            on_progress(min(chunk_idx / n_chunks, 1.0))
        offset += chunk_samples

    return {"segments": all_segments, "language": detected_language_code or language or "de"}


APPLE_MODEL_PREFIX = "apple:"
APPLE_SPEECHANALYZER_MODEL = "apple:speechanalyzer"


def is_apple_model(model_size: str) -> bool:
    """Prueft, ob die Apple SpeechAnalyzer-Engine (nur macOS) genutzt werden soll."""
    return model_size.startswith(APPLE_MODEL_PREFIX)


# ISO-639-1 -> volle BCP-47-Locale, wie sie SpeechTranscriber erwartet. Anders
# als WhisperX/das Server-Modell kennt SpeechAnalyzer (Stand macOS 26) keine
# automatische Spracherkennung - die Sprache muss vorher feststehen, "de" ist
# der Default, wenn keine/automatische Sprache gewaehlt wurde.
_APPLE_LOCALE_BY_LANGUAGE = {
    "de": "de-DE", "en": "en-US", "fr": "fr-FR", "es": "es-ES",
    "it": "it-IT", "pt": "pt-PT", "ja": "ja-JP", "ko": "ko-KR", "zh": "zh-CN",
}


def _apple_locale_identifier(language: str | None) -> str:
    return _APPLE_LOCALE_BY_LANGUAGE.get((language or "de").lower(), "de-DE")


def _apple_speechanalyzer_binary_path() -> str:
    """Pfad zum kompilierten SpeechAnalyzer-Helfer (siehe macos/README.md)."""
    return os.path.join(_get_base_dir(), "macos", "SpeechAnalyzerTranscribe")


def default_model_size() -> str:
    """Plattform-Default fuer model_size, wenn der Aufrufer keinen expliziten Wunsch hat.

    Auf macOS ist Apples SpeechAnalyzer (Neural Engine) die einzige lokale
    Transkriptions-Engine - kein Whisper-Modell-Download/-Inferenz mehr, siehe
    CLAUDE.md ("Apple Silicon (MPS) acceleration is partial by design"). Auf
    Windows/Linux bleibt WhisperX (lokal) unveraendert der Default.
    """
    return APPLE_SPEECHANALYZER_MODEL if sys.platform == "darwin" else "large-v2"


def transcribe_apple(
    audio_path: str,
    language: str | None = "de",
    on_progress=None,
) -> dict:
    """Transkribiert via Apples SpeechAnalyzer/SpeechTranscriber (macOS 26+, Neural Engine).

    Ruft den kompilierten Swift-Helfer macos/SpeechAnalyzerTranscribe als
    einmaligen Subprozess auf (kein Dauer-Stream wie bei der Systemaudio-
    Aufnahme, daher auch kein Watcher-Thread noetig - der Prozess endet von
    selbst) und liest dessen NDJSON-Ausgabe zeilenweise fuer Fortschritt.

    SpeechTranscriber liefert pro Wort bereits einen Zeitstempel
    (attributeOptions: [.audioTimeRange] im Swift-Helfer) - an echter Hardware
    mit deutschem Testaudio verifiziert als wortgenau. Das Rueckgabe-Dict ist
    deshalb bereits im "ausgerichteten" Format, das WhisperX' wav2vec2-
    Alignment sonst produziert ({"segments": [...{"words": [...]}...],
    "word_segments": [...]}) - der Alignment-Schritt entfaellt fuer diesen Pfad
    komplett (siehe Verzweigung in transcribe()).
    """
    binary = _apple_speechanalyzer_binary_path()
    if not os.path.isfile(binary):
        raise RuntimeError(
            f"SpeechAnalyzer-Helfer nicht gefunden ({binary}). "
            "Erst 'cd macos && ./build.sh' ausfuehren."
        )

    locale_id = _apple_locale_identifier(language)
    try:
        duration = sf.info(audio_path).duration or 1.0
    except Exception:
        duration = 1.0

    proc = subprocess.Popen(
        [binary, audio_path, locale_id],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )

    segments = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "segment":
            continue
        words = [
            {"word": w["word"], "start": w["start"], "end": w["end"], "score": w["score"]}
            for w in obj.get("words", [])
        ]
        segments.append({
            "start": obj["start"], "end": obj["end"], "text": obj["text"], "words": words,
        })
        if on_progress:
            on_progress(min(obj["end"] / duration, 1.0))

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().strip() if proc.stderr else ""
        raise RuntimeError(f"SpeechAnalyzer-Transkription fehlgeschlagen: {stderr}")

    word_segments = [w for seg in segments for w in seg["words"]]
    return {
        "segments": segments,
        "word_segments": word_segments,
        "language": (language or "de").lower(),
    }


def transcribe(
    audio_path: str,
    language: str | None = "de",
    model_size: str = "large-v2",
    diarize: bool = True,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    batch_size: int = 16,
    device: str | None = None,
    on_progress=None,
    api_key: str | None = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> dict:
    """Fuehrt die vollstaendige WhisperX-Pipeline aus.

    1. Transkription (batched inference)
    2. Forced Alignment (word-level timestamps)
    3. Speaker Diarization (optional)

    Args:
        audio_path: Pfad zur Audio-Datei (WAV, MP3, etc.)
        language: Sprache des Audios (z.B. "de", "en"). None = automatische Erkennung
            (nur bei lokalen/Server-Modellen - Apples SpeechAnalyzer kennt keine
            automatische Spracherkennung und faellt in dem Fall auf "de" zurueck)
        model_size: Whisper-Modellgroesse ("large-v2", "large-v3", "medium", "base"),
            ein Server-Modell (Praefix "server:", siehe is_remote_model()) oder
            "apple:speechanalyzer" fuer Apples SpeechAnalyzer (nur macOS 26+,
            siehe is_apple_model()/transcribe_apple())
        diarize: Speaker Diarization aktivieren
        hf_token: HuggingFace Token fuer pyannote (nur beim ersten Download noetig)
        min_speakers: Minimale Anzahl Sprecher (optional)
        max_speakers: Maximale Anzahl Sprecher (optional)
        batch_size: Batch-Groesse fuer Inference (kleiner = weniger VRAM)
        device: "cuda", "mps" oder "cpu" (auto-detect wenn None). Wirkt nur auf
            Alignment/Diarization (plain PyTorch) - die Whisper-Transkription
            selbst laeuft ueber CTranslate2, das kein MPS unterstuetzt und
            deshalb bei "mps" automatisch auf "cpu" zurueckfaellt.

    Returns:
        Dict mit "segments", "language" und ggf. Speaker-Labels
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # CTranslate2 (WhisperX' Inference-Backend) kennt nur "cuda"/"cpu" - kein
    # MPS. Alignment (wav2vec2) und Diarization (pyannote) sind plain PyTorch
    # und koennen MPS dagegen nutzen, daher zwei getrennte Device-Variablen.
    whisper_device = device if device == "cuda" else "cpu"
    torch_device = device

    compute_type = "float16" if whisper_device == "cuda" else "int8"
    remote = is_remote_model(model_size)
    apple = is_apple_model(model_size)

    if apple:
        print(f"Device: Transkription=Apple Neural Engine (SpeechAnalyzer), Diarization={torch_device}")
    elif device == "mps":
        print(f"Device: Whisper={whisper_device} ({compute_type}), Alignment/Diarization={torch_device} (Apple MPS)")
    else:
        print(f"Device: {device} ({compute_type})")
    print(f"Modell: {model_size}")
    print(f"Sprache: {language or 'automatisch erkennen'}")
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
    t0 = time.time()

    # Bundled model path verwenden falls vorhanden
    bundled = _get_bundled_models_dir()

    if apple:
        print("1/3  Transkription via Apple SpeechAnalyzer (Neural Engine)...")
        _prog(0.0, "SpeechAnalyzer transkribiert...")

        def _apple_progress(frac):
            mapped = frac * 0.40
            _prog(mapped, f"Transkription... {frac * 100:.0f}%")

        result = transcribe_apple(audio_path, language, on_progress=_apple_progress)
        # Fuer die Diarization unten wird das Audio als Array benoetigt (der
        # Swift-Helfer laedt/dekodiert die Datei intern selbst nochmal).
        audio = load_audio_universal(audio_path)

        t1 = time.time()
        n_segs = len(result["segments"])
        print(f"     Transkription abgeschlossen ({t1 - t0:.1f}s)")
        print(f"     {n_segs} Segmente erkannt")
        _prog(0.40, f"Transkription fertig - {n_segs} Segmente ({t1 - t0:.0f}s)")
    elif remote:
        remote_model_name = model_size[len(REMOTE_MODEL_PREFIX):]
        print(f"1/3  Transkription via Server-Modell ({remote_model_name})...")
        _prog(0.0, "Audio laden...")
        audio = load_audio_universal(audio_path)

        def _remote_progress(frac):
            mapped = 0.10 + frac * 0.30
            _prog(mapped, f"Sende Audio an Server... {frac * 100:.0f}%")

        _prog(0.10, f"Sende Audio an Server ({remote_model_name})...")
        result = transcribe_remote(
            audio, language, remote_model_name,
            api_key=api_key, base_url=api_base_url,
            on_progress=_remote_progress,
        )

        t1 = time.time()
        n_segs = len(result["segments"])
        print(f"     Transkription abgeschlossen ({t1 - t0:.1f}s)")
        print(f"     {n_segs} Segmente erkannt")
        _prog(0.40, f"Transkription fertig - {n_segs} Segmente ({t1 - t0:.0f}s)")
    else:
        print("1/3  Transkription laeuft...")
        _prog(0.0, "Whisper-Modell laden...")

        whisper_download_root = None
        whisper_local_only = False
        if bundled:
            whisper_path = os.path.join(bundled, "whisper", model_size)
            if os.path.isdir(whisper_path):
                whisper_download_root = whisper_path
                whisper_local_only = True

        model = whisperx.load_model(
            model_size, whisper_device, compute_type=compute_type, language=language,
            download_root=whisper_download_root, local_files_only=whisper_local_only,
        )

        _prog(0.10, "Audio laden...")
        audio = load_audio_universal(audio_path)

        _prog(0.15, "Transkription laeuft...")
        result = model.transcribe(audio, batch_size=batch_size,
                                  progress_callback=_transcribe_progress)

        t1 = time.time()
        n_segs = len(result['segments'])
        print(f"     Transkription abgeschlossen ({t1 - t0:.1f}s)")
        print(f"     {n_segs} Segmente erkannt")
        _prog(0.40, f"Transkription fertig - {n_segs} Segmente ({t1 - t0:.0f}s)")

        # Modell entladen
        del model
        gc.collect()
        if whisper_device == "cuda":
            torch.cuda.empty_cache()

    # --- 2. Alignment ---
    # SpeechAnalyzer liefert bereits wortgenaue Zeitstempel (siehe
    # transcribe_apple()) - der wav2vec2-Alignment-Schritt ist fuer dieses
    # Ergebnis ueberfluessig und wuerde auf macOS zudem ein weiteres
    # PyTorch-Modell laden, das gar nicht gebraucht wird.
    if apple:
        print("2/3  Alignment uebersprungen (SpeechAnalyzer liefert bereits Wort-Zeitstempel)")
        _prog(0.65, "Alignment nicht noetig (Apple SpeechAnalyzer)")
    else:
        print("2/3  Alignment laeuft...")
        _prog(0.45, "Alignment-Modell laden...")
        t2 = time.time()

        # Bundled alignment model path
        align_model_dir = None
        align_cache_only = False
        if bundled:
            align_path = os.path.join(bundled, "align")
            if os.path.isdir(align_path):
                align_model_dir = align_path
                align_cache_only = True

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=torch_device,
            model_dir=align_model_dir, model_cache_only=align_cache_only,
        )

        _prog(0.50, "Wort-Alignment laeuft...")
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, torch_device,
            return_char_alignments=False,
            progress_callback=_align_progress,
        )

        t3 = time.time()
        print(f"     Alignment abgeschlossen ({t3 - t2:.1f}s)")
        _prog(0.65, f"Alignment fertig ({t3 - t2:.0f}s)")

        # Modell entladen
        del model_a
        gc.collect()
        if torch_device == "cuda":
            torch.cuda.empty_cache()
        elif torch_device == "mps":
            torch.mps.empty_cache()

    # --- 3. Speaker Diarization ---
    if diarize:
        if not hf_token:
            print("3/3  Diarization uebersprungen (kein HF_TOKEN gesetzt)")
            print("     Setze HF_TOKEN in .env fuer Speaker-Erkennung")
            _prog(1.0, "Fertig (Diarization uebersprungen - kein Token)")
        else:
            print("3/3  Speaker Diarization laeuft...")
            _prog(0.70, "Diarization-Modell laden...")
            t4 = time.time()

            # Bundled diarization model path
            diarize_cache = None
            if bundled:
                diarize_path = os.path.join(bundled, "diarize")
                if os.path.isdir(diarize_path):
                    diarize_cache = diarize_path

            diarize_model = DiarizationPipeline(
                token=hf_token, device=torch_device, cache_dir=diarize_cache
            )

            _prog(0.80, "Speaker Diarization laeuft...")
            diarize_segments, speaker_embeddings = diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True,
            )

            _prog(0.92, "Sprecher zuordnen...")
            result = whisperx.assign_word_speakers(
                diarize_segments, result, speaker_embeddings=speaker_embeddings)

            # Bekannte Sprecher (Voice-Prints) automatisch erkennen und
            # Segmente/Woerter direkt mit dem erkannten Namen beschriften.
            # speaker_id_map merkt sich die Zuordnung roher Diarization-ID ->
            # aktuell angezeigtes Label (Name oder unveraendert), damit die GUI
            # spaeter das passende Embedding zum "Sprecher merken" findet.
            speaker_id_map = {spk: spk for spk in (speaker_embeddings or {})}
            if speaker_embeddings:
                try:
                    from speaker_profiles import match_speakers
                    suggestions = match_speakers(speaker_embeddings)
                except Exception:
                    suggestions = {}
                for spk_id, name in suggestions.items():
                    speaker_id_map[spk_id] = name
                    for seg in result.get("segments", []):
                        if seg.get("speaker") == spk_id:
                            seg["speaker"] = name
                    for word in result.get("word_segments", []) or []:
                        if word.get("speaker") == spk_id:
                            word["speaker"] = name
            result["speaker_id_map"] = speaker_id_map

            t5 = time.time()
            print(f"     Diarization abgeschlossen ({t5 - t4:.1f}s)")

            # Sprecher zaehlen
            speakers = set()
            for seg in result.get("segments", []):
                if "speaker" in seg:
                    speakers.add(seg["speaker"])
            print(f"     {len(speakers)} Sprecher erkannt: {', '.join(sorted(speakers))}")
            _prog(0.98, f"Diarization fertig - {len(speakers)} Sprecher ({t5 - t4:.0f}s)")

            del diarize_model
            gc.collect()
            if torch_device == "cuda":
                torch.cuda.empty_cache()
            elif torch_device == "mps":
                torch.mps.empty_cache()
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
