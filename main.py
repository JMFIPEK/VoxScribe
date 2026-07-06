"""VoxScribe — CLI Entry Point für ORION.

Befehle:
    python main.py devices                      Audio-Geraete auflisten
    python main.py record --source mic           Mikrofon-Aufnahme
    python main.py record --source system        System-Audio (Teams/Zoom)
    python main.py record --source both          Mikrofon + System-Audio
    python main.py transcribe --input audio.wav  Transkription starten
    python main.py run --source mic              Aufnahme + Transkription
"""

import os
import ssl
import warnings

# --- Warnungen unterdruecken ---
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_LOGS"] = "-all"
os.environ["TORCHAUDIO_NO_BACKEND_CHECK"] = "1"

# --- Firmen-Proxy: SSL-Verifikation deaktivieren (vor allen anderen Imports) ---
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import logging
logging.disable(logging.WARNING)
logging.getLogger().setLevel(logging.ERROR)

import requests
_original_send = requests.adapters.HTTPAdapter.send
def _patched_send(self, request, *args, **kwargs):
    kwargs["verify"] = False
    return _original_send(self, request, *args, **kwargs)
requests.adapters.HTTPAdapter.send = _patched_send

import argparse
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def cmd_devices(_args):
    """Zeigt alle verfuegbaren Audio-Geraete."""
    from recorder import list_devices
    list_devices()


def cmd_record(args):
    """Startet eine Audio-Aufnahme."""
    from recorder import (
        record_microphone,
        record_microphone_and_system,
        record_system_audio,
    )

    output = args.output
    if not output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("recordings", exist_ok=True)
        output = f"recordings/{timestamp}.wav"

    if args.source == "mic":
        record_microphone(output, device_index=args.device)
    elif args.source == "system":
        record_system_audio(output, device_index=args.device)
    elif args.source == "both":
        record_microphone_and_system(output, device_index=args.device)
    else:
        print(f"Unbekannte Quelle: {args.source}")
        sys.exit(1)

    return output


def _cli_progress(state):
    """Erstellt einen on_progress-Callback, der eine einzeilige Fortschrittsanzeige
    ausgibt (per \\r ueberschrieben), damit lange Schritte (v.a. Diarization) nicht
    wie ein Haenger aussehen."""
    def _callback(pct, message):
        # Nur alle ~0.5% aktualisieren, um das Terminal nicht zu fluten
        shown = int(pct * 200)
        if shown == state.get("last") and pct < 1.0:
            return
        state["last"] = shown
        bar_width = 30
        filled = int(bar_width * pct)
        bar = "#" * filled + "-" * (bar_width - filled)
        end = "\n" if pct >= 1.0 else ""
        print(f"\r     [{bar}] {pct * 100:5.1f}%  {message}", end=end, flush=True)
    return _callback


def cmd_transcribe(args):
    """Transkribiert eine Audio-Datei."""
    from transcriber import DEFAULT_API_BASE_URL, transcribe, save_transcript

    if not os.path.isfile(args.input):
        print(f"Datei nicht gefunden: {args.input}")
        sys.exit(1)

    hf_token = args.hf_token or os.getenv("HF_TOKEN") or None
    api_key = args.api_key or os.getenv("KIT_TOOLBOX_API_KEY") or None
    api_base_url = args.api_base_url or os.getenv("KIT_TOOLBOX_BASE_URL") or DEFAULT_API_BASE_URL
    language = None if args.language.lower() in ("auto", "automatisch") else args.language

    result = transcribe(
        audio_path=args.input,
        language=language,
        model_size=args.model,
        diarize=args.diarize,
        hf_token=hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        batch_size=args.batch_size,
        device=args.device_compute,
        api_key=api_key,
        api_base_url=api_base_url,
        on_progress=_cli_progress({}),
    )

    # Output-Pfad bestimmen
    output = args.output
    if not output:
        base = os.path.splitext(args.input)[0]
        output = base

    formats = [f.strip() for f in args.format.split(",")]
    print()
    save_transcript(result, output, formats=formats)

    return result


def cmd_run(args):
    """Aufnahme + sofortige Transkription."""
    # Erst aufnehmen
    audio_path = cmd_record(args)
    print("\n" + "=" * 60 + "\n")

    # Dann transkribieren
    args.input = audio_path
    if not args.output:
        args.output = os.path.splitext(audio_path)[0]
    cmd_transcribe(args)


def main():
    parser = argparse.ArgumentParser(
        description="VoxScribe — 100%% lokal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Verfuegbare Befehle")

    # --- devices ---
    sub_devices = subparsers.add_parser("devices", help="Audio-Geraete auflisten")
    sub_devices.set_defaults(func=cmd_devices)

    # --- record ---
    sub_record = subparsers.add_parser("record", help="Audio aufnehmen")
    sub_record.add_argument(
        "--source", choices=["mic", "system", "both"], default="mic",
        help=("Audio-Quelle: mic, system (WASAPI Loopback) oder "
              "both (Mikrofon + System)")
    )
    sub_record.add_argument("--output", "-o", help="Ausgabe-Pfad (WAV)")
    sub_record.add_argument(
        "--device", type=int, default=None,
        help="Geraete-Index (siehe 'devices' Befehl)"
    )
    sub_record.set_defaults(func=cmd_record)

    # --- transcribe ---
    sub_transcribe = subparsers.add_parser("transcribe", help="Audio transkribieren")
    sub_transcribe.add_argument("--input", "-i", required=True, help="Audio-Datei")
    sub_transcribe.add_argument(
        "--language", "-l", default="de",
        help="Sprache (z.B. de, en) oder 'auto' fuer automatische Erkennung. Default: de"
    )
    sub_transcribe.add_argument(
        "--model", "-m", default="large-v2",
        help=("Whisper-Modell (large-v2, large-v3, medium, base) oder "
              "Server-Modell (z.B. server:kit.whisper-large-v3). Default: large-v2")
    )
    sub_transcribe.add_argument(
        "--diarize", action="store_true", default=True,
        help="Speaker Diarization aktivieren (Default: an)"
    )
    sub_transcribe.add_argument(
        "--no-diarize", dest="diarize", action="store_false",
        help="Speaker Diarization deaktivieren"
    )
    sub_transcribe.add_argument(
        "--hf-token", default=None,
        help="HuggingFace Token (alternativ: HF_TOKEN in .env)"
    )
    sub_transcribe.add_argument(
        "--api-key", default=None,
        help="API-Key fuer Server-Modelle (alternativ: KIT_TOOLBOX_API_KEY in .env)"
    )
    sub_transcribe.add_argument(
        "--api-base-url", default=None,
        help="Basis-URL fuer Server-Modelle (alternativ: KIT_TOOLBOX_BASE_URL in .env)"
    )
    sub_transcribe.add_argument("--min-speakers", type=int, default=None)
    sub_transcribe.add_argument("--max-speakers", type=int, default=None)
    sub_transcribe.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch-Groesse (kleiner = weniger VRAM). Default: 16"
    )
    sub_transcribe.add_argument(
        "--device-compute", choices=["cuda", "mps", "cpu"], default=None,
        help="Compute Device (auto-detect wenn nicht gesetzt). 'mps' beschleunigt "
             "auf Apple Silicon nur Alignment/Diarization, nicht die Whisper-Transkription."
    )
    sub_transcribe.add_argument("--output", "-o", help="Ausgabe-Pfad (ohne Endung)")
    sub_transcribe.add_argument(
        "--format", "-f", default="txt",
        help="Ausgabe-Format(e), kommagetrennt: txt, srt, json. Default: txt"
    )
    sub_transcribe.set_defaults(func=cmd_transcribe)

    # --- run ---
    sub_run = subparsers.add_parser(
        "run", help="Aufnahme + sofortige Transkription"
    )
    sub_run.add_argument(
        "--source", choices=["mic", "system", "both"], default="mic",
        help="Audio-Quelle: mic, system oder both"
    )
    sub_run.add_argument("--output", "-o", default=None)
    sub_run.add_argument("--device", type=int, default=None)
    sub_run.add_argument("--language", "-l", default="de")
    sub_run.add_argument("--model", "-m", default="large-v2")
    sub_run.add_argument("--diarize", action="store_true", default=True)
    sub_run.add_argument("--no-diarize", dest="diarize", action="store_false")
    sub_run.add_argument("--hf-token", default=None)
    sub_run.add_argument("--api-key", default=None)
    sub_run.add_argument("--api-base-url", default=None)
    sub_run.add_argument("--min-speakers", type=int, default=None)
    sub_run.add_argument("--max-speakers", type=int, default=None)
    sub_run.add_argument("--batch-size", type=int, default=16)
    sub_run.add_argument(
        "--device-compute", choices=["cuda", "mps", "cpu"], default=None
    )
    sub_run.add_argument("--format", "-f", default="txt")
    sub_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
