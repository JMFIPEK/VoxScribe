"""VoxScribe — CLI command implementations.

The actual logic behind the `main.py` entry point in the project root (kept
thin there so `python main.py ...` still works exactly as documented). SSL/
proxy setup and warning suppression happen in the root main.py, before this
module (and the requests/torch imports it eventually pulls in) is imported.

Commands:
    python main.py devices                      List audio devices
    python main.py record --source mic           Microphone recording
    python main.py record --source system        System audio (Teams/Zoom)
    python main.py record --source both          Microphone + system audio
    python main.py transcribe --input audio.wav  Start transcription
    python main.py run --source mic              Record + transcribe
"""

import os
import argparse
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def cmd_devices(_args):
    """Shows all available audio devices."""
    from voxscribe.recorder import list_devices
    list_devices()


def cmd_record(args):
    """Starts an audio recording."""
    from voxscribe.recorder import (
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
        print(f"Unknown source: {args.source}")
        sys.exit(1)

    return output


def _cli_progress(state):
    """Creates an on_progress callback that prints a single-line progress
    indicator (overwritten via \\r), so long steps (diarization especially)
    don't look like a hang."""
    def _callback(pct, message):
        # Only update every ~0.5% to avoid flooding the terminal
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
    """Transcribes an audio file."""
    # Must happen before the first torch import, see voxscribe/gpu_setup.py.
    from voxscribe import gpu_setup
    gpu_setup.ensure_correct_torch_backend()

    from voxscribe.transcriber import default_model_size, transcribe, save_transcript

    if not os.path.isfile(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    hf_token = args.hf_token or os.getenv("HF_TOKEN") or None
    language = None if args.language.lower() == "auto" else args.language
    model = args.model or default_model_size()

    result = transcribe(
        audio_path=args.input,
        language=language,
        model_size=model,
        diarize=args.diarize,
        hf_token=hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        batch_size=args.batch_size,
        device=args.device_compute,
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        on_progress=_cli_progress({}),
    )

    # Determine output path
    output = args.output
    if not output:
        base = os.path.splitext(args.input)[0]
        output = base

    formats = [f.strip() for f in args.format.split(",")]
    print()
    save_transcript(result, output, formats=formats)

    return result


def cmd_run(args):
    """Record + transcribe immediately."""
    # Record first
    audio_path = cmd_record(args)
    print("\n" + "=" * 60 + "\n")

    # Then transcribe
    args.input = audio_path
    if not args.output:
        args.output = os.path.splitext(audio_path)[0]
    cmd_transcribe(args)


def main():
    parser = argparse.ArgumentParser(
        description="VoxScribe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- devices ---
    sub_devices = subparsers.add_parser("devices", help="List audio devices")
    sub_devices.set_defaults(func=cmd_devices)

    # --- record ---
    sub_record = subparsers.add_parser("record", help="Record audio")
    sub_record.add_argument(
        "--source", choices=["mic", "system", "both"], default="mic",
        help=("Audio source: mic, system (WASAPI loopback), or "
              "both (microphone + system)")
    )
    sub_record.add_argument("--output", "-o", help="Output path (WAV)")
    sub_record.add_argument(
        "--device", type=int, default=None,
        help="Device index (see the 'devices' command)"
    )
    sub_record.set_defaults(func=cmd_record)

    # --- transcribe ---
    sub_transcribe = subparsers.add_parser("transcribe", help="Transcribe audio")
    sub_transcribe.add_argument("--input", "-i", required=True, help="Audio file")
    sub_transcribe.add_argument(
        "--language", "-l", default="auto",
        help="Language (e.g. en, de) or 'auto' for automatic detection. Default: auto"
    )
    sub_transcribe.add_argument(
        "--model", "-m", default=None,
        help=("Whisper model (large-v2, large-v3, medium, base), OpenVINO model "
              "(openvino:GPU:medium, Intel Arc GPU), a configured provider "
              "(server:<provider-id>, see voxscribe/providers.py/Settings), or "
              "apple:speechanalyzer. Default: your configured default provider, "
              "if any - falls back automatically to the recommended local "
              "model on failure (hardware-/platform-dependent), see "
              "voxscribe.transcriber.default_model_size()")
    )
    sub_transcribe.add_argument(
        "--diarize", action="store_true", default=True,
        help="Enable speaker diarization (default: on)"
    )
    sub_transcribe.add_argument(
        "--no-diarize", dest="diarize", action="store_false",
        help="Disable speaker diarization"
    )
    sub_transcribe.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token (alternatively: HF_TOKEN in .env)"
    )
    sub_transcribe.add_argument(
        "--api-key", default=None,
        help="API key for ad-hoc provider use, overriding a configured provider's key"
    )
    sub_transcribe.add_argument(
        "--api-base-url", default=None,
        help="Base URL for ad-hoc provider use, overriding a configured provider's URL"
    )
    sub_transcribe.add_argument("--min-speakers", type=int, default=None)
    sub_transcribe.add_argument("--max-speakers", type=int, default=None)
    sub_transcribe.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (smaller = less VRAM). Default: 16"
    )
    sub_transcribe.add_argument(
        "--device-compute", choices=["cuda", "xpu", "mps", "cpu"], default=None,
        help="Compute device for alignment/diarization (auto-detected if not "
             "set). 'mps'/'xpu' only accelerate alignment/diarization, not "
             "Whisper transcription itself (except for 'openvino:...' models, "
             "which always run independently via OpenVINO)."
    )
    sub_transcribe.add_argument("--output", "-o", help="Output path (without extension)")
    sub_transcribe.add_argument(
        "--format", "-f", default="txt",
        help="Output format(s), comma-separated: txt, srt, json. Default: txt"
    )
    sub_transcribe.set_defaults(func=cmd_transcribe)

    # --- run ---
    sub_run = subparsers.add_parser(
        "run", help="Record + transcribe immediately"
    )
    sub_run.add_argument(
        "--source", choices=["mic", "system", "both"], default="mic",
        help="Audio source: mic, system, or both"
    )
    sub_run.add_argument("--output", "-o", default=None)
    sub_run.add_argument("--device", type=int, default=None)
    sub_run.add_argument("--language", "-l", default="auto")
    sub_run.add_argument("--model", "-m", default=None)
    sub_run.add_argument("--diarize", action="store_true", default=True)
    sub_run.add_argument("--no-diarize", dest="diarize", action="store_false")
    sub_run.add_argument("--hf-token", default=None)
    sub_run.add_argument("--api-key", default=None)
    sub_run.add_argument("--api-base-url", default=None)
    sub_run.add_argument("--min-speakers", type=int, default=None)
    sub_run.add_argument("--max-speakers", type=int, default=None)
    sub_run.add_argument("--batch-size", type=int, default=16)
    sub_run.add_argument(
        "--device-compute", choices=["cuda", "xpu", "mps", "cpu"], default=None
    )
    sub_run.add_argument("--format", "-f", default="txt")
    sub_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)
