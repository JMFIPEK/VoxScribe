"""Pre-download models for offline/bundled operation.

This script downloads all needed models into the 'bundled_models/' folder, so
the application works without an internet connection and without a
HuggingFace token.

Usage:
    python download_models.py

Requirements:
    - HF_TOKEN in .env or as an environment variable (for pyannote diarization)
    - Internet connection while running this script
"""

import os
import ssl
import sys

# SSL/proxy configuration (same as main.py)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
_original_send = requests.adapters.HTTPAdapter.send
def _patched_send(self, request, *args, **kwargs):
    kwargs["verify"] = False
    return _original_send(self, request, *args, **kwargs)
requests.adapters.HTTPAdapter.send = _patched_send

from dotenv import load_dotenv
load_dotenv()

# This file lives in voxscribe/, bundled_models/ belongs at the project root
# (one level up), same convention as transcriber.py's _get_base_dir().
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bundled_models")


def download_whisper_model(model_size: str = "medium"):
    """Downloads the Whisper model (faster-whisper format)."""
    from faster_whisper.utils import download_model

    dest = os.path.join(MODELS_DIR, "whisper", model_size)
    os.makedirs(dest, exist_ok=True)

    if os.path.exists(os.path.join(dest, "model.bin")):
        print(f"  [OK] Whisper {model_size} already present")
        return dest

    print(f"  Downloading Whisper {model_size}...")
    path = download_model(model_size, output_dir=dest)
    print(f"  [OK] Whisper {model_size} -> {path}")
    return dest


def download_align_models():
    """Downloads the alignment models (torchaudio) for DE and EN."""
    import torchaudio

    dest = os.path.join(MODELS_DIR, "align")
    os.makedirs(dest, exist_ok=True)

    models = {
        "de": "VOXPOPULI_ASR_BASE_10K_DE",
        "en": "WAV2VEC2_ASR_BASE_960H",
    }

    for lang, model_name in models.items():
        print(f"  Downloading alignment model {model_name} ({lang})...")
        bundle = torchaudio.pipelines.__dict__[model_name]
        # Download the model weights to our directory
        bundle.get_model(dl_kwargs={"model_dir": dest})
        print(f"  [OK] {model_name}")

    return dest


def download_diarization_model():
    """Downloads the pyannote diarization model."""
    from huggingface_hub import snapshot_download

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("  [WARNING] No HF_TOKEN found - skipping diarization model!")
        print("  Set HF_TOKEN in .env to download this model.")
        return None

    dest = os.path.join(MODELS_DIR, "diarize")
    os.makedirs(dest, exist_ok=True)

    # Main pipeline
    model_id = "pyannote/speaker-diarization-community-1"
    print(f"  Downloading {model_id}...")
    snapshot_download(
        model_id,
        token=hf_token,
        cache_dir=dest,
    )
    print(f"  [OK] {model_id}")

    # Segmentation model (pipeline dependency)
    seg_model_id = "pyannote/segmentation-3.0"
    print(f"  Downloading {seg_model_id}...")
    snapshot_download(
        seg_model_id,
        token=hf_token,
        cache_dir=dest,
    )
    print(f"  [OK] {seg_model_id}")

    # Embedding model (pipeline dependency)
    emb_model_id = "pyannote/wespeaker-voxceleb-resnet34-LM"
    print(f"  Downloading {emb_model_id}...")
    snapshot_download(
        emb_model_id,
        token=hf_token,
        cache_dir=dest,
    )
    print(f"  [OK] {emb_model_id}")

    return dest


def export_openvino_model(whisper_size: str = "medium"):
    """Exports a Whisper model to OpenVINO IR (INT8-quantized) for Intel Arc
    GPU/NPU transcription (see transcriber.transcribe_openvino()).

    Windows/Intel-specific - needs `optimum-intel[openvino]` (see
    pyproject.toml, only installed on sys_platform == 'win32') and exports
    from the HuggingFace Transformers checkpoint (not the CTranslate2 format
    download_whisper_model() above downloads - that's a different model
    format for a completely different inference backend).
    """
    try:
        from optimum.commands.optimum_cli import main as optimum_cli_main
    except ImportError:
        print("  [SKIPPED] optimum-intel[openvino] not installed "
              "(Windows-only) - Intel Arc GPU/NPU backend not available.")
        return None

    dest = os.path.join(MODELS_DIR, "openvino", f"whisper-{whisper_size}")
    if os.path.exists(os.path.join(dest, "openvino_encoder_model.bin")):
        print(f"  [OK] OpenVINO Whisper {whisper_size} already present")
        return dest

    hf_model_id = {
        "medium": "openai/whisper-medium",
        "large-v2": "openai/whisper-large-v2",
        "large-v3": "openai/whisper-large-v3",
        "base": "openai/whisper-base",
    }.get(whisper_size, f"openai/whisper-{whisper_size}")

    print(f"  Exporting {hf_model_id} to OpenVINO IR (INT8)...")
    # Via the optimum-cli command class instead of main_export() directly,
    # because the INT8/INT4 weight-compression config (OVConfig/
    # quantization_config) is built internally in a fairly complex way from
    # --weight-format (see optimum.commands.export.openvino.OVExportCommand.
    # run) - reimplementing that here would be error-prone, the CLI class
    # already does it correctly.
    original_argv = sys.argv
    try:
        sys.argv = [
            "optimum-cli", "export", "openvino",
            "--model", hf_model_id,
            "--task", "automatic-speech-recognition-with-past",
            "--weight-format", "int8",
            dest,
        ]
        optimum_cli_main()
    finally:
        sys.argv = original_argv
    print(f"  [OK] OpenVINO Whisper {whisper_size} -> {dest}")
    return dest


def main():
    print("=" * 60)
    print("  VoxScribe — Downloading models")
    print("=" * 60)
    print(f"\nTarget folder: {MODELS_DIR}\n")

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[1/4] Whisper model (medium)...")
    download_whisper_model("medium")
    print()

    print("[2/4] Alignment models (DE + EN)...")
    download_align_models()
    print()

    print("[3/4] Diarization model (pyannote)...")
    download_diarization_model()
    print()

    print("[4/4] OpenVINO model for Intel Arc GPU/NPU (medium)...")
    export_openvino_model("medium")
    print()

    print("=" * 60)
    print("  All models downloaded!")
    print(f"  Folder: {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
