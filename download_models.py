"""Modelle vorab herunterladen fuer Offline-/Bundled-Betrieb.

Dieses Skript laedt alle benoetigten Modelle in den Ordner 'bundled_models/' herunter,
damit die Anwendung ohne Internetverbindung und ohne HuggingFace Token funktioniert.

Verwendung:
    python download_models.py

Voraussetzungen:
    - HF_TOKEN in .env oder als Umgebungsvariable (fuer pyannote-Diarization)
    - Internetverbindung beim Ausfuehren dieses Skripts
"""

import os
import ssl
import sys

# SSL/Proxy-Konfiguration (wie in main.py)
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

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundled_models")


def download_whisper_model(model_size: str = "medium"):
    """Laedt das Whisper-Modell (faster-whisper Format) herunter."""
    from faster_whisper.utils import download_model

    dest = os.path.join(MODELS_DIR, "whisper", model_size)
    os.makedirs(dest, exist_ok=True)

    if os.path.exists(os.path.join(dest, "model.bin")):
        print(f"  [OK] Whisper {model_size} bereits vorhanden")
        return dest

    print(f"  Lade Whisper {model_size} herunter...")
    path = download_model(model_size, output_dir=dest)
    print(f"  [OK] Whisper {model_size} -> {path}")
    return dest


def download_align_models():
    """Laedt die Alignment-Modelle (torchaudio) fuer DE und EN herunter."""
    import torchaudio

    dest = os.path.join(MODELS_DIR, "align")
    os.makedirs(dest, exist_ok=True)

    models = {
        "de": "VOXPOPULI_ASR_BASE_10K_DE",
        "en": "WAV2VEC2_ASR_BASE_960H",
    }

    for lang, model_name in models.items():
        print(f"  Lade Alignment-Modell {model_name} ({lang})...")
        bundle = torchaudio.pipelines.__dict__[model_name]
        # Download the model weights to our directory
        bundle.get_model(dl_kwargs={"model_dir": dest})
        print(f"  [OK] {model_name}")

    return dest


def download_diarization_model():
    """Laedt das pyannote Diarization-Modell herunter."""
    from huggingface_hub import snapshot_download

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("  [WARNUNG] Kein HF_TOKEN gefunden - Diarization-Modell wird uebersprungen!")
        print("  Setze HF_TOKEN in .env um das Modell herunterzuladen.")
        return None

    dest = os.path.join(MODELS_DIR, "diarize")
    os.makedirs(dest, exist_ok=True)

    # Haupt-Pipeline
    model_id = "pyannote/speaker-diarization-community-1"
    print(f"  Lade {model_id}...")
    snapshot_download(
        model_id,
        token=hf_token,
        cache_dir=dest,
    )
    print(f"  [OK] {model_id}")

    # Segmentierungs-Modell (Dependency der Pipeline)
    seg_model_id = "pyannote/segmentation-3.0"
    print(f"  Lade {seg_model_id}...")
    snapshot_download(
        seg_model_id,
        token=hf_token,
        cache_dir=dest,
    )
    print(f"  [OK] {seg_model_id}")

    # Embedding-Modell (Dependency der Pipeline)
    emb_model_id = "pyannote/wespeaker-voxceleb-resnet34-LM"
    print(f"  Lade {emb_model_id}...")
    snapshot_download(
        emb_model_id,
        token=hf_token,
        cache_dir=dest,
    )
    print(f"  [OK] {emb_model_id}")

    return dest


def export_openvino_model(whisper_size: str = "medium"):
    """Exportiert ein Whisper-Modell nach OpenVINO IR (INT8-quantisiert) fuer die
    Intel Arc GPU/NPU-Transkription (siehe transcriber.transcribe_openvino()).

    Windows/Intel-spezifisch - braucht `optimum-intel[openvino]` (siehe
    pyproject.toml, nur auf sys_platform == 'win32' installiert) und exportiert
    aus dem HuggingFace-Transformers-Checkpoint (nicht dem CTranslate2-Format,
    das download_whisper_model() oben laedt - das ist ein anderes Modellformat
    fuer einen komplett anderen Inferenz-Backend).
    """
    try:
        from optimum.commands.optimum_cli import main as optimum_cli_main
    except ImportError:
        print("  [UEBERSPRUNGEN] optimum-intel[openvino] nicht installiert "
              "(nur auf Windows vorgesehen) - Intel Arc GPU/NPU-Backend nicht verfuegbar.")
        return None

    dest = os.path.join(MODELS_DIR, "openvino", f"whisper-{whisper_size}")
    if os.path.exists(os.path.join(dest, "openvino_encoder_model.bin")):
        print(f"  [OK] OpenVINO Whisper {whisper_size} bereits vorhanden")
        return dest

    hf_model_id = {
        "medium": "openai/whisper-medium",
        "large-v2": "openai/whisper-large-v2",
        "large-v3": "openai/whisper-large-v3",
        "base": "openai/whisper-base",
    }.get(whisper_size, f"openai/whisper-{whisper_size}")

    print(f"  Exportiere {hf_model_id} nach OpenVINO IR (INT8)...")
    # Ueber die optimum-cli-Befehlsklasse statt main_export() direkt, weil die
    # INT8/INT4-Gewichtskompressions-Konfiguration (OVConfig/quantization_config)
    # intern recht komplex aus --weight-format zusammengebaut wird (siehe
    # optimum.commands.export.openvino.OVExportCommand.run) - das hier
    # nachzubauen waere fehleranfaellig, die CLI-Klasse macht es bereits korrekt.
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
    print("  WhisperX — Modelle herunterladen")
    print("=" * 60)
    print(f"\nZielordner: {MODELS_DIR}\n")

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[1/4] Whisper-Modell (medium)...")
    download_whisper_model("medium")
    print()

    print("[2/4] Alignment-Modelle (DE + EN)...")
    download_align_models()
    print()

    print("[3/4] Diarization-Modell (pyannote)...")
    download_diarization_model()
    print()

    print("[4/4] OpenVINO-Modell fuer Intel Arc GPU/NPU (medium)...")
    export_openvino_model("medium")
    print()

    print("=" * 60)
    print("  Alle Modelle heruntergeladen!")
    print(f"  Ordner: {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
