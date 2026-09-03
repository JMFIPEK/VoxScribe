"""VoxScribe — CLI entry point.

Thin wrapper - the actual command implementations live in voxscribe/cli.py.
This file's job is just the SSL/proxy/warnings bootstrap, which must run
before anything (including voxscribe.cli) imports requests/torch/etc.

Commands:
    python main.py devices                      List audio devices
    python main.py record --source mic           Microphone recording
    python main.py record --source system        System audio (Teams/Zoom)
    python main.py record --source both          Microphone + system audio
    python main.py transcribe --input audio.wav  Start transcription
    python main.py run --source mic              Record + transcribe
"""

import os
import ssl
import warnings

# --- Suppress warnings ---
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TORCH_LOGS"] = "-all"
os.environ["TORCHAUDIO_NO_BACKEND_CHECK"] = "1"

# --- Corporate proxy: disable SSL verification (before any other imports) ---
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

from voxscribe.cli import main

if __name__ == "__main__":
    main()
