"""Persistent configuration for remote transcription providers.

Generalizes what used to be a single hard-coded "KIT ToolBox" server into an
arbitrary list of OpenAI-compatible transcription endpoints the user can add,
edit, and pick a default from in Settings. Each provider is just
{id, name, base_url, api_key, model}; `transcriber.transcribe_remote()`
already accepts base_url/api_key/model as plain parameters (it was never
actually KIT-specific under the hood), so this module only adds the missing
persistence/selection layer on top.

Stored at ~/.voxscribe/providers.json, next to speaker_profiles.json - same
directory, same "small local JSON store" pattern.
"""

import json
import os
import re
import uuid

PROVIDERS_DIR = os.path.join(os.path.expanduser("~"), ".voxscribe")
PROVIDERS_PATH = os.path.join(PROVIDERS_DIR, "providers.json")

# Back-compat: pre-provider-system installs configured a single server via
# these two env vars (see migrate_from_env()).
_LEGACY_ENV_API_KEY = "KIT_TOOLBOX_API_KEY"
_LEGACY_ENV_BASE_URL = "KIT_TOOLBOX_BASE_URL"
_LEGACY_DEFAULT_BASE_URL = "https://ki-toolbox.scc.kit.edu/api/v1"
_LEGACY_DEFAULT_MODEL = "kit.whisper-large-v3"
_LEGACY_NAME = "KIT ToolBox"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "provider"


def _load_raw() -> dict:
    if not os.path.isfile(PROVIDERS_PATH):
        return {"providers": [], "default_provider_id": None}
    try:
        with open(PROVIDERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("providers", [])
        data.setdefault("default_provider_id", None)
        return data
    except Exception:
        return {"providers": [], "default_provider_id": None}


def _save_raw(data: dict) -> None:
    os.makedirs(PROVIDERS_DIR, exist_ok=True)
    with open(PROVIDERS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_providers() -> list[dict]:
    return _load_raw()["providers"]


def get_provider(provider_id: str) -> dict | None:
    for p in list_providers():
        if p["id"] == provider_id:
            return p
    return None


def get_default_provider() -> dict | None:
    data = _load_raw()
    pid = data.get("default_provider_id")
    if pid:
        for p in data["providers"]:
            if p["id"] == pid:
                return p
    return data["providers"][0] if data["providers"] else None


def _unique_id(name: str, existing: list[dict]) -> str:
    base = _slugify(name)
    taken = {p["id"] for p in existing}
    if base not in taken:
        return base
    i = 2
    while f"{base}-{i}" in taken:
        i += 1
    return f"{base}-{i}"


def add_provider(name: str, base_url: str, api_key: str, model: str,
                  make_default: bool = False) -> dict:
    data = _load_raw()
    provider = {
        "id": _unique_id(name, data["providers"]),
        "name": name,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
    }
    data["providers"].append(provider)
    if make_default or data["default_provider_id"] is None:
        data["default_provider_id"] = provider["id"]
    _save_raw(data)
    return provider


def update_provider(provider_id: str, **fields) -> dict | None:
    data = _load_raw()
    for p in data["providers"]:
        if p["id"] == provider_id:
            p.update({k: v for k, v in fields.items() if v is not None})
            _save_raw(data)
            return p
    return None


def delete_provider(provider_id: str) -> None:
    data = _load_raw()
    data["providers"] = [p for p in data["providers"] if p["id"] != provider_id]
    if data["default_provider_id"] == provider_id:
        data["default_provider_id"] = data["providers"][0]["id"] if data["providers"] else None
    _save_raw(data)


def set_default_provider(provider_id: str) -> None:
    data = _load_raw()
    if any(p["id"] == provider_id for p in data["providers"]):
        data["default_provider_id"] = provider_id
        _save_raw(data)


def test_provider(base_url: str, api_key: str, model: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Sends a tiny synthetic audio clip to the provider's
    `/audio/transcriptions` endpoint to verify the URL/API key/model actually
    work end-to-end - used by the "Test" button in the Settings provider
    dialog. Deliberately only uses `requests`/`soundfile`/`numpy` (already
    lightweight dependencies) instead of going through
    `transcriber.transcribe_remote()`, so this stays fast: importing
    `transcriber` pulls in whisperx/torch, a 1-3 minute cold import that would
    make clicking "Test" in the GUI look frozen.

    Returns (ok, message).
    """
    if not base_url:
        return False, "Base URL is required."
    if not api_key:
        return False, "API key is required."

    import io
    import numpy as np
    import soundfile as sf
    import requests

    # 0.5s of silence is enough to exercise the real upload/auth/response
    # path without needing an actual recording.
    silence = np.zeros(int(0.5 * 16000), dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, silence, 16000, format="FLAC")
    buf.seek(0)

    url = base_url.rstrip("/") + "/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model} if model else {}
    files = {"file": ("test.flac", buf, "audio/flac")}

    try:
        resp = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)
    except requests.exceptions.RequestException as e:
        return False, f"Connection failed: {e}"

    if resp.status_code == 200:
        return True, "Success - the provider accepted the request."
    if resp.status_code == 401:
        return False, "Authentication failed (401) - check the API key."
    if resp.status_code == 404:
        return False, "Endpoint not found (404) - check the base URL."
    return False, f"Server returned HTTP {resp.status_code}: {resp.text[:200]}"


def migrate_from_env() -> dict | None:
    """One-time migration: if providers.json doesn't exist yet but the old
    single-server env vars (KIT_TOOLBOX_API_KEY/_BASE_URL) are set, create a
    provider from them automatically so upgrading doesn't silently drop an
    existing working setup. Returns the created provider, or None if there
    was nothing to migrate (providers.json already exists, or no legacy env
    vars set)."""
    if os.path.isfile(PROVIDERS_PATH):
        return None
    api_key = os.getenv(_LEGACY_ENV_API_KEY)
    if not api_key:
        return None
    base_url = os.getenv(_LEGACY_ENV_BASE_URL) or _LEGACY_DEFAULT_BASE_URL
    return add_provider(_LEGACY_NAME, base_url, api_key, _LEGACY_DEFAULT_MODEL,
                         make_default=True)
