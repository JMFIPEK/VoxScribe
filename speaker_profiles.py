"""Persistente Sprecher-Profile (Voice-Prints) fuer automatische Wiedererkennung.

Nutzt die Speaker-Embeddings aus der pyannote-Diarization-Pipeline, um Sprecher
ueber mehrere Aufnahmen hinweg per Stimmabgleich (Cosine-Similarity) wieder-
zuerkennen, statt sie bei jeder Transkription neu manuell benennen zu muessen.
"""

import json
import os

import numpy as np

PROFILES_DIR = os.path.join(os.path.expanduser("~"), ".voxscribe")
PROFILES_PATH = os.path.join(PROFILES_DIR, "speaker_profiles.json")

# Minimale Cosine-Similarity, ab der ein erkannter Sprecher automatisch einem
# bekannten Profil zugeordnet wird. Empirischer Richtwert fuer pyannote-
# Embeddings; ggf. anpassen, falls zu oft falsch/gar nicht zugeordnet wird.
DEFAULT_MATCH_THRESHOLD = 0.75


def load_profiles() -> dict:
    """Laedt gespeicherte Sprecher-Profile: {name: {"embedding": [...], "samples": n}}."""
    if not os.path.isfile(PROFILES_PATH):
        return {}
    try:
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_profiles(profiles: dict) -> None:
    os.makedirs(PROFILES_DIR, exist_ok=True)
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def _cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_speakers(embeddings: dict, profiles: dict | None = None,
                    threshold: float = DEFAULT_MATCH_THRESHOLD) -> dict:
    """Ordnet erkannten Sprecher-Embeddings bekannte Namen zu.

    Args:
        embeddings: {"SPEAKER_00": [...], ...} aus der Diarization
        profiles: bekannte Profile (werden selbst geladen, falls None)
        threshold: minimale Cosine-Similarity fuer eine automatische Zuordnung

    Returns:
        {"SPEAKER_00": "Jonas", ...} nur fuer eindeutig zugeordnete Sprecher.
        Jeder Name wird hoechstens einem Sprecher zugewiesen (bester Treffer
        gewinnt bei Konflikten), damit nicht zwei unterschiedliche Sprecher
        versehentlich denselben Namen bekommen.
    """
    if profiles is None:
        profiles = load_profiles()
    if not profiles or not embeddings:
        return {}

    candidates = []
    for spk, emb in embeddings.items():
        for name, profile in profiles.items():
            sim = _cosine_similarity(emb, profile["embedding"])
            if sim >= threshold:
                candidates.append((sim, spk, name))
    candidates.sort(key=lambda x: x[0], reverse=True)

    matches: dict[str, str] = {}
    used_names = set()
    for _sim, spk, name in candidates:
        if spk in matches or name in used_names:
            continue
        matches[spk] = name
        used_names.add(name)
    return matches


def enroll_speaker(name: str, embedding, profiles: dict | None = None) -> dict:
    """Speichert/aktualisiert das Profil eines Sprechers als laufenden Durchschnitt."""
    if profiles is None:
        profiles = load_profiles()
    name = name.strip()
    if not name:
        return profiles

    new_embedding = np.asarray(embedding, dtype=np.float64)
    existing = profiles.get(name)
    if existing:
        n = existing.get("samples", 1)
        old_embedding = np.asarray(existing["embedding"], dtype=np.float64)
        averaged = ((old_embedding * n) + new_embedding) / (n + 1)
        profiles[name] = {"embedding": averaged.tolist(), "samples": n + 1}
    else:
        profiles[name] = {"embedding": new_embedding.tolist(), "samples": 1}

    save_profiles(profiles)
    return profiles


def delete_profile(name: str, profiles: dict | None = None) -> dict:
    if profiles is None:
        profiles = load_profiles()
    profiles.pop(name, None)
    save_profiles(profiles)
    return profiles
