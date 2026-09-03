"""Persistent speaker profiles (voice prints) for automatic recognition.

Uses the speaker embeddings from the pyannote diarization pipeline to
recognize speakers across multiple recordings by voice match (cosine
similarity), instead of having to manually rename them on every transcription.
"""

import json
import os

import numpy as np

PROFILES_DIR = os.path.join(os.path.expanduser("~"), ".voxscribe")
PROFILES_PATH = os.path.join(PROFILES_DIR, "speaker_profiles.json")

# Minimum cosine similarity above which a detected speaker is automatically
# matched to a known profile. Empirical value for pyannote embeddings; adjust
# if matches happen too often incorrectly or not at all.
DEFAULT_MATCH_THRESHOLD = 0.75


def load_profiles() -> dict:
    """Loads saved speaker profiles: {name: {"embedding": [...], "samples": n}}."""
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
    """Matches detected speaker embeddings to known names.

    Args:
        embeddings: {"SPEAKER_00": [...], ...} from diarization
        profiles: known profiles (loaded automatically if None)
        threshold: minimum cosine similarity for an automatic match

    Returns:
        {"SPEAKER_00": "Jonas", ...} only for unambiguously matched speakers.
        Each name is assigned to at most one speaker (best match wins on
        conflicts), so two different speakers never accidentally get the
        same name.
    """
    if profiles is None:
        profiles = load_profiles()
    if not profiles or not embeddings:
        return {}

    candidates = []
    for spk, emb in embeddings.items():
        for name, profile in profiles.items():
            try:
                sim = _cosine_similarity(emb, profile["embedding"])
            except ValueError:
                # Corrupted/incompatible profile (e.g. wrong embedding
                # dimension) - skip it instead of aborting the whole match
                # for every other speaker/profile.
                continue
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
    """Saves/updates a speaker's profile as a running average."""
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
