"""Persistent speaker profiles (voice prints) for automatic recognition.

Uses the speaker embeddings from the pyannote diarization pipeline to
recognize speakers across multiple recordings by voice match (cosine
similarity), instead of having to manually rename them on every transcription.
"""

import json
import os
from typing import NamedTuple

import numpy as np

PROFILES_DIR = os.path.join(os.path.expanduser("~"), ".voxscribe")
PROFILES_PATH = os.path.join(PROFILES_DIR, "speaker_profiles.json")

# Minimum cosine similarity above which a detected speaker is automatically
# matched to a known profile. Empirical value for pyannote embeddings; adjust
# if matches happen too often incorrectly or not at all.
DEFAULT_MATCH_THRESHOLD = 0.75

# Below the match threshold, a speaker whose best similarity is at least this
# is offered as a Tentative match ("Jonas?") for the user to confirm. Not yet
# calibrated: every match's similarity is logged for tuning it later.
DEFAULT_TENTATIVE_THRESHOLD = 0.60


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


class SpeakerMatches(NamedTuple):
    """Result of match_speakers(): {Diarization ID: name} per kind of match."""
    certain: dict[str, str]
    tentative: dict[str, str]


def _assign_names(candidates: list, taken_speakers: set, taken_names: set,
                  log_suffix: str = "") -> dict[str, str]:
    """Best similarity first: each speaker and each name is used at most once."""
    assigned: dict[str, str] = {}
    for sim, spk, name in sorted(candidates, key=lambda x: x[0], reverse=True):
        if spk in taken_speakers or spk in assigned or name in taken_names:
            continue
        assigned[spk] = name
        taken_names.add(name)
        print(f"     Voice print: {spk} -> {name}{log_suffix} (similarity {sim:.2f})")
    return assigned


def match_speakers(embeddings: dict, profiles: dict | None = None,
                    threshold: float = DEFAULT_MATCH_THRESHOLD,
                    tentative_threshold: float = DEFAULT_TENTATIVE_THRESHOLD) -> SpeakerMatches:
    """Matches detected speaker embeddings to known names.

    Args:
        embeddings: {"SPEAKER_00": [...], ...} from diarization
        profiles: known profiles (loaded automatically if None)
        threshold: minimum cosine similarity for an automatic (certain) match
        tentative_threshold: minimum similarity for a Tentative match

    Returns:
        certain: {"SPEAKER_00": "Jonas", ...} for speakers at or above
        `threshold`. tentative: for the remaining speakers, their best name
        if its similarity is in [tentative_threshold, threshold) and it isn't
        taken - matched for certain or by a closer speaker; there is no
        fallback to the next-best name. Within each, a name is assigned to at
        most one speaker (best match wins on conflicts), so two different
        speakers never accidentally get the same name. Every candidate at or
        above tentative_threshold is logged with its similarity.
    """
    if profiles is None:
        profiles = load_profiles()
    if not profiles or not embeddings:
        return SpeakerMatches({}, {})

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
            if sim >= tentative_threshold:
                candidates.append((sim, spk, name))

    used_names: set = set()
    certain = _assign_names([c for c in candidates if c[0] >= threshold], set(), used_names)
    # A Tentative match only ever offers a speaker's best name: if that one is
    # taken (matched for certain, or by a closer speaker), there is none.
    best: dict = {}
    for sim, spk, name in candidates:
        if spk not in best or sim > best[spk][0]:
            best[spk] = (sim, spk, name)
    tentative = _assign_names([c for c in best.values() if c[0] < threshold],
                              set(certain), used_names, "?")
    # Every candidate is logged, for calibrating the thresholds later.
    for sim, spk, name in sorted(candidates, key=lambda x: x[0], reverse=True):
        if certain.get(spk) != name and tentative.get(spk) != name:
            print(f"     Voice print: {spk} -> {name} (similarity {sim:.2f}, not assigned)")
    return SpeakerMatches(certain, tentative)

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
