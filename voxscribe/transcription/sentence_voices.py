"""Second opinion on diarization from per-sentence voice prints (#62).

pyannote decides who speaks from 10 s windows. Two similar voices (relatives,
measured 2026-11-01: 0.31 similarity between them vs. ~0.68 within) end up as
one speaker there, and one voice on a single track is now and then split in
two (a Call's Playback track alone: 33 s / 37 s of one person). A voice print
per transcript sentence, with the same embedding model, separates both cases.

Used only where it clearly contradicts pyannote: when the sentences form a
different number of substantial groups than pyannote has substantial
speakers. Otherwise pyannote's word-level turns stay as they are. Not for
Calls, whose tracks already settle who is local and who is remote (#61).
"""

import re

import numpy as np

# A sentence shorter than this gets no voice print (too little voice to tell).
SENTENCE_MIN_SECONDS = 1.0
# A pause this long between two words ends a sentence too.
SENTENCE_MAX_GAP_SECONDS = 1.0
# Average-linkage cosine distance at which sentence groups stop merging.
# Measured on 20 clips (2026-11-01): 0.5 splits single voices into several
# groups of 4-6 s, 0.7 merges the two similar voices; 0.6 keeps every
# single-speaker clip in one substantial group and separates every two-person one.
CLUSTER_DISTANCE = 0.6
# A group (or a pyannote speaker) is substantial with this much sentence
# speech, absolute and as a share - the same limits as the track split.
GROUP_MIN_SECONDS = 8.0
GROUP_MIN_FRACTION = 0.10

_SENTENCE_END = re.compile(r"[.?!…]$")


def sentences(result: dict) -> list[list[dict]]:
    """The timed words of every segment, cut into sentences at sentence-end
    punctuation and at pauses over SENTENCE_MAX_GAP_SECONDS."""
    out: list[list[dict]] = []
    for seg in result.get("segments", []):
        cur: list[dict] = []
        for w in seg.get("words") or []:
            if "start" not in w or "end" not in w:
                continue
            if cur and w["start"] - cur[-1]["end"] > SENTENCE_MAX_GAP_SECONDS:
                out.append(cur)
                cur = []
            cur.append(w)
            if _SENTENCE_END.search(w.get("word", "").strip()):
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
    return out


def _majority(labels: list) -> str | None:
    labels = [x for x in labels if x]
    return max(set(labels), key=labels.count) if labels else None


def _substantial(seconds: dict, total: float) -> set:
    return {k for k, s in seconds.items()
            if s >= GROUP_MIN_SECONDS and s >= GROUP_MIN_FRACTION * total}


def _groups(vectors: np.ndarray) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    if len(vectors) < 2:
        return np.ones(len(vectors), dtype=int)
    return fcluster(linkage(vectors, "average", metric="cosine"), CLUSTER_DISTANCE, "distance")


def second_opinion(result: dict, speaker_embeddings: dict | None, embed) -> dict | None:
    """Relabels `result` (in place) by sentence voice prints where they clearly
    contradict pyannote. `embed(start, end)` returns the voice print of that
    stretch of audio. Runs on raw Diarization IDs, before the Stray speaker
    merge and voice-print matching.

    Returns the new {Diarization ID: embedding} (the mean voice print of each
    speaker's sentences, so a split-off speaker can be recognized and enrolled
    too), or None when pyannote's labels stay."""
    sents = [s for s in sentences(result) if s[-1]["end"] - s[0]["start"] >= SENTENCE_MIN_SECONDS]
    if len(sents) < 2:
        return None
    vectors = np.array([np.asarray(embed(s[0]["start"], s[-1]["end"]), dtype=float) for s in sents])
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    groups = _groups(vectors)
    seconds = np.array([s[-1]["end"] - s[0]["start"] for s in sents])
    total = float(seconds.sum())

    pyannote = [_majority([w.get("speaker") for w in s]) for s in sents]
    pyannote_seconds: dict = {}
    for spk, sec in zip(pyannote, seconds):
        if spk:
            pyannote_seconds[spk] = pyannote_seconds.get(spk, 0.0) + sec
    group_seconds = {g: float(seconds[groups == g].sum()) for g in set(groups)}
    big_groups = _substantial(group_seconds, total)
    big_speakers = _substantial(pyannote_seconds, total)
    print("     Sentence voice prints: " + ", ".join(
        f"{group_seconds[g]:.0f}s" for g in sorted(group_seconds, key=lambda g: -group_seconds[g]))
        + f" (pyannote: {len(big_speakers)} speakers, sentences: {len(big_groups)})")
    if not big_groups or len(big_groups) == len(big_speakers):
        return None

    # Small groups join the substantial group whose voice is closest.
    centroids = {g: vectors[groups == g].mean(axis=0) for g in big_groups}
    for g in set(groups) - big_groups:
        mean = vectors[groups == g].mean(axis=0)
        groups[groups == g] = max(big_groups, key=lambda b: float(mean @ centroids[b]))

    # Each group keeps the Diarization ID it shares most speech with; the rest get new IDs.
    used = set(speaker_embeddings or {}) | set(pyannote_seconds)
    label: dict = {}
    for g in sorted(big_groups, key=lambda g: -group_seconds[g]):
        shared: dict = {}
        for spk, sec, gg in zip(pyannote, seconds, groups):
            if gg == g and spk and spk not in label.values():
                shared[spk] = shared.get(spk, 0.0) + sec
        if shared:
            label[g] = max(shared, key=shared.get)
        else:
            n = 0
            while f"SPEAKER_{n:02d}" in used:
                n += 1
            label[g] = f"SPEAKER_{n:02d}"
        used.add(label[g])

    _relabel_words(result, sents, [label[g] for g in groups])
    from voxscribe.tracks import cut_at_speaker_changes
    cut_at_speaker_changes(result)

    embeddings = {label[g]: vectors[groups == g].mean(axis=0).tolist() for g in big_groups}
    if "speaker_embeddings" in result:
        result["speaker_embeddings"] = embeddings
    print("     Sentence voice prints overrule pyannote: " + ", ".join(
        f"{label[g]} {float(seconds[groups == g].sum()):.0f}s"
        for g in sorted(big_groups, key=lambda g: -group_seconds[g])))
    return embeddings


def _relabel_words(result: dict, sents: list[list[dict]], speakers: list[str]) -> None:
    """Gives every word of a voice-printed sentence its group's speaker, and
    every other word (short sentences, words outside the segments) the
    speaker of the nearest voice-printed sentence in time."""
    spans = [(s[0]["start"], s[-1]["end"], spk) for s, spk in zip(sents, speakers)]
    for s, spk in zip(sents, speakers):
        for w in s:
            w["speaker"] = spk
    printed = {id(w) for s in sents for w in s}

    def nearest(start: float, end: float) -> str:
        return min(spans, key=lambda sp: max(sp[0] - end, start - sp[1], 0.0))[2]

    for seg in result.get("segments", []):
        words = seg.get("words") or []
        for w in words:
            if id(w) not in printed:
                w["speaker"] = nearest(w.get("start", seg.get("start", 0.0)),
                                       w.get("end", seg.get("end", 0.0)))
        if not words:
            seg["speaker"] = nearest(seg.get("start", 0.0), seg.get("end", 0.0))
    for w in result.get("word_segments", []) or []:
        if id(w) not in printed and "start" in w:
            w["speaker"] = nearest(w["start"], w.get("end", w["start"]))
