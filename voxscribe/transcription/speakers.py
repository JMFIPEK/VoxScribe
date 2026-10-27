"""Speaker handling after diarization: track split, Stray speaker merge,
voice-print labels, multi-file merge."""

from voxscribe.transcription.audio import AudioSource

# A Diarization ID with less speech than this in total is a Stray speaker
# (GLOSSARY.md). An absolute limit, not a share: 2% of a 1-hour meeting
# (72 s) would merge a real person who only says "yes, agreed".
STRAY_SPEAKER_MAX_SECONDS = 3.0


def split_by_track(result: dict, audio: AudioSource) -> dict[str, str]:
    """"Microphone + system" recordings: use the separate tracks (see
    tracks.py) to split speakers that pyannote merged across the
    local/remote boundary. Runs on raw labels, before voice-print
    matching - split-off speakers have no embedding of their own
    and so are simply left unrecognized. The tracks come from the
    pipeline's AudioSource, so a Loudspeaker Call's echo canceller doesn't
    run a second time.

    Returns {new_label: original_label} for every split (empty if there is no
    track sidecar, or the split failed - which is only a warning)."""
    try:
        from voxscribe.tracks import split_speakers_by_track
        tracks = audio.tracks()
        if tracks is None:
            return {}
        split_speakers = split_speakers_by_track(result, *tracks)
        if split_speakers:
            print("     Split by microphone/system track: " + ", ".join(
                f"{old} -> {old} + {new}" for new, old in split_speakers.items()))
        return split_speakers
    except Exception as e:
        print(f"     [WARNING] Track-based speaker split failed: {e}")
        return {}


def merge_stray_speakers(result: dict, speaker_embeddings: dict | None) -> dict[str, str]:
    """Gives every Stray speaker's segments and words to the speaker of the
    nearest neighbouring turn in time (previous or next, whichever is
    closer); nothing is deleted. Runs on raw Diarization IDs, after the
    track split and before voice-print matching, so a Stray speaker is never
    recognized. Every Diarization ID counts: one that has an embedding or a
    word but wins no segment has 0 s of speech and is merged too. Merged IDs
    are removed from `speaker_embeddings` (also result["speaker_embeddings"]).

    Returns {stray_id: neighbour_id} for every merged speaker that had
    segments or words (if they went to different neighbours, the last one)."""
    segments = result.get("segments", [])
    word_segments = result.get("word_segments", []) or []
    speech: dict[str, float] = {}
    for spk in [*(speaker_embeddings or {}), *(result.get("speaker_embeddings") or {}),
                *(w.get("speaker") for w in word_segments),
                *(w.get("speaker") for seg in segments for w in seg.get("words", []) or [])]:
        if spk:
            speech.setdefault(spk, 0.0)
    for seg in segments:
        spk = seg.get("speaker")
        if spk:
            speech[spk] = speech.get(spk, 0.0) + max(0.0, seg.get("end", 0.0) - seg.get("start", 0.0))
    stray = {spk for spk, secs in speech.items() if secs < STRAY_SPEAKER_MAX_SECONDS}
    kept = [seg for seg in segments if seg.get("speaker") and seg["speaker"] not in stray]
    if not stray or not kept:
        return {}

    def nearest(start: float, end: float) -> str:
        def gap(seg):
            return max(seg.get("start", 0.0) - end, start - seg.get("end", 0.0), 0.0)
        return min(kept, key=gap)["speaker"]

    merged: dict[str, str] = {}
    moved: list[tuple[float, float, str, str]] = []  # (start, end, stray_id, neighbour)
    for seg in segments:
        spk = seg.get("speaker")
        if spk in stray:
            target = nearest(seg.get("start", 0.0), seg.get("end", 0.0))
            seg["speaker"] = target
            merged[spk] = target
            moved.append((seg.get("start", 0.0), seg.get("end", 0.0), spk, target))
        # A word labelled with a Stray speaker belongs to the turn around it.
        for w in seg.get("words", []) or []:
            if w.get("speaker") in stray and seg.get("speaker"):
                merged.setdefault(w["speaker"], seg["speaker"])
                w["speaker"] = seg["speaker"]

    # Words outside the segments follow the turn they belong to.
    for w in word_segments:
        spk = w.get("speaker")
        if spk not in stray:
            continue
        mid = (w.get("start", 0.0) + w.get("end", 0.0)) / 2
        w["speaker"] = next((target for a, b, s, target in moved if s == spk and a <= mid <= b),
                            None) or nearest(w.get("start", 0.0), w.get("end", 0.0))
        merged.setdefault(spk, w["speaker"])

    for embeddings in (speaker_embeddings, result.get("speaker_embeddings")):
        for spk in stray:
            (embeddings or {}).pop(spk, None)
    print("     Stray speakers merged: " + ", ".join(
        f"{spk} ({speech[spk]:.1f}s) -> {merged.get(spk, '-')}" for spk in sorted(stray)))
    return merged


def label_known_speakers(result: dict, speaker_embeddings: dict | None,
                         split_speakers: dict[str, str]) -> None:
    """Automatically recognizes known speakers (voice prints) and labels
    their segments/words directly with the recognized name.

    Sets result["speaker_id_map"]: raw Diarization ID -> currently displayed
    Speaker label, so the GUI can find the right embedding when the user
    names a speaker (the name saves in place and the voice is enrolled,
    unless the speaker has under 10 s of speech).

    A Tentative match is labelled "<Name>?" and recorded in
    result["tentative_speakers"]: Diarization ID -> name without the "?", so
    the GUI can offer to confirm it. Exports write the label as shown."""
    speaker_id_map = {spk: spk for spk in (speaker_embeddings or {})}
    for new_spk in split_speakers:
        speaker_id_map[new_spk] = new_spk
    tentative: dict[str, str] = {}
    if speaker_embeddings:
        try:
            from voxscribe.speaker_profiles import match_speakers
            matches = match_speakers(speaker_embeddings)
            labels = dict(matches.certain)
            tentative = dict(matches.tentative)
        except Exception:
            labels = {}
        labels.update({spk_id: f"{name}?" for spk_id, name in tentative.items()})
        for spk_id, name in labels.items():
            speaker_id_map[spk_id] = name
            for seg in result.get("segments", []):
                if seg.get("speaker") == spk_id:
                    seg["speaker"] = name
            for word in result.get("word_segments", []) or []:
                if word.get("speaker") == spk_id:
                    word["speaker"] = name
    result["speaker_id_map"] = speaker_id_map
    result["tentative_speakers"] = tentative


def append_file_result(combined: dict, result: dict, idx: int, n_files: int, offset: float) -> None:
    """Adds one file's transcription `result` to the `combined` multi-file
    transcript (keys: segments, word_segments, speaker_id_map,
    speaker_embeddings), shifting all timestamps by `offset` seconds.

    Speakers not yet recognized via voice print (Speaker label == raw
    Diarization ID) get a file prefix ("File 2: SPEAKER_00") when there are
    several files, since diarization runs per file and SPEAKER_00 in file 1 is
    not necessarily the same person as in file 2. Recognized names are left
    unchanged and so merge across files."""
    # Rename speakers not yet recognized (label == raw diarization ID) in
    # this file, to avoid collisions with other files; leave already
    # voice-print-recognized names unchanged so the same person is merged
    # across files.
    rename_map = {}
    file_speaker_id_map = result.get("speaker_id_map") or {}
    file_embeddings = result.get("speaker_embeddings") or {}
    for raw_id, current_label in file_speaker_id_map.items():
        if n_files > 1 and raw_id == current_label:
            new_label = f"File {idx + 1}: {current_label}"
            new_key = f"file{idx}:{raw_id}"
        else:
            new_label = current_label
            new_key = current_label
        rename_map[current_label] = new_label
        combined["speaker_id_map"][new_key] = new_label
        if raw_id in file_embeddings:
            combined["speaker_embeddings"][new_key] = file_embeddings[raw_id]

    for seg in result.get("segments", []):
        if rename_map and seg.get("speaker") in rename_map:
            seg["speaker"] = rename_map[seg["speaker"]]
        seg["start"] = seg.get("start", 0.0) + offset
        seg["end"] = seg.get("end", 0.0) + offset
        for w in seg.get("words", []) or []:
            if "start" in w:
                w["start"] += offset
            if "end" in w:
                w["end"] += offset
        combined["segments"].append(seg)

    for w in result.get("word_segments", []) or []:
        if rename_map and w.get("speaker") in rename_map:
            w["speaker"] = rename_map[w["speaker"]]
        if "start" in w:
            w["start"] += offset
        if "end" in w:
            w["end"] += offset
        combined["word_segments"].append(w)
