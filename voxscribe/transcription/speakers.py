"""Speaker handling after diarization: track split, voice-print labels, multi-file merge."""

from voxscribe.transcription.audio import AudioSource


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


def label_known_speakers(result: dict, speaker_embeddings: dict | None,
                         split_speakers: dict[str, str]) -> None:
    """Automatically recognizes known speakers (voice prints) and labels
    their segments/words directly with the recognized name.

    Sets result["speaker_id_map"]: raw Diarization ID -> currently displayed
    Speaker label, so the GUI can find the right embedding when the user
    confirms/corrects a name via the "remember" checkbox.

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
            suggestions = dict(matches.certain)
            tentative = dict(matches.tentative)
        except Exception:
            suggestions = {}
        suggestions.update({spk_id: f"{name}?" for spk_id, name in tentative.items()})
        for spk_id, name in suggestions.items():
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
