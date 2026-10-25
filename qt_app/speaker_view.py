"""Speaker colours, ranks and shares for the Transcription tab (spec #40).

Pure Python, no Qt and no torch/whisperx, so the speaker list, the Timeline
and playback can all share it and it stays testable on synthetic result
dicts (tests/test_speaker_view.py)."""

from __future__ import annotations

from dataclasses import dataclass

# Speakers with their own Timeline lane; the rest share the Others lane.
TOP_LANES = 5
# Less speech than this gives no voice print worth remembering.
MIN_ENROLL_SECONDS = 10.0


@dataclass(frozen=True)
class Speaker:
    """One row of the speaker list / one speaker on the Timeline."""
    label: str              # current Speaker label
    ids: tuple[str, ...]    # its Diarization IDs, most speech first
    color: str
    seconds: float          # speaking time
    share: float            # of all speaking time, 0..1
    rank: int               # 0 = most speaking time


class SpeakerView:
    """Built once per transcription result. Colours are assigned here, per
    Diarization ID in order of speaking time, and never change afterwards."""

    def __init__(self, result: dict, palette: list[str]):
        self._result = result
        speaker_id_map = result.get("speaker_id_map") or {}
        label_to_id = {label: raw_id for raw_id, label in speaker_id_map.items()}

        seconds: dict[str, float] = {raw_id: 0.0 for raw_id in speaker_id_map}
        for seg in result.get("segments", []):
            label = seg.get("speaker")
            if label:
                raw_id = label_to_id.get(label, label)
                seconds[raw_id] = seconds.get(raw_id, 0.0) + _duration(seg)
        self._seconds = seconds

        by_time = sorted(seconds, key=lambda raw_id: -seconds[raw_id])
        self._colors = {raw_id: palette[i % len(palette)] for i, raw_id in enumerate(by_time)}
        self._order = {raw_id: i for i, raw_id in enumerate(by_time)}

    def speakers(self) -> list[Speaker]:
        """All speakers, ranked by share of speaking time. Merged Diarization
        IDs count as one speaker; a rename alone never changes the order
        (ties keep the order the colours were assigned in)."""
        speaker_id_map = self._result.get("speaker_id_map") or {}
        labels = {speaker_id_map.get(raw_id, raw_id) for raw_id in self._seconds}
        groups = [(label, self._ids_of(label)) for label in labels]
        total = sum(self._seconds.values()) or 1.0
        groups.sort(key=lambda g: (-sum(self._seconds[i] for i in g[1]), self._order[g[1][0]]))
        speakers = []
        for rank, (label, ids) in enumerate(groups):
            seconds = sum(self._seconds[i] for i in ids)
            speakers.append(Speaker(label, tuple(ids), self._colors[ids[0]],
                                    seconds, seconds / total, rank))
        return speakers

    def lanes(self, top: int = TOP_LANES) -> tuple[list[Speaker], list[Speaker]]:
        """Timeline split: the `top` speakers by speaking time get a Speaker
        lane each, everyone else shares the Others lane (empty if nobody)."""
        speakers = self.speakers()
        return speakers[:top], speakers[top:]

    def voice_prints(self, label: str) -> list:
        """Embeddings to enroll when `label` is named: those of its
        Diarization IDs with at least MIN_ENROLL_SECONDS of speech. A
        shorter voice print is too unreliable and would spoil the profile."""
        embeddings = self._result.get("speaker_embeddings") or {}
        return [embeddings[raw_id] for raw_id in self._ids_of(label)
                if self._seconds.get(raw_id, 0.0) >= MIN_ENROLL_SECONDS and raw_id in embeddings]

    def enroll_hint(self, label: str) -> str | None:
        """Why naming `label` won't remember its voice, or None if it will."""
        if self.voice_prints(label):
            return None
        if not any(self._seconds.get(raw_id, 0.0) >= MIN_ENROLL_SECONDS for raw_id in self._ids_of(label)):
            return f"Not remembered: under {MIN_ENROLL_SECONDS:.0f} s of speech"
        return "Not remembered: no voice print"

    def _ids_of(self, label: str) -> list[str]:
        """Diarization IDs currently shown as `label`, most speech first.
        Labels missing from speaker_id_map are their own ID."""
        speaker_id_map = self._result.get("speaker_id_map") or {}
        ids = [raw_id for raw_id, current in speaker_id_map.items() if current == label]
        if not ids and label in self._colors and label not in speaker_id_map:
            ids = [label]
        return sorted(ids, key=lambda raw_id: -self._seconds.get(raw_id, 0.0))

    def color_of(self, label: str) -> str | None:
        """The colour of the speaker shown as `label`; after a merge the one
        of the Diarization ID with the most speech."""
        ids = self._ids_of(label)
        return self._colors[ids[0]] if ids else None

    def tentative_name(self, label: str) -> str | None:
        """The name a Tentative match `label` ("Jonas?") offers for
        confirmation ("Jonas"), or None if `label` isn't one."""
        tentative = self._result.get("tentative_speakers") or {}
        for raw_id in self._ids_of(label):
            name = tentative.get(raw_id)
            if name and label == f"{name}?":
                return name
        return None

    def confirm(self, label: str) -> list[tuple[str, object]]:
        """Accepts the Tentative match `label`: "Jonas?" becomes "Jonas" and
        its voice is enrolled like for a typed name (see apply_name())."""
        name = self.tentative_name(label)
        return self.apply_name(label, name) if name else []

    def apply_name(self, label: str, new_name: str) -> list[tuple[str, object]]:
        """Names `label` (see rename()) and returns the (profile name,
        embedding) pairs to enroll: only this speaker's own voice prints
        (on a merge the other speaker's are already in the profile), and
        never under a profile name ending in "?"."""
        new_name = new_name.strip()
        if not new_name or new_name == label:
            return []
        voice_prints = self.voice_prints(label)
        self.rename(label, new_name)
        profile_name = new_name.rstrip("?").strip()
        if not profile_name:
            return []
        return [(profile_name, embedding) for embedding in voice_prints]

    def rename(self, label: str, new_name: str) -> None:
        """Relabels every segment/word of `label` and its speaker_id_map
        entries. A new_name that is already another speaker's label merges
        the two: both Diarization IDs then map to the same Speaker label.
        A renamed speaker is no Tentative match any more."""
        new_name = new_name.strip()
        if not new_name or new_name == label:
            return
        result = self._result
        tentative = result.get("tentative_speakers") or {}
        for raw_id in self._ids_of(label):
            tentative.pop(raw_id, None)
        for seg in result.get("segments", []):
            if seg.get("speaker") == label:
                seg["speaker"] = new_name
            for word in seg.get("words", []) or []:
                if word.get("speaker") == label:
                    word["speaker"] = new_name
        for word in result.get("word_segments", []) or []:
            if word.get("speaker") == label:
                word["speaker"] = new_name
        speaker_id_map = result.setdefault("speaker_id_map", {})
        for raw_id in self._ids_of(label):
            speaker_id_map[raw_id] = new_name


def _duration(seg: dict) -> float:
    return max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
