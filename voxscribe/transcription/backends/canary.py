"""Candidate (#57/#58): NVIDIA Canary-1B-v2 via onnx-asr on ONNX Runtime (CPU).

Reachable by Model ID (`canary:1b-v2`) from the CLI and the benchmark, not
offered in the GUI. Same approach as Parakeet (onnx_asr_common): Silero VAD
cuts the audio into speech segments of at most 25s, which become the
WhisperX-style segments the regular alignment/diarization steps process.
"""

import numpy as np

from voxscribe.transcription.audio import SAMPLE_RATE
from voxscribe.transcription.backend import TranscribeRequest
from voxscribe.transcription.backends.onnx_asr_common import (
    CANARY_VARIANTS, bundled_model_dir, load_vad_asr, recognize_segments,
)
from voxscribe.transcription.model_ids import CANARY_MODEL_PREFIX


# Canary has no language detection: onnx-asr's prompt names the speech
# language and the output language, and defaults both to English - German
# speech then comes out *translated* into English. Auto-detect therefore runs
# one pass per language and keeps, for each speech segment, the version the
# model is most confident in (mean token log-probability). One language for
# the whole recording, chosen from its start, translated the other half of a
# German-then-English meeting (48.7 % WER on such a take, #58). Both passes
# use the same Silero VAD on the same audio, so their segments match one to
# one. Only the study's languages are tried.
CANARY_AUTO_LANGUAGES = ("de", "en")


def _mean_logprob(result) -> float:
    logprobs = getattr(result, "logprobs", None) or []
    return float(np.mean(logprobs)) if len(logprobs) else float("-inf")


def _recognize_scored(asr, audio: np.ndarray, language: str, on_progress=None) -> list[tuple]:
    """[(start, end, text, mean log-probability)] for every VAD segment, also
    the empty ones (so the passes of two languages line up)."""
    duration = len(audio) / SAMPLE_RATE
    out = []
    for seg in asr.with_timestamps().recognize(audio, sample_rate=SAMPLE_RATE, language=language,
                                               target_language=language, pnc=True):
        out.append((float(seg.start), float(seg.end), (seg.text or "").strip(), _mean_logprob(seg)))
        if on_progress and duration > 0:
            on_progress(min(seg.end / duration, 1.0))
    return out


def _transcribe_per_segment_language(asr, audio: np.ndarray, on_progress=None) -> dict:
    """Auto-detect: see CANARY_AUTO_LANGUAGES. Returns segments tagged with
    their own "language" (the alignment step aligns each language with its
    own model) and, as "language", the one with the most speech."""
    n = len(CANARY_AUTO_LANGUAGES)
    passes = {}
    for k, lang in enumerate(CANARY_AUTO_LANGUAGES):
        def progress(frac, k=k):
            if on_progress:
                on_progress((k + frac) / n)
        passes[lang] = _recognize_scored(asr, audio, lang, progress)

    first = passes[CANARY_AUTO_LANGUAGES[0]]
    if any([(s, e) for s, e, _, _ in p] != [(s, e) for s, e, _, _ in first] for p in passes.values()):
        # Shouldn't happen (same VAD, same audio); then one language overall.
        best = max(passes, key=lambda lang: np.mean([sc for *_, sc in passes[lang]] or [-1e9]))
        print(f"     [WARNING] Canary: the language passes cut the audio differently - using '{best}' throughout")
        chosen = [(seg, best) for seg in passes[best]]
    else:
        chosen = []
        for i in range(len(first)):
            lang = max(CANARY_AUTO_LANGUAGES, key=lambda l: passes[l][i][3])
            chosen.append((passes[lang][i], lang))

    segments, speech = [], {lang: 0.0 for lang in CANARY_AUTO_LANGUAGES}
    for (start, end, text, _), lang in chosen:
        if text:
            segments.append({"start": start, "end": end, "text": text, "language": lang})
            speech[lang] += end - start
    main = max(speech, key=speech.get) if segments else "en"
    switches = sum(1 for a, b in zip(segments, segments[1:]) if a["language"] != b["language"])
    print(f"     Canary languages per segment: " + ", ".join(
        f"{lang} {sec:.0f}s" for lang, sec in speech.items()) + f" ({switches} switches)")
    return {"segments": segments, "language": main}


def transcribe_canary(audio: np.ndarray, language: str | None, model_size: str,
                      on_progress=None, on_segments=None) -> dict:
    """Transcribes via NVIDIA Canary-1B-v2 (25 European languages) through
    onnx-asr. Speech and output language are both set to `language` (so it
    transcribes, never translates); without one, each speech segment gets
    its own language, see CANARY_AUTO_LANGUAGES."""
    variant = model_size[len(CANARY_MODEL_PREFIX):]
    name = CANARY_VARIANTS.get(variant)
    if name is None:
        raise ValueError(f"Unknown Canary model: {model_size}")

    asr = load_vad_asr(name, bundled_model_dir("canary", variant))
    if not language:
        result = _transcribe_per_segment_language(asr, audio, on_progress)
        if on_segments and result["segments"]:
            on_segments([dict(s) for s in result["segments"]])
        del asr
        return result
    try:
        segments = recognize_segments(asr, audio, on_progress, on_segments,
                                      language=language, target_language=language, pnc=True)
    except KeyError as e:  # onnx-asr looks the language token up in Canary's vocabulary
        raise ValueError(f"Canary does not support the language '{language}'") from e
    del asr
    return {"segments": segments, "language": language}


class CanaryBackend:
    name = "Canary"
    skips_alignment = False

    def describe(self, model_id: str, device: str) -> str:
        return f"Device: transcription=Canary (ONNX Runtime, CPU), alignment/diarization={device}"

    def transcribe(self, request: TranscribeRequest) -> dict:
        print("1/3  Transcribing via Canary (ONNX Runtime)...")
        request.progress(0.0, "Loading audio...")
        audio = request.audio.load()
        request.progress(0.05, "Loading Canary model... (first use downloads it)")

        def _progress(frac):
            request.progress(0.05 + frac * 0.35, f"Transcribing... {frac * 100:.0f}%")

        return transcribe_canary(audio, request.language, request.model_id,
                                 on_progress=_progress, on_segments=request.on_segments)
