"""Benchmark: several Backends on the same recordings, side by side (#56).

`python main.py benchmark -i FILE [--reference REF] [--models ...]` runs every
Backend whose model is on this machine (plus the configured Providers) on the
first --seconds of each recording and prints one comparison: load and
transcription time, alignment time where the Backend needs it, x realtime,
word error rate against a reference text, what the Backend returns, how far
its own word timings are from wav2vec2's, and each transcript. For a Call it
can also transcribe the two tracks separately and merge them by time.

Only the steps that differ between Backends are measured: speaker
identification is the same for all and never runs here. Nothing is
downloaded unless asked for (--download).

How to run it, and how to get a trustworthy reference: docs/benchmark-with-agent.md.
"""

from __future__ import annotations

import contextlib
import difflib
import io
import os
import platform
import re
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000
WARMUP_SECONDS = 1.0
TRACK_MODES = ("mix", "separate", "both")

# Must match the Model IDs the app uses (voxscribe/transcription/model_ids.py,
# qt_app/constants.py).
LOCAL_MODELS = ["large-v3", "large-v2", "medium", "base",
                "openvino:GPU:medium", "parakeet:tdt-0.6b-v3", "apple:speechanalyzer"]


# --- Word error rate ---------------------------------------------------------

_PUNCTUATION = re.compile(r"[^\w\s']", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    """Words for comparing: casefolded, punctuation removed (apostrophes
    kept, "it's" stays one word). Numbers are compared as written - "8 Uhr"
    and "acht Uhr" differ."""
    text = _PUNCTUATION.sub(" ", text.casefold().replace("’", "'"))
    return [w.strip("'") for w in text.split() if w.strip("'")]


@dataclass
class WordErrors:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def rate(self) -> float:
        errors = self.substitutions + self.deletions + self.insertions
        return errors / self.reference_words if self.reference_words else float(errors > 0)


def word_errors(reference: str, hypothesis: str) -> WordErrors:
    """Word-level edit distance (Levenshtein) between reference and
    hypothesis, split into substitutions, deletions and insertions."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    # cost[j] = (total, subs, dels, ins) for ref[:i] vs hyp[:j]
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub, dele, ins = prev[j - 1], prev[j], cur[j - 1]
            best = min(sub, dele, ins, key=lambda c: c[0])
            if best is sub:
                cur.append((sub[0] + 1, sub[1] + 1, sub[2], sub[3]))
            elif best is dele:
                cur.append((dele[0] + 1, dele[1], dele[2] + 1, dele[3]))
            else:
                cur.append((ins[0] + 1, ins[1], ins[2], ins[3] + 1))
        prev = cur
    _, subs, dels, ins = prev[-1]
    return WordErrors(subs, dels, ins, len(ref))


# --- Word timings ------------------------------------------------------------

def segment_words(segments: list[dict]) -> list[dict]:
    """All words with a start time, in time order."""
    words = [w for s in segments for w in s.get("words", []) if w.get("start") is not None]
    return sorted(words, key=lambda w: w["start"])


def word_timing_offset(own: list[dict], aligned: list[dict]) -> float | None:
    """Median |start difference| in seconds between a Backend's own word
    timings and wav2vec2's for the same text, over the words both have
    (matched in order by their normalized spelling); None if none match."""
    a = [(normalize_words(w.get("word", "")) or [""])[0] for w in own]
    b = [(normalize_words(w.get("word", "")) or [""])[0] for w in aligned]
    diffs = []
    for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        for k in range(block.size):
            if a[block.a + k]:
                diffs.append(abs(own[block.a + k]["start"] - aligned[block.b + k]["start"]))
    return statistics.median(diffs) if diffs else None


def merge_by_time(*segment_lists: list[dict]) -> list[dict]:
    """Segments of separately transcribed tracks as one transcript, in time order."""
    return sorted((s for segs in segment_lists for s in segs), key=lambda s: s.get("start", 0.0))


def transcript_text(segments: list[dict]) -> str:
    return " ".join(s.get("text", "").strip() for s in segments if s.get("text", "").strip())


# --- Which models are on this machine ----------------------------------------

def _hf_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import scan_cache_dir
        return any(r.repo_id == repo_id for r in scan_cache_dir().repos)
    except Exception:
        return False


def model_presence(model_id: str) -> str | None:
    """None if `model_id` can run here without downloading anything, else
    why not (shown in the skipped list)."""
    from voxscribe.paths import get_bundled_models_dir
    from voxscribe.transcription import model_ids as m

    bundled = get_bundled_models_dir() or ""
    if m.is_remote_model(model_id):
        from voxscribe.providers import get_provider
        return None if get_provider(model_id[len(m.REMOTE_MODEL_PREFIX):]) else "no such Provider"
    if m.is_apple_model(model_id):
        if sys.platform != "darwin":
            return "macOS only"
        from voxscribe.transcription.backends.apple import _apple_speechanalyzer_binary_path
        return None if os.path.isfile(_apple_speechanalyzer_binary_path()) else \
            "helper not built (macos/build.sh)"
    if m.is_parakeet_model(model_id):
        variant = model_id[len(m.PARAKEET_MODEL_PREFIX):]
        if os.path.isdir(os.path.join(bundled, "parakeet", variant)):
            return None
        return None if _hf_cached(f"istupakov/parakeet-{variant}-onnx") else "not downloaded"
    if m.is_openvino_model(model_id):
        from voxscribe.hardware_detect import _bundled_openvino_model_exists, get_openvino_devices
        ov_device, size = m._parse_openvino_model(model_id)
        if ov_device not in (get_openvino_devices() or []):
            return f"no OpenVINO {ov_device} device"
        return None if _bundled_openvino_model_exists(size) else "not exported (download_models.py)"
    if ":" in model_id:
        return None  # another registered prefix (e.g. a candidate): let it try
    # Local WhisperX size
    if os.path.isdir(os.path.join(bundled, "whisper", model_id)):
        return None
    try:
        from faster_whisper.utils import _MODELS
    except Exception:
        return "faster-whisper not installed"
    repo = _MODELS.get(model_id)
    if repo is None:
        return "unknown Whisper size"
    return None if _hf_cached(repo) else "not downloaded"


def alignment_presence(language: str) -> str | None:
    """None if the wav2vec2 alignment model for `language` is on disk."""
    try:
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
    except Exception:
        return "whisperx not installed"
    from voxscribe.paths import get_bundled_models_dir
    bundled = get_bundled_models_dir()
    if bundled and os.path.isdir(os.path.join(bundled, "align")):
        return None
    if language in DEFAULT_ALIGN_MODELS_TORCH:
        import torch
        import torchaudio
        bundle = getattr(torchaudio.pipelines, DEFAULT_ALIGN_MODELS_TORCH[language])
        name = os.path.basename(bundle._path)
        path = os.path.join(torch.hub.get_dir(), "checkpoints", name)
        return None if os.path.isfile(path) else f"alignment model for '{language}' not downloaded"
    if language in DEFAULT_ALIGN_MODELS_HF:
        return None if _hf_cached(DEFAULT_ALIGN_MODELS_HF[language]) else \
            f"alignment model for '{language}' not downloaded"
    return f"no alignment model for '{language}'"


def default_models() -> list[str]:
    """The app's local models plus every configured Provider."""
    from voxscribe.providers import list_providers
    from voxscribe.transcription.model_ids import remote_model_id
    return LOCAL_MODELS + [remote_model_id(p["id"]) for p in list_providers()]


# --- Running -----------------------------------------------------------------

@dataclass
class Run:
    """One Backend on one recording in one track mode."""
    model_id: str
    mode: str  # "mix" or "separate"
    status: str = "ok"  # "ok", or why it failed / was skipped
    load_s: float | None = None
    transcribe_s: float | None = None
    align_s: float | None = None
    audio_s: float = 0.0
    returns: str = ""
    language: str | None = None
    timing_offset_s: float | None = None
    errors: WordErrors | None = None
    text: str = ""

    @property
    def realtime(self) -> float | None:
        total = sum(t for t in (self.load_s, self.transcribe_s, self.align_s) if t is not None)
        return self.audio_s / total if self.status.startswith("ok") and total > 0 else None


@dataclass
class Recording:
    path: str
    reference: str | None
    runs: list[Run] = field(default_factory=list)


def _write_wav(samples: np.ndarray) -> str:
    import soundfile as sf
    fd, path = tempfile.mkstemp(prefix="voxscribe-bench-", suffix=".wav")
    os.close(fd)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")
    return path


def _clip_file(path: str, seconds: float) -> str:
    """A temporary WAV of the first `seconds` of `path` (the warm-up run that
    stands for the model's load time)."""
    import soundfile as sf
    samples, _ = sf.read(path, dtype="float32", frames=int(seconds * SAMPLE_RATE))
    return _write_wav(samples)


def _describe_returns(results: list[dict]) -> str:
    """What the Backend returned, e.g. "9 segments, words" (per track "10+9")."""
    counts = "+".join(str(len(r.get("segments", []))) for r in results)
    has_words = any(s.get("words") for r in results for s in r.get("segments", []))
    return f"{counts} segments{', words' if has_words else ''}"


class Benchmark:
    """Runs Backends on recordings. `registry`, `presence` and `align` are
    the real ones by default; tests pass fakes."""

    def __init__(self, registry=None, presence: Callable[[str], str | None] = model_presence,
                 align: Callable | None = None, align_presence=alignment_presence,
                 device: str | None = None, log: Callable[[str], None] = print):
        if registry is None:
            from voxscribe.transcription import backends  # noqa: F401 (registers them)
            from voxscribe.transcription.registry import default_registry
            registry = default_registry
        self.registry = registry
        self.presence = presence
        self.align = align or self._wav2vec2_align
        self.align_presence = align_presence
        self._device = device
        self.log = log

    @property
    def device(self) -> str:
        if self._device is None:
            from voxscribe.transcription.devices import resolve_device
            self._device = resolve_device(None)
        return self._device

    def run(self, paths: list[str], references: list[str | None], models: list[str],
            seconds: float = 120, language: str | None = None, tracks: str = "mix",
            download: bool = False) -> tuple[list[Recording], list[tuple[str, str]]]:
        """Returns (recordings with their runs, skipped (model, reason))."""
        skipped, runnable = [], []
        for model_id in models:
            reason = None if download else self.presence(model_id)
            (skipped.append((model_id, reason)) if reason else runnable.append(model_id))
        recordings = []
        for path, reference in zip(paths, references):
            rec = Recording(path, reference)
            inputs = self._inputs(path, seconds, tracks)
            try:
                for model_id in runnable:
                    for mode, files in inputs.items():
                        self.log(f"  {os.path.basename(path)}: {model_id} ({mode})...")
                        run = self._run_one(model_id, mode, files, language, download)
                        if reference is not None and run.status.startswith("ok"):
                            run.errors = word_errors(reference, run.text)
                        rec.runs.append(run)
                        self.log(f"    -> {run.status}")
            finally:
                for files in inputs.values():
                    for f, _ in files:
                        os.remove(f)
            recordings.append(rec)
        return recordings, skipped

    def _inputs(self, path: str, seconds: float, tracks: str) -> dict[str, list[tuple[str, float]]]:
        """{mode: [(temporary 16kHz WAV, duration)]} - "mix" is what the app
        transcribes (a Call's cleaned/balanced mix), "separate" each track."""
        from voxscribe.transcription.audio import AudioSource
        source = AudioSource(path)
        limit = int(seconds * SAMPLE_RATE) if seconds else None

        def clip(samples):
            samples = samples[:limit] if limit else samples
            return _write_wav(samples), len(samples) / SAMPLE_RATE

        inputs = {}
        if tracks in ("mix", "both"):
            inputs["mix"] = [clip(source.load())]
        if tracks in ("separate", "both"):
            both = source.tracks()
            if both is None:
                self.log(f"  {os.path.basename(path)}: no separate tracks (not a Call) - "
                         "only the mix is compared.")
                if "mix" not in inputs:
                    inputs["mix"] = [clip(source.load())]
            else:
                inputs["separate"] = [clip(both[0]), clip(both[1])]
        return inputs

    def _run_one(self, model_id: str, mode: str, files: list[tuple[str, float]],
                 language: str | None, download: bool) -> Run:
        run = Run(model_id, mode, audio_s=files[0][1])
        try:
            backend = self.registry.resolve(model_id)
        except Exception as e:
            run.status = f"failed: {e}"
            return run
        try:
            warmup = _clip_file(files[0][0], WARMUP_SECONDS)
            try:
                run.load_s = self._timed(backend, model_id, warmup, language)[1]
            finally:
                os.remove(warmup)
            results, total = [], 0.0
            for f, _ in files:
                result, seconds = self._timed(backend, model_id, f, language)
                results.append(result)
                total += seconds
            run.transcribe_s = max(0.0, total - run.load_s * len(files))
        except Exception as e:
            run.status = f"failed: {_first_line(e)}"
            return run

        run.returns = _describe_returns(results)
        run.language = results[0].get("language")
        own_words = [segment_words(r.get("segments", [])) for r in results]

        aligned = []
        if not backend.skips_alignment or any(own_words):
            missing = self.align_presence(run.language or "en") if not download else None
            if missing and not backend.skips_alignment:
                run.status = f"ok, not aligned ({missing})"
            elif not missing:
                t0 = time.time()
                for (f, _), r in zip(files, results):
                    aligned.append(self.align(r, f, run.language or "en", self.device))
                if not backend.skips_alignment:
                    run.align_s = time.time() - t0
        final = aligned if not backend.skips_alignment and aligned else results
        if backend.skips_alignment and aligned:
            offsets = [word_timing_offset(o, segment_words(a.get("segments", [])))
                       for o, a in zip(own_words, aligned)]
            offsets = [o for o in offsets if o is not None]
            run.timing_offset_s = statistics.median(offsets) if offsets else None
        run.text = transcript_text(merge_by_time(*[r.get("segments", []) for r in final]))
        return run

    def _timed(self, backend, model_id: str, path: str, language: str | None) -> tuple[dict, float]:
        """One Backend call on `path`, its output kept out of the report
        (shown only when it fails). A Provider's failure is reported as such,
        never retried with a local model."""
        from voxscribe.transcription.audio import AudioSource
        from voxscribe.transcription.backend import FallbackRequested, TranscribeRequest
        request = TranscribeRequest(audio_path=path, audio=AudioSource(path), language=language,
                                    model_id=model_id, progress=lambda pct, msg: None,
                                    device=self.device)
        captured = io.StringIO()
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(captured):
                result = backend.transcribe(request)
        except FallbackRequested as e:
            raise RuntimeError(f"Provider failed ({e})") from e
        except Exception:
            tail = captured.getvalue().strip().splitlines()[-5:]
            if tail:
                self.log("      " + "\n      ".join(tail))
            raise
        return result, time.time() - t0

    def _wav2vec2_align(self, result: dict, path: str, language: str, device: str) -> dict:
        from voxscribe.transcription.align import align_by_language
        from voxscribe.transcription.audio import load_audio_universal
        plain = {"language": result.get("language") or language,
                 "segments": [{k: v for k, v in s.items() if k != "words"}
                              for s in result.get("segments", [])]}
        with contextlib.redirect_stdout(io.StringIO()):
            return align_by_language(plain, load_audio_universal(path), device, None, False)


def _first_line(e: Exception) -> str:
    return (str(e).strip().splitlines() or [type(e).__name__])[0][:120]


# --- Report ------------------------------------------------------------------

def machine_description() -> str:
    lines = [f"Machine: {platform.system()} {platform.release()} ({platform.machine()}), "
             f"Python {platform.python_version()}"]
    try:
        from voxscribe.hardware_detect import get_cpu_info, get_gpu_info
        cpu = get_cpu_info()
        name = cpu["name"]
        if sys.platform == "darwin":  # platform.processor() only says "arm"
            import subprocess
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip() or name
        lines.append(f"CPU: {name}, {cpu['cores']} cores, RAM {cpu['ram_gb']:.0f} GB")
        gpu = get_gpu_info()
        if gpu and sys.platform != "darwin":  # on a Mac that's the same chip
            lines.append(f"GPU: {gpu.get('name', '?')} ({gpu.get('vram_gb', '?')} GB)")
    except Exception:
        pass
    return "\n".join(lines)


def _fmt(value: float | None, pattern: str) -> str:
    return pattern.format(value) if value is not None else "-"


def format_report(recordings: list[Recording], skipped: list[tuple[str, str]],
                  machine: str = "", device: str = "") -> str:
    out = []
    if machine:
        out.append(machine)
    if device:
        out.append(f"Torch device (alignment): {device}")
    for rec in recordings:
        out.append("")
        out.append(f"## {os.path.basename(rec.path)}"
                   + ("" if rec.reference is not None else "  (no reference: no WER)"))
        out.append("")
        header = ["Model", "Mode", "Load s", "Transcribe s", "Align s", "x realtime", "WER",
                  "Sub/Del/Ins", "Returns", "Lang", "Timing offset", "Status"]
        rows = []
        for r in rec.runs:
            e = r.errors
            rows.append([
                r.model_id, r.mode, _fmt(r.load_s, "{:.1f}"), _fmt(r.transcribe_s, "{:.1f}"),
                _fmt(r.align_s, "{:.1f}"), _fmt(r.realtime, "{:.1f}x"),
                _fmt(e.rate * 100 if e else None, "{:.1f}%"),
                f"{e.substitutions}/{e.deletions}/{e.insertions}" if e else "-",
                r.returns or "-", r.language or "-",
                _fmt(r.timing_offset_s * 1000 if r.timing_offset_s is not None else None,
                     "{:.0f} ms"), r.status])
        out.append(_table(header, rows))
        for r in rec.runs:
            if r.text:
                out.append("")
                out.append(f"### {r.model_id} ({r.mode})")
                out.append(r.text)
    if skipped:
        out.append("")
        out.append("Skipped (not on this machine; --download fetches them): "
                   + ", ".join(f"{m} ({why})" for m, why in skipped))
    return "\n".join(out)


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    line = lambda cells: "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(line(r) for r in rows)])
