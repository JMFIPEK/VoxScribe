# Refactor plan: transcriber into backends

Branch: `refactor/transcriber-backends` (from `main`). Vocabulary: see `GLOSSARY.md`. Decision record: `docs/adr/0001-backend-registry-and-transcriber-facade.md`.

## Goals

- Make `voxscribe/transcriber.py` (1587 lines, 45 functions) easier to change: a new Backend should not mean another branch inside `transcribe()`.
- Make the logic testable.
- Python stays. No behavior change: same outputs, same Model IDs, same GUI behavior, same progress percentages. Bugs found along the way get separate commits.

## Constraints

- **Lazy imports.** Nothing the GUI imports eagerly may import torch or whisperx at module load (cold import takes minutes). A test enforces it.
- **Facade stays.** `voxscribe.transcriber` re-exports the current public names so `main.py`, `voxscribe/cli.py` and `qt_app/` need no changes.
- **Model IDs unchanged**: `server:`, `openvino:`, `parakeet:`, `apple:`; no prefix means local WhisperX.
- Hardware-only paths (Apple, OpenVINO, real provider calls) are not unit-tested.

## Steps

Each step is one small commit (or a few), and the app must work after every one.

1. **Test setup.** Add a pytest dev dependency group to `pyproject.toml`, create `tests/`, make `uv run pytest` pass on an empty suite.
2. **Characterization tests, pure logic only.** `_find_quiet_split`, `_guess_language_from_text`, `_normalize_language_code`, `tracks.split_speakers_by_track`, the speaker-label merge in `transcribe_multi`, the Model ID prefix helpers, `format_transcript`/`_to_srt`/`_format_time`. Record current behavior, do not fix it.
3. **Lazy-import test.** Import the facade and the cheap modules, assert `torch` and `whisperx` are not in `sys.modules`.
4. **Create the package skeleton** `voxscribe/transcription/` and make `voxscribe/transcriber.py` re-export from it (empty at first, then growing).
5. **Extract audio loading** (`load_audio_*`) and **output formats** (`format_transcript`, `save_transcript`, SRT helpers). Lowest risk, no shared state.
6. **Define the Backend interface and registry.** Input: audio, language, model, progress callback. Output: `{"segments", "language"}`. Registry maps Model ID prefix to Backend.
7. **Extract backends one at a time**, running the tests after each: Parakeet, OpenVINO, Apple, remote Provider (largest: chunking, retry, language refine), then local WhisperX.
8. **Extract alignment and diarization** (`_align_by_language`, diarization step, `prewarm_diarization`) and **speaker handling** (track split, voice-print matching, label merge).
9. **Split `transcribe()`** (about 410 lines) into named steps: resolve devices, run Backend, align, diarize, split by track, match voice-prints, Provider-failure fallback to a local model. `transcribe()` only orchestrates. `transcribe_multi()` keeps calling it unchanged.
10. **Update docs**: the Core Modules table and the transcriber description in `CLAUDE.md`, plus any mentions in `docs/`.

## Out of scope (for now)

- `voxscribe/recorder.py` (1329 lines): later, as its own plan.
- Any change to Model ID strings, saved settings, or GUI layout.

## Done when

- `uv run pytest` is green.
- A real transcription gives the same result as before on the same file for at least one local run and one Provider run (manual check).
- The GUI starts and a transcription runs end to end.

## Status

Steps 1-10 are done on this branch. Deviations from the plan and open items:

- **Backend interface.** Not the plain `(audio, language, model, progress)` of step 6: a Backend takes a `TranscribeRequest` (audio path plus an `AudioSource` that decodes once, language, Model ID, `progress(fraction, message)`, batch size, device, ad-hoc Provider credentials), because Apple needs the file path and Provider needs credentials. Each Backend owns its slice of the 0-40% progress range, so percentages and messages are unchanged. Backends also expose `name`, `skips_alignment` (Apple) and `describe()` (the "Device:" line).
- **Provider fallback.** A Provider request failure is raised as `FallbackRequested`; `transcribe()` catches it and retries with `default_local_model_size()`. Unknown-Provider and other errors still propagate.
- **Facade is now lazy.** `voxscribe.transcriber` imports neither torch nor whisperx (the lazy-import test is a normal test now). `prewarm_diarization()` imports torch, whisperx and pyannote, so the GUI's idle-thread prewarm still absorbs the cold import.
- **Small console/status differences.** The "segments detected" line now prints the language for every Backend (it was Parakeet only); the "Alignment skipped" console line uses the Backend's name. Progress percentages and GUI status texts are unchanged.
- **Known bug, not fixed here:** `transcribe_multi` renames the speaker on segments and `word_segments` but not on the words nested inside a segment (pinned by a characterization test).
- **Manual checks.** Local run verified identical to the pre-refactor code (result and progress sequence, Whisper base, diarization, track sidecar). Still to do by hand: a Provider run, the Apple/OpenVINO/Parakeet paths on their hardware, and a full GUI transcription.
