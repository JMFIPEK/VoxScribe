---
status: accepted
---

# Backends are chosen by model-ID prefix through a registry, behind a transcriber facade

`voxscribe/transcriber.py` grew into one large file where every Backend (local WhisperX, Provider, OpenVINO, Parakeet, Apple) is wired into `transcribe()` by hand. We split it into a package with one module per Backend and a registry that maps the existing Model ID prefixes to a Backend. `voxscribe.transcriber` stays as a thin facade that re-exports today's public names.

## Considered Options

- **Clean break with a new API and new Model IDs.** Rejected: Model IDs are saved in user settings and used by the GUI and CLI, so changing them would invalidate existing setups and force edits across `qt_app/` and `voxscribe/cli.py`.
- **Facade plus unchanged Model IDs (chosen).** The refactor stays invisible to callers, so it can land in small steps with no behavior change.

## Consequences

- The facade looks redundant, but removing it breaks `qt_app/controllers.py`, `qt_app/pages/transcribe_page.py` and `voxscribe/cli.py`, which import from `voxscribe.transcriber`.
- Importing the facade or any module the GUI loads eagerly must not import torch or whisperx at module load, because a cold import takes minutes (see CLAUDE.md, PySide6/Qt GUI).
