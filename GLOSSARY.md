# VoxScribe

Desktop and CLI tool that records meetings and turns audio into speaker-labelled transcripts, using a configurable engine for the speech-to-text step.

## Language

### Transcription engines

**Provider**:
A stored remote-server configuration (name, address, key, model) that a user adds in Settings and may mark as default.
_Avoid_: Server, endpoint, remote model

**Backend**:
Any engine that turns audio into text segments: local WhisperX, a Provider, Intel Arc via OpenVINO, Parakeet, or Apple's SpeechAnalyzer.
_Avoid_: Engine, provider (a Provider is one kind of Backend, not a synonym)

**Model ID**:
The full string that selects a Backend together with its model, for example a Provider reference or an OpenVINO device plus size.
_Avoid_: Model name, model string

**Model size**:
Only the Whisper size part of a Model ID, such as medium or large-v3.
_Avoid_: Model (when only the size is meant)

### Speakers

**Diarization ID**:
The raw anonymous label the diarization step assigns to a voice, such as SPEAKER_00.
_Avoid_: Speaker ID, raw speaker

**Speaker label**:
The name currently shown for a speaker in a transcript, either a Diarization ID or a recognized person's name.
_Avoid_: Speaker name, speaker ID
