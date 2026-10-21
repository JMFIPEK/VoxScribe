# VoxScribe

Desktop and CLI tool that records meetings and turns audio into speaker-labelled transcripts, using a configurable engine for the speech-to-text step.

## Language

### Recording

**Playback audio**:
Everything the computer plays through its speakers or headphones, in a meeting mainly the remote participants' voices.
_Avoid_: System audio, loopback, outgoing audio, incoming audio

**Microphone audio**:
Everything a microphone picks up in the user's room, mainly the user's own voice.
_Avoid_: Mic input, outgoing audio, incoming audio

**Recording mode**:
What a recording captures: Playback only (Playback audio alone), Call (Playback audio and Microphone audio as separate tracks), or In room (Microphone audio alone).
_Avoid_: Source, input

**Setup**:
How a Call is heard and spoken: Headset (headphones and their own microphone, so the two tracks stay clean) or Loudspeaker (speakers and a room microphone, so the remote voices also reach the microphone).
_Avoid_: Device type, hardware mode

**Echo**:
Playback audio that the microphone picks up again, so remote voices also land in the Microphone audio; happens with a Loudspeaker Setup.
_Avoid_: Bleed, crosstalk, feedback

### Distribution

**Release**:
A version deliberately published to users under a version tag such as v1.1.0; installs only ever move from one Release to a newer one.
_Avoid_: Build, version (when the published one is meant), latest commit

**Update**:
Moving an install from its current Release to a newer one, offered by the app and carried out on the next start.
_Avoid_: Upgrade, sync, pull

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
