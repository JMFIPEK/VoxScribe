---
status: accepted
---

# Echo is cancelled offline at transcription time, only for a Loudspeaker Setup

In a Call with a Loudspeaker Setup, the microphone picks up the Playback audio again (Echo). That smears the mono mix and makes the track split attribute remote speech to the local speaker. A Call already stores both tracks in `<name>.tracks.flac`, and the playback track is exactly the far-end reference that echo cancellation needs. So echo is cancelled after recording, while transcribing: the delay between playback and microphone track is estimated block by block (it drifts with the two device clocks), the playback track is lined up accordingly, and both are run through WebRTC's echo canceller (`livekit` AudioProcessingModule), and the mix going into the pipeline is rebuilt from the cleaned microphone track. The raw files on disk stay untouched.

## Considered Options

- **Live echo cancellation while recording.** Rejected: hard real-time constraints and WASAPI timing drift, and every mistake would be baked into the saved audio.
- **Muting the microphone track whenever playback is active.** Rejected: it drops double-talk, which is when the local speaker talks over a remote one.
- **Speex (`pyaec`) or a self-written numpy adaptive filter.** Kept as fallbacks: lighter dependencies but weaker on double-talk. The canceller sits behind one function, so it can be swapped.

## Consequences

- Recordings without a stored Setup (sidecars from before the Setup existed) and Headset Calls get no echo cancellation.
- Better cancellation can be re-run on old recordings, because the raw tracks are kept.
- The saved `<name>.wav` still contains the Echo when listened to.
- It relies on the recorder keeping the two tracks in sync apart from slow clock drift. The recorder had offset jumps (Windows silence padding, macOS start offset, silently dropped buffers), which are fixed separately and before this work. Sync hints are deliberately not stored in the sidecar: the canceller's own delay estimate is the only source of truth.
