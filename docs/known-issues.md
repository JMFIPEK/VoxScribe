# Known issues and open follow-ups

## Loudspeaker setup: echo on the microphone track

**Status:** found 2026-10-18 on macOS, not investigated further, not fixed.

In Call mode with Loudspeaker setup, the room microphone hears the speakers, so
the microphone track contains the Playback audio as an echo.

Measured with a YouTube video playing and nobody speaking:

- Cross-correlation of microphone vs. playback track peaks at 0.35 with a
  ~299 ms delay.
- The microphone track alone transcribes the whole video (average word
  confidence 0.95, playback track alone: 0.97).
- The mixed `<name>.wav` - what gets transcribed - came out garbled in places
  ("Wolke besuchen. Jetzt ist ..." instead of "Onkel Wolk besuchen. Ja, es
  ist ..."), because two copies of the same speech 300 ms apart are summed.

**Open questions / next steps**

1. How does `voxscribe/tracks.py` (`split_speakers_by_track`) treat Loudspeaker
   setup? It classifies segments by per-track level, so echo on the
   microphone track may be misattributed to the local speaker.
2. Echo cancellation: subtract the playback track from the microphone track
   (adaptive filter, ~300 ms delay) before mixing and before the speaker split.
3. Check whether Windows (WASAPI) shows the same behavior.
