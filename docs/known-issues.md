# Known issues and open follow-ups

## Loudspeaker setup: echo on the microphone track

**Status:** found 2026-10-18 on macOS, fixed (issues #22-#24, #36, #39). Tuning
on a new Mac recording is still open (#26).

In Call mode with Loudspeaker setup, the room microphone hears the speakers, so
the microphone track contains the Playback audio as an echo. First measured
with a YouTube video playing and nobody speaking (macOS): the microphone track
alone transcribed the whole video, and the mix - two copies of the same speech
~300 ms apart - came out garbled in places.

What happens now:

- **Echo removal** (`voxscribe/echo.py`, ADR 0003) cleans the microphone track
  of a Loudspeaker Call before the track split and before the mix every
  Backend transcribes (Apple SpeechAnalyzer via a temporary file).
- **Track split per run of words** (#36): long segments that hold both sides
  are cut at the side change, so remote speech isn't left with the local
  speaker.
- **Quiet microphone** (#39): if the tracks' speech levels are 12 dB or more
  apart, the quieter one is boosted by up to 20 dB before mixing for
  transcription.

Measured on real Loudspeaker Calls:

| | Mac, built-in mic (66 s, 2026-10-21, before the recorder fixes) | Windows, Intel Smart Sound mic array (211 s, 2026-10-21) |
|---|---|---|
| Raw mic while only the remote side talks | Echo clearly audible | already -69.5 dB, 74 % digital zeros: the mic's own noise suppression |
| `cancel_echo`, remote-only passages | -14.5 dB | -13.1 dB on top |
| `cancel_echo`, local speech | -1.5 dB | -0.6 dB |
| Doubled remote speech in the transcript | - | none |
| Remote words labelled as the local speaker | 34 of 49 (before #36) | 9 of 198, all fragments: a 1 s piece at a turn, 4 words of double-talk inside a local sentence, single words pyannote labels on its own |

Notes:

- Some laptop mic arrays (Intel Smart Sound, Windows Voice Clarity) gate
  everything that isn't speech to digital zeros. Clicks never reach the
  Microphone track; use spoken words as test signals.
- The remaining misattributed fragments would need a per-word choice between
  remote speakers; lowering the split's thresholds instead cuts local
  sentences apart in double-talk.
