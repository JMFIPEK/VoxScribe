"""Transcription internals, split out of voxscribe.transcriber.

`voxscribe.transcriber` stays the public facade and re-exports from here
(docs/adr/0001). Nothing in this package may import torch or whisperx at
module load; heavy imports belong inside functions.
"""
