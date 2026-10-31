#!/bin/bash
# Builds the macOS helper processes: ScreenCaptureKit (system-audio recording),
# MicCapture (microphone with voice processing, for Loudspeaker Calls) and
# SpeechAnalyzer (transcription).
#
# Requirement: Xcode Command Line Tools (`xcode-select --install`).
#
# Usage:
#   cd macos
#   ./build.sh
#
# Produces ./SystemAudioCapture, ./MicCapture and ./SpeechAnalyzerTranscribe - recorder.py
# and transcriber.py look for the binaries exactly there.

set -euo pipefail
cd "$(dirname "$0")"

echo "Building SystemAudioCapture..."
swiftc -O SystemAudioCapture.swift -o SystemAudioCapture

echo "Building MicCapture..."
swiftc -O MicCapture.swift -o MicCapture

echo "Building SpeechAnalyzerTranscribe..."
swiftc -O SpeechAnalyzerTranscribe.swift -o SpeechAnalyzerTranscribe

echo "Done: $(pwd)/SystemAudioCapture, $(pwd)/MicCapture, $(pwd)/SpeechAnalyzerTranscribe"
echo
echo "Quick test SystemAudioCapture (Ctrl+C to stop, should show NO error"
echo "about missing permissions - if it does: System Settings > Privacy &"
echo "Security > Screen & System Audio Recording > allow Terminal/VoxScribe):"
echo "  ./SystemAudioCapture > /tmp/test.raw"
echo
echo "Quick test SpeechAnalyzerTranscribe (with a 16kHz mono WAV file):"
echo "  ./SpeechAnalyzerTranscribe /path/to/audio.wav en-US"
