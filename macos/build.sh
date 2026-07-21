#!/bin/bash
# Baut die macOS-Hilfsprozesse: ScreenCaptureKit (System-Audio-Aufnahme) und
# SpeechAnalyzer (Transkription).
#
# Voraussetzung: Xcode Command Line Tools (`xcode-select --install`).
#
# Nutzung:
#   cd macos
#   ./build.sh
#
# Erzeugt ./SystemAudioCapture und ./SpeechAnalyzerTranscribe - recorder.py
# bzw. transcriber.py suchen die Binaries genau dort.

set -euo pipefail
cd "$(dirname "$0")"

echo "Baue SystemAudioCapture..."
swiftc -O SystemAudioCapture.swift -o SystemAudioCapture

echo "Baue SpeechAnalyzerTranscribe..."
swiftc -O SpeechAnalyzerTranscribe.swift -o SpeechAnalyzerTranscribe

echo "Fertig: $(pwd)/SystemAudioCapture, $(pwd)/SpeechAnalyzerTranscribe"
echo
echo "Kurzer Test SystemAudioCapture (Ctrl+C zum Stoppen, sollte KEINE"
echo "Fehlermeldung zu fehlenden Berechtigungen zeigen - falls doch:"
echo "Systemeinstellungen > Datenschutz & Sicherheit > Bildschirm- und"
echo "Systemaudioaufnahme > Terminal/VoxScribe erlauben):"
echo "  ./SystemAudioCapture > /tmp/test.raw"
echo
echo "Kurzer Test SpeechAnalyzerTranscribe (mit einer 16kHz-Mono-WAV-Datei):"
echo "  ./SpeechAnalyzerTranscribe /pfad/zu/audio.wav de-DE"
