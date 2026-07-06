#!/bin/bash
# Baut den ScreenCaptureKit-Hilfsprozess fuer System-Audio-Aufnahme unter macOS.
#
# Voraussetzung: Xcode Command Line Tools (`xcode-select --install`).
#
# Nutzung:
#   cd macos
#   ./build.sh
#
# Erzeugt ./SystemAudioCapture - recorder.py sucht das Binary genau dort
# (siehe _macos_system_audio_binary_path() in recorder.py).

set -euo pipefail
cd "$(dirname "$0")"

echo "Baue SystemAudioCapture..."
swiftc -O SystemAudioCapture.swift -o SystemAudioCapture

echo "Fertig: $(pwd)/SystemAudioCapture"
echo
echo "Kurzer Test (Ctrl+C zum Stoppen, sollte KEINE Fehlermeldung zu fehlenden"
echo "Berechtigungen zeigen - falls doch: Systemeinstellungen > Datenschutz &"
echo "Sicherheit > Bildschirm- und Systemaudioaufnahme > Terminal/VoxScribe erlauben):"
echo "  ./SystemAudioCapture > /tmp/test.raw"
