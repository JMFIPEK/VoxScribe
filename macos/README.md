# System-Audio-Aufnahme unter macOS

`SystemAudioCapture.swift` ist ein kleiner Kommandozeilen-Helfer, der System-Audio
via Apples ScreenCaptureKit aufnimmt und als rohe 16kHz-Mono-Int16-PCM-Bytes nach
stdout schreibt. `recorder.py` startet ihn als Subprozess (siehe
`_record_system_macos()`), analog dazu, wie unter Windows PyAudioWPatch und unter
macOS/Linux `sounddevice` fuers Mikrofon genutzt werden.

**Status: auf echter Apple-Silicon-Hardware (macOS 26) gebaut, getestet und
debuggt.** Zwei Bugs wurden dabei gefunden und behoben: ein Swift-Kompilierfehler
(String/`Data`-Typkonflikt im Fehlerlogging) und ein Python-Importfehler
(`recorder.py` referenzierte `pyaudio.PyAudio` in einer Typannotation, obwohl
`pyaudio` unter macOS gar nicht importiert wird - das crashte den kompletten
Import bereits beim Programmstart, siehe `from __future__ import annotations`
in `recorder.py`). Nach den Fixes funktioniert die Aufnahme end-to-end:
Berechtigungsabfrage, echte (nicht-stille) Audiodaten, sauberes Beenden in
~10ms ohne haengenden Prozess.

## Voraussetzungen

- macOS 13 (Ventura) oder neuer
- Xcode Command Line Tools: `xcode-select --install`
- Berechtigung "Bildschirm- und Systemaudioaufnahme" fuer den Prozess, der das
  Binary ausfuehrt (Terminal beim manuellen Testen, spaeter die gepackte
  VoxScribe-App) unter Systemeinstellungen > Datenschutz & Sicherheit

## Bauen

```bash
cd macos
./build.sh
```

Erzeugt `macos/SystemAudioCapture` - genau dort sucht `recorder.py` danach.

## Manuell testen (unabhaengig von VoxScribe)

```bash
cd macos
./SystemAudioCapture > /tmp/test.raw
# ein paar Sekunden laufen lassen waehrend z.B. Musik/ein Video laeuft, dann Ctrl+C
```

Die Datei `/tmp/test.raw` enthaelt danach rohe 16-bit-PCM-Samples (kein WAV-Header!).
Zum Anhoeren z.B. mit ffmpeg in eine WAV-Datei konvertieren:

```bash
ffmpeg -f s16le -ar 16000 -ac 1 -i /tmp/test.raw /tmp/test.wav
```

## Bekannte offene Punkte / worauf beim Debuggen zu achten ist

- **Berechtigungsdialog**: Verifiziert - das Programm bricht ohne erteilte
  Berechtigung mit einem klaren Fehler ab (TCC-Error -3801, "Benutzer:in hat
  TCCs fuer die Aufnahme durch Apps, Fenster, Displays abgelehnt"). Die
  Berechtigung muss manuell unter Systemeinstellungen > Datenschutz &
  Sicherheit > Bildschirm- und Systemaudioaufnahme fuer den ausfuehrenden
  Prozess (Terminal, spaeter die gepackte App) erteilt werden; danach muss
  der Prozess (Terminal) neu gestartet werden, damit macOS das TCC-Update
  uebernimmt.
- **Sample-Format**: Verifiziert - ScreenCaptureKit liefert bei
  `sampleRate = 16000`/`channelCount = 1` tatsaechlich zuverlaessig
  16kHz-Mono-Float32, die Konvertierung nach Int16 in
  `stream(_:didOutputSampleBuffer:of:)` erzeugt korrekte, hoerbare PCM-Daten.
- **`excludingDesktopWindows`/`onScreenWindowsOnly`**: Aktuell wird das erste
  gefundene Display genutzt (`content.displays.first`) - bei Mehrschirm-Setups
  waere ggf. eine Display-Auswahl noetig (noch nicht getestet).
- **Sauberes Beenden**: Verifiziert - SIGTERM beendet den Prozess sauber
  (`signal(...)`-Handler + `exit(0)`), `subprocess.terminate()` von
  `recorder.py` aus fuehrt zu keinem haengenden Prozess und `stop()` kehrt in
  ~10ms zurueck.
- **Nur "System-Audio" allein implementiert**: Die kombinierte Aufnahme
  "Mikrofon + System" (wie unter Windows) ist auf macOS noch NICHT umgesetzt -
  bewusste Scope-Entscheidung, um zuerst den Basis-Fall zu verifizieren.
