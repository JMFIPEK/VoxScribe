# System-Audio-Aufnahme unter macOS (experimentell, ungetestet)

`SystemAudioCapture.swift` ist ein kleiner Kommandozeilen-Helfer, der System-Audio
via Apples ScreenCaptureKit aufnimmt und als rohe 16kHz-Mono-Int16-PCM-Bytes nach
stdout schreibt. `recorder.py` startet ihn als Subprozess (siehe
`_record_system_macos()`), analog dazu, wie unter Windows PyAudioWPatch und unter
macOS/Linux `sounddevice` fuers Mikrofon genutzt werden.

**Status: wurde ohne Zugriff auf echte Mac-/Xcode-Hardware geschrieben.**
Muss auf einem echten Mac gebaut, getestet und wahrscheinlich noch debuggt werden,
bevor es produktiv genutzt werden kann.

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

- **Berechtigungsdialog**: Ob macOS die "Bildschirmaufnahme"-Berechtigung beim
  ersten Start automatisch abfragt oder das Programm nur mit einem Fehler
  abbricht (dann muesste man es manuell in den Systemeinstellungen freigeben,
  danach das Terminal/den Prozess neu starten), ist ungetestet.
- **Sample-Format**: Es wird angenommen, dass ScreenCaptureKit die Audiodaten
  als Float32 liefert und dass `SCStreamConfiguration.sampleRate`/`channelCount`
  zuverlaessig auf 16kHz/mono resampled. Falls nicht, muss die Float->Int16-
  Konvertierung in `stream(_:didOutputSampleBuffer:of:)` angepasst werden
  (z.B. falls tatsaechlich mehrkanalig oder mit anderer Sample-Rate geliefert
  wird).
- **`excludingDesktopWindows`/`onScreenWindowsOnly`**: Aktuell wird das erste
  gefundene Display genutzt (`content.displays.first`) - bei Mehrschirm-Setups
  waere ggf. eine Display-Auswahl noetig.
- **Sauberes Beenden**: SIGTERM/SIGINT sollten das Programm sauber beenden
  (siehe `signal(...)`-Handler); ob `Task`/`RunLoop` das unter echten
  Bedingungen tatsaechlich sauber abbricht (z.B. kein haengender Prozess nach
  `subprocess.terminate()`), ist ungetestet.
- **Nur "System-Audio" allein implementiert**: Die kombinierte Aufnahme
  "Mikrofon + System" (wie unter Windows) ist auf macOS noch NICHT umgesetzt -
  bewusste Scope-Entscheidung, um zuerst den Basis-Fall zu verifizieren.
