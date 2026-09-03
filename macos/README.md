# System-audio recording on macOS

`SystemAudioCapture.swift` is a small command-line helper that captures system
audio via Apple's ScreenCaptureKit and writes it to stdout as raw 16kHz mono
int16 PCM bytes. `recorder.py` launches it as a subprocess (see
`_record_system_macos()`), the same way PyAudioWPatch is used on Windows and
`sounddevice` is used for the microphone on macOS/Linux.

**Status: built, tested, and debugged on real Apple Silicon hardware (macOS
26).** Two bugs were found and fixed along the way: a Swift compile error
(string/`Data` type mismatch in the error-logging path) and a Python
import-time crash (`recorder.py` referenced `pyaudio.PyAudio` in a type
annotation even though `pyaudio` is never imported on macOS - this crashed
the whole import at program start, see `from __future__ import annotations`
in `recorder.py`). After the fixes, recording works end-to-end: permission
prompt, real (non-silent) audio data, clean shutdown in ~10ms with no
orphaned process.

## Requirements

- macOS 13 (Ventura) or newer
- Xcode Command Line Tools: `xcode-select --install`
- The "Screen & System Audio Recording" permission for the process running
  the binary (Terminal while testing manually, later the packaged VoxScribe
  app), under System Settings > Privacy & Security

## Building

```bash
cd macos
./build.sh
```

Produces `macos/SystemAudioCapture` - exactly where `recorder.py` looks for it.

## Testing manually (independent of VoxScribe)

```bash
cd macos
./SystemAudioCapture > /tmp/test.raw
# let it run for a few seconds while e.g. music/a video is playing, then Ctrl+C
```

`/tmp/test.raw` then contains raw 16-bit PCM samples (no WAV header!). To
listen to it, convert it to a WAV file with e.g. ffmpeg:

```bash
ffmpeg -f s16le -ar 16000 -ac 1 -i /tmp/test.raw /tmp/test.wav
```

## Known open items / things to watch when debugging

- **Permission dialog**: Verified - the program aborts with a clear error if
  the permission hasn't been granted (TCC error -3801, "User declined TCC
  for capturing screens/windows/apps"). The permission must be granted
  manually under System Settings > Privacy & Security > Screen & System
  Audio Recording for the process running it (Terminal, later the packaged
  app); the process (Terminal) then needs to be restarted for macOS to pick
  up the TCC update.
- **Sample format**: Verified - at `sampleRate = 16000`/`channelCount = 1`,
  ScreenCaptureKit reliably delivers 16kHz mono float32, and the conversion
  to int16 in `stream(_:didOutputSampleBuffer:of:)` produces correct,
  audible PCM data.
- **`excludingDesktopWindows`/`onScreenWindowsOnly`**: Currently uses the
  first display found (`content.displays.first`) - a multi-monitor setup
  might need display selection (not yet tested).
- **Clean shutdown**: Verified - SIGTERM terminates the process cleanly
  (`signal(...)` handler + `exit(0)`), `subprocess.terminate()` from
  `recorder.py` doesn't leave a hanging process, and `stop()` returns in
  ~10ms.
- **Combined "microphone + system" recording**: Implemented in
  `recorder.py::AudioRecorder._record_both_macos()` (microphone via
  `sounddevice`, system audio via this subprocess pipe, mixed afterward like
  on Windows). Unlike the plain system-audio path above, this has **not**
  been tested on real hardware yet - when first testing it for real, watch
  especially for sync drift between the two sources and behavior when
  recording permission is missing for only one side.
