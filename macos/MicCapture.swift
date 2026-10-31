// MicCapture.swift
//
// Command-line helper for VoxScribe: records a microphone on macOS with
// Apple's voice processing (AVAudioEngine, the same processing Teams/Zoom/
// FaceTime use) and writes raw 16kHz mono int16 PCM to stdout - the same
// format and protocol as SystemAudioCapture, so recorder.py reads both alike.
//
// Why (#54): while a call app holds the built-in microphone with voice
// processing (Teams on the MacBook's own speakers and mic), macOS hands every
// app that reads the microphone *without* voice processing an almost silent,
// gated signal (measured: speech at -60 dB instead of -20 dB). An app that
// asks for voice processing itself gets a normal, echo-cancelled signal.
// recorder.py uses this helper for Loudspeaker Calls only.
//
// Usage: ./MicCapture [--device-name "MacBook Pro-Mikrofon"] > mic.raw
//   Without --device-name (or if no input device has that name) the system's
//   default input is used. Stop via SIGTERM/SIGINT.
//
// Requirements: macOS 13+, the Microphone permission for the process that
// launches it (VoxScribe or Terminal).

import AVFoundation
import CoreAudio
import Foundation

let outRate = 16000.0

func log(_ message: String) {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
}

/// The CoreAudio ID of the input device called `name`, or nil.
func inputDeviceID(named name: String) -> AudioDeviceID? {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(
        AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr else { return nil }
    var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    guard AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr else { return nil }
    for id in ids {
        var nameAddress = AudioObjectPropertyAddress(
            mSelector: kAudioObjectPropertyName,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var cfName: Unmanaged<CFString>?
        var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(id, &nameAddress, 0, nil, &nameSize, &cfName) == noErr,
              let deviceName = cfName?.takeRetainedValue() as String?,
              deviceName == name else { continue }
        // Only devices with input channels.
        var streamsAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain)
        var streamsSize: UInt32 = 0
        if AudioObjectGetPropertyDataSize(id, &streamsAddress, 0, nil, &streamsSize) == noErr,
           streamsSize > 0 {
            return id
        }
    }
    return nil
}

var deviceName: String?
var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    if arg == "--device-name" { deviceName = args.popFirst() }
}

let engine = AVAudioEngine()
let input = engine.inputNode

if let name = deviceName {
    if var id = inputDeviceID(named: name), let unit = input.audioUnit {
        let status = AudioUnitSetProperty(
            unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
            &id, UInt32(MemoryLayout<AudioDeviceID>.size))
        if status != noErr { log("Could not select '\(name)' (\(status)), using the default input.") }
    } else {
        log("No input device '\(name)', using the default input.")
    }
}

do {
    try input.setVoiceProcessingEnabled(true)
} catch {
    log("Error enabling voice processing: \(error)")
    exit(1)
}
if #available(macOS 14.0, *) {
    // Don't turn the call app's (or anyone's) playback down while recording.
    input.voiceProcessingOtherAudioDuckingConfiguration =
        AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
            enableAdvancedDucking: false, duckingLevel: .min)
}

// With voice processing the input node reports several channels; channel 0
// is the processed microphone signal.
let inFormat = input.outputFormat(forBus: 0)
guard let monoIn = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inFormat.sampleRate,
                                 channels: 1, interleaved: false),
      let monoOut = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: outRate,
                                  channels: 1, interleaved: true),
      let converter = AVAudioConverter(from: monoIn, to: monoOut) else {
    log("Error: unsupported input format \(inFormat)")
    exit(1)
}

let stdoutHandle = FileHandle.standardOutput
input.installTap(onBus: 0, bufferSize: 1024, format: inFormat) { buffer, _ in
    guard let src = buffer.floatChannelData?[0], buffer.frameLength > 0,
          let mono = AVAudioPCMBuffer(pcmFormat: monoIn, frameCapacity: buffer.frameLength) else { return }
    mono.frameLength = buffer.frameLength
    mono.floatChannelData![0].update(from: src, count: Int(buffer.frameLength))

    let capacity = AVAudioFrameCount(Double(buffer.frameLength) * outRate / inFormat.sampleRate) + 16
    guard let out = AVAudioPCMBuffer(pcmFormat: monoOut, frameCapacity: capacity) else { return }
    var fed = false
    var error: NSError?
    converter.convert(to: out, error: &error) { _, status in
        if fed { status.pointee = .noDataNow; return nil }
        fed = true
        status.pointee = .haveData
        return mono
    }
    if error == nil, out.frameLength > 0, let samples = out.int16ChannelData?[0] {
        stdoutHandle.write(Data(bytes: samples, count: Int(out.frameLength) * 2))
    }
}

do {
    try engine.start()
} catch {
    log("Error starting the microphone (Microphone permission?): \(error)")
    exit(1)
}
log("Recording started (voice processing, 16kHz mono).")

signal(SIGTERM) { _ in exit(0) }
signal(SIGINT) { _ in exit(0) }
RunLoop.main.run()
