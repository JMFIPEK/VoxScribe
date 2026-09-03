// SystemAudioCapture.swift
//
// Minimal command-line helper for VoxScribe: captures system audio on macOS
// via ScreenCaptureKit and writes raw 16kHz mono int16 PCM samples to stdout.
// `recorder.py` launches this program as a subprocess and reads the bytes
// like an audio stream - the same way PyAudioWPatch is used on Windows and
// `sounddevice` is used for the microphone on macOS/Linux.
//
// Built, tested, and debugged on real Apple Silicon hardware (macOS 26) -
// see build.sh and the README in this folder for details and open items.
//
// Requirements:
//   - macOS 13 (Ventura) or newer (SCStreamConfiguration.capturesAudio was
//     only added in macOS 13; ScreenCaptureKit itself exists since 12.3, but
//     without audio support)
//   - The process running this binary (e.g. Terminal, or the packaged
//     VoxScribe app) needs the "Screen Recording" permission under System
//     Settings > Privacy & Security > Screen & System Audio Recording.
//     macOS usually prompts for this on first launch, otherwise it refuses
//     with a clear error (see stream(_:didStopWithError:)).
//
// Usage: ./SystemAudioCapture > audio.raw
//   Stop via SIGTERM/SIGINT (recorder.py does this via subprocess.terminate()).
//   The raw data is 16-bit signed little-endian PCM, mono, 16000 Hz -
//   identical to the format the rest of the VoxScribe pipeline expects
//   (SAMPLE_RATE in transcriber.py / OUT_RATE in recorder.py).

import Foundation
import ScreenCaptureKit
import CoreMedia

@available(macOS 13.0, *)
final class SystemAudioRecorder: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let stdoutHandle = FileHandle.standardOutput

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            throw NSError(
                domain: "VoxScribe", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No display found."])
        }
        // Don't exclude any window - we want the entire system playback
        // (Teams/Zoom/browser/etc.), not just a specific window.
        let filter = SCContentFilter(display: display, excludingWindows: [])

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        // Prevents this process's own output (if it makes sound itself) from
        // capturing itself - not relevant here, but doesn't hurt.
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 16000
        config.channelCount = 1

        let newStream = SCStream(filter: filter, configuration: config, delegate: self)
        try newStream.addStreamOutput(
            self, type: .audio,
            sampleHandlerQueue: DispatchQueue(label: "voxscribe.audio.capture"))
        try await newStream.startCapture()
        self.stream = newStream

        FileHandle.standardError.write(
            "Recording started (16kHz mono).\n".data(using: .utf8)!)
    }

    func stop() async {
        try? await stream?.stopCapture()
    }

    // MARK: - SCStreamOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }

        var blockBuffer: CMBlockBuffer?
        var audioBufferList = AudioBufferList()
        let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer,
            bufferListSizeNeededOut: nil,
            bufferListOut: &audioBufferList,
            bufferListSize: MemoryLayout<AudioBufferList>.size,
            blockBufferAllocator: kCFAllocatorDefault,
            blockBufferMemoryAllocator: kCFAllocatorDefault,
            flags: 0,
            blockBufferOut: &blockBuffer)
        guard status == noErr else { return }

        let buffers = UnsafeMutableAudioBufferListPointer(&audioBufferList)
        guard let buffer = buffers.first, let dataPointer = buffer.mData else { return }

        // ScreenCaptureKit delivers float32 samples (typically in the range
        // -1.0...1.0). We convert them to int16 PCM here manually, so it
        // matches recorder.py's existing WAV pipeline 1:1 (SAMPLE_WIDTH = 2
        // bytes / 16-bit everywhere else in the project).
        let floatCount = Int(buffer.mDataByteSize) / MemoryLayout<Float32>.size
        guard floatCount > 0 else { return }
        let floatPointer = dataPointer.bindMemory(to: Float32.self, capacity: floatCount)

        var int16Samples = [Int16](repeating: 0, count: floatCount)
        for i in 0..<floatCount {
            let clamped = max(-1.0, min(1.0, floatPointer[i]))
            int16Samples[i] = Int16(clamped * 32767.0)
        }

        int16Samples.withUnsafeBufferPointer { ptr in
            let data = Data(buffer: ptr)
            stdoutHandle.write(data)
        }
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        // Most common cause: missing "Screen Recording" permission.
        let message = "Stream error (possibly missing Screen Recording permission " +
            "under System Settings > Privacy & Security): " +
            "\(error.localizedDescription)\n"
        FileHandle.standardError.write(message.data(using: .utf8)!)
        exit(1)
    }
}

// MARK: - Entry point

if #available(macOS 13.0, *) {
    let recorder = SystemAudioRecorder()

    Task {
        do {
            try await recorder.start()
        } catch {
            FileHandle.standardError.write(
                "Error starting system-audio recording: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }

    // Shut down cleanly when Python sends SIGTERM/SIGINT via
    // subprocess.terminate()/.send_signal() (see recorder.py: _record_system_macos).
    signal(SIGTERM) { _ in exit(0) }
    signal(SIGINT) { _ in exit(0) }

    RunLoop.main.run()
} else {
    FileHandle.standardError.write(
        "System-audio recording requires macOS 13 (Ventura) or newer.\n".data(using: .utf8)!)
    exit(1)
}
