// SystemAudioCapture.swift
//
// Minimaler Kommandozeilen-Helfer fuer VoxScribe: nimmt System-Audio unter
// macOS via ScreenCaptureKit auf und schreibt rohe 16kHz-Mono-Int16-PCM-Samples
// nach stdout. `recorder.py` startet dieses Programm als Subprozess und liest
// die Bytes wie einen Audio-Stream - genau wie es unter Windows PyAudioWPatch
// und unter macOS/Linux `sounddevice` fuers Mikrofon tut.
//
// Gebaut, getestet und debuggt auf echter Apple-Silicon-Hardware (macOS 26) -
// siehe build.sh und README in diesem Ordner fuer Details und offene Punkte.
//
// Voraussetzungen:
//   - macOS 13 (Ventura) oder neuer (SCStreamConfiguration.capturesAudio kam
//     erst in macOS 13 hinzu; ScreenCaptureKit selbst existiert seit 12.3,
//     aber ohne Audio-Support)
//   - Der Prozess, der dieses Binary ausfuehrt (z.B. Terminal, oder die
//     gepackte VoxScribe-App), braucht die "Bildschirmaufnahme"-Berechtigung
//     unter Systemeinstellungen > Datenschutz & Sicherheit > Bildschirm- und
//     Systemaudioaufnahme. macOS fragt das i.d.R. beim ersten Start ab bzw.
//     verweigert sonst mit einem klaren Fehler (siehe stream(_:didStopWithError:)).
//
// Nutzung: ./SystemAudioCapture > audio.raw
//   Stoppen per SIGTERM/SIGINT (das macht recorder.py per subprocess.terminate()).
//   Die Rohdaten sind 16-bit signed little-endian PCM, mono, 16000 Hz -
//   identisch zum Format, das der Rest der VoxScribe-Pipeline erwartet
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
                userInfo: [NSLocalizedDescriptionKey: "Kein Display gefunden."])
        }
        // Kein Fenster ausschliessen - wir wollen die komplette
        // System-Wiedergabe (Teams/Zoom/Browser/etc.), nicht nur ein
        // bestimmtes Fenster.
        let filter = SCContentFilter(display: display, excludingWindows: [])

        let config = SCStreamConfiguration()
        config.capturesAudio = true
        // Verhindert, dass die eigene Ausgabe dieses Prozesses (falls er
        // selbst Ton macht) sich selbst aufnimmt - hier nicht relevant, aber
        // schadet nicht.
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
            "Aufnahme gestartet (16kHz mono).\n".data(using: .utf8)!)
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

        // ScreenCaptureKit liefert Float32-Samples (typischerweise im Bereich
        // -1.0...1.0). Wir wandeln das hier manuell nach Int16 PCM um, damit
        // es 1:1 zu recorder.py's bestehender WAV-Pipeline passt (SAMPLE_WIDTH
        // = 2 Bytes / 16-bit ueberall sonst im Projekt).
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
        // Haeufigste Ursache: fehlende "Bildschirmaufnahme"-Berechtigung.
        let message = "Stream-Fehler (moeglicherweise fehlende Bildschirmaufnahme-Berechtigung " +
            "unter Systemeinstellungen > Datenschutz & Sicherheit): " +
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
                "Fehler beim Start der Systemaudio-Aufnahme: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }

    // Sauber beenden, wenn Python per subprocess.terminate()/.send_signal()
    // SIGTERM/SIGINT schickt (siehe recorder.py: _record_system_macos).
    signal(SIGTERM) { _ in exit(0) }
    signal(SIGINT) { _ in exit(0) }

    RunLoop.main.run()
} else {
    FileHandle.standardError.write(
        "Systemaudio-Aufnahme erfordert macOS 13 (Ventura) oder neuer.\n".data(using: .utf8)!)
    exit(1)
}
