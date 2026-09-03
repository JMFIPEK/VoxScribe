// SpeechAnalyzerTranscribe.swift
//
// One-shot (batch) command-line helper for VoxScribe: transcribes an
// already-recorded audio file entirely offline via Apple's
// SpeechAnalyzer/SpeechTranscriber (Speech framework, macOS 26+, runs on the
// Neural Engine) and writes one JSON line (NDJSON) per recognized segment to
// stdout. transcriber.py reads this line by line to show the user real
// progress during transcription - unlike SystemAudioCapture.swift (which
// delivers a continuous stream of raw PCM bytes), this is a one-shot batch
// job: it exits on its own once the file has been fully processed, there's
// no SIGTERM handling.
//
// Usage: ./SpeechAnalyzerTranscribe <audio-file> [locale, e.g. en-US]
//   Each line on stdout is a JSON object:
//     {"type":"segment","start":0.0,"end":1.2,"text":"...",
//      "words":[{"word":"...","start":0.0,"end":0.4,"score":0.99}, ...]}
//   A final line follows at the end: {"type":"done"}
//   Errors (e.g. unsupported language, missing language model, unreadable
//   audio file) go to stderr with a non-zero exit code - Python reads that
//   on failure (same pattern as SystemAudioCapture.swift).
//
// Word-level timestamps come directly from the `audioTimeRange` attribute on
// the AttributedString runs SpeechTranscriber returns - verified on real
// hardware (macOS 26.5.1, Apple Silicon) with synthetic (`say`-generated)
// test audio: the timestamps are already word-accurate, so a subsequent
// wav2vec2 alignment step (as needed for WhisperX) isn't required for this
// result - see the matching comment in transcriber.py::transcribe_apple().

import Foundation
import Speech
import AVFoundation

struct WordJSON: Encodable {
    let word: String
    let start: Double
    let end: Double
    let score: Double
}

struct SegmentJSON: Encodable {
    let type = "segment"
    let start: Double
    let end: Double
    let text: String
    let words: [WordJSON]
}

struct DoneJSON: Encodable {
    let type = "done"
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

func printJSON<T: Encodable>(_ value: T) {
    guard let data = try? JSONEncoder().encode(value) else { return }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

@available(macOS 26.0, *)
func run() async {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        fail("Usage: SpeechAnalyzerTranscribe <audio-file> [locale, e.g. en-US]")
    }
    let audioPath = args[1]
    let localeIdentifier = args.count >= 3 ? args[2] : "en-US"
    let requestedLocale = Locale(identifier: localeIdentifier)

    guard SpeechTranscriber.isAvailable else {
        fail("SpeechTranscriber isn't available on this system (requires macOS 26+ on Apple Silicon).")
    }

    guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: requestedLocale) else {
        fail("Language '\(localeIdentifier)' isn't supported by SpeechAnalyzer.")
    }

    // attributeOptions: [.audioTimeRange, .transcriptionConfidence] provides
    // per-word (AttributedString run) timestamps + confidence - this
    // completely replaces the WhisperX pipeline's wav2vec2 alignment step
    // (see the comment above and transcriber.py::transcribe_apple()).
    let transcriber = SpeechTranscriber(
        locale: locale,
        transcriptionOptions: [],
        reportingOptions: [],
        attributeOptions: [.audioTimeRange, .transcriptionConfidence]
    )

    let status = await AssetInventory.status(forModules: [transcriber])
    if status != .installed {
        do {
            if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                FileHandle.standardError.write(
                    "Downloading language model for \(locale.identifier)...\n".data(using: .utf8)!)
                try await request.downloadAndInstall()
            }
        } catch {
            fail("Error downloading the language model for \(locale.identifier): \(error)")
        }
    }

    let audioFile: AVAudioFile
    do {
        audioFile = try AVAudioFile(forReading: URL(fileURLWithPath: audioPath))
    } catch {
        fail("Could not read audio file (\(audioPath)): \(error)")
    }

    let analyzer: SpeechAnalyzer
    do {
        analyzer = try await SpeechAnalyzer(
            inputAudioFile: audioFile,
            modules: [transcriber],
            finishAfterFile: true
        )
    } catch {
        fail("Error initializing SpeechAnalyzer: \(error)")
    }

    let resultsTask = Task {
        do {
            for try await result in transcriber.results {
                var words: [WordJSON] = []
                for run in result.text.runs {
                    let text = String(result.text[run.range].characters)
                        .trimmingCharacters(in: .whitespaces)
                    guard !text.isEmpty else { continue }
                    let timeRange = run.audioTimeRange ?? result.range
                    words.append(WordJSON(
                        word: text,
                        start: timeRange.start.seconds,
                        end: timeRange.end.seconds,
                        score: run.transcriptionConfidence ?? 0.0
                    ))
                }
                printJSON(SegmentJSON(
                    start: result.range.start.seconds,
                    end: result.range.end.seconds,
                    text: String(result.text.characters).trimmingCharacters(in: .whitespaces),
                    words: words
                ))
            }
        } catch {
            fail("Error during transcription: \(error)")
        }
    }

    do {
        try await analyzer.finalizeAndFinishThroughEndOfInput()
    } catch {
        fail("Error finalizing transcription: \(error)")
    }
    await resultsTask.value

    printJSON(DoneJSON())
}

if #available(macOS 26.0, *) {
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        await run()
        semaphore.signal()
    }
    semaphore.wait()
} else {
    FileHandle.standardError.write(
        "SpeechAnalyzer transcription requires macOS 26 (Tahoe) or newer.\n".data(using: .utf8)!)
    exit(1)
}
