// SpeechAnalyzerTranscribe.swift
//
// Einmaliger (batch) Kommandozeilen-Helfer fuer VoxScribe: transkribiert eine
// bereits aufgenommene Audio-Datei komplett offline via Apples
// SpeechAnalyzer/SpeechTranscriber (Speech-Framework, macOS 26+, laeuft auf
// der Neural Engine) und schreibt pro erkanntem Segment eine JSON-Zeile
// (NDJSON) nach stdout. transcriber.py liest das zeilenweise, um dem Nutzer
// waehrend der Transkription echten Fortschritt anzuzeigen - anders als
// SystemAudioCapture.swift (das einen Dauer-Stream roher PCM-Bytes liefert)
// ist dies aber ein einmaliger Batch-Job: er endet von selbst, sobald die
// Datei fertig verarbeitet ist, es gibt kein SIGTERM-Handling.
//
// Nutzung: ./SpeechAnalyzerTranscribe <audio-datei> [locale, z.B. de-DE]
//   Jede Zeile auf stdout ist ein JSON-Objekt:
//     {"type":"segment","start":0.0,"end":1.2,"text":"...",
//      "words":[{"word":"...","start":0.0,"end":0.4,"score":0.99}, ...]}
//   Am Ende folgt eine Abschlusszeile: {"type":"done"}
//   Fehler (z.B. nicht unterstuetzte Sprache, fehlendes Sprachmodell, nicht
//   lesbare Audio-Datei) gehen nach stderr, Exit-Code != 0 - Python liest das
//   bei einem Fehler aus (gleiches Muster wie bei SystemAudioCapture.swift).
//
// Word-Level-Timestamps kommen direkt aus der `audioTimeRange`-Attribution
// der von SpeechTranscriber gelieferten AttributedString-Runs - verifiziert
// an echter Hardware (macOS 26.5.1, Apple Silicon) mit synthetischem
// (`say`-generiertem) deutschem Testaudio: die Zeitstempel sind bereits
// wortgenau, ein nachtraeglicher wav2vec2-Alignment-Schritt (wie er fuer
// WhisperX noetig ist) ist fuer dieses Ergebnis nicht erforderlich - siehe
// den entsprechenden Kommentar in transcriber.py::transcribe_apple().

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
        fail("Nutzung: SpeechAnalyzerTranscribe <audio-datei> [locale, z.B. de-DE]")
    }
    let audioPath = args[1]
    let localeIdentifier = args.count >= 3 ? args[2] : "de-DE"
    let requestedLocale = Locale(identifier: localeIdentifier)

    guard SpeechTranscriber.isAvailable else {
        fail("SpeechTranscriber ist auf diesem System nicht verfuegbar (benoetigt macOS 26+ auf Apple Silicon).")
    }

    guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: requestedLocale) else {
        fail("Sprache '\(localeIdentifier)' wird von SpeechAnalyzer nicht unterstuetzt.")
    }

    // attributeOptions: [.audioTimeRange, .transcriptionConfidence] liefert
    // pro Wort (AttributedString-Run) Zeitstempel + Konfidenz mit - das ersetzt
    // den wav2vec2-Alignment-Schritt der WhisperX-Pipeline komplett (siehe
    // Kommentar oben und transcriber.py::transcribe_apple()).
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
                    "Lade Sprachmodell fuer \(locale.identifier) herunter...\n".data(using: .utf8)!)
                try await request.downloadAndInstall()
            }
        } catch {
            fail("Fehler beim Herunterladen des Sprachmodells fuer \(locale.identifier): \(error)")
        }
    }

    let audioFile: AVAudioFile
    do {
        audioFile = try AVAudioFile(forReading: URL(fileURLWithPath: audioPath))
    } catch {
        fail("Konnte Audio-Datei nicht lesen (\(audioPath)): \(error)")
    }

    let analyzer: SpeechAnalyzer
    do {
        analyzer = try await SpeechAnalyzer(
            inputAudioFile: audioFile,
            modules: [transcriber],
            finishAfterFile: true
        )
    } catch {
        fail("Fehler beim Initialisieren von SpeechAnalyzer: \(error)")
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
            fail("Fehler waehrend der Transkription: \(error)")
        }
    }

    do {
        try await analyzer.finalizeAndFinishThroughEndOfInput()
    } catch {
        fail("Fehler beim Abschliessen der Transkription: \(error)")
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
        "SpeechAnalyzer-Transkription erfordert macOS 26 (Tahoe) oder neuer.\n".data(using: .utf8)!)
    exit(1)
}
