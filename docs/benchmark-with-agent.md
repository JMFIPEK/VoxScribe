# Benchmarking transcription models (for AI agents)

This page is for an AI agent asked to compare VoxScribe's transcription
models on someone's recordings: which one is most accurate, which one is
fast enough, on this machine. Follow it step by step. The command is
`python main.py benchmark` (spec #56, code in `voxscribe/benchmark.py`).

## Rules

1. **Ask before any audio leaves the machine.** By default the benchmark
   includes every configured Provider (`~/.voxscribe/providers.json`), and a
   Provider uploads the audio to a server. Before the first run, list the
   Providers and ask the user whether they may be used. If not, pass
   `--models` without them.
2. **Ask before downloading.** Without `--download` nothing is downloaded:
   models that aren't on the machine are listed as skipped. Large ones are
   several GB (Whisper large-v3 ~3 GB, Parakeet ~640 MB, alignment models
   ~360 MB per language). Name what would be downloaded and ask.
3. **Never use a model's output as the reference.** Word error rates against
   a model's own text are meaningless. See "Getting a reference" below.
4. **Recordings stay out of git.** They contain other people's voices. Keep
   them in `~/.voxscribe/benchmark/recordings/`, never in the repo, and never
   attach audio to an issue or a comment.

## Running it

From the repo root (on Windows `uv run --no-sync python`, as in the README):

```bash
# Every model on this machine plus the Providers, first 120 s, auto language
python main.py benchmark -i ~/.voxscribe/benchmark/recordings/T2.wav

# With a reference text (word error rate), German, whole file
python main.py benchmark -i T2.wav -r T2_de.txt -l de --seconds 0

# Chosen models only
python main.py benchmark -i T2.wav -m parakeet:tdt-0.6b-v3,apple:speechanalyzer

# A Call: transcribe the mix (what the app does) and each track separately
python main.py benchmark -i T4.wav -r ref.txt --tracks both

# Several recordings in one run: each -r belongs to the -i at the same position
python main.py benchmark -i de.wav -r de.txt -i en.wav -r en.txt
```

Model IDs: `large-v3`, `large-v2`, `medium`, `base` (local Whisper),
`parakeet:tdt-0.6b-v3`, `apple:speechanalyzer` (macOS 26+),
`openvino:GPU:medium` (Intel Arc), `server:<provider id>` (a Provider), and
the candidates of #57 (not in the app's model list): `canary:1b-v2` (NVIDIA
Canary-1B-v2 via onnx-asr, CPU, int8; from `istupakov/canary-1b-v2-onnx`,
fetched with `--download` on first use).

**Several models through one Provider:** `server:<provider id>#<model>` runs
the Provider with `<model>` instead of the one it stores, for this run only
(the app itself never uses this form). The Provider just has to exist in
Settings → Providers, with its base URL and API key. For OpenRouter (Provider
id `openrouter`, base URL `https://openrouter.ai/api/v1`):

```bash
python main.py benchmark -i T4.wav -r ref.txt -l de -m "server:openrouter#qwen/qwen3-asr-1.7b,server:openrouter#nvidia/nemotron-3.5-transcribe,server:openrouter#mistralai/voxtral-small-24b-2507-stt,server:openrouter#openai/whisper-large-v3"
```

Models that refuse `response_format=verbose_json` (HTTP 400) are asked again
with `json`; they then return text only (one segment per piece in the
"returns" column), and the audio goes
out in 30 s pieces whose timings come from wav2vec2 alignment. OpenRouter caps
an upload at 25 MB and gives each request 60 s upstream; VoxScribe's pieces
(FLAC, 30-180 s) stay well under the size cap, and a timed-out piece is
retried like any other server error. Never paste the API key into a command,
a report or the issue.

**Language:** Canary can't detect the language either: without `-l` it
transcribes the first 30 s once in German and once in English and keeps the
one it is more confident in (other languages need `-l`). Pass `-l` for
Canary whenever the language is known. Apple SpeechAnalyzer can't detect the language. For a
recording that switches language, cut it per language with `--seconds`, or
run it once per language and compare only the matching part.

## Reading the output

The header names the machine (OS, CPU, RAM, GPU), so results from several
machines can be put side by side. Then one table per recording:

| Column | Meaning |
|---|---|
| Load s | A 1-second warm-up run: model loading plus first-call overhead (for a Provider: one request's latency) |
| Transcribe s | The real run minus the load time |
| Align s | wav2vec2 alignment, only for Backends that need it |
| x realtime | Audio seconds / (load + transcribe + align). Above 1 is faster than realtime |
| WER | Word error rate against the reference; Sub/Del/Ins split it up |
| Returns | What the Backend delivers: segments (per track for `separate`), word timings yes/no |
| Lang | The language the Backend reported or that was guessed from its text |
| Timing offset | Median distance between a Backend's own word timings and wav2vec2's (only if the alignment model is on the machine) |
| Status | `ok`, `ok, not aligned (...)` (alignment model missing; the WER still counts), or `failed: ...` |

Below the table come each model's texts, then the skipped models with the
reason.

**Word error rate details:** lower case, punctuation removed, numbers
compared as written ("8 Uhr" vs "acht Uhr" is one error). Many deletions in
the mix but few in `separate` means the model loses words where voices
alternate. That was the case for SpeechAnalyzer on a Mac Loudspeaker Call
(#54).

## Getting a reference

A reference is the text that was really said, in time order, as a plain
UTF-8 text file. Any of these works:

- **A read script.** `docs/benchmark/script.md` is a German/English dialog
  for this purpose. Ready-made references of its two parts (speaker names
  and stage directions removed) are `~/.voxscribe/benchmark/references/script_de.txt`
  and `script_en.txt`. Make them again with the snippet below if they're
  missing. Who read A and B doesn't matter for the WER. Ask the user where
  they read differently from the script, and fix those spots in a copy of
  the reference.
- **A corrected draft.** Transcribe the recording with the best available
  model, give the user the text, and let them correct it while listening.
  Only the corrected text is a reference.
- **A cut.** For a long recording, correct only the first minutes and run
  with `--seconds` set to match.

Making the script references:

```bash
uv run --no-sync python - <<'EOF'
import os, re
s = open("docs/benchmark/script.md", encoding="utf-8").read()
def part(title):
    body = s.split(title, 1)[1].split("\n## ", 1)[0]
    return "\n".join(re.sub(r"\*\*\[[^\]]+\]\*\*\s*", "", m.group(1))
                     for line in body.splitlines()
                     if (m := re.match(r"\*\*[AB]:\*\*\s*(.*)", line.strip()))) + "\n"
d = os.path.expanduser("~/.voxscribe/benchmark/references"); os.makedirs(d, exist_ok=True)
open(f"{d}/script_de.txt", "w", encoding="utf-8").write(part("## Part 1 - Deutsch"))
open(f"{d}/script_en.txt", "w", encoding="utf-8").write(part("## Part 2 - English"))
EOF
```

## Reporting

Post results as a comment on the study issue (#58), or wherever the user
asks:
- the header lines (machine)
- the table
- which reference was used and which spots were read differently
- your reading of the result

No audio. Long transcripts only on request: they may contain what other
people said.
