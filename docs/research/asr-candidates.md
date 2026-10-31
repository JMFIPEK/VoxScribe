# ASR candidates for VoxScribe (issue #57)

**As of 2026-10-31.** Scope: open-weight speech-to-text models (plus Provider-only ones as reference points) that could beat our shipped NVIDIA Parakeet TDT 0.6B v3 on German meeting audio, or match it while running much faster, on our three target devices: MacBook Pro M2 Pro, 16 GB, macOS 27; Windows + NVIDIA (CUDA 12.8); Windows + Core Ultra 7 258V / Arc 140V (OpenVINO, no NVIDIA). Diarization, forced alignment and the Provider route (OpenRouter and other OpenAI-compatible `/audio/transcriptions` servers) are covered briefly. Every number comes from the publisher's model card, paper or official docs unless marked *unverified*. Public benchmarks (FLEURS, CoVoST2, MLS, Common Voice) are read speech. Our own first benchmark (KIT ToolBox / whisper-large-v3 4.0 %, Parakeet v3 7.0-7.3 %, SpeechAnalyzer 20 % on two-voice mixes) shows the gap is larger on our audio than on FLEURS. So treat the published numbers below as a ranking hint only, not as the answer.

Summary: whisper-large-v3 is still hard to beat on German among *small* open models. Even NVIDIA's own paper shows it ahead of Parakeet v3 on FLEURS-de (4.30 vs 5.04 %). The newer candidates that plausibly beat both are Qwen3-ASR-1.7B (FLEURS-de 3.92 %), Cohere Transcribe (leaderboard DE 3.84 %), Voxtral Small 24B (DE 3.01 %), and Nemotron 3.5 Transcribe (FLEURS-de 2.73 %, vendor-reported). Only the first two fit on our devices. The cheapest wins come in two kinds. Some models run on runtimes we already ship: Canary-1B-v2 via `onnx-asr`, whisper-large-v3-turbo via faster-whisper/whisperx, Qwen3-ASR via OpenVINO GenAI. Others need no Backend code at all: OpenAI-compatible local servers (`mlx-audio`, WhisperKit's `argmax-cli`) and OpenRouter can be added as Providers today.

---

## 1. Recommendation

Ordered by expected value for German meeting audio. **[same runtime]** = no new heavy dependency (onnx-asr, faster-whisper/whisperx, optimum-intel/OpenVINO, or the existing Provider code).

### Quick benchmarks via the Provider route, no Backend code
Our `remote` Backend already POSTs multipart `file`/`model`/`response_format=verbose_json` to `/audio/transcriptions` (`voxscribe/transcription/backends/remote.py`). OpenRouter's STT endpoint accepts exactly that ([OpenRouter STT](https://openrouter.ai/docs/guides/overview/multimodal/stt.md)). So the following can be benchmarked **now** by adding OpenRouter as a Provider and only changing the model name:
1. `nvidia/nemotron-3.5-transcribe`: best published German number (FLEURS-de 2.73 %), too big for any of our devices (see §4.7).
2. `qwen/qwen3-asr-1.7b`: quality check before integrating it locally.
3. `mistralai/voxtral-small-24b-2507-stt` and `mistralai/voxtral-mini-3b-2507`.
4. `nvidia/parakeet-tdt-0.6b-v3` and `openai/whisper-large-v3`: control rows, so we can compare the same model on OpenRouter vs. local vs. KIT ToolBox.
5. Optional upper bound (closed): `elevenlabs/scribe-v2`, `assemblyai/universal-3-6-pro` (leaderboard DE 2.27 / 2.34 % for the earlier versions).

(Model IDs are from OpenRouter's live models API, `GET /api/v1/models?output_modalities=transcription`, fetched 2026-10-31.) Caveats: the 60 s upstream timeout and 25 MB multipart cap per request fit our 30 s/180 s FLAC chunks. Audio leaves the machine. Not every model returns `verbose_json` segments (see §6).

### Mac M2 Pro (16 GB)
1. **Canary-1B-v2 via `onnx-asr`** (`nemo-canary-1b-v2`) **[same runtime]**: same package and code path as our Parakeet Backend. FLEURS-de 4.40 % vs Parakeet 5.04 %, but CoVoST-de 5.53 % vs 4.84 %. A coin flip on paper, cheap to test.
2. **Qwen3-ASR-1.7B via `mlx-audio`** (MLX, GPU): FLEURS-de 3.92 / MLS-de 6.05 / CV-de 5.85 %, has language ID, 20 min per pass, German word timestamps via Qwen3-ForcedAligner. Test first through `mlx_audio.server` (OpenAI-compatible) as a localhost Provider.
3. **whisper-large-v3-turbo via WhisperKit `argmax-cli` local server, or `mlx-whisper`**: whisper quality class (our best result so far) on ANE/GPU. The WhisperKit server is OpenAI-compatible with word timestamps, so it can run as a localhost Provider with zero code.
4. **Parakeet v3 via `parakeet-mlx`**: same weights as today, so speed only. Do it only if Parakeet stays the default.
5. Cohere Transcribe (MLX port exists per the model card) and Granite Speech 4.1 2B (`mlx-audio`) are second tier. Cohere has no timestamps and no language ID. Granite has no published German WER.

### Windows + NVIDIA (CUDA 12.8)
1. **whisper-large-v3-turbo via faster-whisper/whisperx** **[same runtime]**: `large-v3-turbo` is a built-in faster-whisper name. It is one new entry in the model list, and alignment and diarization stay unchanged. It tells us whether turbo keeps large-v3's German lead at ~1.4x the speed.
2. **Canary-1B-v2 via `onnx-asr` on CUDA** **[same runtime]**.
3. **Qwen3-ASR-1.7B via `qwen-asr` / transformers** (native transformers support since 2026-06-26): new torch-based Backend, moderate dependency weight.
4. **Cohere Transcribe via transformers ≥ 5.4** (`CohereAsrForConditionalGeneration`): only if (3) is promising and we accept forcing the language.
5. Voxtral Mini 3B (~9.5 GB VRAM) only if the GPU has ≥ 12 GB. Otherwise test it via the Provider.
6. Nemotron 3.5 Transcribe locally only on a ≥ 24 GB GPU (IQ4 GGUF 19.7 GB + 1.7 GB encoder, NeMo-Speech.cpp). Otherwise test it via the Provider.

### Windows + Intel Arc 140V (OpenVINO)
1. **Qwen3-ASR-1.7B / 0.6B via `openvino-genai`**: officially supported architecture (`Qwen3ASRForConditionalGeneration`), word timestamps via Qwen3 Forced Aligner. The most promising GPU-accelerated German upgrade on this machine. New package, but the same OpenVINO toolchain family we already ship.
2. **whisper-large-v3-turbo via our existing optimum-intel path** **[same runtime]**: same `WhisperForConditionalGeneration` architecture as the medium model we export today. It is not on OpenVINO GenAI's example list, so the export needs to be verified.
3. **Canary-1B-v2 / Parakeet v3 via `onnx-asr` on CPU** **[same runtime]**: the baseline. onnx-asr does not document an OpenVINO execution provider.

### Drop from consideration
| Model | Why |
|---|---|
| Kyutai STT | English/French only, no German ([repo](https://github.com/kyutai-labs/delayed-streams-modeling)). |
| Meta Omnilingual ASR | 40 s input limit (except "Unlimited" variants), 7B = 25-30 GiB, leaderboard DE 4.55 % (LLM 7B) / 6.05 % (CTC 7B): worse than Parakeet. |
| NVIDIA Canary-Qwen-2.5B | English only. |
| Canary-1B-flash / 180M-flash | Older (4 languages), designed for < 40 s, and v2 supersedes them for German. |
| Nemotron 3.5 ASR Streaming 0.6B | FLEURS-de 8.3-9.8 % (worse than Parakeet v3). Streaming isn't needed. |
| Phi-4-multimodal-instruct | Leaderboard DE 3.96 % is good, but ~40 s recommended ASR input, 5.6B params, `trust_remote_code`, not on OpenRouter's STT list. Voxtral and Qwen3-ASR cover the same "LLM-ASR" slot more cheaply. |
| Nemotron 3.5 Transcribe *locally* on the Mac / Arc laptop | Smallest usable GGUF ≥ 13.6 GB + encoder: doesn't fit a 16 GB Mac. No OpenVINO/Intel path. Use it via the Provider only. |
| DiariZen (diarization) | Weights CC BY-NC 4.0 (research only). |

---

## 2. Comparison table

DE/EN WER in %, lower is better. "LB" = Open ASR Leaderboard paper v4, snapshot 2026-03-27 ([arXiv 2510.06961v4](https://arxiv.org/html/2510.06961v4)): DE = multilingual track average for German, EN = English short-form average. Sizes marked ≈ are params × bytes per weight estimates, not published sizes.

| Candidate | Licence | Params / size | DE WER | EN WER | Runtimes (today) | Word timestamps | Cheapest integration |
|---|---|---|---|---|---|---|---|
| **Parakeet TDT 0.6B v3** (shipped) | CC-BY-4.0 | 0.6B; INT8 ONNX ~640 MB | FLEURS 5.04, CoVoST 4.84, LB 4.20 | FLEURS 4.85, LB 6.32 | onnx-asr (CPU/CUDA/CoreML/DirectML), NeMo, parakeet-mlx, FluidAudio CoreML | yes (TDT tokens) | shipped |
| **Canary-1B-v2** | CC-BY-4.0 | 978M; ≈1 GB INT8 | FLEURS 4.40, CoVoST 5.53, LB 4.10 | FLEURS 4.50, LB 7.15 | onnx-asr (`nemo-canary-1b-v2`), NeMo, mlx-audio ("Canary") | NeMo: yes (via NFA + aux CTC). onnx-asr: token timestamps | **onnx-asr [same runtime]** |
| **Whisper large-v3** (shipped) | MIT | 1.55B | FLEURS 4.30, CoVoST 5.94, LB 4.26 | LB 7.44 | faster-whisper, OpenVINO, mlx-whisper, WhisperKit | via wav2vec2 (whisperx) or DTW | shipped |
| **Whisper large-v3-turbo** | MIT | 809M; WhisperKit compressed 626 MB | not published per language ("similar to large-v2") | LB 7.83 | faster-whisper (`large-v3-turbo`), onnx-asr (`onnx-community/whisper-large-v3-turbo`), mlx-whisper, WhisperKit, optimum-intel (unverified) | yes (DTW) / wav2vec2 | **faster-whisper [same runtime]** |
| **Qwen3-ASR-1.7B** | Apache-2.0 | ~2B BF16; ≈4 GB | FLEURS 3.92, MLS 6.05, CV 5.85, LB 4.12 | FLEURS 3.35, LB 5.76 | qwen-asr (transformers/vLLM), transformers (native), OpenVINO GenAI, mlx-audio | via Qwen3-ForcedAligner-0.6B (DE supported, ≤ 5 min) | OpenVINO GenAI (Arc) / mlx-audio (Mac) / transformers (CUDA) |
| Qwen3-ASR-0.6B | Apache-2.0 | ~0.6B | FLEURS 6.48, MLS 9.52, CV 9.44 | FLEURS 4.39 | as above | as above | as above, not better than Parakeet on DE |
| **Cohere Transcribe 03-2026** | Apache-2.0 (gated: accept conditions) | ~2B BF16; ≈4 GB | LB 3.84 | LB 5.42 | transformers ≥ 5.4, vLLM, MLX, GGUF (transcribe.cpp) | **no** | transformers; needs wav2vec2 alignment |
| **Voxtral Mini 3B 2507** | Apache-2.0 | "3B" (card metadata 5B); ~9.5 GB GPU RAM | not in text (charts) | LB (paper Table 5) 7.05 | vLLM, transformers ≥ 4.54, mlx-audio | not documented | Provider (OpenRouter) |
| **Voxtral Small 24B 2507** | Apache-2.0 | 24B; ~55 GB GPU RAM | LB 3.01 | LB 6.62 | vLLM, transformers | not documented | Provider only |
| Voxtral Mini 4B Realtime 2602 | Apache-2.0 | ~4B BF16 (≈8.9 GB, *unverified*) | FLEURS 6.19 @480 ms | FLEURS 4.90 @480 ms | vLLM, transformers ≥ 5.2, MLX ports | not documented | drop (streaming; worse DE) |
| **Nemotron 3.5 Transcribe 30B-A4B** | OpenMDW-1.1 (commercial OK) | 34B total / ~4.3B active; GGUF 13.6-35 GB + 1.7 GB encoder | FLEURS 2.73, LB-style DE mean 2.81 (vendor) | 9-set mean 4.71, FLEURS 3.35 (vendor) | vLLM, transformers, SGLang-Omni, NeMo-Speech.cpp (GGUF), mlx-audio | yes (CTC head) + speaker attribution | Provider only |
| Granite Speech 4.1 2B | Apache-2.0 | 2B BF16 | not in text (charts) | LB 5.33 (card) | transformers, vLLM, GGUF, mlx-audio | only "plus" variant | mlx-audio / transformers, second tier |
| Phi-4-multimodal-instruct | MIT | 5.6B | LB 3.96 | LB 6.02 | transformers (remote code), vLLM, ONNX variant | not documented | drop |
| Omnilingual ASR 7B | Apache-2.0 | 6.5-7.8B; 25-30 GiB | LB 4.55 (LLM) / 6.05 (CTC) | – | fairseq2 | not documented | drop |

---

## 3. Per-candidate details

### 3.1 NVIDIA Parakeet TDT 0.6B v3 (baseline, shipped)
- Licence CC-BY-4.0 (commercial and non-commercial use). 600M params, 25 European languages incl. de/en. Released 2025-08-14. Word, segment and char timestamps (NeMo). Up to 24 min with full attention on an A100, up to 3 h with local attention ([model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)).
- WER: FLEURS de 5.04 / en 4.85, CoVoST de 4.84 / en 6.80 ([paper App. B, Tables 11/13](https://arxiv.org/html/2509.14128v2)). Leaderboard EN 6.32, RTFx 3330, DE 4.20 ([LB v4](https://arxiv.org/html/2510.06961v4)).
- Runtimes: `onnx-asr` 0.12.0 (2026-07-15, [PyPI](https://pypi.org/project/onnx-asr/)), model name `nemo-parakeet-tdt-0.6b-v3`, execution providers CPU, CUDA, TensorRT, CoreML, DirectML, ROCm, WebGPU. OpenVINO is not mentioned ([repo](https://github.com/istupakov/onnx-asr), [usage](https://istupakov.github.io/onnx-asr/usage/)). `parakeet-mlx` 0.5.3 (2026-10-01), Apache-2.0, default `mlx-community/parakeet-tdt-0.6b-v3`, word/sentence timestamps, 120 s chunks with 15 s overlap, needs ffmpeg for the CLI ([repo](https://github.com/senstella/parakeet-mlx)). FluidAudio CoreML/ANE port `FluidInference/parakeet-tdt-0.6b-v3-coreml`, Swift SDK, claims ~190x RT on M4 Pro ([repo](https://github.com/FluidInference/FluidAudio)).
- "Parakeet Ultra" (Moondream post-training of v3, CC-BY-4.0, CoreML builds): third-party FLEURS-24 means of 11.67 vs 14.81 % and 10.77 vs 13.00 % ([TerminatorPower](https://huggingface.co/TerminatorPower/parakeet-ultra-coreml), [Glimpse](https://huggingface.co/Glimpse-Dictation/Parakeet-Ultra-coreml)). *Unverified*, community claims. Not checked against Moondream's own card. Worth a look only if Parakeet stays the default.

### 3.2 NVIDIA Canary-1B-v2
- CC-BY-4.0, ~978M params, 25 languages, ASR + translation. Word and segment timestamps for transcription. Long-form via dynamic chunking with 1 s overlap, automatic for files > 40 s. Released 2025-08-14. NeMo main / ≥ 2.5, transformers from source ([model card](https://huggingface.co/nvidia/canary-1b-v2)). Timestamps come from NeMo Forced Aligner with an auxiliary CTC model ([arXiv 2509.14128](https://arxiv.org/abs/2509.14128)).
- **The FLEURS-de claim from secondary blogs is correct**: Canary-1B-v2 4.40 vs Parakeet v3 5.04 (Table 11). But the same table has whisper-large-v3 at **4.30**, and on CoVoST2-de Canary is *worse* (5.53 vs 4.84; whisper-large-v3 5.94) ([paper App. B](https://arxiv.org/html/2509.14128v2)). Leaderboard DE: Canary 4.10, Parakeet 4.20, whisper-large-v3 4.26 ([LB v4](https://arxiv.org/html/2510.06961v4)). Expect a small gain at most.
- Speed: LB RTFx 749 vs Parakeet 3330 (≈4.5x slower than Parakeet, still ≈5x faster than whisper-large-v3 at 146).
- onnx-asr: `nemo-canary-1b-v2` is in the supported list. Also `istupakov/canary-1b-flash-onnx` and `istupakov/canary-180m-flash-onnx` ([usage](https://istupakov.github.io/onnx-asr/usage/)). onnx-asr gives token-level timestamps (`.with_timestamps()`), which needs checking for the AED decoder. If they are missing, keep wav2vec2 alignment. Most onnx-asr models accept ~20-30 s without VAD, and we already cut with Silero VAD for Parakeet.
- Integration: a second model name in `backends/parakeet.py` (same onnx-asr loader). Canary needs a source-language argument in its prompt (to verify in onnx-asr). Without one we'd keep our text-based language guess.
- Canary-1B-flash (883M, CC-BY-4.0, en/de/fr/es only, MLS-de 4.36, CV16.1-de 4.09, designed for < 40 s, experimental timestamps) ([card](https://huggingface.co/nvidia/canary-1b-flash)): superseded by v2 for us.

### 3.3 Whisper large-v3-turbo
- MIT, ~809M params, large-v3 with 4 instead of 32 decoder layers. 30 s receptive field, so long-form uses sliding or chunked decoding. Word timestamps via `return_timestamps="word"` ([model card](https://huggingface.co/openai/whisper-large-v3-turbo)). LB EN 7.83 / RTFx 200 vs large-v3 7.44 / 146 ([LB v4](https://arxiv.org/html/2510.06961v4)). Not in the LB German table.
- OpenAI (2024-10-01): turbo "performs similarly to large-v2" across languages, with larger losses on some (Thai, Cantonese). It is **not trained for translation** ([announcement](https://github.com/openai/whisper/discussions/2363)). No published German number. Our benchmark must measure it.
- **The faster-whisper name is verified**: `large-v3-turbo` and `turbo` both map to `mobiuslabsgmbh/faster-whisper-large-v3-turbo` in `faster_whisper/utils.py` ([source](https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/faster_whisper/utils.py)). Also `distil-large-v3.5` exists (English only). faster-whisper latest 1.2.1 (2025-10-31, [PyPI](https://pypi.org/project/faster-whisper/)).
- Other runtimes: onnx-asr `onnx-community/whisper-large-v3-turbo` (the docs warn some old onnx-community fp16 exports are broken) ([usage](https://istupakov.github.io/onnx-asr/usage/)). `mlx-whisper` 0.4.3, MIT, `word_timestamps=True` ([PyPI](https://pypi.org/project/mlx-whisper/)). WhisperKit (MIT) recommends compressed Large v3 Turbo (626 MB) on macOS. Its `argmax-cli` local server implements `/v1/audio/transcriptions` with `json`/`verbose_json` and word/segment timestamps, model fixed at launch ([repo](https://github.com/argmaxinc/WhisperKit)). OpenVINO GenAI lists whisper-large-v3 but not turbo ([supported models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)). It is the same architecture, so the export is expected to work but is unverified.
- Caveats: the usual Whisper risks (hallucination on silence, one language per 30 s window, translating instead of transcribing when the language is mis-set, see our issue #38).

### 3.4 Qwen3-ASR (1.7B / 0.6B) and Qwen3-ForcedAligner
- Apache-2.0. 1.7B card shows ~2B params BF16. 30 languages **including German (de)** plus 22 Chinese dialects. Released 2026-01-29. Native transformers support announced 2026-06-26 ([GitHub](https://github.com/QwenLM/Qwen3-ASR), [model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)). (The model card's own language list as rendered omitted German. The GitHub README lists `de`, and the tech report has German results, so I trust those.)
- German WER (tech report Table A.2): **1.7B: FLEURS 3.92, MLS 6.05, Common Voice 5.85**. 0.6B: 6.48 / 9.52 / 9.44 ([arXiv 2601.21337](https://arxiv.org/html/2601.21337)). LB DE 4.12, EN 5.76 at RTFx 148 ([LB v4](https://arxiv.org/html/2510.06961v4)). English FLEURS 3.35 / CV 7.39 ([card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)).
- Language ID built in (1.7B avg 97.9 % accuracy) ([card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)). Max **20 min per pass** ([tech report](https://arxiv.org/html/2601.21337)). Streaming only with the vLLM backend, without timestamps ([card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)).
- Timestamps: separate **Qwen3-ForcedAligner-0.6B** (Apache-2.0, 11 languages incl. German, up to 5 min per call). Its own benchmark (alignment error, ms, lower is better): German raw Qwen 46.5 vs NFA 122.7 vs WhisperX 165.1. On 300 s concatenated audio: 52.9 vs 246.7 vs 2708.4 ([aligner card](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)). Vendor benchmark, competitor versions not stated.
- Runtimes: `qwen-asr` 0.0.6 (transformers or vLLM backend, [PyPI](https://pypi.org/project/qwen-asr/)). transformers native. **OpenVINO GenAI** supports `Qwen3ASRForConditionalGeneration` (0.6B and 1.7B), word timestamps via Qwen3 Forced Aligner for de/en/fr/es/it and more, ~1200 s per chunk (180 s with word timestamps), no beam search ([use case](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/), [supported models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)). `openvino-genai` 2026.4.1.0 ([PyPI](https://pypi.org/project/openvino-genai/)). **mlx-audio** lists Qwen3-ASR and Qwen3-ForcedAligner ([repo](https://github.com/Blaizzy/mlx-audio)). No ONNX/onnx-asr support found.
- Integration: Mac via mlx-audio (or its server as a Provider). Arc via openvino-genai (new Backend, export via optimum-cli + qwen-asr). CUDA via qwen-asr/transformers. Pure torch, moderate weight.
- Caveats: LLM decoder, so hallucination and repetition risk on silence is plausible (not documented). Code-switching behaviour within one pass isn't documented, and the 20 min cap needs chunking for long meetings.

### 3.5 Cohere Transcribe (cohere-transcribe-03-2026)
- Apache-2.0, but **gated** (accept conditions + login). ~2B BF16, 14 languages incl. de/en. Released 2026-03-26 ([model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026), [blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release)).
- EN LB avg 5.42 (AMI 8.15) ([card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)). DE: LB 3.84 ([LB v4](https://arxiv.org/html/2510.06961v4)). The vendor gives German only as a chart.
- **No timestamps, no diarization, no language ID** (language must be passed). Code-switched audio is inconsistent. Tends to transcribe non-speech sounds, so the card recommends a VAD/noise gate. Splits long audio automatically ([card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)).
- Runtimes: transformers ≥ 5.4 (`CohereAsrForConditionalGeneration`), vLLM (OpenAI-style transcription endpoint), MLX, Rust, transformers.js/WebGPU, GGUF via transcribe.cpp ([card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)). Not on OpenRouter's STT list (2026-10-31).
- Verdict: strong German on paper, but no language ID is a poor fit for our mixed-language meetings. Second tier.

### 3.6 Mistral Voxtral
- **Voxtral Mini 3B 2507**: Apache-2.0, 8 languages incl. de/en, dedicated transcription mode with automatic language detection, 30 min transcription / 40 min understanding (32k context), ~9.5 GB GPU RAM bf16, vLLM ≥ 0.10 / transformers ≥ 4.54. Timestamps not mentioned ([card](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)). The card's metadata says "5B params" (likely incl. the audio encoder, *unverified*). EN LB 7.05 ([paper Table 5](https://arxiv.org/html/2509.14128v2)).
- **Voxtral Small 24B 2507**: Apache-2.0, ~55 GB GPU RAM ([card](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)). LB DE **3.01** (best open model in LB v4), EN 6.62, RTFx 54 ([LB v4](https://arxiv.org/html/2510.06961v4)). Provider only.
- **Voxtral Mini 4B Realtime 2602**: Apache-2.0, ~4B, 13 languages, streaming with 80 ms-2.4 s delay. FLEURS-de 6.19 / en 4.90 at 480 ms, vs 3.54 / 3.32 for the offline (closed) Voxtral Mini Transcribe 2.0. vLLM, transformers ≥ 5.2 ([card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)). Worse than Parakeet on German. Drop.
- **Voxtral Mini Transcribe V2** (2026-02-04): API only (closed weights), diarization, word timestamps, up to 3 h, $0.003/min, ~4 % FLEURS ([Mistral](https://mistral.ai/news/voxtral-transcribe-2)). An NYU RITS article claims it is open-weights. That contradicts Mistral, and I trust Mistral. OpenRouter also lists `mistralai/voxtral-mini-transcribe-3-0` / `voxtral-small-transcribe-3-0`. I found no Mistral primary source for a "3.0" release (*unverified*: licence and weights unknown).

### 3.7 NVIDIA Nemotron 3.5 Transcribe (30B-A4B) and Nemotron 3.5 ASR Streaming 0.6B
- **The release date is confirmed: 2026-10-28** ([model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B), [HF blog](https://huggingface.co/blog/nvidia/nemotron-transcribe-granary)). Licence **OpenMDW-1.1**, "ready for commercial or non-commercial use" ([card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)). OpenMDW is MIT/Apache-like, but not OSI-approved yet (review ongoing Aug 2026) ([OSI list](https://lists.opensource.org/pipermail/license-review_lists.opensource.org/2026-August/006125.html), [repo](https://github.com/OpenMDW/openmdw)).
- **Params: 34B total / ~4.3B active** (not "~30B" exactly; the name says 30B-A4B, and secondary sources vary 19-34B). 40 languages. Word timestamps via a CTC head. Speaker-attributed transcription. Up to 1 h per input (65,536-token context). vLLM 0.32, transformers, SGLang-Omni, GGML/NeMo-Speech.cpp. Reference hardware H100 80 GB, also RTX 5090, L40S (FP8) etc. ([card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)).
- WER (vendor): FLEURS-de **2.73**, multilingual DE mean 2.81. EN 9-set mean 4.71, FLEURS-en 3.35 ([card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)). The blog quotes LB EN 5.33 and RTFx 462 ([blog](https://huggingface.co/blog/nvidia/nemotron-transcribe-granary)). The two EN numbers are different test sets/snapshots, so prefer the card. The blog says it is "built for recorded audio, not live streaming".
- Local size: community GGUF Q8_0 35 GB, IQ4_NL 19.7 GB, Q2_0_M 13.6 GB, + 1.7 GB F16 encoder. Metal unverified by the author ([AtomicChat GGUF](https://huggingface.co/AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF), community). Not viable on a 16 GB Mac or the Arc laptop. On OpenRouter as `nvidia/nemotron-3.5-transcribe`.
- **Nemotron 3.5 ASR Streaming 0.6B** (2026-06-04): OpenMDW-1.1, cache-aware FastConformer-RNNT, 40 locales. FLEURS-de 9.81→8.31 / en 9.43→7.91 from 80 ms to 1.12 s chunks. NeMo 26.06, transformers ≥ 5.13, GGUF, Linux + NVIDIA ([card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)). Worse than Parakeet offline. Drop.

### 3.8 IBM Granite Speech
- Granite Speech 4.1 2B (2026-04-29): Apache-2.0, en/fr/de/es/pt/ja. LB EN mean 5.33, RTFx 231. German only in charts. transformers ≥ 4.52.1, vLLM (OpenAI-compatible), GGUF, mlx-audio ≥ 0.4.1. The `-plus` variant adds word timestamps and speaker-attributed ASR. `-nar` is non-autoregressive for throughput ([card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)). LB v4 has Granite 4.0 1B EN 5.52, 3.3 8B 5.76, 3.3 2B 6.00, but no German row ([LB v4](https://arxiv.org/html/2510.06961v4)).
- Verdict: English-strong. German quality is unknown, so test it only if it's cheap (mlx-audio server on the Mac).

### 3.9 Microsoft Phi-4-multimodal-instruct
- MIT, ~5.6B, speech in en/zh/de/fr/it/ja/es/pt. ~40 s recommended audio for general use (30 min for summarization). transformers (`trust_remote_code`), vLLM, a separate ONNX repo. Released Feb 2025 ([card](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)). LB DE 3.96, EN 6.02 ([LB v4](https://arxiv.org/html/2510.06961v4)). The "strong German" report holds, but it needs 40 s chunking and has no timestamps. Dropped (see §1).

### 3.10 Others checked
- **Kyutai STT**: `stt-1b-en_fr`, `stt-2.6b-en`, CC-BY-4.0, word timestamps, PyTorch/Rust/MLX. **No German** ([repo](https://github.com/kyutai-labs/delayed-streams-modeling)).
- **Meta Omnilingual ASR**: Apache-2.0 code and models, CTC and LLM 300M-7B (v2 Dec 2025), < 40 s input except "Unlimited" variants, 7B 25-30 GiB ([repo](https://github.com/facebookresearch/omnilingual-asr)). LB DE 4.55/6.05.
- **Canary-Qwen-2.5B**: English only (CC-BY-4.0) ([card](https://huggingface.co/nvidia/canary-qwen-2.5b)).
- **Fun-ASR-Nano / SenseVoice-Small**: supported by OpenVINO GenAI ([supported models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)). Not researched for German. Mostly Chinese-focused (*unverified*).
- **Closed reference points** (LB v4 DE): ElevenLabs Scribe v2 2.27, AssemblyAI Universal 3 Pro 2.34, Speechmatics Enhanced 2.84 ([LB v4](https://arxiv.org/html/2510.06961v4)).

---

## 4. Diarization

| Option | Licence | Speakers | Runtime | Notes |
|---|---|---|---|---|
| **pyannote speaker-diarization-community-1** (in use) | CC-BY-4.0, **gated** (accept conditions, HF token) | no hard max (`min/max_speakers`) | pyannote.audio 4.x (4.0.7, 2026-06-30), CPU/GPU, offline from a local clone | DER vs 3.1: AMI-IHM 18.8→17.0, AMI-SDM 22.7→19.9, DIHARD3 21.4→20.2. Has `exclusive_speaker_diarization` output for aligning with transcripts ([card](https://huggingface.co/pyannote/speaker-diarization-community-1)) |
| **NVIDIA Nemotron-3-Diarization** (2026-09-23) | OpenMDW-1.1 (commercial OK) | **8** | NeMo, NeMo-Speech.cpp, transformers (from source). Card lists Linux + NVIDIA GPU, no ONNX | ~100M params, offline (30.4 s buffer) and streaming (0.32-1.04 s). DER AMI-SDM 11.14 vs Sortformer v2.1 21.42, DIHARD3 12.73. Vendor numbers, partly on forced-alignment references, so not directly comparable to pyannote's ([card](https://huggingface.co/nvidia/nemotron-3-diarization)). Community ONNX/MLX conversions exist (*unverified*) |
| NVIDIA Streaming Sortformer 4spk v2.1 | NVIDIA Open Model License | **4** (degrades at ≥ 5) | NeMo / PyTorch | DER AMI-IHM 15.90, DIHARD3 19.49 at 30.4 s latency ([card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)). The 4-speaker cap is too low for meetings |
| DiariZen (BUT) | code MIT, **weights CC BY-NC 4.0 (research only)** | – | modified pyannote 3.1 | DER AMI-SDM 13.9 (Large-s80-v2) vs pyannote 3.1 22.4 ([repo](https://github.com/BUTSpeechFIT/DiariZen)). Not usable commercially |

FluidAudio and WhisperKit's SpeakerKit both ship pyannote community-1 on CoreML ([FluidAudio](https://github.com/FluidInference/FluidAudio), [WhisperKit](https://github.com/argmaxinc/WhisperKit)). That could speed up diarization on the Mac without changing the model, but needs a Swift helper. Our voice-print matching depends on pyannote embeddings, so a switch to Nemotron-3-Diarization would also need a new embedding source. Recommendation: keep community-1. Benchmark Nemotron-3-Diarization on the NVIDIA PC only if diarization errors dominate our results.

## 5. Forced alignment

| Option | Licence | German | Notes |
|---|---|---|---|
| wav2vec2 (whisperx, in use) | per model | yes | Baseline. Degrades over minutes-long segments (our text-only Provider experience) |
| **Qwen3-ForcedAligner-0.6B** | Apache-2.0 | yes (11 languages) | ≤ 5 min per call. Vendor alignment error German 46.5 ms vs WhisperX 165.1 ms, 300 s audio 52.9 vs 2708 ms ([card](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)). Runs in qwen-asr, OpenVINO GenAI, mlx-audio. **The most promising replacement**, also for text-only Providers and for Cohere Transcribe |
| NeMo Forced Aligner (NFA) | NeMo (Apache-2.0 toolkit) | needs a German CTC model | CTC / hybrid models in CTC mode, token/word/segment, hour-long audio. "Only English has been tested" ([docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html)). Heavy (NeMo) |
| ctc-forced-aligner (MMS-300m) | code BSD-2, **default weights CC-BY-NC 4.0** | yes (1130 languages, romanized) | Not for commercial use with the default model ([repo](https://github.com/MahmoudAshraf97/ctc-forced-aligner)) |

Backends that already give word timestamps (Parakeet/Canary via onnx-asr tokens, faster-whisper `word_timestamps`, Nemotron 3.5 Transcribe, Granite plus) could skip alignment like the Apple Backend does. Whether their word timing is good enough for our speaker assignment is a benchmark question.

## 6. Provider route (OpenRouter / OpenAI-compatible)

- **OpenRouter has a real `/api/v1/audio/transcriptions` endpoint**, not only chat-input audio ([STT guide](https://openrouter.ai/docs/guides/overview/multimodal/stt.md)). Its general audio guide documents only `input_audio` in Chat Completions / Responses ([audio guide](https://openrouter.ai/docs/guides/overview/multimodal/audio)). The STT guide is the relevant one. It accepts multipart `file`, `model`, `language`, `temperature`, `response_format` (`json` | `verbose_json`), and `timestamp_granularities`. `text`/`srt`/`vtt` return 400, and `prompt` is ignored. `verbose_json` adds `language`, `duration`, `segments` (optional `speaker`) and `words`, depending on the provider. Some models (e.g. `gpt-4o-transcribe`, `mai-transcribe-1.5`) reject `verbose_json`. Limits: 25 MB multipart, **60 s upstream timeout per request** ([STT guide](https://openrouter.ai/docs/guides/overview/multimodal/stt.md)). Our Provider code (multipart, `verbose_json`, 30/180 s FLAC chunks, text-only fallback probe) should work unchanged. To verify: whether each model returns segments, and whether 180 s chunks stay under 60 s of processing.
- OpenRouter STT models relevant to us (live API, 2026-10-31; price per audio second as returned in `pricing.prompt`, interpreted as per second because `openai/whisper-1` = 0.0001 = OpenAI's $0.006/min): `nvidia/nemotron-3.5-transcribe` (0.0000333 ≈ $0.12/h), `nvidia/parakeet-tdt-0.6b-v3` ($0.09/h), `qwen/qwen3-asr-1.7b` ($0.027/h), `qwen/qwen3-asr-0.6b`, `mistralai/voxtral-mini-3b-2507` ($0.06/h), `mistralai/voxtral-small-24b-2507-stt` ($0.18/h), `openai/whisper-large-v3` ($0.027/h), `openai/whisper-large-v3-turbo`, `nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b`, plus closed ones (ElevenLabs Scribe v2, AssemblyAI Universal 3.5/3.6 Pro, Deepgram Nova-3, Google Chirp 3, gpt-transcribe, Voxtral Transcribe 3.0). **Not offered**: Canary-1B-v2, Cohere Transcribe, Granite Speech, Phi-4 (source: `GET https://openrouter.ai/api/v1/models?output_modalities=transcription`).
- **Localhost OpenAI-compatible servers** let us benchmark Mac models as Providers with no Backend code: `mlx_audio.server` exposes `POST /v1/audio/transcriptions` for all mlx-audio STT models (Qwen3-ASR, Parakeet, Voxtral, Granite, Canary, Nemotron 3.5 Transcribe, ...) ([mlx-audio](https://github.com/Blaizzy/mlx-audio)); mlx-audio 0.6.3, 2026-10-28 ([PyPI](https://pypi.org/project/mlx-audio/)). WhisperKit's `argmax-cli` serves `/v1/audio/transcriptions` with `verbose_json` and word timestamps ([WhisperKit](https://github.com/argmaxinc/WhisperKit)). vLLM serves Cohere Transcribe, Granite and Qwen3-ASR with OpenAI-style transcription endpoints ([Cohere card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026), [Granite card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b)). That needs Linux/WSL with a GPU.
- KIT ToolBox: which models it serves beyond whisper-large-v3 was not researched here (it is not a public source).

## 7. Open questions only our benchmark can answer

1. Does any candidate beat whisper-large-v3 (KIT ToolBox, 4.0 %) on our conversational German, not just FLEURS? Do Canary-1B-v2 or large-v3-turbo beat Parakeet v3 (7.0 %)?
2. Robustness to alternating speakers in a two-voice mix (SpeechAnalyzer dropped passages): especially LLM-decoders (Qwen3-ASR, Voxtral, Nemotron), which may skip or repeat.
3. Mixed German/English meetings: language ID per chunk (Qwen3-ASR, Voxtral auto-detect) vs. forced language (Cohere, Canary).
4. Hallucination on silence and pauses in real calls (Whisper family, Cohere per its card, LLM-ASR).
5. Whether onnx-asr's Canary token timestamps, faster-whisper word timestamps, or Qwen3-ForcedAligner beat wav2vec2 for our speaker assignment (`speakers.py` and the track split).
6. Real speed on the M2 Pro (onnx-asr CPU vs parakeet-mlx vs mlx-audio vs WhisperKit ANE) and on the Arc 140V (OpenVINO GenAI Qwen3-ASR vs our OpenVINO Whisper medium), including model load.
7. Whether the large-v3-turbo OpenVINO export works with our `optimum-intel` path (not on OpenVINO's list).
8. OpenRouter: segments vs text-only per model, and the 60 s timeout with our chunk sizes.

## 8. Sources

Model cards and papers (publisher):
- Parakeet TDT 0.6B v3: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Canary-1B-v2: https://huggingface.co/nvidia/canary-1b-v2
- Canary-1B-v2 & Parakeet-TDT-0.6B-v3 paper (v2, 2025-09-25), App. B Tables 11-13: https://arxiv.org/abs/2509.14128, https://arxiv.org/html/2509.14128v2
- Canary-1B-flash: https://huggingface.co/nvidia/canary-1b-flash
- Canary-Qwen-2.5B: https://huggingface.co/nvidia/canary-qwen-2.5b
- Whisper large-v3-turbo: https://huggingface.co/openai/whisper-large-v3-turbo ; announcement (2024-10-01): https://github.com/openai/whisper/discussions/2363
- Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR , https://huggingface.co/Qwen/Qwen3-ASR-1.7B , tech report https://arxiv.org/html/2601.21337
- Qwen3-ForcedAligner: https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
- Cohere Transcribe: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 , https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release (2026-03-26)
- Voxtral: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507 , https://huggingface.co/mistralai/Voxtral-Small-24B-2507 , https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602 , https://mistral.ai/news/voxtral-transcribe-2 (2026-02-04)
- Nemotron 3.5 Transcribe: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B , https://huggingface.co/blog/nvidia/nemotron-transcribe-granary (2026-10-28)
- Nemotron 3.5 ASR Streaming 0.6B: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- Granite Speech 4.1 2B: https://huggingface.co/ibm-granite/granite-speech-4.1-2b
- Phi-4-multimodal-instruct: https://huggingface.co/microsoft/Phi-4-multimodal-instruct
- Omnilingual ASR: https://github.com/facebookresearch/omnilingual-asr
- Kyutai STT: https://github.com/kyutai-labs/delayed-streams-modeling
- Open ASR Leaderboard paper v4 (snapshot 2026-03-27, published 2026-03-30): https://arxiv.org/html/2510.06961v4 ; live leaderboard (not machine-readable at fetch time): https://huggingface.co/spaces/hf-audio/open_asr_leaderboard

Runtimes:
- onnx-asr: https://github.com/istupakov/onnx-asr , https://istupakov.github.io/onnx-asr/usage/ , https://pypi.org/project/onnx-asr/
- faster-whisper: https://github.com/SYSTRAN/faster-whisper , model map https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/faster_whisper/utils.py
- parakeet-mlx: https://github.com/senstella/parakeet-mlx
- mlx-whisper: https://pypi.org/project/mlx-whisper/
- mlx-audio: https://github.com/Blaizzy/mlx-audio
- WhisperKit: https://github.com/argmaxinc/WhisperKit
- FluidAudio: https://github.com/FluidInference/FluidAudio
- OpenVINO GenAI: https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/ , https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/
- PyPI versions (fetched 2026-10-31): onnx-asr, parakeet-mlx, mlx-audio, faster-whisper, qwen-asr, openvino-genai, pyannote.audio, nemo-toolkit

Diarization / alignment:
- pyannote community-1: https://huggingface.co/pyannote/speaker-diarization-community-1
- Nemotron-3-Diarization: https://huggingface.co/nvidia/nemotron-3-diarization
- Streaming Sortformer v2.1: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- DiariZen: https://github.com/BUTSpeechFIT/DiariZen
- NeMo Forced Aligner: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html
- ctc-forced-aligner: https://github.com/MahmoudAshraf97/ctc-forced-aligner
- OpenMDW licence: https://github.com/OpenMDW/openmdw , OSI review https://lists.opensource.org/pipermail/license-review_lists.opensource.org/2026-August/006125.html

Provider:
- OpenRouter STT: https://openrouter.ai/docs/guides/overview/multimodal/stt.md ; audio input: https://openrouter.ai/docs/guides/overview/multimodal/audio ; models API: https://openrouter.ai/api/v1/models?output_modalities=transcription

Secondary / community (marked unverified above):
- Parakeet Ultra CoreML: https://huggingface.co/TerminatorPower/parakeet-ultra-coreml , https://huggingface.co/Glimpse-Dictation/Parakeet-Ultra-coreml
- Nemotron 3.5 Transcribe GGUF: https://huggingface.co/AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF
