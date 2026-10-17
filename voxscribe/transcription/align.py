"""Forced alignment (wav2vec2) of transcribed segments to word timestamps."""

import numpy as np

from voxscribe.transcription.devices import free_torch_memory


def align_by_language(result: dict, audio: np.ndarray, torch_device: str,
                       align_model_dir: str | None, align_cache_only: bool,
                       progress_callback=None) -> dict:
    """WhisperX forced alignment, one wav2vec2 model per segment language.

    Segments may carry their own "language" (transcribe_remote() in
    auto-detect mode, for meetings that switch languages); all others use
    result["language"]. Each language group is aligned with its own model,
    then everything is merged back in time order. If no model can be loaded
    for a minor language, its segments fall back to the main language's model.
    """
    import whisperx

    main = result["language"]
    groups: dict[str, list[dict]] = {}
    for seg in result["segments"]:
        groups.setdefault(seg.get("language") or main, []).append(seg)
    # Main language first, so a failing minor language can fall back to it.
    order = sorted(groups, key=lambda lang: (lang != main, lang))
    if len(order) > 1:
        print("     Languages: " + ", ".join(f"{lang} ({len(groups[lang])} segments)" for lang in order))

    total = sum(len(g) for g in groups.values()) or 1
    done = 0
    segments, word_segments = [], []
    for lang in order:
        group = groups[lang]
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=lang, device=torch_device,
                model_dir=align_model_dir, model_cache_only=align_cache_only,
            )
        except Exception as e:
            if lang == main:
                raise
            print(f"     [WARNING] No alignment model for '{lang}' ({e}) - using '{main}'")
            model_a, metadata = whisperx.load_align_model(
                language_code=main, device=torch_device,
                model_dir=align_model_dir, model_cache_only=align_cache_only,
            )

        def _group_progress(pct, _start=done, _n=len(group)):
            if progress_callback:
                progress_callback((_start + pct / 100 * _n) / total * 100)

        aligned = whisperx.align(
            group, model_a, metadata, audio, torch_device,
            return_char_alignments=False,
            progress_callback=_group_progress,
        )
        for seg in aligned["segments"]:
            seg["language"] = lang
        segments.extend(aligned["segments"])
        word_segments.extend(aligned.get("word_segments", []))
        done += len(group)
        del model_a
        free_torch_memory(torch_device)

    segments.sort(key=lambda s: s.get("start", 0.0))
    word_segments.sort(key=lambda w: w.get("start", 0.0))
    return {"segments": segments, "word_segments": word_segments, "language": main}
