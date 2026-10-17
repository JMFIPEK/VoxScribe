"""Language helpers: normalizing server-reported names and guessing from text."""

import functools
import re


@functools.cache
def _language_tables() -> tuple[dict[str, str], dict[str, str]]:
    """(ISO code -> name, lowercase full name -> ISO code), from whisper's own
    list. Loaded on first use: importing whisperx at module load would slow
    every importer of this module.

    Full names (e.g. "german", as returned by OpenAI-compatible APIs) are
    mapped to ISO-639-1 codes (e.g. "de"), which is what whisperx expects for
    alignment."""
    from whisperx.utils import LANGUAGES
    return LANGUAGES, {name.lower(): code for code, name in LANGUAGES.items()}


def _normalize_language_code(raw: str | None, fallback: str | None = None) -> str | None:
    """Normalizes a language identifier (name or code) returned by a server API."""
    if not raw:
        return fallback
    names, name_to_code = _language_tables()
    raw = raw.strip().lower()
    if raw in names:
        return raw
    return name_to_code.get(raw, fallback)


# Very common function words per language - enough to tell the GUI's
# languages (see qt_app/constants.py LANGUAGES) apart reliably from a few
# sentences of transcribed text. Used where the transcription backend
# returns no language of its own (text-only providers, Parakeet), since the
# alignment step needs the correct language-specific wav2vec2 model.
_STOPWORDS = {
    "de": {"der", "die", "das", "und", "ist", "nicht", "ich", "wir", "es", "ein", "eine",
           "zu", "mit", "auf", "für", "auch", "dass", "sie", "den", "dem", "von", "haben",
           "aber", "noch", "jetzt", "also", "oder", "wenn", "schon", "mal", "kann", "sind"},
    "en": {"the", "and", "is", "not", "we", "it", "a", "to", "of", "that", "you", "this",
           "for", "with", "on", "have", "are", "be", "but", "so", "what", "if", "can",
           "there", "they", "do", "was", "would", "just", "or", "will", "at"},
    "fr": {"le", "la", "les", "et", "est", "pas", "je", "nous", "un", "une", "des", "que",
           "pour", "avec", "sur", "dans", "ce", "qui", "mais", "vous", "il", "elle", "du"},
    "es": {"el", "la", "los", "las", "y", "es", "no", "yo", "un", "una", "que", "de",
           "para", "con", "en", "por", "pero", "lo", "se", "del", "como", "está", "muy"},
    "it": {"il", "la", "le", "e", "è", "non", "io", "noi", "un", "una", "che", "di",
           "per", "con", "su", "ma", "sono", "gli", "della", "anche", "questo", "molto"},
}


_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _guess_language_from_text(text: str) -> str | None:
    """Best-guess ISO-639-1 code from transcribed text, or None if there's
    too little evidence (short/empty text, or no clear winner)."""
    words = _WORD_RE.findall((text or "").lower())
    if len(words) < 8:
        return None
    scores = {lang: sum(1 for w in words if w in sw) for lang, sw in _STOPWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_score), (_, second_score) = ranked[0], ranked[1]
    if best_score < 3 or best_score < 1.5 * second_score:
        return None
    return best
