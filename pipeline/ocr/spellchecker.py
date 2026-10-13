# Copied from C:\Projects\OCR (user's earlier OCR project); pkg_resources replaced by importlib.resources.
"""Corpus-based spell correction and Hebrew final-letter fixing for OCR output."""

from __future__ import annotations

import re
from pathlib import Path

# Matches any character in the Hebrew Unicode block (U+0590–U+05FF).
_HEB_CHAR = re.compile(r"[֐-׿]")
# Matches a token made up entirely of ASCII letters (no digits, no punctuation).
_ENG_WORD = re.compile(r"^[A-Za-z]+$")

# Hebrew final-letter pairs.
_TO_FINAL: dict[str, str] = {
    "כ": "ך",
    "מ": "ם",
    "נ": "ן",
    "פ": "ף",
    "צ": "ץ",
}
_TO_NON_FINAL: dict[str, str] = {v: k for k, v in _TO_FINAL.items()}

# Letters visually confused by OCR that may be swapped at any position in a word.
# A swap is applied only when the result is confirmed by the dictionary.
_VISUAL_SWAPS: dict[str, str] = {
    "ס": "ם",
    "ם": "ס",
}

# Single-character Hebrew prefixes that attach directly to words.
_HEBREW_PREFIXES: frozenset[str] = frozenset("בכלהמוש")

# Matches a contiguous run of Hebrew characters (no punctuation, no hyphens).
_HEB_WORD = re.compile(r"[֐-׿]+")


class HebrewFinalFixer:
    """
    Corrects non-final letter forms at Hebrew word endings using a dictionary.

    OCR frequently produces the non-final form of כ מ נ פ צ at the end of a
    word instead of the correct sofit forms ך ם ן ף ץ.  For each token that
    ends in one of these five letters, this class checks whether the word
    appears in the supplied frequency dictionary:

    * If it *does* appear → keep it unchanged (it may be a valid form).
    * If it *does not* appear → replace the last letter with its final form.

    This runs unconditionally after OCR, regardless of whether the spell
    checker is enabled.
    """

    def __init__(self, dict_path: str | Path, min_count: int = 200) -> None:
        """
        Load the word list from *dict_path*.

        Args:
            dict_path: Path to a tab-separated frequency dictionary
                       (``<word>\\t<count>`` per line, UTF-8).
            min_count: Only words whose corpus count is at or above this value
                       are considered valid.  This filters out low-frequency
                       corpus noise (OCR/transcription errors in the source
                       corpus) while keeping genuine words — including loanwords
                       that legitimately end in non-final letters (e.g. קטשופ).
                       Default: 200.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(dict_path)
        if not path.exists():
            raise FileNotFoundError(f"Hebrew dictionary not found: {path}")

        words: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            try:
                if int(parts[1]) >= min_count:
                    words.add(parts[0])
            except ValueError:
                continue
        self._words: frozenset[str] = frozenset(words)

    def _in_dict(self, word: str) -> bool:
        """Return True if *word* or its base form (without a single Hebrew prefix) is in the dictionary."""
        if word in self._words:
            return True
        if len(word) > 1 and word[0] in _HEBREW_PREFIXES:
            return word[1:] in self._words
        return False

    def _fix_token(self, token: str) -> str:
        # token is a pure Hebrew character sequence (no punctuation or hyphens).
        if not token:
            return token

        # Word is valid as-is — nothing to fix.
        if token in self._words:
            return token

        last = token[-1]

        # Orthographic rule: non-final letter at word end → always convert to
        # final form (מ→ם, נ→ן, כ→ך, פ→ף, צ→ץ).  No dictionary check needed;
        # standard Hebrew never ends a word in these non-final forms.
        final = _TO_FINAL.get(last)
        if final is not None:
            return token[:-1] + final

        # Visual confusion: ס↔ם look identical in many fonts and are frequently
        # swapped by OCR.  Try every position; return the first dictionary match.
        for i, ch in enumerate(token):
            swap = _VISUAL_SWAPS.get(ch)
            if swap is not None:
                candidate = token[:i] + swap + token[i + 1:]
                if self._in_dict(candidate):
                    return candidate

        return token

    def fix(self, text: str) -> str:
        """Apply final-letter correction to every Hebrew word in *text*."""
        return _HEB_WORD.sub(lambda m: self._fix_token(m.group()), text)


def _is_hebrew(token: str) -> bool:
    """Return True if the token contains at least one Hebrew character."""
    return bool(_HEB_CHAR.search(token))


def _normalize_hebrew_finals(token: str) -> str:
    """
    Enforce correct final-letter forms throughout a Hebrew token.

    - Every character except the last: final form → non-final (ך→כ, ם→מ, …).
    - Last character: non-final → final (כ→ך, מ→ם, …).

    This corrects the two common OCR mistakes:
      * A final-form letter used mid-word (e.g. דםות → דמות).
      * A non-final-form letter used at word end (e.g. צרצ → צרץ).
    """
    if not token:
        return token
    chars = list(token)
    for i in range(len(chars) - 1):
        chars[i] = _TO_NON_FINAL.get(chars[i], chars[i])
    chars[-1] = _TO_FINAL.get(chars[-1], chars[-1])
    return "".join(chars)


def _preserve_case(original: str, correction: str) -> str:
    """Apply the capitalisation pattern of *original* to *correction*."""
    if original.isupper():
        return correction.upper()
    if original[0].isupper():
        return correction.capitalize()
    return correction


class SpellChecker:
    """
    Word-level spell corrector backed by ``symspellpy``.

    Supports English (built-in frequency dictionary bundled with symspellpy)
    and Hebrew (requires a user-supplied frequency dictionary file).  For
    mixed Hebrew/English documents the engine detects the script of each
    token and applies the appropriate sub-corrector.

    Tokens that are not pure alphabetic (numbers, punctuation, math symbols,
    mixed alphanumeric) are passed through unchanged so formulas and proper
    nouns are not mangled.

    Hebrew dictionary
    -----------------
    A symspellpy-format frequency dictionary is a plain text file where each
    line is ``<word>\\t<count>``.  A pre-built Hebrew one can be downloaded
    from:
      https://github.com/Dotan-Peleh/hebrew-frequency-dictionary
    Pass the path via ``--spellcheck-dict``.
    """

    def __init__(
        self,
        language: str,
        hebrew_dict_path: str | Path | None = None,
    ) -> None:
        """
        Initialise the spell checker for the given language.

        Args:
            language: ``"english"``, ``"hebrew"``, or ``"hebrew_english"``.
            hebrew_dict_path: Path to a Hebrew frequency dictionary file.
                              Required when ``language`` is ``"hebrew"`` or
                              ``"hebrew_english"``.

        Raises:
            FileNotFoundError: If Hebrew is requested but no dictionary path
                               is provided or the file does not exist.
            ImportError: If ``symspellpy`` is not installed.
        """
        self._language = language
        self._eng: object | None = None
        self._heb: object | None = None

        needs_english = language in ("english", "hebrew_english")
        needs_hebrew = language in ("hebrew", "hebrew_english")

        if needs_hebrew and not hebrew_dict_path:
            raise FileNotFoundError(
                "Hebrew spell checking requires a frequency dictionary.\n"
                "Download one and pass it with --spellcheck-dict <path>.\n"
                "Format: one '<word>\\t<count>' pair per line."
            )

        if needs_english:
            self._eng = self._load_english()

        if needs_hebrew:
            heb_path = Path(hebrew_dict_path)  # type: ignore[arg-type]
            if not heb_path.exists():
                raise FileNotFoundError(f"Hebrew dictionary not found: {heb_path}")
            self._heb = self._load_custom(heb_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_english():
        """Load the English frequency dictionary bundled with symspellpy."""
        from importlib.resources import files  # pkg_resources is deprecated/removed on new setuptools
        from symspellpy import SymSpell  # type: ignore[import]

        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = str(files("symspellpy") / "frequency_dictionary_en_82_765.txt")
        sym.load_dictionary(dict_path, term_index=0, count_index=1)
        return sym

    @staticmethod
    def _load_custom(path: Path):
        """Load a custom frequency dictionary from *path*."""
        from symspellpy import SymSpell  # type: ignore[import]

        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        sym.load_dictionary(str(path), term_index=0, count_index=1, separator="\t", encoding="utf-8")
        return sym

    def _correct_token(self, token: str) -> str:
        """
        Return the best correction for a single alphabetic token.

        For Hebrew tokens the pipeline is:
          1. Normalise final-letter forms (fixes mid-word sofit / wrong word-end form).
          2. Look up the normalised token in the Hebrew dictionary.
          3. Apply final-letter normalisation to the suggestion as well (dictionary
             entries are generally correct, but this is a safety net).

        For English tokens the original capitalisation is preserved.

        Tokens that are numbers, punctuation, or mixed script are returned unchanged.

        Args:
            token: A single word-like string.

        Returns:
            Corrected token, or the (possibly normalised) original if no
            suggestion is found within the edit distance.
        """
        from symspellpy import Verbosity  # type: ignore[import]

        if _is_hebrew(token) and self._heb is not None:
            normalised = _normalize_hebrew_finals(token)
            suggestions = self._heb.lookup(normalised, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                return _normalize_hebrew_finals(suggestions[0].term)
            return normalised

        if _ENG_WORD.match(token) and self._eng is not None:
            suggestions = self._eng.lookup(token.lower(), Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                return _preserve_case(token, suggestions[0].term)

        return token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct(self, text: str) -> str:
        """
        Apply spell correction to an OCR-extracted text string.

        Each line is tokenised on whitespace.  Pure alphabetic tokens are
        corrected; everything else is left as-is.  Line breaks and blank
        lines (paragraph separators) are preserved.

        Args:
            text: Multi-line OCR text, e.g. as returned by
                  :meth:`~src.ocr_base.OCREngine.extract_text`.

        Returns:
            Spell-corrected text with identical line structure.
        """
        corrected_lines: list[str] = []
        for line in text.splitlines():
            # Split on whitespace but keep the whitespace tokens so we can
            # reconstruct the original spacing exactly.
            parts = re.split(r"(\s+)", line)
            corrected_lines.append(
                "".join(
                    self._correct_token(p) if p.strip() else p
                    for p in parts
                )
            )
        return "\n".join(corrected_lines)


