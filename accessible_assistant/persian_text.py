"""Persian text normalisation for questions after STT and for OCR display (topics 2/15/20/33).

Raw text is always kept by callers; this produces a cleaned copy: Arabic letter forms to
Persian, diacritics/tatweel removed, digits unified, ZWNJ for «می» prefixes and «ها» plurals,
spacing around punctuation. Numbers are never changed in value.
"""
from __future__ import annotations

import re

_LETTERS = str.maketrans({"ي": "ی", "ى": "ی", "ئ": "ئ", "ك": "ک", "ۀ": "هٔ", "ة": "ه", "ـ": None})
_DIACRITICS = re.compile("[ً-ْٰ]")  # harakat; ezafe marks are added later only for TTS
_TO_PERSIAN_DIGITS = str.maketrans("0123456789٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹۰۱۲۳۴۵۶۷۸۹")
_ZWNJ = "‌"
_MI = re.compile(r"(?<![\w‌])(ن?می)\s+(?=[؀-ۿ])")
_PLURAL = re.compile(r"(?<=[؀-ۿ])\s+(ها|های|هایی|هایم|هایت|هایش|هایمان|هایتان|هایشان)(?![؀-ۿ])")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([،؛:.!؟?])")
_SPACE_AFTER_PUNCT = re.compile(r"([،؛!؟?])(?=[^\s\d۰-۹])")


def normalize_letters(text: str) -> str:
    return _DIACRITICS.sub("", text.translate(_LETTERS))


def normalize(text: str, persian_digits: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = normalize_letters(text).replace("?", "؟").replace(",", "،") if _mostly_persian(text) else normalize_letters(text)
    if persian_digits:
        text = text.translate(_TO_PERSIAN_DIGITS)
    text = _MI.sub(lambda m: m.group(1) + _ZWNJ, text)
    text = _PLURAL.sub(lambda m: _ZWNJ + m.group(1), text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_PUNCT.sub(r"\1 ", text)
    text = re.sub(r"‌{2,}", _ZWNJ, text)
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _mostly_persian(text: str) -> bool:
    persian = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return persian >= latin
