from __future__ import annotations

from dataclasses import asdict, dataclass

_FULLWIDTH_ASCII_OFFSET = 0xFEE0
_IDEOGRAPHIC_SPACE = "\u3000"
_FULLWIDTH_DIGIT_START = ord("\uFF10")
_FULLWIDTH_DIGIT_END = ord("\uFF19")
_FULLWIDTH_UPPER_START = ord("\uFF21")
_FULLWIDTH_UPPER_END = ord("\uFF3A")
_FULLWIDTH_LOWER_START = ord("\uFF41")
_FULLWIDTH_LOWER_END = ord("\uFF5A")
_FULLWIDTH_ASCII_START = ord("\uFF01")
_FULLWIDTH_ASCII_END = ord("\uFF5E")

_FULLWIDTH_TO_HALFWIDTH_MAP = {
    **{codepoint: codepoint - _FULLWIDTH_ASCII_OFFSET for codepoint in range(_FULLWIDTH_ASCII_START, _FULLWIDTH_ASCII_END + 1)},
    ord(_IDEOGRAPHIC_SPACE): ord(" "),
}


@dataclass(frozen=True)
class OcrTextNormalizationResult:
    text: str
    normalization_applied: bool
    normalization_strategy: str | None
    fullwidth_ascii_count: int
    fullwidth_latin_count: int
    fullwidth_digit_count: int
    fullwidth_ascii_punctuation_count: int
    fullwidth_space_count: int

    def to_metadata(self) -> dict[str, int | bool | str | None]:
        return asdict(self)


def normalize_ocr_text(text: str) -> OcrTextNormalizationResult:
    source = text or ""
    fullwidth_latin_count = 0
    fullwidth_digit_count = 0
    fullwidth_ascii_punctuation_count = 0
    fullwidth_space_count = 0

    for char in source:
        codepoint = ord(char)
        if _FULLWIDTH_UPPER_START <= codepoint <= _FULLWIDTH_UPPER_END or _FULLWIDTH_LOWER_START <= codepoint <= _FULLWIDTH_LOWER_END:
            fullwidth_latin_count += 1
            continue
        if _FULLWIDTH_DIGIT_START <= codepoint <= _FULLWIDTH_DIGIT_END:
            fullwidth_digit_count += 1
            continue
        if _FULLWIDTH_ASCII_START <= codepoint <= _FULLWIDTH_ASCII_END:
            fullwidth_ascii_punctuation_count += 1
            continue
        if char == _IDEOGRAPHIC_SPACE:
            fullwidth_space_count += 1

    normalized_text = source.translate(_FULLWIDTH_TO_HALFWIDTH_MAP)
    normalization_applied = normalized_text != source
    fullwidth_ascii_count = fullwidth_latin_count + fullwidth_digit_count + fullwidth_ascii_punctuation_count
    return OcrTextNormalizationResult(
        text=normalized_text,
        normalization_applied=normalization_applied,
        normalization_strategy="fullwidth_ascii_fold" if normalization_applied else None,
        fullwidth_ascii_count=fullwidth_ascii_count,
        fullwidth_latin_count=fullwidth_latin_count,
        fullwidth_digit_count=fullwidth_digit_count,
        fullwidth_ascii_punctuation_count=fullwidth_ascii_punctuation_count,
        fullwidth_space_count=fullwidth_space_count,
    )
