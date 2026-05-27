from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings

_SUSPICIOUS_FRAGMENT_RE = re.compile(r"\b(?:[A-Za-z0-9]{1,2}\s+){3,}[A-Za-z0-9]{1,2}\b")
_UNIT_TOKEN_RE = re.compile(r"\b([IlOo0-9]{1,4})(mg|kg|g|ml|mL|L|%)\b")
_PLUS_MINUS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[土士]\s*(\d+(?:\.\d+)?)")
_SPACE_BEFORE_UNIT_RE = re.compile(r"(\d(?:[\d.]*\d)?)\s+([a-zA-Z%]+)\b")


@dataclass(frozen=True)
class RuleRepairResult:
    text: str
    repairs: int


def apply_custom_ocr_rules(text: str) -> RuleRepairResult:
    if not settings.ocr_rules_enabled or not text:
        return RuleRepairResult(text=text, repairs=0)

    repairs = 0
    updated = text

    updated, count = _PLUS_MINUS_RE.subn(r"\1±\2", updated)
    repairs += count

    updated, count = _SPACE_BEFORE_UNIT_RE.subn(lambda match: _normalize_numeric_unit(match.group(1), match.group(2)), updated)
    repairs += count

    unit_repairs = 0

    def repair_unit(match: re.Match[str]) -> str:
        nonlocal unit_repairs
        original = match.group(0)
        corrected = _repair_unit_token(match)
        if corrected != original:
            unit_repairs += 1
        return corrected

    updated = _UNIT_TOKEN_RE.sub(repair_unit, updated)
    repairs += unit_repairs

    return RuleRepairResult(text=updated, repairs=repairs)


def looks_suspicious_ocr_text(text: str) -> bool:
    if not text:
        return False
    if _SUSPICIOUS_FRAGMENT_RE.search(text):
        return True
    fragments = text.split()
    if not fragments:
        return False
    short_fragment_ratio = sum(1 for part in fragments if len(part) <= 2) / max(1, len(fragments))
    return short_fragment_ratio > 0.45


def _normalize_numeric_unit(number: str, unit: str) -> str:
    return f"{number}{unit}"


def _repair_unit_token(match: re.Match[str]) -> str:
    number_part = match.group(1)
    unit = match.group(2)
    normalized_digits = (
        number_part.replace("I", "1")
        .replace("l", "1")
        .replace("O", "0")
        .replace("o", "0")
    )
    return f"{normalized_digits}{unit}"
