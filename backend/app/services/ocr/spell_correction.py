from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from app.config import settings
from app.services.ocr.medical_terms import is_known_medical_term, load_medical_terms

_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9+\-]{2,}\b")


@dataclass(frozen=True)
class SpellCorrection:
    original: str
    corrected: str
    strategy: str | None = None


class SymSpellCorrector:
    def __init__(self) -> None:
        self._enabled = bool(settings.ocr_symspell_enabled)
        self._terms = load_medical_terms()
        self._symspell = None
        self._verbosity = None
        if self._enabled:
            self._initialize_symspell()

    def _initialize_symspell(self) -> None:
        try:
            from symspellpy import SymSpell, Verbosity
        except Exception:
            return

        symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for term in self._terms:
            symspell.create_dictionary_entry(term.lower(), 1)
        self._symspell = symspell
        self._verbosity = Verbosity.CLOSEST

    def correct_text(self, text: str) -> tuple[str, int]:
        if not self._enabled:
            return text, 0

        corrections = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal corrections
            token = match.group(0)
            corrected = self.correct_token(token)
            if corrected.corrected != token:
                corrections += 1
            return corrected.corrected

        return _TOKEN_RE.sub(replace, text), corrections

    def correct_token(self, token: str) -> SpellCorrection:
        if not self._enabled or not token:
            return SpellCorrection(original=token, corrected=token)

        if is_known_medical_term(token):
            return SpellCorrection(original=token, corrected=token)

        normalized = token.lower()
        if self._symspell is not None and self._verbosity is not None:
            suggestions = self._symspell.lookup(normalized, self._verbosity, max_edit_distance=2)
            if suggestions:
                candidate = suggestions[0].term or normalized
                corrected = _restore_token_case(token, candidate)
                if is_known_medical_term(corrected):
                    return SpellCorrection(original=token, corrected=corrected, strategy="symspell")

        confusion_candidate = _match_by_confusion_signature(token, self._terms)
        if confusion_candidate:
            return SpellCorrection(original=token, corrected=confusion_candidate, strategy="confusion_signature")

        fallback = difflib.get_close_matches(token.upper(), list(self._terms), n=1, cutoff=0.72)
        if fallback:
            return SpellCorrection(original=token, corrected=fallback[0], strategy="medical_dictionary")
        return SpellCorrection(original=token, corrected=token)


def _restore_token_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.title()
    return replacement


def _match_by_confusion_signature(token: str, terms: set[str]) -> str | None:
    signature = _confusion_signature(token)
    for term in terms:
        if _confusion_signature(term) == signature:
            return term
    return None


def _confusion_signature(token: str) -> str:
    normalized = token.upper()
    normalized = normalized.replace("II", "H")
    normalized = normalized.translate(str.maketrans({"I": "1", "L": "1", "O": "0"}))
    return normalized
