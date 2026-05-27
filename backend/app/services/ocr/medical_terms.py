from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from app.config import _REPO_ROOT, settings

_BACKEND_DIR = Path(__file__).resolve().parents[3]
BUNDLED_WORDLIST_DIR = _BACKEND_DIR / "resources" / "medical-wordlist"
BUNDLED_RESEARCH_CORE_WORDLIST = BUNDLED_WORDLIST_DIR / "research-core.txt"

_OCR_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+\-]{2,}$")


def bundled_research_core_wordlist_path() -> Path:
    return BUNDLED_RESEARCH_CORE_WORDLIST


def resolve_medical_wordlist_path() -> Path | None:
    configured = str(settings.ocr_medical_wordlist_path or "").strip()
    if not configured:
        return None
    path = Path(configured)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def iter_wordlist_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        candidate
        for candidate in path.rglob("*.txt")
        if candidate.is_file() and not candidate.name.startswith(".")
    )


def normalize_wordlist_term(raw: str) -> str | None:
    line = str(raw or "").strip()
    if not line or line.startswith("#"):
        return None
    folded = unicodedata.normalize("NFKD", line).encode("ascii", "ignore").decode("ascii")
    normalized = folded.strip().upper()
    if not normalized or not _OCR_TOKEN_RE.fullmatch(normalized):
        return None
    return normalized


def parse_wordlist_text(text: str) -> set[str]:
    terms: set[str] = set()
    for raw_line in text.splitlines():
        normalized = normalize_wordlist_term(raw_line)
        if normalized:
            terms.add(normalized)
    return terms


def parse_wordlist_file(path: Path) -> set[str]:
    return parse_wordlist_text(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_default_medical_terms() -> set[str]:
    path = bundled_research_core_wordlist_path()
    if not path.exists():
        return set()
    return parse_wordlist_file(path)


@lru_cache(maxsize=1)
def load_medical_terms() -> set[str]:
    path = resolve_medical_wordlist_path()
    if path is None or not path.exists():
        return load_default_medical_terms()

    files = iter_wordlist_files(path)
    if not files:
        return load_default_medical_terms()

    terms: set[str] = set()
    for file_path in files:
        terms.update(parse_wordlist_file(file_path))
    return terms or load_default_medical_terms()


def is_known_medical_term(token: str) -> bool:
    normalized = str(token or "").strip().upper()
    if not normalized:
        return False
    return normalized in load_medical_terms()
