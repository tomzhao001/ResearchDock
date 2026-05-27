from __future__ import annotations

from pathlib import Path

import pytest

from app.services.ocr import medical_terms


@pytest.fixture(autouse=True)
def clear_medical_terms_cache() -> None:
    medical_terms.load_medical_terms.cache_clear()
    medical_terms.load_default_medical_terms.cache_clear()
    yield
    medical_terms.load_medical_terms.cache_clear()
    medical_terms.load_default_medical_terms.cache_clear()


def test_load_medical_terms_from_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wordlist_dir = tmp_path / "wordlists"
    wordlist_dir.mkdir()
    (wordlist_dir / "alpha.txt").write_text("ADHD\n# comment\n", encoding="utf-8")
    (wordlist_dir / "beta.txt").write_text("METHYLPHENIDATE\n", encoding="utf-8")
    (wordlist_dir / "ignored.swp").write_text("SHOULDNOTLOAD\n", encoding="utf-8")

    monkeypatch.setattr(medical_terms.settings, "ocr_medical_wordlist_path", str(wordlist_dir))

    terms = medical_terms.load_medical_terms()

    assert "ADHD" in terms
    assert "METHYLPHENIDATE" in terms
    assert "SHOULDNOTLOAD" not in terms


def test_load_medical_terms_from_single_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wordlist_file = tmp_path / "terms.txt"
    wordlist_file.write_text("ASD\nBMI\n", encoding="utf-8")

    monkeypatch.setattr(medical_terms.settings, "ocr_medical_wordlist_path", str(wordlist_file))

    terms = medical_terms.load_medical_terms()

    assert terms == {"ASD", "BMI"}


def test_load_default_medical_terms_from_bundled_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(medical_terms.settings, "ocr_medical_wordlist_path", "")

    terms = medical_terms.load_medical_terms()

    assert "ADHD" in terms
    assert "METHYLPHENIDATE" in terms
    assert "ADDERALL" in terms


def test_normalize_wordlist_term_filters_invalid_tokens() -> None:
    assert medical_terms.normalize_wordlist_term("adhd") == "ADHD"
    assert medical_terms.normalize_wordlist_term(" carbon dioxide ") is None
    assert medical_terms.normalize_wordlist_term("ab") is None
    assert medical_terms.normalize_wordlist_term("# comment") is None
