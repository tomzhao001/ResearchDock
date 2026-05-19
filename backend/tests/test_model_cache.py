from __future__ import annotations

from app.config import Settings
from app.services.model_cache import configure_model_cache_env, resolve_model_cache_paths


def test_resolve_model_cache_paths_uses_subdirectories_by_default(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models"
    monkeypatch.setenv("MODEL_CACHE_PATH", str(root))
    monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)
    monkeypatch.delenv("EASYOCR_MODULE_PATH", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    cfg = Settings()
    paths = resolve_model_cache_paths(cfg)

    assert paths.root == root.resolve()
    assert paths.docling == (root / "docling").resolve()
    assert paths.easyocr == (root / "easyocr").resolve()
    assert paths.hf_home == (root / "huggingface").resolve()


def test_configure_model_cache_env_sets_process_env(tmp_path, monkeypatch) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("MODEL_CACHE_PATH", str(root))
    monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)
    monkeypatch.delenv("EASYOCR_MODULE_PATH", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    import os

    paths = configure_model_cache_env(config=Settings())

    assert os.environ["DOCLING_ARTIFACTS_PATH"] == str(paths.docling)
    assert os.environ["EASYOCR_MODULE_PATH"] == str(paths.easyocr)
    assert os.environ["HF_HOME"] == str(paths.hf_home)
    assert paths.docling.is_dir()
    assert paths.easyocr.is_dir()
    assert paths.hf_home.is_dir()
