from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, settings


@dataclass(frozen=True)
class ModelCachePaths:
    root: Path
    docling: Path
    easyocr: Path
    hf_home: Path


def resolve_model_cache_paths(config: Settings | None = None) -> ModelCachePaths:
    cfg = config or settings
    root = Path(cfg.model_cache_path).resolve()
    docling = Path(cfg.docling_artifacts_path).resolve() if str(cfg.docling_artifacts_path).strip() else root / "docling"
    easyocr = Path(cfg.easyocr_module_path).resolve() if str(cfg.easyocr_module_path).strip() else root / "easyocr"
    hf_home = Path(cfg.hf_home).resolve() if str(cfg.hf_home).strip() else root / "huggingface"
    return ModelCachePaths(root=root, docling=docling, easyocr=easyocr, hf_home=hf_home)


def configure_model_cache_env(*, config: Settings | None = None) -> ModelCachePaths:
    paths = resolve_model_cache_paths(config)
    for directory in (paths.root, paths.docling, paths.easyocr, paths.hf_home):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["DOCLING_ARTIFACTS_PATH"] = str(paths.docling)
    os.environ["EASYOCR_MODULE_PATH"] = str(paths.easyocr)
    os.environ["HF_HOME"] = str(paths.hf_home)
    os.environ["HF_HUB_CACHE"] = str(paths.hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(paths.hf_home / "transformers")
    return paths


def model_cache_metadata(config: Settings | None = None) -> dict[str, str]:
    paths = resolve_model_cache_paths(config)
    return {
        "model_cache_root": str(paths.root),
        "docling_artifacts_path": str(paths.docling),
        "easyocr_module_path": str(paths.easyocr),
        "hf_home": str(paths.hf_home),
    }
