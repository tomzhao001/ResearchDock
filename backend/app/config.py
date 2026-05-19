from pathlib import Path
from typing import Self

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import URL

# backend/app/config.py -> repo root is two levels up from app/
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            str(_REPO_ROOT / ".env"),
            str(_BACKEND_DIR / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_name: str = "ResearchDock API"
    postgres_user: str = "paper_user"
    postgres_password: str = "change_me"
    postgres_db: str = "paper_archive"
    postgres_port: int = 5432
    # 本机 uvicorn：127.0.0.1 + POSTGRES_PORT；Compose 内 backend 由 docker-compose 注入 POSTGRES_HOST=db
    postgres_host: str = "127.0.0.1"
    # 若设置 DATABASE_URL 且非空，则优先生效，忽略上方分项拼接
    database_url: str = ""

    @model_validator(mode="after")
    def _resolve_database_url(self) -> Self:
        explicit = (self.database_url or "").strip()
        if explicit:
            object.__setattr__(self, "database_url", explicit)
            return self
        url = URL.create(
            drivername="postgresql",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )
        object.__setattr__(self, "database_url", url.render_as_string(hide_password=False))
        return self
    app_secret_key: str = "change_me"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    public_origin: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("PUBLIC_ORIGIN", "FRONTEND_ORIGIN"),
    )
    cookie_name: str = "rd_access_token"
    cookie_secure: bool = False  # set COOKIE_SECURE=true behind HTTPS in production
    file_storage_path: Path = _REPO_ROOT / "data" / "files"
    redis_url: str = "redis://localhost:6379/1"
    celery_broker_url: str = ""
    celery_result_backend: str = ""
    celery_task_always_eager: bool = False
    db_auto_migrate_on_startup: bool = True
    db_auto_stamp_existing_schema: bool = False
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_provider: str = "auto"
    # Chat completion（对话、论文摘要等）单独超时；嵌入/轻量请求仍用 OPENAI_TIMEOUT_SECONDS / GLM_TIMEOUT_SECONDS
    llm_chat_timeout_seconds: int = 300
    llm_chat_max_retries: int = 2
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "auto"
    openai_timeout_seconds: int = 120
    openai_verify_ssl: bool = True
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_api_key: str = ""
    glm_model: str = ""
    glm_embedding_model: str = "embedding-3"
    glm_embedding_dimensions: int = 1024
    glm_rerank_model: str = "rerank"
    glm_timeout_seconds: int = 120
    glm_verify_ssl: bool = True
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_top_k: int = 5
    rag_sparse_top_k: int = 20
    rag_dense_top_k: int = 20
    rag_fusion_window: int = 20
    rag_rrf_k: int = 60
    rag_rerank_top_n: int = 10
    rag_text_search_config: str = "simple"
    rag_crosslingual_query_rewrite_enabled: bool = True
    rag_crosslingual_max_subqueries: int = 2
    rag_hyde_enabled: bool = False
    rag_attribution_max_evidence: int = 4
    rag_attribution_min_support_score: float = 0.55
    rag_attribution_min_total_support_score: float = 0.9
    rag_attribution_verifier_min_support_score: float = 0.5
    document_extractor: str = "docling"
    docling_do_ocr: bool = True
    docling_do_table_structure: bool = True
    docling_do_picture_description: bool = True
    docling_ocr_engine: str = "easyocr"
    docling_ocr_languages: str = "ch_sim,en"
    docling_force_full_page_ocr: bool = False
    docling_generate_picture_images: bool = True
    docling_images_scale: float = 2.0
    docling_document_timeout_seconds: int = 240
    docling_device: str = "auto"
    model_cache_path: Path = _REPO_ROOT / "data" / "models"
    docling_artifacts_path: str = ""
    easyocr_module_path: str = ""
    hf_home: str = ""
    picture_vlm_provider: str = "glm_4_6v"
    picture_vlm_base_url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    picture_vlm_api_key: str = ""
    picture_vlm_model: str = "glm-4.6v"
    picture_vlm_timeout_seconds: int = 120
    picture_vlm_max_retries: int = 2
    picture_vlm_prompt_version: str = "picture-rag-v1"

    @model_validator(mode="after")
    def _resolve_paths_and_queue(self) -> Self:
        storage_path = self.file_storage_path
        if not storage_path.is_absolute():
            storage_path = (_REPO_ROOT / storage_path).resolve()
        object.__setattr__(self, "file_storage_path", storage_path)

        model_cache_path = self.model_cache_path
        if not model_cache_path.is_absolute():
            model_cache_path = (_REPO_ROOT / model_cache_path).resolve()
        object.__setattr__(self, "model_cache_path", model_cache_path)

        broker = (self.celery_broker_url or "").strip() or self.redis_url
        backend = (self.celery_result_backend or "").strip() or broker
        object.__setattr__(self, "celery_broker_url", broker)
        object.__setattr__(self, "celery_result_backend", backend)
        return self


settings = Settings()
