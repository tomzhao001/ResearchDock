from pathlib import Path
from typing import Self

from pydantic import model_validator
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
    frontend_origin: str = "http://localhost:3000"
    cookie_name: str = "rd_access_token"
    cookie_secure: bool = False  # set COOKIE_SECURE=true behind HTTPS in production


settings = Settings()
