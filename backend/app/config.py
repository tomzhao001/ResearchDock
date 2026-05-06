from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

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
    # 本地 uvicorn：host 使用 127.0.0.1 或 localhost；Compose 内由 DATABASE_URL 覆盖为 db
    database_url: str = "postgresql://paper_user:change_me@127.0.0.1:5432/paper_archive"
    app_secret_key: str = "change_me"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    frontend_origin: str = "http://localhost:3000"
    cookie_name: str = "rd_access_token"
    cookie_secure: bool = False  # set COOKIE_SECURE=true behind HTTPS in production


settings = Settings()
