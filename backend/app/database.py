import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config import settings

# 避免中文 Windows 下连接阶段报错/握手信息为非 UTF-8 时，
# psycopg2/libpq 按 UTF-8 解码抛出 UnicodeDecodeError。
os.environ.setdefault("PGCLIENTENCODING", "UTF8")

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args={"client_encoding": "utf8"},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
