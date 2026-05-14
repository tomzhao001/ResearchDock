import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect

from app.config import settings
from app.database import engine

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ALEMBIC_INI_PATH = _BACKEND_DIR / "alembic.ini"
_ALEMBIC_SCRIPT_LOCATION = _BACKEND_DIR / "alembic"


def _build_alembic_config(connection=None) -> Config:
    config = Config(str(_ALEMBIC_INI_PATH))
    config.set_main_option("script_location", str(_ALEMBIC_SCRIPT_LOCATION))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def _get_table_names(connection) -> set[str]:
    inspector = inspect(connection)
    try:
        table_names = set(inspector.get_table_names(schema="public"))
    except (NotImplementedError, SQLAlchemyError):
        table_names = set()
    if table_names:
        return table_names
    return set(inspector.get_table_names())


def run_startup_migrations() -> None:
    if not settings.db_auto_migrate_on_startup:
        logger.info("Skipping startup database migrations because DB_AUTO_MIGRATE_ON_STARTUP=false.")
        return

    with engine.begin() as connection:
        table_names = _get_table_names(connection)
        has_alembic_version = "alembic_version" in table_names
        managed_tables = table_names - {"alembic_version"}
        alembic_config = _build_alembic_config(connection)

        if managed_tables and not has_alembic_version:
            if settings.db_auto_stamp_existing_schema:
                logger.warning(
                    "Detected an existing unmanaged schema; stamping the current database to Alembic head "
                    "because DB_AUTO_STAMP_EXISTING_SCHEMA=true."
                )
                command.stamp(alembic_config, "head")
                return

            raise RuntimeError(
                "Detected a non-empty PostgreSQL schema without alembic_version. "
                "Run a one-time Alembic stamp from the backend directory with "
                "`alembic stamp head`, or set DB_AUTO_STAMP_EXISTING_SCHEMA=true "
                "for an explicit one-time automatic stamp."
            )

        command.upgrade(alembic_config, "head")
