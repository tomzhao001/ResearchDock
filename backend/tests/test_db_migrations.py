from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app import db_migrations
from app.main import app


def test_run_startup_migrations_upgrades_empty_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    calls: list[str] = []

    monkeypatch.setattr("app.db_migrations.engine", engine)
    monkeypatch.setattr("app.config.settings.db_auto_migrate_on_startup", True)
    monkeypatch.setattr("app.config.settings.db_auto_stamp_existing_schema", False)
    monkeypatch.setattr("app.db_migrations.command.upgrade", lambda _cfg, revision: calls.append(revision))
    monkeypatch.setattr("app.db_migrations.command.stamp", lambda _cfg, revision: calls.append(f"stamp:{revision}"))

    db_migrations.run_startup_migrations()

    assert calls == ["head"]


def test_run_startup_migrations_rejects_nonempty_unmanaged_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE legacy_table (id INTEGER PRIMARY KEY)"))

    monkeypatch.setattr("app.db_migrations.engine", engine)
    monkeypatch.setattr("app.config.settings.db_auto_migrate_on_startup", True)
    monkeypatch.setattr("app.config.settings.db_auto_stamp_existing_schema", False)

    with pytest.raises(RuntimeError, match="alembic_version"):
        db_migrations.run_startup_migrations()


def test_run_startup_migrations_stamps_existing_schema_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy_stamp.db'}")
    calls: list[str] = []

    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE legacy_table (id INTEGER PRIMARY KEY)"))

    monkeypatch.setattr("app.db_migrations.engine", engine)
    monkeypatch.setattr("app.config.settings.db_auto_migrate_on_startup", True)
    monkeypatch.setattr("app.config.settings.db_auto_stamp_existing_schema", True)
    monkeypatch.setattr("app.db_migrations.command.upgrade", lambda _cfg, revision: calls.append(revision))
    monkeypatch.setattr("app.db_migrations.command.stamp", lambda _cfg, revision: calls.append(f"stamp:{revision}"))

    db_migrations.run_startup_migrations()

    assert calls == ["stamp:head"]


def test_app_lifespan_runs_startup_migrations(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("app.main.run_startup_migrations", lambda: calls.append("called"))

    with TestClient(app):
        pass

    assert calls == ["called"]
