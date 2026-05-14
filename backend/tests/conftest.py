from datetime import datetime, timezone
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.auth import pwd_context
from app.database import Base, get_db
from app.main import app
from app.models import Organization, User
from app.permissions import ROLE_ORG_OWNER


@pytest.fixture()
def session_factory(tmp_path: Path) -> Generator[sessionmaker, None, None]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture()
def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_factory: sessionmaker,
) -> Generator[TestClient, None, None]:
    storage_path = tmp_path / "files"
    storage_path.mkdir(parents=True, exist_ok=True)

    def override_get_db() -> Generator[Session, None, None]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr("app.config.settings.file_storage_path", storage_path)
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def db_session(session_factory: sessionmaker) -> Generator[Session, None, None]:
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def organization(db_session: Session) -> Organization:
    record = Organization(name="Test Org", slug="test-org", is_active=True)
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    return record


@pytest.fixture()
def second_organization(db_session: Session) -> Organization:
    record = Organization(name="Other Org", slug="other-org", is_active=True)
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    return record


@pytest.fixture()
def user(db_session: Session, organization: Organization) -> User:
    record = User(
        organization_id=organization.id,
        username="admin",
        password_hash=pwd_context.hash("123456"),
        role=ROLE_ORG_OWNER,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    return record


@pytest.fixture()
def second_user(db_session: Session, second_organization: Organization) -> User:
    record = User(
        organization_id=second_organization.id,
        username="other-admin",
        password_hash=pwd_context.hash("654321"),
        role=ROLE_ORG_OWNER,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)
    return record
