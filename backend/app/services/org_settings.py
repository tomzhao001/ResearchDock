from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import OrganizationSettings
from app.schemas import OrganizationQuestionItem


def _normalize_question_id(value: object) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("Question id is required")
    return normalized


def _normalize_question_text(value: object) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("Question text is required")
    return normalized


def normalize_question_items(items: list[OrganizationQuestionItem] | list[dict] | None) -> list[dict[str, str]]:
    normalized_items: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for item in items or []:
        item_id = _normalize_question_id(item.id if isinstance(item, OrganizationQuestionItem) else item.get("id"))
        if item_id in seen_ids:
            raise ValueError(f"Duplicate question id: {item_id}")
        seen_ids.add(item_id)
        question = _normalize_question_text(item.question if isinstance(item, OrganizationQuestionItem) else item.get("question"))
        normalized_items.append({"id": item_id, "question": question})
    return normalized_items


def get_or_create_organization_settings(db: Session, *, organization_id: int) -> OrganizationSettings:
    settings = db.scalar(select(OrganizationSettings).where(OrganizationSettings.organization_id == organization_id))
    if settings is not None:
        return settings

    now = datetime.now(timezone.utc)
    settings = OrganizationSettings(
        organization_id=organization_id,
        auto_extraction_questions_json=[],
        created_at=now,
        updated_at=now,
    )
    db.add(settings)
    db.commit()
    db.refresh(settings)
    return settings


def get_organization_question_items(db: Session, *, organization_id: int) -> list[dict[str, str]]:
    settings = db.scalar(select(OrganizationSettings).where(OrganizationSettings.organization_id == organization_id))
    if settings is None:
        return []
    questions = settings.auto_extraction_questions_json
    if not isinstance(questions, list):
        return []
    return normalize_question_items(questions)


def update_organization_question_items(
    db: Session,
    *,
    organization_id: int,
    questions: list[OrganizationQuestionItem],
) -> OrganizationSettings:
    settings = get_or_create_organization_settings(db, organization_id=organization_id)
    settings.auto_extraction_questions_json = normalize_question_items(questions)
    settings.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(settings)
    return settings
