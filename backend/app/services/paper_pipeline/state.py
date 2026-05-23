from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from sqlalchemy.orm import Session

from app.models import Job, Paper, PaperAsset


class PaperJobGraphState(TypedDict, total=False):
    db: Session
    job_id: int
    job: Job
    paper: Paper
    asset: PaperAsset
    extractor: Any
    picture_adapter: Any
    document: Any
    file_path: Path
    paper_text: str
    structured_summary: dict[str, Any] | None
    summary: dict[str, Any]
    question_result: dict[str, Any]
    questions: list[dict[str, Any]]
    next_job_id: int | None
