from datetime import datetime

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class UserPublic(BaseModel):
    id: int
    username: str

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    message: str


class UploadAcceptedResponse(BaseModel):
    paper_id: int
    job_id: int
    filename: str
    status: str


class JobPublic(BaseModel):
    id: int
    job_type: str | None
    paper_id: int | None
    status: str | None
    error_message: str | None
    retry_count: int
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    items: list[JobPublic]


class PaperListItem(BaseModel):
    id: int
    title: str | None
    abstract_raw: str | None
    status: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PaperListResponse(BaseModel):
    items: list[PaperListItem]


class PaperDetailResponse(BaseModel):
    id: int
    title: str | None
    authors: str | None
    abstract_raw: str | None
    source_url: str | None
    pdf_url: str | None
    doi: str | None
    published_at: datetime | None
    status: str | None
    created_at: datetime
    updated_at: datetime
    original_filename: str | None
    preview_text: str | None
    extraction_metadata: dict | None
    structured_summary: dict | None
    latest_job: JobPublic | None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    model: str | None
