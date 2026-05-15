from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class OrganizationPublic(BaseModel):
    id: int
    name: str
    slug: str

    model_config = {"from_attributes": True}


class UserSessionPublic(BaseModel):
    id: int
    username: str
    role: str
    permissions: list[str]
    organization: OrganizationPublic

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    message: str


class UploadAcceptedResponse(BaseModel):
    paper_id: int
    job_id: int
    filename: str
    status: str


class UploadConflictDetail(BaseModel):
    message: str
    existing_paper_id: int
    filename: str


class JobAcceptedResponse(BaseModel):
    paper_id: int
    job_id: int
    job_type: str
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
    organization_id: int
    title: str | None
    original_filename: str | None = None
    abstract_raw: str | None
    published_at: datetime | None = None
    status: str | None
    ocr_status: str | None = None
    summary_status: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PaperListResponse(BaseModel):
    items: list[PaperListItem]


class PaperDetailResponse(BaseModel):
    id: int
    organization_id: int
    title: str | None
    authors: str | None
    abstract_raw: str | None
    source_url: str | None
    pdf_url: str | None
    doi: str | None
    published_at: datetime | None
    status: str | None
    ocr_status: str | None
    summary_status: str | None
    created_at: datetime
    updated_at: datetime
    original_filename: str | None
    preview_text: str | None
    extraction_metadata: dict | None
    structured_summary: dict | None
    latest_job: JobPublic | None
    latest_ocr_job: JobPublic | None
    latest_summary_job: JobPublic | None


class TaskStatusEvent(BaseModel):
    type: Literal["task-status"] = "task-status"
    paper_id: int
    job_id: int | None
    job_type: str | None
    job_status: str | None
    paper_status: str | None
    ocr_status: str | None
    summary_status: str | None
    error_message: str | None
    updated_at: datetime
    job: JobPublic | None
    paper_list_item: PaperListItem
    paper_detail: PaperDetailResponse


class ChatProgressEvent(BaseModel):
    type: Literal["chat-progress"] = "chat-progress"
    topic_id: int
    phase: str
    status: str
    message: str
    detail: str | None = None
    created_at: datetime


class PaperUpdateRequest(BaseModel):
    title: str | None = None
    authors: str | None = None
    doi: str | None = None
    source_url: str | None = None
    published_at: datetime | None = None


class ChatTopicCreateRequest(BaseModel):
    title: str | None = None


class ChatTopicPublic(BaseModel):
    id: int
    title: str
    message_count: int
    last_message_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatTopicListResponse(BaseModel):
    items: list[ChatTopicPublic]


class ChatCitation(BaseModel):
    evidence_id: str | None = None
    chunk_id: int
    paper_id: int
    paper_title: str | None
    source_url: str | None
    snippet: str
    score: float | None = None
    support_score: float | None = None
    page_from: int | None = None
    page_to: int | None = None
    section_path: str | None = None
    selection_reason: str | None = None
    claim_texts: list[str] = []


class ChatMessageCreateRequest(BaseModel):
    message: str


class ChatMessagePublic(BaseModel):
    id: int
    topic_id: int
    role: str
    content: str
    model: str | None = None
    answer_mode: str | None = None
    used_knowledge_base: bool = False
    citations: list[ChatCitation] = []
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatMessageListResponse(BaseModel):
    items: list[ChatMessagePublic]


class ChatTurnResponse(BaseModel):
    topic: ChatTopicPublic
    user_message: ChatMessagePublic
    assistant_message: ChatMessagePublic
