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
