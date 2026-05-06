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
