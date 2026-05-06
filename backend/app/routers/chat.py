from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.deps import get_current_user
from app.models import User
from app.schemas import ChatRequest, ChatResponse
from app.services.llm import chat_with_model

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def create_chat_completion(
    payload: ChatRequest,
    _: Annotated[User, Depends(get_current_user)],
):
    try:
        answer, model = await run_in_threadpool(chat_with_model, payload.message)
    except RuntimeError as exc:
        detail = str(exc) or "Chat service is unavailable"
        status_code = status.HTTP_400_BAD_REQUEST if detail == "message is required" else status.HTTP_503_SERVICE_UNAVAILABLE
        raise HTTPException(status_code=status_code, detail=detail) from exc

    return ChatResponse(answer=answer, model=model)
