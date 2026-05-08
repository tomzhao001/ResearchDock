import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from redis.asyncio import Redis

from app.database import SessionLocal
from app.deps import get_token_from_websocket, resolve_current_user
from app.services.task_events import TASK_STATUS_CHANNEL

router = APIRouter(tags=["ws"])


@router.websocket("/api/ws/tasks")
async def task_status_stream(websocket: WebSocket):
    db = SessionLocal()
    try:
        resolve_current_user(db, get_token_from_websocket(websocket))
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    finally:
        db.close()

    await websocket.accept()

    redis = Redis.from_url(websocket.app.state.redis_url, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe(TASK_STATUS_CHANNEL)
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message is not None and isinstance(message.get("data"), str):
                await websocket.send_text(message["data"])
            await asyncio.sleep(0.1)
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        await pubsub.unsubscribe(TASK_STATUS_CHANNEL)
        await pubsub.aclose()
        await redis.aclose()
