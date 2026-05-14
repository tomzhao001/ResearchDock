from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db_migrations import run_startup_migrations
from app.routers import auth_routes, chat, health, jobs, papers, ws
from app.services.http_clients import close_shared_http_clients


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        run_startup_migrations()
        yield
    finally:
        close_shared_http_clients()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.state.redis_url = settings.redis_url

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.public_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth_routes.router)
app.include_router(papers.router)
app.include_router(jobs.router)
app.include_router(chat.router)
app.include_router(ws.router)
