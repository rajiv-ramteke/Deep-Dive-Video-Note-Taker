"""
Deep-Dive Video Note Taker
==========================
Main FastAPI Application Entry Point
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.utils.logger import get_logger
from backend.utils.config import settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager — runs on startup and shutdown."""
    # ── STARTUP ──────────────────────────────────────────────
    logger.info("🚀 Starting Deep-Dive Video Note Taker...")

    # Ensure all data directories exist
    dirs = [
        settings.UPLOAD_DIR,
        settings.AUDIO_DIR,
        settings.TRANSCRIPT_DIR,
        settings.SUMMARY_DIR,
        "data/embeddings",
        "outputs/final_notes",
        "outputs/timestamps",
        "outputs/action_items",
        "outputs/reports",
        "frontend/static/uploads",
        "models/whisper",
        "models/summarization_model",
        "models/embedding_model",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    logger.info("✅ Directories initialised")
    logger.info(f"🌐 Server running at http://{settings.APP_HOST}:{settings.APP_PORT}")

    yield

    # ── SHUTDOWN ─────────────────────────────────────────────
    logger.info("🛑 Shutting down Deep-Dive Video Note Taker...")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Deep-Dive Video Note Taker",
    description=(
        "An AI-powered system that converts long-form videos into "
        "structured notes, timestamped highlights, and actionable insights."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files & Templates ──────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


# ── Root redirect ─────────────────────────────────────────────────────────────
from fastapi import Request
from fastapi.responses import HTMLResponse


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": "Deep-Dive Video Note Taker",
        "version": "1.0.0",
    }
