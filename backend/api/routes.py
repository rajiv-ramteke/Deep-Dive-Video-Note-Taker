"""
backend/api/routes.py
======================
Central FastAPI router — registers all API sub-routes.
"""

from fastapi import APIRouter

from backend.api.endpoints import (
    process_router,
    query_router,
    status_router,
    notes_router,
)

router = APIRouter()

# Mount sub-routers
router.include_router(process_router, prefix="/process", tags=["Processing"])
router.include_router(query_router,   prefix="/query",   tags=["RAG Query"])
router.include_router(status_router,  prefix="/status",  tags=["Job Status"])
router.include_router(notes_router,   prefix="/notes",   tags=["Notes"])
