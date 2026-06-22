"""
main.py — Application Launcher
===============================
Entry point to start the Deep-Dive Video Note Taker server.
"""

import uvicorn
from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("   Deep-Dive Video Note Taker  v1.0.0")
    logger.info("=" * 60)
    logger.info(f"  Host : {settings.APP_HOST}")
    logger.info(f"  Port : {settings.APP_PORT}")
    logger.info(f"  Debug: {settings.DEBUG}")
    logger.info(f"  LLM  : {settings.LLM_PROVIDER}")
    logger.info(f"  ASR  : Whisper ({settings.WHISPER_MODEL})")
    logger.info("=" * 60)

    import os
    reload_mode = settings.DEBUG and not os.environ.get("SPACE_ID")
    
    uvicorn.run(
        "app:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=reload_mode,
        log_level="info",
    )
