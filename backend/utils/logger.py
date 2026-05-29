"""
backend/utils/logger.py
========================
Loguru-based logging configuration for the entire application.
"""

import sys
from loguru import logger as _logger

# Remove default handler
_logger.remove()

# Console handler — coloured, human-readable
_logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)

# File handler — rotating log file
_logger.add(
    "logs/app.log",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
    level="INFO",
    enqueue=True,
)


def get_logger(name: str):
    """Return a bound logger for a specific module."""
    return _logger.bind(name=name)
