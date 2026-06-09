"""
backend/utils/logger.py
========================
Loguru-based logging configuration for the entire application.
"""

import io
import sys
from loguru import logger as _logger

# Re-wrap stdout with UTF-8 so emoji in log messages don't crash on Windows cp1252
_stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# Console handler — coloured, human-readable
_logger.add(
    _stdout_utf8,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=False,  # disable ANSI colours when redirecting to UTF-8 wrapper
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
