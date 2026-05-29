"""
frontend/app_frontend.py
=========================
Frontend helper utilities — Jinja2 template rendering helpers,
static path resolvers, and response wrappers for the web UI.
"""

from pathlib import Path

# ── Path resolution ────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"
UPLOADS_DIR   = STATIC_DIR / "uploads"


def get_templates_dir() -> str:
    return str(TEMPLATES_DIR)


def get_static_dir() -> str:
    return str(STATIC_DIR)


def get_upload_dir() -> str:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return str(UPLOADS_DIR)


# ── Template context helpers ───────────────────────────────────────────────────

def base_context() -> dict:
    """Return a base template context dictionary."""
    return {
        "app_name":    "Deep-Dive Video Note Taker",
        "app_version": "1.0.0",
        "app_tagline": "AI-Powered Video Analysis",
    }
