"""
backend/utils/helper.py
========================
Shared utility functions used across the application.
"""

import os
import re
import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ── File helpers ──────────────────────────────────────────────────────────────

def generate_job_id() -> str:
    """Generate a unique job ID for each processing request."""
    return str(uuid.uuid4())


def get_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file for deduplication."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_dir(path: str) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_output_path(base_dir: str, job_id: str, suffix: str) -> str:
    """Build a standardised output file path."""
    ensure_dir(base_dir)
    return os.path.join(base_dir, f"{job_id}{suffix}")


def safe_filename(name: str) -> str:
    """Sanitise a filename by removing/replacing unsafe characters."""
    name = re.sub(r"[^\w\-_\. ]", "_", name)
    name = name.strip().replace(" ", "_")
    return name


def get_file_size_mb(file_path: str) -> float:
    """Return file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


# ── Time helpers ──────────────────────────────────────────────────────────────

def seconds_to_timestamp(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS string."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def timestamp_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS or MM:SS to float seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts)


def format_duration(seconds: float) -> str:
    """Human-readable duration string."""
    ts = seconds_to_timestamp(seconds)
    return ts


# ── JSON helpers ──────────────────────────────────────────────────────────────

def save_json(data: Any, file_path: str) -> None:
    """Save a Python object as a JSON file."""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug(f"JSON saved: {file_path}")


def load_json(file_path: str) -> Any:
    """Load JSON from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, file_path: str) -> None:
    """Save plain text to a file."""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.debug(f"Text saved: {file_path}")


# ── Text helpers ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excessive whitespace and special characters from text."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars, appending ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def word_count(text: str) -> int:
    """Return word count of a string."""
    return len(text.split())


# ── Result builders ───────────────────────────────────────────────────────────

def build_success_response(data: Dict, message: str = "Success") -> Dict:
    return {
        "status": "success",
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }


def build_error_response(error: str, details: Optional[str] = None) -> Dict:
    return {
        "status": "error",
        "message": error,
        "details": details,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Extractive NLP helpers for API key-free fallback mode ───────────────────

COMMON_STOPWORDS = {
    "the", "and", "a", "of", "to", "is", "in", "it", "you", "that", "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
    "at", "be", "this", "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your",
    "can", "said", "there", "use", "an", "each", "which", "she", "do", "how", "their", "if", "will", "up", "other", "about", "out",
    "many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look", "two",
    "more", "write", "go", "see", "number", "no", "way", "could", "people", "my", "than", "first", "water", "been", "call",
    "who", "oil", "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may", "part", "to", "do", "does",
    "then", "now", "just", "very", "also", "any", "from", "our", "us", "we"
}

def extract_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def extract_top_words(text: str, n: int = 3) -> List[str]:
    """Extract top frequent words from the text, ignoring stopwords."""
    from collections import Counter
    words = re.findall(r'\b\w+\b', text.lower())
    words = [w for w in words if w not in COMMON_STOPWORDS and len(w) > 3]
    word_counts = Counter(words)
    return [w.capitalize() for w, _ in word_counts.most_common(n)]

def get_key_sentences(text: str, num_sentences: int = 3) -> List[str]:
    """Score and extract top sentences based on word frequency."""
    from collections import Counter
    sentences = extract_sentences(text)
    if len(sentences) <= num_sentences:
        return sentences
    
    words = re.findall(r'\b\w+\b', text.lower())
    words = [w for w in words if w not in COMMON_STOPWORDS and len(w) > 2]
    word_counts = Counter(words)
    
    scores = []
    for s in sentences:
        s_words = re.findall(r'\b\w+\b', s.lower())
        score = sum(word_counts.get(w, 0) for w in s_words)
        if len(s_words) > 0:
            score = score / (len(s_words) ** 0.5)
        scores.append((score, s))
        
    top_scored = sorted(scores, key=lambda x: x[0], reverse=True)[:num_sentences]
    top_sentences = [s for _, s in sorted(top_scored, key=lambda x: sentences.index(x[1]))]
    return top_sentences
