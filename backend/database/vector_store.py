"""
backend/database/vector_store.py
===================================
High-level vector store abstraction used by the RAG pipeline.
Stores both vectors and associated metadata (chunk dicts).
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from backend.database.faiss_db import FAISSDatabase
from backend.utils.helper import ensure_dir, save_json, load_json
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Combines a FAISS index with a metadata store.
    Provides add / search / clear operations.
    """

    def __init__(self, index_path: str = None, meta_path: str = None):
        self.index_path = index_path or "data/embeddings/faiss.index"
        self.meta_path  = meta_path  or "data/embeddings/metadata.json"
        self._db: Optional[FAISSDatabase] = None
        self._meta: List[Dict] = []

    # ── Indexing ──────────────────────────────────────────────

    def add(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        """
        Add vectors and their metadata to the store.

        Args:
            vectors:  np.ndarray of shape (N, D)
            metadata: List of N dicts (chunk info, text, timestamps…)
        """
        assert len(vectors) == len(metadata), "Vector/metadata length mismatch"
        dim = vectors.shape[1]

        self._db = FAISSDatabase(dim=dim, index_path=self.index_path)
        self._db.build(vectors)
        self._meta = metadata

        logger.info(f"VectorStore: {len(metadata)} items indexed")

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self._db:
            self._db.save()
        save_json(self._meta, self.meta_path)
        logger.info("VectorStore saved to disk")

    def load(self) -> bool:
        """Load index and metadata from disk."""
        if not os.path.exists(self.index_path):
            return False
        # Peek at metadata to get dimension
        if os.path.exists(self.meta_path):
            self._meta = load_json(self.meta_path)

        # Detect dim from a dummy load
        import faiss
        idx = faiss.read_index(self.index_path)
        dim = idx.d

        self._db = FAISSDatabase(dim=dim, index_path=self.index_path)
        self._db.load()
        logger.info(f"VectorStore loaded: {len(self._meta)} items")
        return True

    # ── Search ────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search the store and return metadata for top-k matches.

        Returns:
            List of metadata dicts with an added 'score' field.
        """
        if not self._db or not self._db.is_ready:
            logger.warning("VectorStore not ready. Call add() or load() first.")
            return []

        distances, indices = self._db.search(query_vector, top_k)
        results = []
        for dist, idx in zip(distances, indices):
            if 0 <= idx < len(self._meta):
                item = dict(self._meta[idx])
                item["score"] = float(dist)
                results.append(item)

        return results

    # ── Utilities ─────────────────────────────────────────────

    def clear(self) -> None:
        """Reset the store."""
        self._db   = None
        self._meta = []
        logger.info("VectorStore cleared")

    @property
    def size(self) -> int:
        return self._db.size if self._db else 0

    @property
    def is_empty(self) -> bool:
        return self.size == 0
