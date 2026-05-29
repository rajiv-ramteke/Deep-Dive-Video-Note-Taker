"""
backend/database/faiss_db.py
==============================
FAISS index management — create, save, load, search.
"""

import os
from typing import List, Optional, Tuple

import numpy as np

from backend.utils.config import settings
from backend.utils.helper import ensure_dir
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSDatabase:
    """
    Low-level FAISS index wrapper.
    Handles creation, persistence, and k-NN search.
    """

    def __init__(self, dim: int = 384, index_path: str = None):
        self.dim = dim
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self._index = None

    # ── Build ─────────────────────────────────────────────────

    def build(self, vectors: np.ndarray) -> None:
        """Build a flat inner-product FAISS index from vectors."""
        import faiss
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim, \
            f"Expected shape (N, {self.dim}), got {vectors.shape}"

        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)

        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(vectors)
        logger.info(f"FAISS index built: {self._index.ntotal} vectors")

    # ── Persist ───────────────────────────────────────────────

    def save(self) -> None:
        """Write FAISS index to disk."""
        import faiss
        ensure_dir(os.path.dirname(self.index_path))
        faiss.write_index(self._index, self.index_path)
        logger.info(f"FAISS index saved → {self.index_path}")

    def load(self) -> bool:
        """Load FAISS index from disk. Returns True if successful."""
        import faiss
        if not os.path.exists(self.index_path):
            logger.warning(f"No index at {self.index_path}")
            return False
        self._index = faiss.read_index(self.index_path)
        logger.info(f"FAISS index loaded: {self._index.ntotal} vectors")
        return True

    # ── Search ────────────────────────────────────────────────

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> Tuple[List[float], List[int]]:
        """
        Nearest-neighbour search.

        Returns:
            (distances, indices) — lists of length top_k.
        """
        import faiss
        if self._index is None:
            raise RuntimeError("FAISS index not built or loaded.")

        q = query_vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)

        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(q, k)
        return distances[0].tolist(), indices[0].tolist()

    # ── Properties ────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0
