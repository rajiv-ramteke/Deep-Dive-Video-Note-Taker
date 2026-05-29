"""
backend/services/rag_pipeline.py
==================================
Retrieval-Augmented Generation (RAG) pipeline.
Embeds transcript chunks into FAISS, then retrieves relevant
segments to answer user queries or improve summarization context.
"""

from typing import Dict, List, Optional

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    RAG pipeline built on FAISS vector store + SentenceTransformers embeddings.
    Enables semantic search over video transcripts.
    """

    def __init__(self):
        self._embedder = None
        self._index = None
        self._chunks: List[Dict] = []

    # ── Public API ────────────────────────────────────────────

    def index_chunks(self, chunks: List[Dict]) -> None:
        """
        Embed and index all transcript chunks into FAISS.

        Args:
            chunks: List of chunk dicts (must have 'text' key).
        """
        import faiss
        import numpy as np

        logger.info(f"Indexing {len(chunks)} chunks into FAISS...")
        self._chunks = chunks
        texts = [c["text"] for c in chunks]

        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # Inner-product = cosine on normalised vecs
        self._index.add(embeddings)

        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim} ✅")

    def save_index(self, path: str = None) -> None:
        """Persist FAISS index to disk."""
        import faiss
        from backend.utils.helper import ensure_dir, save_json
        import os

        path = path or settings.FAISS_INDEX_PATH
        ensure_dir(os.path.dirname(path))
        faiss.write_index(self._index, path)

        # Save chunk metadata alongside the index
        meta_path = path.replace(".index", "_meta.json")
        save_json(self._chunks, meta_path)
        logger.info(f"FAISS index saved: {path}")

    def load_index(self, path: str = None) -> bool:
        """Load a persisted FAISS index from disk."""
        import faiss
        from backend.utils.helper import load_json
        import os

        path = path or settings.FAISS_INDEX_PATH
        meta_path = path.replace(".index", "_meta.json")

        if not os.path.exists(path):
            logger.warning(f"No FAISS index found at {path}")
            return False

        self._index = faiss.read_index(path)
        if os.path.exists(meta_path):
            self._chunks = load_json(meta_path)
        logger.info(f"FAISS index loaded: {self._index.ntotal} vectors ✅")
        return True

    def query(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search over indexed chunks.

        Args:
            query:  Natural language query string.
            top_k:  Number of top results to return.

        Returns:
            List of chunk dicts with an added 'score' field.
        """
        import numpy as np

        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty. Run index_chunks() first.")
            return []

        embedder = self._get_embedder()
        q_vec = embedder.encode([query], normalize_embeddings=True)
        q_vec = q_vec.astype("float32")

        distances, indices = self._index.search(q_vec, min(top_k, self._index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = float(dist)
            results.append(chunk)

        logger.debug(f"RAG query '{query[:50]}...' → {len(results)} results")
        return results

    def get_context_for_summary(self, topic: str, top_k: int = 3) -> str:
        """
        Retrieve relevant chunks and format as a context string for LLMs.
        """
        results = self.query(topic, top_k=top_k)
        if not results:
            return ""
        context_parts = [
            f"[{r['start_ts']} → {r['end_ts']}]: {r['text']}"
            for r in results
        ]
        return "\n\n".join(context_parts)

    # ── Private ───────────────────────────────────────────────

    def _get_embedder(self):
        """Lazy-load SentenceTransformer embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._embedder = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                cache_folder="models/embedding_model",
            )
            logger.info("Embedding model loaded ✅")
        return self._embedder
