"""
tests/test_rag.py
==================
Unit tests for the RAG pipeline (FAISS indexing + semantic search).
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from backend.services.rag_pipeline import RAGPipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {"chunk_id": 0, "text": "Machine learning is a subset of artificial intelligence.",
     "start_ts": "00:00:00", "end_ts": "00:01:00", "start": 0.0, "end": 60.0},
    {"chunk_id": 1, "text": "Deep learning uses neural networks with many layers.",
     "start_ts": "00:01:00", "end_ts": "00:02:00", "start": 60.0, "end": 120.0},
    {"chunk_id": 2, "text": "Natural language processing deals with text and speech.",
     "start_ts": "00:02:00", "end_ts": "00:03:00", "start": 120.0, "end": 180.0},
    {"chunk_id": 3, "text": "FAISS is a library for efficient similarity search.",
     "start_ts": "00:03:00", "end_ts": "00:04:00", "start": 180.0, "end": 240.0},
    {"chunk_id": 4, "text": "Whisper is an automatic speech recognition model by OpenAI.",
     "start_ts": "00:04:00", "end_ts": "00:05:00", "start": 240.0, "end": 300.0},
]


def _make_mock_embedder(dim: int = 64):
    """Return a mock SentenceTransformer that produces random fixed embeddings."""
    mock = MagicMock()
    np.random.seed(42)

    def encode(texts, **kwargs):
        vecs = np.random.rand(len(texts), dim).astype("float32")
        # Normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock.encode.side_effect = encode
    return mock


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRAGPipeline:

    @patch("backend.services.rag_pipeline.RAGPipeline._get_embedder")
    def test_index_builds_successfully(self, mock_get):
        mock_get.return_value = _make_mock_embedder()
        rag = RAGPipeline()
        rag.index_chunks(SAMPLE_CHUNKS)
        assert rag._index is not None
        assert rag._index.ntotal == len(SAMPLE_CHUNKS)

    @patch("backend.services.rag_pipeline.RAGPipeline._get_embedder")
    def test_query_returns_results(self, mock_get):
        mock_get.return_value = _make_mock_embedder()
        rag = RAGPipeline()
        rag.index_chunks(SAMPLE_CHUNKS)
        results = rag.query("speech recognition", top_k=3)
        assert len(results) <= 3
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)

    @patch("backend.services.rag_pipeline.RAGPipeline._get_embedder")
    def test_query_results_have_scores(self, mock_get):
        mock_get.return_value = _make_mock_embedder()
        rag = RAGPipeline()
        rag.index_chunks(SAMPLE_CHUNKS)
        results = rag.query("neural network", top_k=2)
        for r in results:
            assert isinstance(r["score"], float)

    @patch("backend.services.rag_pipeline.RAGPipeline._get_embedder")
    def test_top_k_respected(self, mock_get):
        mock_get.return_value = _make_mock_embedder()
        rag = RAGPipeline()
        rag.index_chunks(SAMPLE_CHUNKS)
        results = rag.query("anything", top_k=2)
        assert len(results) <= 2

    def test_query_on_empty_index_returns_empty(self):
        rag = RAGPipeline()
        results = rag.query("test", top_k=3)
        assert results == []

    @patch("backend.services.rag_pipeline.RAGPipeline._get_embedder")
    def test_save_and_load_index(self, mock_get, tmp_path):
        mock_get.return_value = _make_mock_embedder()
        index_path = str(tmp_path / "test.index")

        rag = RAGPipeline()
        rag.index_chunks(SAMPLE_CHUNKS)
        rag.save_index(index_path)

        import os
        assert os.path.exists(index_path)

        rag2 = RAGPipeline()
        loaded = rag2.load_index(index_path)
        assert loaded
        assert rag2._index.ntotal == len(SAMPLE_CHUNKS)

    @patch("backend.services.rag_pipeline.RAGPipeline._get_embedder")
    def test_get_context_returns_string(self, mock_get):
        mock_get.return_value = _make_mock_embedder()
        rag = RAGPipeline()
        rag.index_chunks(SAMPLE_CHUNKS)
        ctx = rag.get_context_for_summary("speech recognition", top_k=2)
        assert isinstance(ctx, str)
        assert len(ctx) > 0
