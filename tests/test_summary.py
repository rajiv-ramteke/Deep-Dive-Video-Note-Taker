"""
tests/test_summary.py
=======================
Unit tests for the Summarizer and TextChunker services.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.services.text_chunker import TextChunker
from backend.services.summarizer   import Summarizer


# ── TextChunker Tests ─────────────────────────────────────────────────────────

SAMPLE_TRANSCRIPT = {
    "text": "word " * 3000,
    "language": "en",
    "segments": [
        {
            "id": i,
            "start": i * 10.0,
            "end": (i + 1) * 10.0,
            "start_ts": f"00:{i:02d}:00",
            "end_ts":   f"00:{i+1:02d}:00",
            "text":     "word " * 100,
            "words":    [],
        }
        for i in range(30)
    ],
}


class TestTextChunker:

    def test_chunks_produced(self):
        """Chunker should produce at least 1 chunk for a non-empty transcript."""
        chunker = TextChunker(max_chunk_size=500, overlap=50)
        chunks  = chunker.chunk_transcript(SAMPLE_TRANSCRIPT)
        assert len(chunks) > 0

    def test_chunk_has_required_keys(self):
        """Each chunk must have the required metadata keys."""
        chunker = TextChunker(max_chunk_size=500, overlap=50)
        chunks  = chunker.chunk_transcript(SAMPLE_TRANSCRIPT)
        required = {"chunk_id", "text", "start", "end", "start_ts", "end_ts", "segments"}
        for chunk in chunks:
            assert required.issubset(chunk.keys()), f"Missing keys: {required - chunk.keys()}"

    def test_chunk_ids_are_sequential(self):
        """Chunk IDs should be sequential starting from 0."""
        chunker = TextChunker(max_chunk_size=500, overlap=50)
        chunks  = chunker.chunk_transcript(SAMPLE_TRANSCRIPT)
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_empty_transcript_returns_empty_list(self):
        """Empty transcript should produce an empty chunk list."""
        chunker = TextChunker()
        result = chunker.chunk_transcript({"text": "", "segments": []})
        assert result == []

    def test_plain_text_chunking(self):
        """chunk_text() should split a plain string into multiple parts."""
        chunker = TextChunker(max_chunk_size=10, overlap=2)
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, str)
            assert len(c) > 0

    def test_chunk_text_word_count_limit(self):
        """Each chunk should not exceed max_chunk_size words."""
        chunker = TextChunker(max_chunk_size=10, overlap=2)
        text = " ".join([f"w{i}" for i in range(100)])
        chunks = chunker.chunk_text(text)
        for c in chunks:
            assert len(c.split()) <= 10


# ── Summarizer Tests ──────────────────────────────────────────────────────────

class TestSummarizer:

    @patch("backend.services.summarizer.Summarizer._summarize_openai")
    def test_summarize_chunks_calls_backend(self, mock_openai):
        """summarize_chunks should call the LLM backend for each chunk."""
        mock_openai.return_value = "• Point 1\n• Point 2"

        import backend.utils.config as cfg
        cfg.settings.LLM_PROVIDER = "openai"
        cfg.settings.OPENAI_API_KEY = "sk-test"

        s = Summarizer()
        chunks = [
            {"chunk_id": 0, "text": "Sample text A", "start_ts": "00:00:00", "end_ts": "00:01:00"},
            {"chunk_id": 1, "text": "Sample text B", "start_ts": "00:01:00", "end_ts": "00:02:00"},
        ]
        result = s.summarize_chunks(chunks)
        assert len(result) == 2
        for r in result:
            assert "summary" in r
            assert r["summary"] == "• Point 1\n• Point 2"

    @patch("backend.services.summarizer.Summarizer._summarize_huggingface")
    def test_hf_fallback(self, mock_hf):
        """summarize_chunks should use HuggingFace when LLM_PROVIDER=huggingface."""
        mock_hf.return_value = "HF summary text"

        import backend.utils.config as cfg
        cfg.settings.LLM_PROVIDER = "huggingface"

        s = Summarizer()
        chunks = [{"chunk_id": 0, "text": "Test", "start_ts": "00:00:00", "end_ts": "00:00:30"}]
        result = s.summarize_chunks(chunks)
        assert result[0]["summary"] == "HF summary text"
        mock_hf.assert_called_once()
