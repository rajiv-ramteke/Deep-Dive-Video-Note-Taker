"""
backend/services/text_chunker.py
==================================
Splits long transcripts into overlapping chunks for LLM processing.
Preserves segment-level metadata (timestamps) per chunk.
"""

from typing import Dict, List

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TextChunker:
    """
    Splits a Whisper transcript into overlapping text chunks
    while preserving start/end timestamps for each chunk.
    """

    def __init__(
        self,
        max_chunk_size: int = None,
        overlap: int = None,
    ):
        self.max_chunk_size = max_chunk_size or settings.MAX_CHUNK_SIZE
        self.overlap = overlap or settings.CHUNK_OVERLAP

    # ── Public API ────────────────────────────────────────────

    def chunk_transcript(self, transcript: Dict) -> List[Dict]:
        """
        Chunk a transcript dict (as returned by WhisperTranscriber).

        Returns:
            List of chunk dicts, each with:
                - chunk_id:  int index
                - text:      chunk text
                - start:     float seconds (start of chunk)
                - end:       float seconds (end of chunk)
                - start_ts:  HH:MM:SS string
                - end_ts:    HH:MM:SS string
                - segments:  original Whisper segment IDs in this chunk
        """
        segments = transcript.get("segments", [])
        if not segments:
            logger.warning("Empty transcript — nothing to chunk.")
            return []

        chunks = []
        current_words: List[str] = []
        current_start: float = segments[0]["start"]
        current_end: float = 0.0
        current_seg_ids: List[int] = []

        for seg in segments:
            seg_words = seg["text"].split()
            current_words.extend(seg_words)
            current_end = seg["end"]
            current_seg_ids.append(seg["id"])

            if len(current_words) >= self.max_chunk_size:
                chunk_text = " ".join(current_words[: self.max_chunk_size])
                chunks.append(
                    self._make_chunk(
                        len(chunks),
                        chunk_text,
                        current_start,
                        current_end,
                        current_seg_ids[:],
                    )
                )
                # Overlap: keep last N words as context for next chunk
                overlap_words = current_words[-self.overlap:] if self.overlap else []
                current_words = overlap_words
                current_start = seg["end"]
                current_seg_ids = []

        # Flush remaining words
        if current_words:
            chunk_text = " ".join(current_words)
            chunks.append(
                self._make_chunk(
                    len(chunks),
                    chunk_text,
                    current_start,
                    current_end,
                    current_seg_ids,
                )
            )

        logger.info(
            f"Transcript chunked into {len(chunks)} chunks "
            f"(max_size={self.max_chunk_size}, overlap={self.overlap})"
        )
        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """
        Simple text-only chunking (no timestamp preservation).
        Useful for plain string input.
        """
        words = text.split()
        chunks = []
        step = self.max_chunk_size - self.overlap

        for i in range(0, len(words), step):
            chunk = " ".join(words[i: i + self.max_chunk_size])
            if chunk:
                chunks.append(chunk)

        logger.debug(f"Text split into {len(chunks)} plain chunks.")
        return chunks

    # ── Private ───────────────────────────────────────────────

    @staticmethod
    def _make_chunk(
        idx: int,
        text: str,
        start: float,
        end: float,
        seg_ids: List[int],
    ) -> Dict:
        from backend.utils.helper import seconds_to_timestamp
        return {
            "chunk_id": idx,
            "text":     text,
            "start":    start,
            "end":      end,
            "start_ts": seconds_to_timestamp(start),
            "end_ts":   seconds_to_timestamp(end),
            "segments": seg_ids,
        }
