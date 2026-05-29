"""
backend/services/timestamp_mapper.py
=======================================
Maps summarized content back to precise video timestamps.
Identifies the most important segments (highlights) in the video.
"""

from typing import Dict, List

from backend.utils.helper import seconds_to_timestamp
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TimestampMapper:
    """
    Aligns key summary topics with their source video timestamps.
    Produces a list of timestamped highlights for easy navigation.
    """

    # ── Public API ────────────────────────────────────────────

    def map_timestamps(self, summarized_chunks: List[Dict]) -> List[Dict]:
        """
        Build a list of timestamped highlights from summarized chunks.

        Args:
            summarized_chunks: Chunks with 'summary', 'start', 'end' fields.

        Returns:
            List of highlight dicts:
                - timestamp:    HH:MM:SS string (start of chunk)
                - end_ts:       HH:MM:SS string (end of chunk)
                - start:        float seconds
                - end:          float seconds
                - title:        First line / sentence of the summary
                - summary:      Full chunk summary
                - chunk_id:     Source chunk index
        """
        highlights = []
        for chunk in summarized_chunks:
            summary = chunk.get("summary", "").strip()
            if not summary:
                continue

            title = self._extract_title(summary)
            highlights.append({
                "chunk_id":  chunk["chunk_id"],
                "timestamp": chunk["start_ts"],
                "end_ts":    chunk["end_ts"],
                "start":     chunk["start"],
                "end":       chunk["end"],
                "title":     title,
                "summary":   summary,
            })

        logger.info(f"Timestamp mapping: {len(highlights)} highlights generated")
        return highlights

    def find_key_moments(
        self,
        transcript_segments: List[Dict],
        keywords: List[str],
    ) -> List[Dict]:
        """
        Search transcript segments for keyword occurrences and return timestamps.

        Args:
            transcript_segments: Raw Whisper segments list.
            keywords:            List of words/phrases to search for.

        Returns:
            List of {keyword, timestamp, start, text} dicts.
        """
        moments = []
        for seg in transcript_segments:
            text_lower = seg["text"].lower()
            for kw in keywords:
                if kw.lower() in text_lower:
                    moments.append({
                        "keyword":   kw,
                        "timestamp": seg["start_ts"],
                        "start":     seg["start"],
                        "text":      seg["text"].strip(),
                    })
                    break  # Only report each segment once

        logger.debug(f"Found {len(moments)} keyword moments for {keywords}")
        return moments

    def generate_chapter_markers(self, highlights: List[Dict]) -> List[Dict]:
        """
        Group highlights into chapters based on time gaps.
        A new chapter starts when there is a gap of > 3 minutes.
        """
        if not highlights:
            return []

        chapters = []
        chapter_num = 1
        chapter_start = highlights[0]
        chapter_items = [highlights[0]]

        for i in range(1, len(highlights)):
            gap = highlights[i]["start"] - highlights[i - 1]["end"]
            if gap > 180:  # 3-minute gap → new chapter
                chapters.append({
                    "chapter":   chapter_num,
                    "title":     f"Chapter {chapter_num}",
                    "timestamp": chapter_start["timestamp"],
                    "start":     chapter_start["start"],
                    "highlights": chapter_items,
                })
                chapter_num += 1
                chapter_start = highlights[i]
                chapter_items = []
            chapter_items.append(highlights[i])

        # Final chapter
        chapters.append({
            "chapter":    chapter_num,
            "title":      f"Chapter {chapter_num}",
            "timestamp":  chapter_start["timestamp"],
            "start":      chapter_start["start"],
            "highlights": chapter_items,
        })

        logger.info(f"Generated {len(chapters)} chapters")
        return chapters

    # ── Private ───────────────────────────────────────────────

    @staticmethod
    def _extract_title(summary: str) -> str:
        """
        Extract a short title from the summary.
        Uses the first non-empty, non-bullet line up to 80 chars.
        """
        for line in summary.splitlines():
            line = line.strip().lstrip("•-*# ")
            if len(line) > 10:
                return line[:80] + ("…" if len(line) > 80 else "")
        return summary[:60]
