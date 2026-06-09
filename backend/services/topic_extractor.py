"""
backend/services/topic_extractor.py
======================================
Extracts key topics from a video transcript and summarizes each one
in a structured, easy-to-understand format using LLM.
"""

import os
import re
from typing import Dict, List

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

TOPIC_EXTRACTION_PROMPT = """You are an expert educator creating structured study notes from a video transcript.
Analyze the transcript below and identify the KEY TOPICS discussed.

CRITICAL INSTRUCTION: You MUST generate the output in the following language: {language}

For each topic, provide:
- A clear topic title
- A simple 1-2 sentence summary
- 3-5 key points that explain the concept in simple, easy-to-understand language

Format your output STRICTLY as a JSON array:
[
  {{
    "topic": "Topic Title Here",
    "summary": "One or two sentence overview of this topic.",
    "key_points": [
      "First key point explained simply",
      "Second key point explained simply",
      "Third key point explained simply"
    ]
  }}
]

Transcript:
\"\"\"
{text}
\"\"\"

JSON Array (return ONLY valid JSON, no markdown fences):"""


class TopicExtractor:
    """
    Extracts structured topic summaries from video transcripts.
    Uses OpenAI when available; falls back to a basic heuristic.
    """

    def __init__(self):
        pass  # No cached client — fresh one per call

    # ── Public API ────────────────────────────────────────────

    def extract(self, chunks: List[Dict], language: str = "English") -> List[Dict]:
        """
        Extract structured topic summaries from all transcript chunks.

        Args:
            chunks: List of chunk dicts from TextChunker.
            language: The language to generate topics in.

        Returns:
            List of topic dicts with keys: topic, summary, key_points.
        """
        combined_text = " ".join([c["text"] for c in chunks])
        text_to_process = combined_text[:15000]

        has_key = bool(os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY)
        if settings.LLM_PROVIDER == "openai" and has_key:
            topics = self._extract_with_llm(text_to_process, language)
        else:
            logger.warning("OpenAI not configured. Using fallback topic extraction.")
            topics = self._fallback_topics(text_to_process)

        logger.info(f"Extracted {len(topics)} topics total")
        return topics

    # ── Private ───────────────────────────────────────────────

    def _extract_with_llm(self, text: str, language: str) -> List[Dict]:
        """Use OpenAI to extract structured topics."""
        import json as _json

        try:
            from openai import OpenAI
            kwargs = {"api_key": os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY}
            if settings.OPENAI_BASE_URL:
                kwargs["base_url"] = settings.OPENAI_BASE_URL
            client = OpenAI(**kwargs)

            prompt = TOPIC_EXTRACTION_PROMPT.format(text=text[:12000], language=language)
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content
            if not raw:
                return self._fallback_topics(text)
            raw = raw.strip()
            # Strip any accidental markdown fences
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            topics = _json.loads(raw)
            return topics if isinstance(topics, list) else []
        except Exception as e:
            logger.warning(f"LLM topic extraction failed: {e}. Using fallback.")
            return self._fallback_topics(text)

    def _fallback_topics(self, text: str) -> List[Dict]:
        """Simple fallback: return a single generic topic."""
        return [
            {
                "topic": "Video Content Overview",
                "summary": "A summary of the main content discussed in this video.",
                "key_points": [
                    "This video covers several educational topics.",
                    "Process the video with an OpenAI API key for detailed topic extraction.",
                ],
            }
        ]
