"""
backend/services/action_item_extractor.py
==========================================
Extracts action items, decisions, and follow-up tasks from
transcript text using LLM-based NLP.
"""

import re
from typing import Dict, List

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

ACTION_ITEM_PROMPT = """You are an expert meeting analyst. Analyse the following transcript and 
extract ALL action items, tasks, decisions, and follow-up points. Keep descriptions very simple and easy to understand.

CRITICAL INSTRUCTION: You MUST generate the action items in the following language: {language}

Format your output STRICTLY as a JSON array with objects having these keys:
  - "type":        one of ["action", "decision", "follow_up", "reminder"]
  - "description": simple description of the item
  - "owner":       person responsible (use "Unassigned" if unclear)
  - "priority":    one of ["high", "medium", "low"]

Transcript:
\"\"\"
{text}
\"\"\"

JSON Array (return ONLY valid JSON, no markdown):"""

# ── Regex fallback patterns ───────────────────────────────────────────────────
ACTION_PATTERNS = [
    r"(?:we need to|we should|you should|please|make sure to|don't forget to|"
    r"action item[:\s]|todo[:\s]|follow[\s-]up[:\s]|next step[:\s]|"
    r"will|shall|must|have to|going to)\s+(.+?)(?:\.|$)",
]

DECISION_PATTERNS = [
    r"(?:we decided|decision[:\s]|agreed|we agreed|it was decided|"
    r"resolved|concluded|the conclusion is)\s+(.+?)(?:\.|$)",
]


class ActionItemExtractor:
    """
    Extracts structured action items from transcript text.
    Uses LLM when available; falls back to regex heuristics.
    """

    def __init__(self):
        self._openai_client = None

    # ── Public API ────────────────────────────────────────────

    def extract(self, chunks: List[Dict], language: str = "English") -> List[Dict]:
        """
        Extract action items from all transcript chunks.

        Args:
            chunks: List of chunk dicts from TextChunker.
            language: The language to generate action items in.

        Returns:
            List of action item dicts.
        """
        all_items = []

        # We don't want to run this for every single chunk if there are many,
        # so we combine or limit. For simplicity, we process the whole text in chunks or combined.
        # Let's combine the text and limit to 15k chars for the prompt.
        combined_text = " ".join([c["text"] for c in chunks])
        text_to_process = combined_text[:15000]

        import os
        has_key = bool(os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY)
        if settings.LLM_PROVIDER == "openai" and has_key:
            items = self._extract_with_llm(text_to_process, language)
            all_items.extend(items)
        else:
            logger.warning("LLM provider is not OpenAI or API key is missing. Using regex fallback.")
            items = self._extract_with_regex(combined_text)
            all_items.extend(items)

        logger.info(f"Extracted {len(all_items)} action items total")
        return all_items

    def extract_from_full_text(self, text: str) -> List[Dict]:
        """Extract from a single text block (no chunk metadata)."""
        dummy_chunk = {"text": text, "start_ts": "00:00:00", "chunk_id": 0}
        return self._extract_from_chunk(dummy_chunk)

    # ── Private ───────────────────────────────────────────────

    def _extract_from_chunk(self, chunk: Dict) -> List[Dict]:
        """Try LLM extraction; fall back to regex."""
        import os
        has_key = bool(os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY)
        if settings.LLM_PROVIDER == "openai" and has_key:
            items = self._extract_with_llm(chunk["text"])
        else:
            items = self._extract_with_regex(chunk["text"])

        # Attach timestamp to each item
        for item in items:
            item["timestamp"] = chunk.get("start_ts", "00:00:00")
            item["chunk_id"]  = chunk.get("chunk_id", 0)

        return items

    def _extract_with_llm(self, text: str, language: str = "English") -> List[Dict]:
        """Use OpenAI to extract structured action items."""
        import json as _json
        try:
            from openai import OpenAI
            import os
            if self._openai_client is None:
                kwargs = {"api_key": os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY}
                if settings.OPENAI_BASE_URL:
                    kwargs["base_url"] = settings.OPENAI_BASE_URL
                self._openai_client = OpenAI(**kwargs)

            prompt = ACTION_ITEM_PROMPT.format(text=text[:3000], language=language)
            response = self._openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            raw = response.choices[0].message.content.strip()
            # Strip any accidental markdown fences
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            items = _json.loads(raw)
            return items if isinstance(items, list) else []
        except Exception as e:
            logger.warning(f"LLM action item extraction failed: {e}. Using regex.")
            return self._extract_with_regex(text)

    def _extract_with_regex(self, text: str) -> List[Dict]:
        """Regex-based heuristic extraction as fallback."""
        items = []
        text_lower = text.lower()

        for pattern in ACTION_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                desc = match.group(1).strip()
                if len(desc) > 10:
                    items.append({
                        "type":        "action",
                        "description": desc.capitalize(),
                        "owner":       "Unassigned",
                        "priority":    "medium",
                    })

        for pattern in DECISION_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                desc = match.group(1).strip()
                if len(desc) > 10:
                    items.append({
                        "type":        "decision",
                        "description": desc.capitalize(),
                        "owner":       "Unassigned",
                        "priority":    "high",
                    })

        return items[:10]   # Cap regex results per chunk

    @staticmethod
    def _deduplicate(items: List[Dict]) -> List[Dict]:
        """Remove near-duplicate action items by description similarity."""
        seen = set()
        unique = []
        for item in items:
            key = item["description"].lower()[:60]
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique
