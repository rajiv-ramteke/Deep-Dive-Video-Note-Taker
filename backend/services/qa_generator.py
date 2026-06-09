"""
backend/services/qa_generator.py
==================================
Generates Questions and Answers from transcript text using LLM.
"""

import re
from typing import Dict, List

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

QA_PROMPT = """You are an expert educator. Analyse the following transcript and 
generate 3 to 5 key questions and their corresponding answers based on the content.

CRITICAL INSTRUCTION: You MUST generate the questions and answers in the following language: {language}

Format your output STRICTLY as a JSON array with objects having these keys:
  - "question": the question text
  - "answer": the correct answer based on the transcript

Transcript:
\"\"\"
{text}
\"\"\"

JSON Array (return ONLY valid JSON, no markdown):"""

class QAGenerator:
    """
    Generates Q&A pairs from transcript chunks.
    """

    def __init__(self):
        pass  # No cached client — fresh one per call

    # ── Public API ────────────────────────────────────────────

    def generate_qa(self, chunks: List[Dict], language: str = "English") -> List[Dict]:
        """
        Generate Q&A pairs from all transcript chunks.

        Args:
            chunks: List of chunk dicts.
            language: The language to generate Q&A in.

        Returns:
            List of Q&A dicts.
        """
        all_qa = []

        # We don't want to run this for every single chunk if there are many,
        # but for simplicity we can combine the summaries or process up to a certain limit.
        # Alternatively, we can use the final structured notes as the source.
        
        # Let's extract from the first few chunks or a combined summary to avoid high costs.
        # Here we just iterate over chunks like the action item extractor, but limit it.
        
        # Combine the text of all chunks to get a comprehensive context, but keep it within limits.
        combined_text = " ".join([c["text"] for c in chunks])
        
        # If the text is very long, chunk it into larger sections or just take the first part.
        # For Q&A, taking a comprehensive slice is good.
        max_chars = 15000
        text_to_process = combined_text[:max_chars]

        if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            qa_pairs = self._generate_with_llm(text_to_process, language)
            all_qa.extend(qa_pairs)
        else:
            logger.warning("LLM provider is not OpenAI or API key is missing. Q&A generation skipped.")

        logger.info(f"Generated {len(all_qa)} Q&A pairs total")
        return all_qa

    # ── Private ───────────────────────────────────────────────

    def _generate_with_llm(self, text: str, language: str) -> List[Dict]:
        """Use OpenAI to generate Q&A pairs."""
        import json as _json
        try:
            from openai import OpenAI
            kwargs = {"api_key": settings.OPENAI_API_KEY}
            if settings.OPENAI_BASE_URL:
                kwargs["base_url"] = settings.OPENAI_BASE_URL
            client = OpenAI(**kwargs)

            prompt = QA_PROMPT.format(text=text, language=language)
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            raw = response.choices[0].message.content
            if not raw:
                return []
            raw = raw.strip()
            # Strip any accidental markdown fences
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            items = _json.loads(raw)
            return items if isinstance(items, list) else []
        except Exception as e:
            logger.error(f"LLM Q&A generation failed: {e}")
            return []
