"""
backend/services/quiz_generator.py
==================================
Generates multiple-choice quiz questions based on the video transcript.
"""

import re
import json
from typing import Dict, List

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

QUIZ_PROMPT = """You are an expert educator. Based on the video transcript provided, 
create a 5-question multiple-choice quiz that tests the user's understanding of the key concepts.

CRITICAL INSTRUCTION: You MUST generate the quiz in the following language: {language}

For each question, provide exactly 4 options (A, B, C, D) and specify the correct answer.
Format your output STRICTLY as a JSON array of objects with the following keys:
- "question": The question text
- "options": An array of exactly 4 strings representing the choices
- "correct_index": An integer (0, 1, 2, or 3) indicating which option is correct

Transcript:
\"\"\"
{text}
\"\"\"

JSON Array (return ONLY valid JSON, no markdown):"""

class QuizGenerator:
    """
    Generates interactive quiz questions from transcripts.
    """

    def __init__(self):
        pass  # No cached client — we create a fresh one per call

    def generate_quiz(self, chunks: List[Dict], language: str = "English") -> List[Dict]:
        """
        Generates quiz questions based on the text context.
        """
        combined_text = " ".join([c["text"] for c in chunks])
        text_to_process = combined_text[:15000]

        import os
        has_key = bool(os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY)
        if settings.LLM_PROVIDER == "openai" and has_key:
            return self._generate_with_llm(text_to_process, language)
        else:
            logger.warning("No OpenAI key. Skipping quiz generation.")
            return self._fallback_quiz()

    def _generate_with_llm(self, text: str, language: str) -> List[Dict]:
        try:
            from openai import OpenAI
            import os
            kwargs = {"api_key": os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY}
            if settings.OPENAI_BASE_URL:
                kwargs["base_url"] = settings.OPENAI_BASE_URL
            client = OpenAI(**kwargs)

            prompt = QUIZ_PROMPT.format(text=text, language=language)
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content
            if not raw:
                return self._fallback_quiz()
            raw = raw.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            items = json.loads(raw)
            return items if isinstance(items, list) else []
        except Exception as e:
            logger.error(f"Quiz generation failed: {e}")
            return self._fallback_quiz()

    def _fallback_quiz(self) -> List[Dict]:
        return [
            {
                "question": "What is required to generate detailed quizzes?",
                "options": ["An OpenAI API Key", "More RAM", "A shorter video", "Nothing"],
                "correct_index": 0
            }
        ]
