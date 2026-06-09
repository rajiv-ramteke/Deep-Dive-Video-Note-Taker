"""
backend/services/quiz_generator.py
==================================
Generates multiple-choice quiz questions based on the video transcript.
"""

import os
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

        has_key = bool(os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY)
        if settings.LLM_PROVIDER == "openai" and has_key:
            return self._generate_with_llm(text_to_process, language)
        else:
            logger.warning("No OpenAI key. Generating local fallback quiz.")
            return self._fallback_quiz(text_to_process)

    def _generate_with_llm(self, text: str, language: str) -> List[Dict]:
        try:
            from openai import OpenAI
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
                return self._fallback_quiz(text)
            raw = raw.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            items = json.loads(raw)
            return items if isinstance(items, list) else []
        except Exception as e:
            logger.error(f"Quiz generation failed: {e}")
            return self._fallback_quiz(text)

    def _fallback_quiz(self, text: str) -> List[Dict]:
        """Generate interactive quiz questions locally using extractive NLP."""
        from backend.utils.helper import extract_sentences, extract_top_words
        
        sentences = extract_sentences(text)
        if not sentences:
            return [
                {
                    "question": "What is required to generate detailed quizzes?",
                    "options": ["An OpenAI API Key", "More RAM", "A shorter video", "Nothing"],
                    "correct_index": 0
                }
            ]

        # Extract top frequent words to use as key concepts and distractors
        all_top_words = extract_top_words(text, 15)
        if len(all_top_words) < 5:
            return [
                {
                    "question": "What is required to generate detailed quizzes?",
                    "options": ["An OpenAI API Key", "More RAM", "A shorter video", "Nothing"],
                    "correct_index": 0
                }
            ]

        quiz = []
        used_words = set()
        question_count = 0
        
        for word in all_top_words:
            if question_count >= 5:
                break
                
            matching_sentence = None
            for s in sentences:
                if re.search(r'\b' + re.escape(word) + r'\b', s, re.IGNORECASE):
                    words_in_s = s.split()
                    if 10 <= len(words_in_s) <= 35:
                        matching_sentence = s
                        break
                        
            if matching_sentence and word not in used_words:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                question_text = pattern.sub("________", matching_sentence)
                
                distractors = [w for w in all_top_words if w != word and w not in used_words][:3]
                while len(distractors) < 3:
                    extra_words = [w for w in all_top_words if w != word and w not in distractors]
                    if extra_words:
                        distractors.append(extra_words[0])
                    else:
                        distractors.append("Concept")
                
                options = [word] + distractors
                
                import random
                correct_word = word
                random.seed(question_count)
                random.shuffle(options)
                correct_index = options.index(correct_word)
                
                quiz.append({
                    "question": f"Complete the statement: \"{question_text}\"",
                    "options": options,
                    "correct_index": correct_index
                })
                
                used_words.add(word)
                question_count += 1
                
        if not quiz:
            quiz = [
                {
                    "question": "What is required to generate detailed quizzes?",
                    "options": ["An OpenAI API Key", "More RAM", "A shorter video", "Nothing"],
                    "correct_index": 0
                }
            ]
            
        return quiz
