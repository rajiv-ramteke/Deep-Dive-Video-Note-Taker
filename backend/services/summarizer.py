"""
backend/services/summarizer.py
================================
LLM-based summarization of transcript chunks.
Supports OpenAI GPT and HuggingFace (BART) backends.
"""

import os
from typing import Dict, List, Optional

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompt Templates ──────────────────────────────────────────────────────────

CHUNK_SUMMARY_PROMPT = """You are an expert educator and note-taker analyzing a video transcript.
Extract ALL important information from this transcript excerpt. Be thorough and detailed.

CRITICAL INSTRUCTION: You MUST write everything in the following language: {language}

Transcript:
\"\"\"
{text}
\"\"\"

Provide a detailed summary covering:
- All key concepts and definitions explained
- Important facts, numbers, formulas, or data mentioned
- Examples given by the speaker
- Any processes or steps described

Format as clear bullet points. Do NOT skip any important detail:"""

FINAL_SUMMARY_PROMPT = """You are a world-class educator creating comprehensive study notes from a video.
Based on the chunk summaries below, produce beautifully structured, in-depth notes.

CRITICAL INSTRUCTION: You MUST write EVERYTHING in the following language: {language}

Use this exact Markdown structure:
## 📌 Overview
(2-3 sentence overview of what this video covers)

## 🎯 Key Concepts
(Explain each main concept in simple, clear language with examples)

## 📖 Detailed Notes
(Use ### subheadings for each topic. Include all facts, formulas, definitions, processes)

## 💡 Important Points to Remember
(Bullet list of the most critical takeaways)

Chunk Summaries:
\"\"\"
{summaries}
\"\"\"

Write the complete, detailed notes now:"""


class Summarizer:
    """
    Summarizes transcript chunks using LLMs.
    Supports OpenAI GPT (API) and HuggingFace BART (local).
    """

    def __init__(self):
        self._openai_client = None
        self._hf_pipeline = None

    # ── Public API ────────────────────────────────────────────

    def summarize_chunks(self, chunks: List[Dict], language: str = "English") -> List[Dict]:
        """
        Summarize each transcript chunk individually.

        Args:
            chunks: List of chunk dicts from TextChunker.
            language: Language to output the summary in.

        Returns:
            Same list with 'summary' key added to each chunk dict.
        """
        logger.info(f"Summarizing {len(chunks)} chunks using {settings.LLM_PROVIDER} in {language}...")
        results = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"  Summarizing chunk {i + 1}/{len(chunks)}")
            summary = self._summarize_text(chunk["text"], language)
            results.append({**chunk, "summary": summary})
        logger.info("Chunk summarization complete ✅")
        return results

    def generate_final_notes(self, summarized_chunks: List[Dict], language: str = "English") -> str:
        """
        Combine all chunk summaries into a single structured document.

        Args:
            summarized_chunks: Output of summarize_chunks().
            language: Language to output the notes in.

        Returns:
            Markdown-formatted final notes string.
        """
        combined = "\n\n".join(
            f"[{c['start_ts']} → {c['end_ts']}]\n{c['summary']}"
            for c in summarized_chunks
            if c.get("summary")
        )
        prompt = FINAL_SUMMARY_PROMPT.format(summaries=combined, language=language)
        logger.info("Generating final structured notes...")
        final = self._llm_complete(prompt)
        logger.info("Final notes generated ✅")
        return final

    def translate_notes(self, notes: str, language: str) -> str:
        """
        Translates the final notes into the specified target language.
        """
        prompt = (
            f"You are an expert translator. Translate the following Markdown notes into {language}. "
            f"Maintain all Markdown formatting (headings, bullet points, bold, etc.) exactly as they appear in the original.\n\n"
            f"Notes:\n\"\"\"\n{notes}\n\"\"\"\n\nTranslated Notes:"
        )
        logger.info(f"Translating notes to {language}...")
        translated = self._llm_complete(prompt)
        logger.info(f"Notes translated to {language} ✅")
        return translated

    def answer_question(self, query: str, context: str) -> str:
        """
        Answers a user's question using video context + general AI knowledge
        for a more detailed and educational response.
        """
        prompt = (
            f"You are a friendly, knowledgeable human expert having a conversation with the user about a video they just watched.\n\n"
            f"The user has a question. Below is the relevant portion of the video transcript.\n"
            f"Your job is to:\n"
            f"1. Answer the question naturally, clearly, and conversationally, as if you are explaining it to a friend.\n"
            f"2. Use the transcript context to inform your answer. If the transcript is missing details, seamlessly bring in your own general knowledge to give a great, helpful explanation.\n"
            f"3. Avoid acting like a robot (don't say 'As an AI' or 'Based on the context'). Just give the answer directly and warmly.\n"
            f"4. You can use some light formatting (like bolding or a few bullet points if it helps readability), but keep the tone natural and human-like.\n\n"
            f"Video Transcript Context:\n\"\"\"\n{context}\n\"\"\"\n\n"
            f"User's Question: {query}\n\n"
            f"Your natural answer:"
        )
        logger.info(f"Answering query: '{query}'")
        answer = self._llm_complete(prompt)
        logger.info("Answer generated ✅")
        return answer

    # ── Internal dispatcher ───────────────────────────────────

    def _summarize_text(self, text: str, language: str) -> str:
        """Dispatch to the configured LLM backend."""
        import os
        has_key = bool(os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY)
        if settings.LLM_PROVIDER == "openai" and has_key:
            return self._summarize_openai(text, language)
        else:
            return self._summarize_huggingface(text)

    def _llm_complete(self, prompt: str) -> str:
        """Run a full prompt through the configured LLM."""
        import os
        has_key = bool(os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY)
        if settings.LLM_PROVIDER == "openai" and has_key:
            return self._openai_chat(prompt)
        else:
            return self._summarize_huggingface(prompt)

    # ── OpenAI Backend ────────────────────────────────────────

    def _get_openai_client(self):
        from openai import OpenAI
        import os
        kwargs = {"api_key": os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY}
        if settings.OPENAI_BASE_URL:
            kwargs["base_url"] = settings.OPENAI_BASE_URL
        return OpenAI(**kwargs)

    def _summarize_openai(self, text: str, language: str) -> str:
        prompt = CHUNK_SUMMARY_PROMPT.format(text=text[:3000], language=language)
        return self._openai_chat(prompt)

    def _openai_chat(self, prompt: str) -> str:
        try:
            client = self._get_openai_client()
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional academic note-taker and summarizer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"[Summarization failed: {str(e)[:100]}]"

    # ── HuggingFace Backend ───────────────────────────────────

    def _get_hf_pipeline(self):
        # We now return a tuple of (model, tokenizer) instead of a pipeline
        if self._hf_pipeline is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info(f"Loading HuggingFace model: {settings.HF_SUMMARIZATION_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(settings.HF_SUMMARIZATION_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(settings.HF_SUMMARIZATION_MODEL)
            self._hf_pipeline = (model, tokenizer)
            logger.info("HuggingFace summarization model loaded ✅")
        return self._hf_pipeline

    def _summarize_huggingface(self, text: str) -> str:
        try:
            model, tokenizer = self._get_hf_pipeline()
            inputs = tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            )
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=4,
                min_length=30,
                max_length=256,
                early_stopping=True
            )
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"HuggingFace summarization error: {e}")
            return f"[Summarization failed: {str(e)[:100]}]"
