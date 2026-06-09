"""
backend/services/translator.py
==============================
Translates complex JSON structures (topics, highlights, qa_pairs, transcript)
using OpenAI.
"""

import os
import json
from typing import Dict, List, Any

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

TRANSLATE_JSON_PROMPT = """You are an expert translator. 
Translate the string values in the provided JSON array into {language}.
Keep the exact same JSON structure, keys, and array length. Only translate the text values.

CRITICAL INSTRUCTION: Return ONLY valid JSON, do NOT wrap in markdown fences.

Input JSON:
{json_str}
"""

class Translator:
    def __init__(self):
        self._openai_client = None

    def _get_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            kwargs = {"api_key": os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY}
            if settings.OPENAI_BASE_URL:
                kwargs["base_url"] = settings.OPENAI_BASE_URL
            self._openai_client = OpenAI(**kwargs)
        return self._openai_client

    def translate_json_array(self, data: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
        """Translate a JSON array of dicts by feeding it to the LLM."""
        if not data:
            return []
        
        has_key = bool(os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY)
        if not has_key:
            logger.warning("No OpenAI API key for translation. Returning original data.")
            return data
        
        # Batch translation if too large? For now, we will do it in one go for 
        # topics, qa_pairs, and highlights since they are small.
        # For transcript segments, we might need batching, but let's try direct first.
        try:
            client = self._get_client()
            prompt = TRANSLATE_JSON_PROMPT.format(
                language=language,
                json_str=json.dumps(data, ensure_ascii=False)
            )
            
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            
            raw = response.choices[0].message.content.strip()
            # Clean markdown
            import re
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
            
            translated_data = json.loads(raw)
            return translated_data
        except Exception as e:
            logger.error(f"Failed to translate JSON array: {e}")
            return data

    def translate_transcript(self, segments: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
        """
        Translates transcript segments in batches to avoid token limits.
        """
        if not segments:
            return []
        
        batch_size = 30
        translated_segments = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            # Create a simplified structure to reduce tokens
            simplified = [{"id": j, "text": seg["text"]} for j, seg in enumerate(batch)]
            
            try:
                translated_batch = self.translate_json_array(simplified, language)
                
                # Re-merge
                for j, seg in enumerate(batch):
                    new_seg = seg.copy()
                    if j < len(translated_batch):
                        new_seg["text"] = translated_batch[j].get("text", seg["text"])
                    translated_segments.append(new_seg)
            except Exception as e:
                logger.error(f"Failed to translate transcript batch: {e}")
                translated_segments.extend(batch)
                
        return translated_segments
