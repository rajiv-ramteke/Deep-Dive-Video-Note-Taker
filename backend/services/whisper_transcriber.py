"""
backend/services/whisper_transcriber.py
=========================================
Transcribes audio using OpenAI Whisper with word-level timestamps.
"""

import os
import json
from typing import Dict, List, Optional

from backend.utils.config import settings
from backend.utils.helper import ensure_dir, save_json, seconds_to_timestamp
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperTranscriber:
    """
    Wraps OpenAI Whisper for audio-to-text transcription.
    Produces full transcript text + word/segment-level timestamps.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self._model = None
        self.output_dir = settings.TRANSCRIPT_DIR
        ensure_dir(self.output_dir)

    # ── Public API ────────────────────────────────────────────

    def transcribe(self, audio_path: str, job_id: str) -> Dict:
        """
        Transcribe audio file to text.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting Whisper transcription...")
        
        # Check if we can use the much faster and lighter OpenAI API
        use_api = settings.LLM_PROVIDER == "openai" and bool(self.api_key or os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY)
        
        if use_api:
            logger.info("Using OpenAI Whisper API for transcription (fast & memory efficient).")
            try:
                transcript = self._transcribe_api(audio_path, job_id)
            except Exception as e:
                logger.warning(f"OpenAI API transcription failed ({e}), falling back to local model...")
                transcript = self._transcribe_local(audio_path, job_id)
        else:
            logger.info(f"Using local Whisper model ({settings.WHISPER_MODEL}).")
            transcript = self._transcribe_local(audio_path, job_id)

        out_path = os.path.join(self.output_dir, f"{job_id}.json")
        save_json(transcript, out_path)

        logger.info(
            f"Transcription complete: {len(transcript['segments'])} segments, "
            f"language={transcript['language']}"
        )
        return transcript

    def _transcribe_api(self, audio_path: str, job_id: str) -> Dict:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key or os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY)
        
        # OpenAI API has a 25MB limit. Since we convert to 16kHz mono, this usually allows up to ~1.5 hours of audio.
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"]
            )
            
        # Convert OpenAI API response to our expected format
        raw = response.model_dump()
        return self._parse_result(raw)

    def _transcribe_local(self, audio_path: str, job_id: str) -> Dict:
        model = self._get_model()
        raw = model.transcribe(
            audio_path,
            verbose=False,
            word_timestamps=True,
            task="transcribe",
            fp16=False,
        )
        return self._parse_result(raw)

    # ── Private Methods ───────────────────────────────────────

    def _get_model(self):
        """Lazy-load the Whisper model (cached after first call)."""
        if self._model is None:
            import whisper
            import torch
            
            # Let PyTorch use all available CPU cores for faster transcription
            self._patch_ffmpeg_path()
            logger.info(f"Loading local Whisper model '{settings.WHISPER_MODEL}'...")
            self._model = whisper.load_model(
                settings.WHISPER_MODEL,
                device=settings.WHISPER_DEVICE,
                download_root="models/whisper",
            )
            logger.info("Whisper model loaded ✅.")
        return self._model

    @staticmethod
    def _patch_ffmpeg_path() -> None:
        """
        Ensure FFmpeg is findable by Whisper's internal subprocess call.
        Priority: imageio-ffmpeg bundle → system PATH (already there).
        """
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg_exe and os.path.exists(ffmpeg_exe):
                ffmpeg_dir = os.path.dirname(ffmpeg_exe)
                current_path = os.environ.get("PATH", "")
                if ffmpeg_dir not in current_path:
                    os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
                    logger.info(f"Patched PATH with imageio-ffmpeg: {ffmpeg_dir}")
        except Exception as e:
            logger.warning(f"Could not patch ffmpeg path: {e}")

    def _parse_result(self, raw: Dict) -> Dict:
        """Parse raw Whisper output into a clean structured format."""
        segments = []
        for seg in raw.get("segments", []):
            segments.append({
                "id":    seg["id"],
                "start": seg["start"],
                "end":   seg["end"],
                "start_ts": seconds_to_timestamp(seg["start"]),
                "end_ts":   seconds_to_timestamp(seg["end"]),
                "text":  seg["text"].strip(),
                "words": [
                    {
                        "word":  w.get("word", "").strip(),
                        "start": w.get("start", 0),
                        "end":   w.get("end", 0),
                    }
                    for w in seg.get("words", [])
                ],
            })

        full_text = " ".join(s["text"] for s in segments)

        return {
            "text":     full_text,
            "segments": segments,
            "language": raw.get("language", "en"),
            "duration": segments[-1]["end"] if segments else 0,
        }

    def load_transcript(self, job_id: str) -> Optional[Dict]:
        """Load a previously saved transcript from disk."""
        path = os.path.join(self.output_dir, f"{job_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
