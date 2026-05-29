"""
backend/services/audio_extractor.py
=====================================
Extracts audio track from video/audio files.

Priority order:
  1. imageio-ffmpeg  — bundled binary, no system install needed
  2. System FFmpeg   — if installed on PATH
  3. MoviePy         — last resort fallback
  4. Direct copy     — for WAV files already in the right format
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from backend.utils.config import settings
from backend.utils.helper import ensure_dir, seconds_to_timestamp
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Audio extensions that can be passed directly to Whisper without extraction
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}


class AudioExtractor:
    """
    Handles video-to-audio and audio format conversion.
    Uses imageio-ffmpeg bundled binary — no manual FFmpeg install required.
    """

    def __init__(self):
        self.output_dir = settings.AUDIO_DIR
        ensure_dir(self.output_dir)
        self._ffmpeg_exe = self._find_ffmpeg()

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, video_path: str, job_id: str) -> str:
        """
        Extract / convert audio from a video or audio file.

        For audio files (.wav, .mp3 …): converts/copies to 16kHz mono WAV.
        For video files (.mp4, .mkv …): extracts audio track first.

        Returns:
            Path to the output .wav file ready for Whisper.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File not found: {video_path}")

        ext = Path(video_path).suffix.lower()
        audio_path = os.path.join(self.output_dir, f"{job_id}.wav")

        logger.info(f"Processing file: {video_path}  (ext={ext})")

        # ── Case 1: already a valid WAV — just copy/convert with ffmpeg ──────
        if ext in AUDIO_EXTENSIONS:
            logger.info("Input is an audio file — converting to 16kHz mono WAV.")
            self._convert_audio(video_path, audio_path)
        # ── Case 2: video — extract audio track ──────────────────────────────
        elif ext in VIDEO_EXTENSIONS:
            logger.info("Input is a video file — extracting audio track.")
            self._convert_audio(video_path, audio_path)
        else:
            # Unknown extension — try anyway
            logger.warning(f"Unknown extension '{ext}', attempting extraction.")
            self._convert_audio(video_path, audio_path)

        if not os.path.exists(audio_path):
            raise RuntimeError(
                "Audio extraction failed — output file was not created. "
                "Ensure the input file is a valid video or audio file."
            )

        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Audio ready: {audio_path}  ({size_mb:.1f} MB)")
        return audio_path

    def get_video_duration(self, video_path: str) -> Optional[float]:
        """Return media duration in seconds."""
        ffmpeg = self._ffmpeg_exe
        if not ffmpeg:
            return None

        # ffprobe lives next to ffmpeg
        ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
        if not os.path.exists(ffprobe):
            # Try system ffprobe
            ffprobe = "ffprobe"

        try:
            result = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            duration = float(result.stdout.strip())
            logger.debug(f"Duration: {seconds_to_timestamp(duration)}")
            return duration
        except Exception as e:
            logger.warning(f"Could not get duration: {e}")
            return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_ffmpeg(self) -> Optional[str]:
        """
        Locate FFmpeg binary — prefer imageio-ffmpeg bundle, then system PATH.
        """
        # 1. Try imageio-ffmpeg bundled binary
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                logger.info(f"Using imageio-ffmpeg binary: {exe}")
                return exe
        except Exception:
            pass

        # 2. Try system ffmpeg
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            logger.info(f"Using system FFmpeg: {system_ffmpeg}")
            return system_ffmpeg

        logger.warning(
            "FFmpeg not found via imageio-ffmpeg or system PATH. "
            "Install imageio-ffmpeg (already in requirements.txt) and restart."
        )
        return None

    def _convert_audio(self, input_path: str, output_path: str) -> None:
        """
        Convert any media file to 16kHz mono WAV using FFmpeg.
        Falls back to MoviePy, then raw WAV copy.
        """
        ffmpeg = self._ffmpeg_exe

        # ── Method 1: FFmpeg (preferred) ─────────────────────────────────────
        if ffmpeg:
            try:
                cmd = [
                    ffmpeg,
                    "-i", input_path,
                    "-vn",                   # strip video
                    "-acodec", "pcm_s16le",  # PCM 16-bit WAV
                    "-ar", "16000",          # 16 kHz — Whisper-friendly
                    "-ac", "1",              # mono
                    "-y",                    # overwrite
                    output_path,
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode == 0:
                    logger.debug("FFmpeg conversion complete.")
                    return
                else:
                    logger.warning(f"FFmpeg error: {result.stderr[:200]}")
            except Exception as e:
                logger.warning(f"FFmpeg failed: {e}")

        # ── Method 2: MoviePy fallback ────────────────────────────────────────
        try:
            logger.info("Trying MoviePy fallback…")
            from moviepy.editor import AudioFileClip, VideoFileClip

            ext = Path(input_path).suffix.lower()
            if ext in AUDIO_EXTENSIONS:
                clip = AudioFileClip(input_path)
            else:
                clip = VideoFileClip(input_path).audio

            clip.write_audiofile(
                output_path,
                fps=16000,
                nbytes=2,
                codec="pcm_s16le",
                logger=None,
            )
            clip.close()
            logger.debug("MoviePy conversion complete.")
            return
        except Exception as e:
            logger.warning(f"MoviePy failed: {e}")

        # ── Method 3: Raw WAV copy (only works if already 16kHz mono WAV) ────
        if Path(input_path).suffix.lower() == ".wav":
            logger.info("Copying WAV file directly (may not be 16kHz mono)…")
            shutil.copy2(input_path, output_path)
            return

        raise RuntimeError(
            "Processing failed: Could not convert audio. "
            "Please ensure imageio-ffmpeg is installed (`pip install imageio-ffmpeg`)."
        )
