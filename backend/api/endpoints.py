"""
backend/api/endpoints.py
=========================
All FastAPI endpoint handlers.
Handles video upload, pipeline execution, RAG query, job status, and note retrieval.
"""

import os
import json
import asyncio
from typing import Dict, Optional
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.utils.config import settings
from backend.utils.helper import (
    generate_job_id, ensure_dir, safe_filename,
    get_file_size_mb, build_success_response, build_error_response,
    load_json,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ── In-memory job store (replace with DB in production) ──────────────────────
_jobs: Dict[str, Dict] = {}

# ── Sub-routers ───────────────────────────────────────────────────────────────
process_router = APIRouter()
query_router   = APIRouter()
status_router  = APIRouter()
notes_router   = APIRouter()


# =============================================================================
# PROCESS — Upload + run full pipeline
# =============================================================================

@process_router.post("/upload")
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    openai_key: Optional[str] = Form(None),
    notes_language: Optional[str] = Form("English"),
):
    """
    Upload a video file and start the AI processing pipeline.
    Returns a job_id to poll for status.
    """
    # Validate file type
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".m4a"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    # Save uploaded file chunk by chunk to avoid out-of-memory errors
    job_id   = generate_job_id()
    filename = safe_filename(Path(file.filename).stem) + ext
    video_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}{ext}")
    ensure_dir(settings.UPLOAD_DIR)

    size_bytes = 0
    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
            size_bytes += len(chunk)
            if size_bytes > settings.max_file_size_bytes:
                os.remove(video_path)
                raise HTTPException(413, f"File too large: max {settings.MAX_FILE_SIZE_MB} MB")
            f.write(chunk)

    size_mb = size_bytes / (1024 * 1024)
    logger.info(f"Job {job_id}: uploaded '{filename}' ({size_mb:.1f} MB)")

    # Store API key in job info instead of global os.environ
    # to prevent race conditions between concurrent requests.

    # Register job
    _jobs[job_id] = {
        "job_id":         job_id,
        "filename":       filename,
        "status":         "queued",
        "progress":       0,
        "message":        "Queued for processing",
        "video_path":     video_path,
        "notes_language": notes_language or "English",
        "openai_key":     openai_key,
        "result":         None,
        "error":          None,
    }

    # Run pipeline in background
    background_tasks.add_task(_run_pipeline, job_id, video_path, filename)

    return JSONResponse(
        build_success_response(
            {"job_id": job_id, "filename": filename, "size_mb": round(size_mb, 2)},
            message="Upload successful. Processing started.",
        )
    )


def _run_pipeline(job_id: str, video_path: str, filename: str):
    """Background task: runs the full AI pipeline for a given job."""
    from backend.services.audio_extractor      import AudioExtractor
    from backend.services.whisper_transcriber  import WhisperTranscriber
    from backend.services.text_chunker         import TextChunker
    from backend.services.summarizer           import Summarizer
    from backend.services.rag_pipeline         import RAGPipeline
    from backend.services.quiz_generator       import QuizGenerator
    from backend.services.topic_extractor        import TopicExtractor
    from backend.services.note_generator       import NoteGenerator

    def update(progress: int, message: str):
        _jobs[job_id]["progress"] = progress
        _jobs[job_id]["message"]  = message
        _jobs[job_id]["status"]   = "processing"
        logger.info(f"[{job_id}] {progress}% — {message}")

    try:
        update(5,  "Extracting audio from video...")
        extractor  = AudioExtractor()
        audio_path = extractor.extract(video_path, job_id)
        duration   = extractor.get_video_duration(video_path)

        api_key = _jobs[job_id].get("openai_key")

        update(20, "Transcribing audio with Whisper ASR...")
        transcriber = WhisperTranscriber(api_key=api_key)
        transcript  = transcriber.transcribe(audio_path, job_id)

        update(40, "Chunking transcript...")
        chunker = TextChunker()
        chunks  = chunker.chunk_transcript(transcript)

        update(50, "Building RAG vector index...")
        rag = RAGPipeline()
        rag.index_chunks(chunks)
        rag.save_index()

        update(60, "Summarizing chunks with LLM...")
        summarizer = Summarizer(api_key=api_key)

        # Language: prefer the user-selected language; fall back to auto-detected transcript language
        user_language  = _jobs[job_id].get("notes_language", "English")
        lang_code      = transcript.get("language", "en")
        lang_fallback_map = {
            "en": "English", "hi": "Hindi", "mr": "Marathi",
            "es": "Spanish", "fr": "French", "de": "German",
            "zh": "Chinese", "ja": "Japanese",
        }
        language = user_language if user_language else lang_fallback_map.get(lang_code, "English")
        logger.info(f"[{job_id}] Notes language: {language}")

        summarized_chunks = summarizer.summarize_chunks(chunks, language=language)

        update(75, "Generating final structured notes...")
        final_notes = summarizer.generate_final_notes(summarized_chunks, language=language)

        update(82, "Generating interactive quiz...")
        q_gen = QuizGenerator(api_key=api_key)
        quiz_data = q_gen.generate_quiz(chunks, language=language)

        update(90, "Extracting topic summaries...")
        topic_ext = TopicExtractor(api_key=api_key)
        topics    = topic_ext.extract(chunks, language=language)

        update(91, "Extracting action items...")
        from backend.services.action_item_extractor import ActionItemExtractor
        action_ext = ActionItemExtractor(api_key=api_key)
        action_items = action_ext.extract(chunks, language=language)

        update(92, "Mapping timestamps and chapters...")
        from backend.services.timestamp_mapper import TimestampMapper
        ts_mapper = TimestampMapper()
        highlights = ts_mapper.map_timestamps(summarized_chunks)
        chapters = ts_mapper.generate_chapter_markers(highlights)

        update(93, "Generating Q&A pairs...")
        from backend.services.qa_generator import QAGenerator
        qa_gen = QAGenerator(api_key=api_key)
        
        qa_pairs = qa_gen.generate_qa(chunks, language=language)

        update(95, "Assembling final notes document...")
        generator = NoteGenerator()
        result    = generator.generate(
            job_id=job_id,
            filename=filename,
            transcript=transcript,
            summarized_chunks=summarized_chunks,
            final_notes=final_notes,
            quiz=quiz_data,
            topics=topics,
            qa_pairs=qa_pairs,
            action_items=action_items,
            highlights=highlights,
            chapters=chapters,

            duration=duration,
        )

        _jobs[job_id].update({
            "status":   "complete",
            "progress": 100,
            "message":  "Processing complete!",
            "result":   result["data"],
            "markdown": result["markdown"],
        })
        logger.info(f"[{job_id}] ✅ Pipeline complete")

    except Exception as e:
        logger.error(f"[{job_id}] Pipeline error: {e}", exc_info=True)
        _jobs[job_id].update({
            "status":  "error",
            "message": f"Processing failed: {str(e)}",
            "error":   str(e),
        })


# =============================================================================
# STATUS — Poll job progress
# =============================================================================

@status_router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Get the current status and progress of a processing job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")

    return JSONResponse(
        build_success_response({
            "job_id":   job_id,
            "status":   job["status"],
            "progress": job["progress"],
            "message":  job["message"],
            "filename": job.get("filename", ""),
        })
    )


@status_router.get("/")
async def list_jobs():
    """List all processing jobs."""
    jobs_summary = [
        {
            "job_id":   jid,
            "filename": j.get("filename"),
            "status":   j["status"],
            "progress": j["progress"],
        }
        for jid, j in _jobs.items()
    ]
    return JSONResponse(build_success_response({"jobs": jobs_summary}))


# =============================================================================
# NOTES — Retrieve results
# =============================================================================

@notes_router.get("/{job_id}")
async def get_notes(job_id: str):
    """Retrieve the structured notes for a completed job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job["status"] != "complete":
        raise HTTPException(409, f"Job not complete yet. Status: {job['status']}")

    return JSONResponse(
        build_success_response({
            "job_id":      job_id,
            "filename":    job.get("filename"),
            "notes":       job["result"],
            "markdown":    job.get("markdown", ""),
        })
    )


@notes_router.get("/{job_id}/download")
async def download_notes(job_id: str):
    """Download notes as a Markdown file."""
    md_path = os.path.join(settings.OUTPUT_DIR, "final_notes", f"{job_id}_notes.md")
    if not os.path.exists(md_path):
        raise HTTPException(404, "Notes file not found. Process the video first.")
    return FileResponse(
        md_path,
        media_type="text/markdown",
        filename=f"notes_{job_id}.md",
    )


@notes_router.get("/{job_id}/download/json")
async def download_notes_json(job_id: str):
    """Download notes as a JSON file."""
    json_path = os.path.join(settings.OUTPUT_DIR, "final_notes", f"{job_id}_notes.json")
    if not os.path.exists(json_path):
        raise HTTPException(404, "JSON notes file not found.")
    return FileResponse(
        json_path,
        media_type="application/json",
        filename=f"notes_{job_id}.json",
    )

class TranslateRequest(BaseModel):
    job_id: str
    language: str


@notes_router.post("/translate")
async def translate_notes_endpoint(req: TranslateRequest):
    """Translate the final notes to a specified language."""
    from backend.services.summarizer import Summarizer
    
    job = _jobs.get(req.job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {req.job_id}")
    if job["status"] != "complete":
        raise HTTPException(409, "Job not complete yet.")

    markdown_notes = job.get("markdown", "")
    if not markdown_notes:
        raise HTTPException(404, "No notes found to translate.")

    try:
        api_key = job.get("openai_key")
        summarizer = Summarizer(api_key=api_key)
        translated_markdown = summarizer.translate_notes(markdown_notes, req.language)
        
        # New: Translate all structured data so the whole UI flips
        from backend.services.translator import Translator
        translator = Translator(api_key=api_key)
        
        notes_data = job.get("result", {})
        
        translated_quiz = translator.translate_json_array(notes_data.get("quiz", []), req.language)
        translated_topics = translator.translate_json_array(notes_data.get("topics", []), req.language)
        translated_qa = translator.translate_json_array(notes_data.get("qa_pairs", []), req.language)
        translated_transcript = translator.translate_transcript(notes_data.get("transcript_segments", []), req.language)
        
        # Update the job state
        job["markdown"] = translated_markdown
        job["result"]["quiz"] = translated_quiz
        job["result"]["topics"] = translated_topics
        job["result"]["qa_pairs"] = translated_qa
        job["result"]["transcript_segments"] = translated_transcript

        return JSONResponse(
            build_success_response({
                "job_id": req.job_id,
                "language": req.language,
                "translated_markdown": translated_markdown,
                "translated_quiz": translated_quiz,
                "translated_topics": translated_topics,
                "translated_qa_pairs": translated_qa,
                "translated_transcript": translated_transcript
            })
        )
    except Exception as e:
        logger.error(f"Translation error for job {req.job_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Translation failed: {str(e)}")

# =============================================================================
# QUERY — RAG semantic search
# =============================================================================

class QueryRequest(BaseModel):
    job_id: str
    query:  str
    top_k:  int = 5


@query_router.post("/")
async def query_transcript(req: QueryRequest):
    """
    Perform a semantic search over an indexed video transcript.
    Returns the most relevant segments for the given query.
    """
    from backend.services.rag_pipeline import RAGPipeline
    from backend.services.summarizer import Summarizer

    job = _jobs.get(req.job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {req.job_id}")
    if job["status"] != "complete":
        raise HTTPException(409, "Job not complete yet.")

    rag = RAGPipeline()
    loaded = rag.load_index()
    if not loaded:
        raise HTTPException(503, "RAG index not available for this job.")

    results = rag.query(req.query, top_k=req.top_k)
    
    answer = ""
    if results:
        # Build context from chunks
        context_parts = [f"[{r.get('start_ts', '')} → {r.get('end_ts', '')}]: {r.get('text', '')}" for r in results]
        context = "\n\n".join(context_parts)
        try:
            api_key = job.get("openai_key")
            summarizer = Summarizer(api_key=api_key)
            answer = summarizer.answer_question(req.query, context)
        except Exception as e:
            logger.error(f"Error generating answer for query '{req.query}': {e}")
            answer = "Sorry, I encountered an error while trying to answer your question."
    else:
        answer = "No relevant sections found in the video to answer your question."

    return JSONResponse(
        build_success_response({
            "query":   req.query,
            "answer":  answer,
            "results": results,
        })
    )
