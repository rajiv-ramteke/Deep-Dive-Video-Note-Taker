"""
backend/services/note_generator.py
=====================================
Assembles all pipeline outputs into a final, structured note document.
Produces Markdown and JSON formats.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

from backend.utils.config import settings
from backend.utils.helper import ensure_dir, save_json, save_text, format_duration
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class NoteGenerator:
    """
    Final stage of the pipeline.
    Combines transcript, summaries, timestamps, and action items
    into a polished, structured note document.
    """

    def __init__(self):
        self.output_dir = settings.OUTPUT_DIR

    # ── Public API ────────────────────────────────────────────

    def generate(
        self,
        job_id: str,
        filename: str,
        transcript: Dict,
        summarized_chunks: List[Dict],
        final_notes: str,
        quiz: List[Dict],
        topics: List[Dict] = None,
        qa_pairs: List[Dict] = None,
        action_items: List[Dict] = None,
        highlights: List[Dict] = None,
        chapters: List[Dict] = None,
        duration: Optional[float] = None,
    ) -> Dict:
        """
        Assemble all pipeline outputs into the final note document.

        Returns:
            Dict containing:
                - markdown_path: path to .md file
                - json_path:     path to .json file
                - markdown:      Markdown string
                - data:          Full structured JSON data
        """
        logger.info(f"Generating final notes for job: {job_id}")

        qa_pairs = qa_pairs or []

        markdown = self._build_markdown(
            filename=filename,
            transcript=transcript,
            final_notes=final_notes,
            quiz=quiz,
            topics=topics or [],
            qa_pairs=qa_pairs,
            action_items=action_items or [],
            chapters=chapters or [],
            duration=duration,
        )

        data = self._build_json(
            job_id=job_id,
            filename=filename,
            transcript=transcript,
            summarized_chunks=summarized_chunks,
            final_notes=final_notes,
            quiz=quiz,
            topics=topics or [],
            qa_pairs=qa_pairs,
            action_items=action_items or [],
            highlights=highlights or [],
            chapters=chapters or [],
            duration=duration,
        )

        # Save outputs
        md_dir   = os.path.join(self.output_dir, "final_notes")
        json_dir = os.path.join(self.output_dir, "final_notes")
        ensure_dir(md_dir)

        md_path   = os.path.join(md_dir, f"{job_id}_notes.md")
        json_path = os.path.join(json_dir, f"{job_id}_notes.json")
        qz_path   = os.path.join(self.output_dir, f"{job_id}_quiz.json")
        ai_path   = os.path.join(self.output_dir, f"{job_id}_topics.json")
        qa_path   = os.path.join(self.output_dir, f"{job_id}_qa.json")

        ensure_dir(os.path.join(self.output_dir, "qa_pairs"))

        save_text(markdown, md_path)
        save_json(data, json_path)
        save_json(quiz, qz_path)
        save_json(topics or [], ai_path)
        save_json(qa_pairs, qa_path)

        logger.info(f"Notes saved: {md_path}")

        return {
            "markdown_path": md_path,
            "json_path":     json_path,
            "markdown":      markdown,
            "data":          data,
        }

    # ── Private: Markdown builder ─────────────────────────────

    def _build_markdown(
        self,
        filename: str,
        transcript: Dict,
        final_notes: str,
        quiz: List[Dict],
        topics: List[Dict],
        qa_pairs: List[Dict],
        action_items: List[Dict],
        chapters: List[Dict],
        duration: Optional[float],
    ) -> str:
        from datetime import timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        dur_str = format_duration(duration) if duration else "Unknown"
        lang = transcript.get("language", "en").upper()
        word_count = len(transcript.get("text", "").split())

        lines = [
            f"# 📝 Video Notes: {filename}",
            "",
            f"> **Generated:** {now}  |  **Duration:** {dur_str}  "
            f"|  **Language:** {lang}  |  **Words:** {word_count:,}",
            "",
            "---",
            "",
            "## 📋 Structured Summary",
            "",
            final_notes,
            "",
            "---",
            "",
        ]

        # Interactive Quiz
        lines += ["## 🎯 Interactive Quiz", ""]
        if quiz:
            for idx, q in enumerate(quiz):
                lines.append(f"### Q{idx+1}: {q.get('question')}")
                for opt_idx, opt in enumerate(q.get('options', [])):
                    lines.append(f"- {opt}")
                lines.append("")
                # Put the answer in a details block so it's hidden by default in MD
                correct_opt = q.get('options', [])[q.get('correct_index', 0)] if q.get('options') else 'Unknown'
                lines.append(f"<details><summary>Reveal Answer</summary>")
                lines.append(f"**Correct Answer:** {correct_opt}")
                lines.append("</details>")
                lines.append("")
        else:
            lines += ["_No quiz generated._", ""]

        # Topic Summaries
        lines += ["## 📚 Topic Summaries", ""]
        if topics:
            for item in topics:
                lines.append(f"### 🔹 {item.get('topic', 'Topic')}")
                lines.append("")
                lines.append(f"> {item.get('summary', '')}")
                lines.append("")
                for kp in item.get('key_points', []):
                    lines.append(f"- {kp}")
                lines.append("")
        else:
            lines += ["_No topics extracted._", ""]

        lines += ["---", ""]

        # Q&A section
        if qa_pairs:
            lines += ["## ❓ Generated Q&A", ""]
            for qa in qa_pairs:
                lines.append(f"**Q: {qa.get('question', '')}**")
                lines.append(f"A: {qa.get('answer', '')}")
                lines.append("")
            lines += ["---", ""]

        # Action Items
        if action_items:
            lines += ["## ✅ Action Items", ""]
            for action in action_items:
                lines.append(f"- [ ] **{action.get('task', 'Task')}**")
                meta = []
                if action.get('assignee'): meta.append(f"Assignee: {action['assignee']}")
                if action.get('deadline'): meta.append(f"Due: {action['deadline']}")
                if meta:
                    lines.append(f"  > _{' | '.join(meta)}_")
            lines += ["", "---", ""]

        # Chapters
        if chapters:
            lines += ["## 🕒 Chapters", ""]
            for chapter in chapters:
                lines.append(f"### {chapter.get('timestamp', '')} - {chapter.get('title', 'Chapter')}")
                for hl in chapter.get('highlights', []):
                    lines.append(f"- **{hl.get('timestamp', '')}**: {hl.get('title', '')}")
                lines.append("")
            lines += ["---", ""]

        # Full Transcript (collapsible)
        lines += [
            "## 📜 Full Transcript",
            "",
            "<details>",
            "<summary>Click to expand full transcript</summary>",
            "",
            "```",
            transcript.get("text", ""),
            "```",
            "",
            "</details>",
            "",
            "---",
            "_Generated by Deep-Dive Video Note Taker — AI-Powered Video Analysis_",
        ]

        return "\n".join(lines)

    # ── Private: JSON builder ─────────────────────────────────

    def _build_json(self, **kwargs) -> Dict:
        return {
            "job_id":           kwargs["job_id"],
            "filename":         kwargs["filename"],
            "generated_at":     datetime.utcnow().isoformat(),
            "duration_seconds": kwargs.get("duration"),
            "language":         kwargs["transcript"].get("language", "en"),
            "word_count":       len(kwargs["transcript"].get("text", "").split()),
            "final_notes":      kwargs["final_notes"],
            "quiz":             kwargs.get("quiz", []),
            "topics":           kwargs.get("topics", []),
            "qa_pairs":         kwargs.get("qa_pairs", []),
            "action_items":     kwargs.get("action_items", []),
            "highlights":       kwargs.get("highlights", []),
            "chapters":         kwargs.get("chapters", []),
            "chunk_summaries": [
                {
                    "chunk_id": c["chunk_id"],
                    "start_ts": c["start_ts"],
                    "end_ts":   c["end_ts"],
                    "summary":  c.get("summary", ""),
                }
                for c in kwargs["summarized_chunks"]
            ],
            "transcript_segments": kwargs["transcript"].get("segments", []),
        }
