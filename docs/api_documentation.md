# Deep-Dive Video Note Taker — API Documentation

## Base URL
```
http://localhost:8000/api/v1
```

---

## Endpoints

### 1. Upload & Process Video

**POST** `/process/upload`

Upload a video file and start the AI processing pipeline.

**Form Data:**
| Field       | Type   | Required | Description                  |
|-------------|--------|----------|------------------------------|
| `file`      | File   | ✅       | Video/audio file (≤500 MB)   |
| `openai_key`| String | ❌       | OpenAI API key (overrides .env) |

**Response:**
```json
{
  "status": "success",
  "message": "Upload successful. Processing started.",
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "my_lecture.mp4",
    "size_mb": 45.3
  }
}
```

---

### 2. Job Status

**GET** `/status/{job_id}`

Poll the processing status of a job.

**Response:**
```json
{
  "status": "success",
  "data": {
    "job_id": "550e8400...",
    "status": "processing",
    "progress": 60,
    "message": "Summarizing chunks with LLM...",
    "filename": "my_lecture.mp4"
  }
}
```

**Status values:** `queued` → `processing` → `complete` / `error`

---

### 3. List All Jobs

**GET** `/status/`

List all processing jobs in the current session.

---

### 4. Get Notes

**GET** `/notes/{job_id}`

Retrieve the structured notes for a completed job.

**Response:**
```json
{
  "status": "success",
  "data": {
    "job_id": "550e8400...",
    "filename": "my_lecture.mp4",
    "notes": {
      "final_notes": "## Summary\n...",
      "highlights": [...],
      "action_items": [...],
      "chapters": [...],
      "transcript_segments": [...]
    },
    "markdown": "# Video Notes: my_lecture.mp4\n..."
  }
}
```

---

### 5. Download Notes (Markdown)

**GET** `/notes/{job_id}/download`

Download notes as a `.md` file.

---

### 6. Download Notes (JSON)

**GET** `/notes/{job_id}/download/json`

Download full structured notes as a `.json` file.

---

### 7. RAG Semantic Search

**POST** `/query/`

Search the video content using natural language.

**Request Body:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "What did they say about neural networks?",
  "top_k": 5
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "query": "What did they say about neural networks?",
    "results": [
      {
        "chunk_id": 3,
        "text": "Neural networks consist of layers of interconnected nodes...",
        "start_ts": "00:12:30",
        "end_ts": "00:14:00",
        "score": 0.87
      }
    ]
  }
}
```

---

### 8. Health Check

**GET** `/health`

```json
{
  "status": "healthy",
  "app": "Deep-Dive Video Note Taker",
  "version": "1.0.0"
}
```

---

## Interactive API Docs

When the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
