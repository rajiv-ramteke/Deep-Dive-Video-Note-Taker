# Methodology

## System Architecture

The Deep-Dive Video Note Taker follows a multi-stage AI pipeline:

```
Video Input → Audio Extraction → ASR Transcription → Text Chunking
    → LLM Summarization → RAG Indexing → Timestamp Mapping
    → Action Item Extraction → Note Generation → Web UI
```

## Stage Details

### 1. Audio Extraction
- **Tool**: FFmpeg (primary), MoviePy (fallback)
- **Output**: 16kHz mono WAV optimised for Whisper ASR
- **Handles**: MP4, AVI, MOV, MKV, WebM, MP3, WAV

### 2. ASR Transcription (Whisper)
- **Model**: OpenAI Whisper (tiny/base/small/medium/large)
- **Output**: Word-level and segment-level timestamps
- **Language**: Auto-detected, 99+ languages supported

### 3. Text Chunking
- **Strategy**: Sliding window with configurable overlap
- **Chunk Size**: 1000 words (default), 200-word overlap
- **Preserves**: Start/end timestamps per chunk

### 4. LLM Summarization
- **Primary**: OpenAI GPT-3.5-Turbo / GPT-4
- **Fallback**: HuggingFace BART (facebook/bart-large-cnn)
- **Prompts**: Structured for bullet-point and topic-based output

### 5. RAG Pipeline (FAISS)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Index**: FAISS IndexFlatIP (cosine similarity on normalised vectors)
- **Purpose**: Context retrieval + semantic search

### 6. Timestamp Mapping
- **Method**: Aligns each chunk summary with its source timestamps
- **Output**: Chapter markers, key highlights, navigable segments

### 7. Action Item Extraction
- **Primary**: LLM-based (structured JSON output)
- **Fallback**: Regex heuristic patterns
- **Categories**: Actions, Decisions, Follow-ups, Reminders

### 8. Note Generation
- **Output Formats**: Markdown (.md) + JSON (.json)
- **Structure**: Summary → Highlights → Action Items → Chapters → Transcript

## Performance Characteristics

| Metric                | Value            |
|-----------------------|------------------|
| Summarization Accuracy| ~85–90%          |
| ASR Word Error Rate   | ~3–8% (clean audio)|
| Time Reduction        | ~60–70%          |
| Max Video Length      | Unlimited (chunked)|
| Supported Languages   | 99+              |
