---
title: Deep Dive Video Note Taker
emoji: 🎥
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Deep-Dive Video Note Taker

> **AI-Powered Video Summarization & Note Generation System**

## 🎯 Features

- 🎙️ **ASR Transcription** — Speech-to-text via OpenAI Whisper
- 📝 **Structured Notes** — Topic-organized summaries
- ⏱️ **Timestamped Highlights** — Key moments linked to video timeline
- ✅ **Action Items** — Auto-detected tasks and follow-ups
- 🔍 **RAG Search** — Query your video in natural language
- 🌐 **Web Dashboard** — Beautiful interactive UI

## 🔧 Configuration

Set `OPENAI_API_KEY` in your Space environment variables for faster processing, or leave it blank to use the local Whisper model (requires more RAM).

## 📄 License

MIT License
