"""
tests/test_transcription.py
=============================
Unit tests for the Whisper transcription service.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from backend.services.whisper_transcriber import WhisperTranscriber
from backend.utils.helper import seconds_to_timestamp


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_WHISPER_OUTPUT = {
    "text": "Hello world. This is a test transcript.",
    "language": "en",
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 2.5,
            "text": "Hello world.",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "world",  "start": 0.6, "end": 1.0},
            ],
        },
        {
            "id": 1,
            "start": 2.6,
            "end": 5.0,
            "text": "This is a test transcript.",
            "words": [],
        },
    ],
}


@pytest.fixture
def transcriber(tmp_path):
    """Return a WhisperTranscriber with output dir set to tmp_path."""
    t = WhisperTranscriber()
    t.output_dir = str(tmp_path)
    return t


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_parse_result_structure(transcriber):
    """_parse_result should return expected keys and types."""
    result = transcriber._parse_result(SAMPLE_WHISPER_OUTPUT)
    assert "text" in result
    assert "segments" in result
    assert "language" in result
    assert "duration" in result
    assert isinstance(result["segments"], list)
    assert result["language"] == "en"


def test_parse_result_timestamps(transcriber):
    """Segments should have HH:MM:SS timestamp strings."""
    result = transcriber._parse_result(SAMPLE_WHISPER_OUTPUT)
    seg = result["segments"][0]
    assert seg["start_ts"] == seconds_to_timestamp(0.0)
    assert seg["end_ts"]   == seconds_to_timestamp(2.5)


def test_parse_result_full_text(transcriber):
    """Full text should concatenate segment texts."""
    result = transcriber._parse_result(SAMPLE_WHISPER_OUTPUT)
    assert "Hello world" in result["text"]
    assert "test transcript" in result["text"]


def test_parse_result_empty_segments(transcriber):
    """Empty segment list should return empty text and 0 duration."""
    empty = {"text": "", "language": "en", "segments": []}
    result = transcriber._parse_result(empty)
    assert result["text"] == ""
    assert result["duration"] == 0
    assert result["segments"] == []


@patch("backend.services.whisper_transcriber.WhisperTranscriber._get_model")
def test_transcribe_saves_json(mock_model, transcriber, tmp_path):
    """transcribe() should save a JSON file in the output directory."""
    # Mock the model
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = SAMPLE_WHISPER_OUTPUT
    mock_model.return_value = mock_whisper

    # Create a dummy audio file
    audio_path = str(tmp_path / "test_audio.wav")
    with open(audio_path, "wb") as f:
        f.write(b"\x00" * 100)

    result = transcriber.transcribe(audio_path, "test_job_001")

    json_path = str(tmp_path / "test_job_001.json")
    assert os.path.exists(json_path), "Transcript JSON was not saved"
    with open(json_path) as f:
        data = json.load(f)
    assert "text" in data
    assert "segments" in data


def test_load_transcript_returns_none_if_missing(transcriber):
    """load_transcript() should return None for unknown job_id."""
    result = transcriber.load_transcript("nonexistent_job_999")
    assert result is None


def test_helper_seconds_to_timestamp():
    """seconds_to_timestamp helper should format correctly."""
    assert seconds_to_timestamp(0)     == "00:00:00"
    assert seconds_to_timestamp(61)    == "00:01:01"
    assert seconds_to_timestamp(3661)  == "01:01:01"
    assert seconds_to_timestamp(86399) == "23:59:59"
