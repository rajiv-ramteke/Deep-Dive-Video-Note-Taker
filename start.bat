@echo off
REM ============================================================
REM  Deep-Dive Video Note Taker — Windows Quick-Start Script
REM ============================================================

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║    Deep-Dive Video Note Taker  v1.0.0           ║
echo  ║    AI-Powered Video Analysis                     ║
echo  ╚══════════════════════════════════════════════════╝
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Check FFmpeg
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] FFmpeg not found on PATH. Audio extraction may fail.
    echo        Download from: https://ffmpeg.org/download.html
    echo.
)

REM Create virtual environment if missing
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install requirements
echo [SETUP] Installing dependencies...
python -m pip install --upgrade pip "setuptools<70.0.0" wheel
pip install -r requirements.txt

REM Copy .env if missing
if not exist ".env" (
    echo [SETUP] Creating .env from .env.example...
    copy .env.example .env
    echo [INFO]  Edit .env to add your OpenAI API key
)

REM Create directories
if not exist "logs" mkdir logs
if not exist "data\videos"    mkdir data\videos
if not exist "data\audio"     mkdir data\audio
if not exist "data\transcripts" mkdir data\transcripts
if not exist "data\summaries" mkdir data\summaries
if not exist "data\embeddings" mkdir data\embeddings
if not exist "outputs\final_notes"  mkdir outputs\final_notes
if not exist "outputs\timestamps"   mkdir outputs\timestamps
if not exist "outputs\action_items" mkdir outputs\action_items
if not exist "outputs\reports"      mkdir outputs\reports
if not exist "models\whisper"       mkdir models\whisper
if not exist "models\embedding_model" mkdir models\embedding_model

echo.
echo [START] Launching server at http://localhost:7860
echo         Press Ctrl+C to stop.
echo.

python main.py

pause
