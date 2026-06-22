FROM python:3.10-slim

# Create a user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code with user ownership
COPY --chown=user:user . .

# Create necessary data directories and ensure correct permissions
RUN mkdir -p data/videos data/audio data/transcripts \
    data/summaries data/embeddings \
    outputs/final_notes outputs/timestamps \
    outputs/action_items outputs/reports \
    frontend/static/uploads \
    models/whisper models/summarization_model models/embedding_model logs \
    models/huggingface models/torch \
    && chown -R user:user /app

# Switch to the non-root user
USER user

# Set environment variables for cache directories
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/app/models/huggingface \
    TORCH_HOME=/app/models/torch

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "main.py"]
