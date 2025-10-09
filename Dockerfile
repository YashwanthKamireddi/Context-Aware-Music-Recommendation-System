# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system-level dependencies required for LightGBM and scientific Python
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies separately to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code and assets
COPY config ./config
COPY src ./src
COPY backend ./backend
COPY frontend ./frontend
COPY models ./models
COPY data ./data
COPY reports ./reports
COPY README.md ./README.md

# Create a non-root user for security (optional but recommended)
RUN useradd -ms /bin/bash vibesync
USER vibesync


ENV PORT=7860
EXPOSE 7860

# Start FastAPI via Uvicorn, honoring the PORT provided by the platform (defaults to 7860 for Spaces)
CMD ["sh", "-c", "uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
