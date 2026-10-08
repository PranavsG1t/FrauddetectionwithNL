# syntax=docker/dockerfile:1
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MODEL_DIR=outputs/models

# LightGBM needs the OpenMP runtime (libgomp), which slim images leave out.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: dependencies. Cached until requirements-api.txt changes, so code
# edits don't trigger a full reinstall.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Layer 2: code + ONLY the artifacts the API loads (not the whole outputs/ dir).
COPY src ./src
COPY outputs/models/lightgbm.joblib outputs/models/shap_explainer.joblib ./outputs/models/
COPY outputs/rag_index ./outputs/rag_index

# Don't run as root inside the container.
RUN useradd --create-home --uid 1000 appuser
USER appuser

# Cloud Run injects $PORT (8080). Locally we map it: -p 8000:8080.
# `exec` makes uvicorn PID 1 so it receives shutdown signals properly.
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
