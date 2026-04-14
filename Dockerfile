# ─────────────────────────────────────────────────────────────────────────────
# Maison Elara — Production Dockerfile (Optimized for Azure Container Apps)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Python slim base ─────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps needed for FAISS-CPU, Pillow (Vision), and SQLite (Checkpointer)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Environment & Model Caching ─────────────────────────────────────
# Set explicit cache paths so they are baked into the image layers.
ENV HF_HOME=/app/model_cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# ── Stage 3: Install Python dependencies ─────────────────────────────────────
# We install the 2026 CPU-only PyTorch stack first to ensure no CUDA bloat.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the application requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 4: Pre-download HuggingFace models ─────────────────────────────────
# This ensures zero download time when the Azure container starts (Cold Start).
RUN python -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
print('CLIP downloaded')"

RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('bge-small-en-v1.5 downloaded')"

RUN python -c "\
from sentence_transformers import CrossEncoder; \
CrossEncoder('BAAI/bge-reranker-base'); \
print('bge-reranker-base downloaded')"

# ── Stage 5: Copy application code ───────────────────────────────────────────

# NOW copy everything else (including your .db and .faiss files)
COPY . .

# ── Runtime Configuration ────────────────────────────────────────────────────
# Note: Use Azure File Share for /app/data if you want persistent storage!
ENV DATABASE_URL=sqlite:///./data/shop.db \
    IMAGE_BASE_PATH=./website/images \
    FAISS_PRODUCT_INDEX=./product_index.faiss \
    IMAGE_IDS_PKL=./image_ids.pkl \
    RAG_DOCS_PATH=./company-data \
    RAG_FAISS_PATH=./data/faiss_vector_store_index \
    RAG_BM25_PATH=./data/bm25.pkl

# Expose the port used in your server.py
EXPOSE 8080

# ── Health check (Crucial for Azure to know your app is ready) ───────────────
# Given the heavy ML models, we use a longer start-period of 120s
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

RUN chmod +x entrypoint.sh
ENTRYPOINT ["sh", "entrypoint.sh"]