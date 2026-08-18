FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root dedicated application user
RUN groupadd -g 10001 chexpert && \
    useradd -u 10001 -g chexpert -d /app -s /sbin/nologin -M chexpert

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ ./app/
COPY static/ ./static/
COPY outputs/ ./outputs/
COPY scripts/ ./scripts/

# Setup directories with strict non-root ownership
RUN mkdir -p /app/checkpoints /app/outputs/evaluation /app/.torch /app/tmp && \
    chown -R chexpert:chexpert /app

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    TORCH_HOME=/app/.torch \
    CHEXPERT_CHECKPOINT=/app/checkpoints/chexpert_convnext_small.pt \
    TMPDIR=/app/tmp

USER chexpert

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health | grep '"model_loaded":true' || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
