FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ ./app/
COPY static/ ./static/
COPY outputs/ ./outputs/
COPY scripts/ ./scripts/

# Create checkpoints directory
RUN mkdir -p checkpoints outputs/evaluation

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.torch \
    CHEXPERT_CHECKPOINT=/app/checkpoints/chexpert_convnext_small.pt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
