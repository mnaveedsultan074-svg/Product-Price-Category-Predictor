# ============================================================
# ML Micro-service Dockerfile
# Base image: Python 3.11 slim
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# System dependencies
#   curl   – used by Docker healthcheck
#   git    – needed by some pip packages that build from source
#   libgomp1 – required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache the pip layer
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all ML source files
COPY . .

# Ensure runtime directories exist
RUN mkdir -p data/raw data/processed models reports

# Non-root user for security
RUN useradd --create-home --shell /bin/bash mluser && \
    chown -R mluser:mluser /app
USER mluser

# Flask listens on this port
EXPOSE 5001

# Default: serve predictions (the full pipeline is run via docker-compose command)
CMD ["python", "predict_service.py"]
