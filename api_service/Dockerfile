# Use a lightweight Python image
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-privileged user
RUN groupadd -g 1001 appuser && \
    useradd -u 1001 -g appuser -s /bin/bash -m appuser

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to production image
FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-privileged user
RUN groupadd -g 1001 appuser && \
    useradd -u 1001 -g appuser -s /bin/bash -m appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Set default environment variables
ENV API_VERSION=1.0.0 \
    PROJECT_NAME="SIFT API Service" \
    LOG_LEVEL=INFO \
    LOG_JSON=false \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Run the application with reduced privileges
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
