FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including FFmpeg for video streaming
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    python3-dev \
    ffmpeg \
    libopencv-dev \
    build-essential \
    libmp3lame-dev \
    libx264-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/models /app/data/screenshots /tmp/rtsp_stream && \
    chmod -R 755 /app

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "Checking if mock data should be generated..."\n\
if [ "$GENERATE_MOCK_DATA" = "true" ]; then\n\
    echo "Generating mock data for YC demo..."\n\
    python /app/backend/scripts/generate_mock_data.py\n\
    python /app/backend/scripts/generate_mock_screenshots.py\n\
    echo "Mock data generation complete!"\n\
fi\n\
\n\
# Start the API server\n\
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8000

# Use our entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]