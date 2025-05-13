FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including FFmpeg for video streaming
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    python3-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install ultralytics separately to ensure we get the latest version
RUN pip install --no-cache-dir ultralytics

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/screenshots data/models

# Check if the model is already mounted in the container
# If not, it will be handled at runtime through volume mounting

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]