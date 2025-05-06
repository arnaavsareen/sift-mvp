# Dockerfile for SIFT PPE Detection Service
FROM python:3.11-slim-bullseye

# Add non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required directories and set proper permissions
RUN mkdir -p logs temp models && \
    chown -R appuser:appuser /app

# Set proper permissions
RUN chmod -R 755 /app

# Switch to non-root user for security
USER appuser

# Download YOLOv8 model if not present
RUN python -c "import os; from ultralytics import YOLO; os.makedirs('models', exist_ok=True); YOLO('yolov8n.pt').save('models/yolov8n.pt') if not os.path.exists('models/yolov8n.pt') else None"

# Expose the application port
EXPOSE 8000

# Healthcheck to ensure the application is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run database migrations on startup
CMD ["sh", "-c", "alembic upgrade head && python -m uvicorn app.api.api:app --host 0.0.0.0 --port 8000"]
