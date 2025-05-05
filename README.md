# SIFT: Real-time AI Compliance Monitoring for Manufacturing

sift-mvp

SIFT is an AWS-native, microservice-based MVP for real-time compliance monitoring in manufacturing. The architecture consists of four main services: data ingestion (from cameras and sensors), AI inference (for real-time analysis), storage (persisting results and metadata), and a web dashboard (for visualization and alerts). Each service is containerized and designed for scalability and independent deployment.

---

## Recent Changes
- Implemented a robust ingestion pipeline (`ingestion_service/ingest_camera.py`) that captures video frames from a camera or video file, uploads them to AWS S3, and sends a message to AWS SQS for each frame.
- Added `.gitignore` files at both the service and project root to exclude sensitive files, virtual environments, outputs, and editor/OS artifacts from version control.
- Updated documentation and environment setup instructions.

---

## Codebase Overview

### Service Directories
- `ingestion_service/`: Handles data ingestion from manufacturing cameras/sensors and pushes to AWS S3/SQS.
    - **Key files:**
        - `ingest_camera.py`: Ingestion script for capturing frames and pushing to AWS.
        - `requirements.txt`: Python dependencies.
        - `.env`: Environment variables (excluded from git).
        - `.gitignore`: Excludes `.env`, video files, outputs, and venvs.
    - **How it works:**
        - Reads from a webcam or video file (set in `.env` as `CAMERA_URL`).
        - Captures frames at a configurable interval.
        - Uploads each frame to S3 (`frames/` prefix).
        - Sends SQS messages with S3 keys for downstream processing.
- `model_service/`: Runs AI inference (e.g., YOLOv5) on incoming data and returns compliance results.
- `api_service/`: Provides REST APIs for data access, storage, and integration with RDS.
- `web_dashboard/`: React-based dashboard for real-time monitoring and visualization.

---

## .gitignore Usage
- Sensitive files like `.env`, local video files, frame outputs, and virtual environments are excluded from version control.
- Project root `.gitignore` also covers Terraform state, node_modules, and editor/OS files.

---

## Next Steps
- [x] Implement ingestion pipeline to push camera/video data to AWS S3/SQS.
- [ ] Build and containerize the model service for AI inference.
- [ ] Develop REST APIs for data access and integration.
- [ ] Create a web dashboard for monitoring and visualization.
- [ ] Add error notifications, monitoring, and alerting.
- [ ] Expand documentation as new components are added.

---

## Getting Started
See `ingestion_service/README.md` for setup and usage instructions for the ingestion pipeline.


