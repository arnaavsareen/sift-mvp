# SIFT - Safety Inspection & Factory Tracking

SIFT is a real-time AI-powered compliance and task monitoring platform for SMB manufacturers. It uses computer vision to detect safety violations like missing PPE, fall protection issues, and hazmat non-compliance.

## Features

- Real-time video stream processing with YOLO object detection
- Detection of safety violations (hard hats, vests, fall protection)
- Alert generation with screenshots and metadata
- Compliance dashboard with metrics and trends
- Camera management interface
- Alert review and resolution workflow

## Tech Stack

- **Backend**: Python, FastAPI, OpenCV, YOLOv8
- **Frontend**: React, TailwindCSS, Chart.js
- **Database**: PostgreSQL
- **Deployment**: Docker, Docker Compose

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Git
- CCTV footage or webcam for testing

### Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/sift.git
cd sift
```

2. Create required directories

```bash
mkdir -p data/screenshots data/models
```

3. Download YOLOv8 model

```bash
# Option 1: Direct download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -P data/models/

# Option 2: Using gdown
pip install gdown
gdown https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -O data/models/yolov8s.pt
```

4. Run with Docker Compose

```bash
docker-compose up -d
```

5. Access the application
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/api
   - API Documentation: http://localhost:8000/docs

## Development Setup

### Backend

1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the backend server

```bash
uvicorn backend.main:app --reload
```

### Frontend

1. Navigate to the frontend directory

```bash
cd frontend
```

2. Install dependencies

```bash
npm install
```

3. Start the development server

```bash
npm start
```

## Working with Cameras

SIFT supports several types of video sources:

- RTSP streams from IP cameras
- HTTP video streams
- Local video files
- Webcams

When adding a camera, use the appropriate URL format:

- RTSP: `rtsp://username:password@192.168.1.100:554/stream`
- HTTP: `http://192.168.1.100/video.mjpg`
- Local file: `/path/to/video.mp4`
- Webcam: `0` (for default webcam)

## Adding Custom Detection Models

To use a custom YOLOv8 model:

1. Place your trained model in the `data/models/` directory
2. Update the `.env` file to point to your model:

```
MODEL_PATH=data/models/your_custom_model.pt
```

## Project Structure

```
sift/
├── .env                    # Environment variables
├── docker-compose.yml      # Docker composition
├── requirements.txt        # Python dependencies
│
├── backend/                # FastAPI application
│   ├── main.py             # Entry point
│   ├── config.py           # Configuration
│   ├── models.py           # Database models
│   ├── routers/            # API endpoints
│   └── services/           # Business logic
│
├── frontend/               # React application
│   ├── public/             # Static assets
│   └── src/                # React components
│
├── data/                   # Data storage
│   ├── models/             # ML models
│   └── screenshots/        # Alert screenshots
│
└── deployment/             # Deployment files
    └── docker/             # Dockerfiles
```
