# SIFT - Safety Inspection & Factory Tracking

SIFT is a real-time AI-powered compliance and task monitoring platform for SMB manufacturers. It uses computer vision to detect safety violations like missing PPE, fall protection issues, and hazmat non-compliance.

## Features

- Real-time video stream processing with YOLO object detection
- Detection of safety violations (hard hats, vests, fall protection)
- Alert generation with screenshots and metadata
- Compliance dashboard with metrics and trends
- Camera management interface
- Alert review and resolution workflow
- Zone-based safety rules for different workplace areas
- Performance monitoring and optimization

## Tech Stack

- **Backend**: Python, FastAPI, OpenCV, YOLOv8, SQLAlchemy
- **Frontend**: React, TailwindCSS, Chart.js, React Router
- **Database**: SQLite (development), PostgreSQL (production)
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

## YC Demo Mode with Mock Data

For demo purposes, you can automatically populate the dashboard with realistic mock data:

1. The system is configured to generate mock data by default when using Docker Compose.

2. To control mock data generation, use the `GENERATE_MOCK_DATA` environment variable:

```bash
# Enable mock data generation (default)
GENERATE_MOCK_DATA=true docker-compose up -d

# Disable mock data generation
GENERATE_MOCK_DATA=false docker-compose up -d
```

3. To manually generate mock data after startup:

```bash
# Generate mock data in a running container
docker exec -it sift-mvp-backend-1 python /app/backend/scripts/generate_mock_data.py
docker exec -it sift-mvp-backend-1 python /app/backend/scripts/generate_mock_screenshots.py
```

The mock data includes:
- Multiple cameras with different locations
- Safety violation alerts with timestamps
- Realistic alert patterns (more during work hours)
- Mock screenshots for visual display

For more details, see [Mock Data Generation Documentation](backend/scripts/README.md).

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

## Detection Capabilities

SIFT can detect the following safety violations:

- Missing hard hats in required areas
- Missing safety vests in high-visibility zones
- Missing masks, goggles, and gloves in chemical areas
- Missing boots in construction zones

The system uses zone-based rules that can be customized for different workplace areas, allowing for specific PPE requirements per zone.

## Adding Custom Detection Models

To use a custom YOLOv8 model:

1. Place your trained model in the `data/models/` directory
2. Update the `.env` file to point to your model:

```
MODEL_PATH=data/models/your_custom_model.pt
```

## Alert Management

The platform provides a comprehensive alert management system:

- Real-time alert generation with screenshots
- Alert categorization by violation type
- Resolution workflow for tracking compliance improvements
- Historical alert data for trend analysis
- Performance metrics for safety compliance

## Current Project Structure

```
sift-mvp/
├── README.md                           # Project documentation
├── docker-compose.yml                  # Docker composition
├── requirements.txt                    # Python dependencies
│
├── backend/                            # FastAPI application
│   ├── __init__.py                     # Python package marker
│   ├── app.db                          # SQLite database
│   ├── config.py                       # Configuration settings
│   ├── database.py                     # Database connection
│   ├── main.py                         # Application entry point
│   ├── models.py                       # SQLAlchemy models
│   │
│   ├── routers/                        # API endpoints
│   │   ├── __init__.py                 # Package marker
│   │   ├── alerts.py                   # Alert management routes
│   │   ├── api.py                      # API utilities
│   │   ├── cameras.py                  # Camera management routes
│   │   └── dashboard.py                # Dashboard data routes
│   │
│   └── services/                       # Business logic
│       ├── __init__.py                 # Package marker
│       ├── alert.py                    # Alert management
│       ├── config_service.py           # Configuration management
│       ├── detection.py                # Object detection with YOLO
│       ├── model_service.py            # ML model management
│       ├── notification_service.py     # Alert notifications
│       ├── performance_service.py      # Performance monitoring
│       ├── video.py                    # Video stream processing
│       └── zone_service.py             # Safety zone management
│
├── data/                               # Data storage
│   └── screenshots/                    # Alert screenshots
│
├── deployment/                         # Deployment files
│   └── docker/                         # Dockerfiles
│       ├── backend.Dockerfile          # Backend container
│       └── frontend.Dockerfile         # Frontend container
│
└── frontend/                           # React application
    ├── build/                          # Production build
    ├── package.json                    # NPM dependencies
    ├── postcss.config.js               # PostCSS configuration
    ├── public/                         # Static assets
    ├── src/                            # Source code
    │   ├── App.jsx                     # Main application component
    │   ├── api/                        # API clients
    │   │   └── api.js                  # API service
    │   ├── components/                 # React components
    │   │   ├── AlertsDetail.jsx        # Alert detail view
    │   │   ├── AlertsList.jsx          # Alerts list view
    │   │   ├── CamerasDetail.jsx       # Camera detail view
    │   │   ├── CamerasList.jsx         # Cameras list view
    │   │   ├── Dashboard.jsx           # Main dashboard view
    │   │   ├── Layout.jsx              # Application layout
    │   │   └── NotFound.jsx            # 404 page
    │   ├── index.css                   # Global styles
    │   ├── index.js                    # Application entry point
    │   └── reportWebVitals.js          # Performance reporting
    └── tailwind.config.js              # Tailwind CSS configuration
```

## Key Components and Functionality

### Backend Services

- **Detection Service**: Uses YOLOv8 to detect people and PPE items in video frames, analyzes for safety violations
- **Video Service**: Handles camera streams, frame processing, and maintains processing threads
- **Alert Service**: Generates and manages safety violation alerts
- **Zone Service**: Manages safety zones and their associated rules
- **Notification Service**: Sends alerts through configured channels
- **Performance Service**: Monitors system performance metrics

### Frontend Components

- **Dashboard**: Real-time overview with key metrics, recent alerts, and compliance trends
- **Camera Management**: Interface for adding, configuring, and monitoring cameras
- **Alert Management**: Review and resolution of safety violations
- **Zone Configuration**: Define safety zones within camera views

### API Endpoints

- `/api/cameras`: Camera CRUD operations and status monitoring
- `/api/alerts`: Alert generation, retrieval, and resolution
- `/api/dashboard`: Analytics and metrics for the dashboard UI

The application uses FastAPI's dependency injection for database access and React's component architecture for a responsive UI. All components communicate through RESTful API endpoints.
