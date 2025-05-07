from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import logging
from sqlalchemy.orm import Session

from backend.database import get_db, engine
from backend.models import Camera, Alert, Zone, Base
from backend.routers import cameras, alerts, dashboard
from backend.config import SCREENSHOTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title="SIFT API",
    description="Safety Inspection & Factory Tracking API",
    version="0.1.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Mount static files (screenshots)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
app.mount("/screenshots", StaticFiles(directory=SCREENSHOTS_DIR), name="screenshots")

# Include routers
app.include_router(cameras.router, prefix="/api/cameras", tags=["Cameras"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])


@app.get("/api/health")
def health_check():
    """API health check endpoint."""
    return {"status": "healthy", "service": "SIFT API"}


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Starting SIFT API service")
    
    try:
        # Create database tables if they don't exist
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created or verified")
        
        # Make sure directories exist
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        logger.info(f"Screenshots directory: {SCREENSHOTS_DIR}")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Shutting down SIFT API service")
    
    try:
        # Stop all camera processors
        from backend.services.video import get_all_processors, stop_processor
        processors = get_all_processors()
        
        for camera_id in list(processors.keys()):
            stop_processor(camera_id)
            logger.info(f"Stopped processor for camera {camera_id}")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)