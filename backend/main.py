# backend/main.py
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import logging
from sqlalchemy.orm import Session
import time
import base64
import asyncio
from typing import Dict, List

from backend.database import get_db, engine
from backend.models import Base
from backend.routers import cameras, alerts, dashboard, api
from backend.config import SCREENSHOTS_DIR, MODEL_PATH
from backend.services.detection import get_detection_service
from backend.services.model_service import get_model_service
from pathlib import Path

# Create database tables
Base.metadata.create_all(bind=engine)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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
app.include_router(api.router, prefix="/api", tags=["API"])

# Active WebSocket connections
active_connections: Dict[int, List[WebSocket]] = {}

@app.websocket("/ws/cameras/{camera_id}/stream")
async def websocket_camera_stream(websocket: WebSocket, camera_id: int, db: Session = Depends(get_db)):
    """WebSocket endpoint for streaming camera frames."""
    from backend.services.video import get_processor
    import cv2
    
    # Accept the connection
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for camera {camera_id}")
    
    # Initialize connection list for this camera if it doesn't exist
    if camera_id not in active_connections:
        active_connections[camera_id] = []
    
    # Add this connection to the list
    active_connections[camera_id].append(websocket)
    
    try:
        # Keep the connection open and stream frames
        while True:
            # Get the processor for this camera
            processor = get_processor(camera_id)
            
            if not processor:
                # If processor isn't running, send an error message
                await websocket.send_json({"type": "error", "message": "Camera not processing"})
                # Wait a bit before trying again
                await asyncio.sleep(1.0)
                continue
                
            # Get the current frame
            frame = processor.get_current_frame()
            
            if frame is not None:
                try:
                    # Ensure frame is in correct format for encoding
                    if frame.ndim == 2:  # If grayscale, convert to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    # Encode frame to JPEG with moderate compression
                    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    
                    if success:
                        # Convert to base64 for sending over WebSocket
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Send the frame
                        await websocket.send_json({
                            "type": "frame",
                            "frame": frame_base64,
                            "timestamp": time.time()
                        })
                    else:
                        logger.warning(f"Failed to encode frame for camera {camera_id}")
                except Exception as e:
                    logger.error(f"Error encoding frame for camera {camera_id}: {str(e)}")
                    # Don't exit the loop, try again with the next frame
            
            # Short delay to control frame rate (adjust as needed)
            await asyncio.sleep(0.15)  # ~6-7 FPS (increased from 0.05 which was ~20 FPS)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for camera {camera_id}")
        # Remove this connection from the list when disconnected
        if camera_id in active_connections:
            active_connections[camera_id].remove(websocket)
            
            # Clean up if no more connections for this camera
            if not active_connections[camera_id]:
                del active_connections[camera_id]
    
    except Exception as e:
        logger.error(f"WebSocket error for camera {camera_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Try to close connection gracefully
        try:
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")
        except:
            pass

# Function to broadcast alert to all connected clients for a camera
async def broadcast_alert(camera_id: int, alert_data: dict):
    """Broadcast alert to all clients connected to a camera stream."""
    if camera_id not in active_connections:
        return
        
    for connection in active_connections[camera_id]:
        try:
            await connection.send_json({
                "type": "alert",
                "data": alert_data
            })
        except Exception as e:
            logger.error(f"Error broadcasting alert: {str(e)}")

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
        
        # Check if the model file exists
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            logger.warning(f"Model file not found: {MODEL_PATH}")
            # Look for any .pt files in the models directory
            model_dir = model_path.parent
            pt_files = list(model_dir.glob("*.pt"))
            if pt_files:
                logger.info(f"Found alternative model file: {pt_files[0]}")
                # Use the first .pt file found
                os.environ["MODEL_PATH"] = str(pt_files[0])
            else:
                logger.error("No model files found in the models directory!")
        
        # Initialize the detection service
        logger.info("Initializing detection service...")
        detection_service = get_detection_service()
        model_service = get_model_service()
        
        # Load the default model
        logger.info("Loading YOLO model...")
        success, model = model_service.load_model()
        if success:
            logger.info("Model loaded successfully")
        else:
            logger.error("Failed to load model")
            
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


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