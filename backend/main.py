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
import queue

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

# Alert queue for thread-safe communication
alert_queue = queue.Queue()

def get_alert_queue():
    """Get the global alert queue for thread-safe communication."""
    global alert_queue
    return alert_queue

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
    
    # Track failed attempts to reduce log noise
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 5
    last_frame_hash = None  # For tracking frame changes
    
    try:
        # Keep the connection open and stream frames
        while True:
            # Get the processor for this camera
            processor = get_processor(camera_id)
            
            if not processor:
                # If processor isn't running, send an error message
                try:
                    await websocket.send_json({"type": "error", "message": "Camera not processing"})
                except Exception as e:
                    logger.error(f"Error sending error message for camera {camera_id}: {str(e)}")
                    # Connection might be closed, break the loop
                    break
                
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
                    
                    # Resize the frame to reduce data size
                    height, width = frame.shape[:2]
                    if width > 800:  # Only resize if larger than 800px
                        scale = 800 / width
                        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                    
                    # Compute a simple hash of the frame to check if it's significantly different
                    # This reduces bandwidth by not sending duplicate frames
                    import hashlib
                    small_frame = cv2.resize(frame, (32, 32))  # Tiny version for hashing
                    frame_hash = hashlib.md5(small_frame.tobytes()).hexdigest()
                    
                    # Only send if the frame has changed significantly
                    if frame_hash != last_frame_hash:
                        last_frame_hash = frame_hash
                        
                        # Encode frame to JPEG with lower quality for faster transmission
                        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        
                        if success:
                            # Convert to base64 for sending over WebSocket
                            frame_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Send the frame with error handling
                            try:
                                await websocket.send_json({
                                    "type": "frame",
                                    "frame": frame_base64,
                                    "timestamp": time.time()
                                })
                                # Reset error counter on success
                                consecutive_errors = 0
                            except WebSocketDisconnect:
                                logger.info(f"WebSocket disconnected while sending frame for camera {camera_id}")
                                break
                            except Exception as e:
                                consecutive_errors += 1
                                if consecutive_errors <= MAX_CONSECUTIVE_ERRORS:
                                    logger.error(f"Error sending frame for camera {camera_id}: {str(e)}")
                                
                                # Try to ping only if we haven't hit error limit
                                if consecutive_errors <= 3:
                                    try:
                                        # Send a ping to check connection
                                        await websocket.send_json({"type": "ping"})
                                    except:
                                        # If ping fails, break out of the loop
                                        logger.info(f"WebSocket connection appears broken for camera {camera_id}")
                                        break
                                else:
                                    # Too many consecutive errors, just break
                                    logger.info(f"Too many consecutive WebSocket errors for camera {camera_id}, disconnecting")
                                    break
                        else:
                            consecutive_errors += 1
                            if consecutive_errors <= MAX_CONSECUTIVE_ERRORS:
                                logger.warning(f"Failed to encode frame for camera {camera_id}")
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors <= MAX_CONSECUTIVE_ERRORS:
                        logger.error(f"Error encoding frame for camera {camera_id}: {str(e)}")
            
            # Longer delay between frames to reduce connection load, with 
            # dynamic adjustment based on detected connection issues
            delay = 0.2  # Base delay (~5 FPS)
            if consecutive_errors > 0:
                # Increase delay when experiencing errors
                delay = min(1.0, 0.2 + consecutive_errors * 0.1)  # Up to 1 second (1 FPS)
            
            await asyncio.sleep(delay)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for camera {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket error for camera {camera_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Try to close connection gracefully
        try:
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")
        except:
            pass
    finally:
        # Remove this connection from the list when disconnected (always do this cleanup)
        if camera_id in active_connections and websocket in active_connections[camera_id]:
            active_connections[camera_id].remove(websocket)
            logger.info(f"Removed WebSocket connection for camera {camera_id}, {len(active_connections[camera_id])} connections remaining")
            
            # Clean up if no more connections for this camera
            if not active_connections[camera_id]:
                del active_connections[camera_id]
                logger.info(f"Removed all WebSocket connections for camera {camera_id}")

# Function to broadcast alert to all connected clients for a camera
async def broadcast_alert(camera_id: int, alert_data: dict):
    """Broadcast alert to all clients connected to a camera stream."""
    if camera_id not in active_connections:
        logger.debug(f"No active WebSocket connections for camera {camera_id} to broadcast alert")
        return
    
    connection_count = len(active_connections[camera_id])
    logger.info(f"Broadcasting alert to {connection_count} active connections for camera {camera_id}")
    
    # Make a copy of the connections list since we might modify it during iteration
    connections = list(active_connections[camera_id])
    
    for connection in connections:
        try:
            await connection.send_json({
                "type": "alert",
                "data": alert_data
            })
            logger.debug(f"Alert sent to a connection for camera {camera_id}")
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected while broadcasting alert for camera {camera_id}")
            # Remove dead connection
            if camera_id in active_connections and connection in active_connections[camera_id]:
                active_connections[camera_id].remove(connection)
        except Exception as e:
            logger.error(f"Error broadcasting alert: {str(e)}")
            # Try to determine if connection is still alive
            try:
                await connection.send_json({"type": "ping"})
            except:
                # Connection is dead, remove it
                if camera_id in active_connections and connection in active_connections[camera_id]:
                    active_connections[camera_id].remove(connection)
                    logger.info(f"Removed dead WebSocket connection for camera {camera_id}")

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
            
        # Start background task to process alerts
        asyncio.create_task(process_alerts_from_queue())
        logger.info("Alert processing background task started")
            
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# Background task to process alerts from the queue
async def process_alerts_from_queue():
    """Process alerts from the queue and broadcast them."""
    logger.info("Starting alert queue processor")
    while True:
        try:
            # Non-blocking check for new alerts (timeout of 0.1 seconds)
            try:
                camera_id, alert_data = alert_queue.get(timeout=0.1)
                
                logger.info(f"Processing alert from queue for camera {camera_id} (violation: {alert_data.get('violation_type', 'unknown')}, confidence: {alert_data.get('confidence', 0):.2f})")
                
                # Broadcast the alert
                await broadcast_alert(camera_id, alert_data)
                
                # Mark task as done
                alert_queue.task_done()
                
                logger.info(f"Successfully broadcasted alert for camera {camera_id}")
            except queue.Empty:
                # Queue is empty, wait a bit before checking again
                pass
                
            # Short sleep to avoid CPU spinning
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error processing alert from queue: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Keep the background task running despite errors
            await asyncio.sleep(1.0)

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