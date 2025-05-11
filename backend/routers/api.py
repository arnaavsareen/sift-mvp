from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
import os
import time
from datetime import datetime, timedelta
import io
import cv2
import numpy as np
import base64

from backend.database import get_db
from backend.models import Camera, Alert, Zone
from backend.services.detection import get_detection_service
from backend.services.video import get_processor, start_processor, stop_processor, get_all_processors
from backend.services.alert import get_alert_service
from backend.services.zone_service import get_zone_service
from backend.services.config_service import get_config_service
from backend.services.model_service import get_model_service
from backend.services.performance_service import get_performance_service
from backend.services.notification_service import get_notification_service
from backend.config import SCREENSHOTS_DIR, BASE_DIR

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Common responses
STANDARD_RESPONSES = {
    400: {"description": "Bad Request"},
    401: {"description": "Unauthorized"},
    403: {"description": "Forbidden"},
    404: {"description": "Not Found"},
    500: {"description": "Internal Server Error"}
}

# --- System management endpoints ---

@router.get(
    "/system/status",
    tags=["System Management"],
    summary="Get overall system status",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def get_system_status(db: Session = Depends(get_db)):
    """
    Get overall system status including:
    - Performance metrics
    - Camera status
    - Detection status
    - Resource usage
    """
    try:
        # Get performance service
        perf_service = get_performance_service()
        status = perf_service.get_system_status()
        
        # Get camera status
        processors = get_all_processors()
        cameras = db.query(Camera).all()
        
        active_cameras = 0
        for camera in cameras:
            if camera.id in processors:
                active_cameras += 1
        
        # Get model service
        model_service = get_model_service()
        models = model_service.get_models()
        
        # Add to status
        status.update({
            "cameras": {
                "total": len(cameras),
                "active": active_cameras
            },
            "models": {
                "total": len(models),
                "active": len([m for m in models if m.get("is_loaded", False)])
            },
            "uptime": {
                "api_server": "Up",  # We're running, so API is up
                "database": "Up"     # We made a query, so DB is up
            }
        })
        
        return status
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system status"
        )


@router.get(
    "/system/performance",
    tags=["System Management"],
    summary="Get detailed performance metrics",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def get_performance_metrics(
    metric: Optional[str] = None,
    hours: int = Query(24, description="Number of hours to look back"),
    db: Session = Depends(get_db)
):
    """
    Get detailed performance metrics for system monitoring.
    Optionally filter by specific metric and time range.
    """
    try:
        perf_service = get_performance_service()
        
        if metric:
            return perf_service.get_historical_metrics(metric, hours)
        else:
            return perf_service.get_metrics()
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve performance metrics"
        )


@router.post(
    "/system/restart",
    tags=["System Management"],
    summary="Restart specific system components",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def restart_component(
    component: str = Body(..., description="Component to restart (detection, monitoring, all)"),
    db: Session = Depends(get_db)
):
    """
    Restart specific system components.
    Options: detection, monitoring, all
    """
    try:
        result = {"restarted": [], "failed": []}
        
        if component in ["detection", "all"]:
            # Restart detection service
            detection_service = get_detection_service()
            model_service = get_model_service()
            
            # Unload all models and reload default
            model_service.unload_all_models()
            _, model = model_service.load_model()
            
            if model:
                result["restarted"].append("detection")
            else:
                result["failed"].append("detection")
        
        if component in ["monitoring", "all"]:
            # Restart all video processors
            processors = get_all_processors()
            processor_ids = list(processors.keys())
            
            for camera_id in processor_ids:
                try:
                    # Get camera
                    camera = db.query(Camera).filter(Camera.id == camera_id).first()
                    if not camera:
                        continue
                    
                    # Stop processor
                    stop_processor(camera_id)
                    
                    # Start processor
                    detection_service = get_detection_service()
                    alert_service = get_alert_service(db)
                    
                    success = start_processor(
                        camera_id=camera.id,
                        camera_url=camera.url,
                        detection_service=detection_service,
                        alert_service=alert_service
                    )
                    
                    if success:
                        result["restarted"].append(f"camera_{camera_id}")
                    else:
                        result["failed"].append(f"camera_{camera_id}")
                        
                except Exception as e:
                    logger.error(f"Error restarting camera {camera_id}: {str(e)}")
                    result["failed"].append(f"camera_{camera_id}")
        
        return result
    except Exception as e:
        logger.error(f"Error restarting component {component}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restart component: {component}"
        )


# --- Model management endpoints ---

@router.get(
    "/models",
    tags=["Model Management"],
    summary="Get available models",
    response_model=List[Dict[str, Any]],
    responses=STANDARD_RESPONSES
)
def get_models(db: Session = Depends(get_db)):
    """
    Get list of all available detection models.
    """
    try:
        model_service = get_model_service()
        return model_service.get_models()
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve models"
        )


@router.get(
    "/models/{model_id}",
    tags=["Model Management"],
    summary="Get model details",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def get_model(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific model.
    """
    try:
        model_service = get_model_service()
        model = model_service.get_model_by_id(model_id)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_id}"
            )
        
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model: {model_id}"
        )


@router.post(
    "/models/default/{model_id}",
    tags=["Model Management"],
    summary="Set default model",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def set_default_model(
    model_id: str,
    task_type: Optional[str] = Body(None, description="Task type (e.g., object_detection)"),
    db: Session = Depends(get_db)
):
    """
    Set the default model for a specific task type.
    """
    try:
        model_service = get_model_service()
        success = model_service.set_default_model(model_id, task_type)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found or cannot be set as default: {model_id}"
            )
        
        return {"status": "success", "message": f"Set {model_id} as default model"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting default model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set default model: {model_id}"
        )


@router.post(
    "/models/upload",
    tags=["Model Management"],
    summary="Upload a new model",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
async def upload_model(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(...),
    name: Optional[str] = Query(None, description="Custom name for the model"),
    db: Session = Depends(get_db)
):
    """
    Upload a new YOLO model file.
    """
    try:
        # Check file extension
        if not model_file.filename.endswith('.pt'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only .pt files are supported."
            )
        
        # Create a temporary file
        temp_file_path = os.path.join(BASE_DIR, "data", "temp", model_file.filename)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        
        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            f.write(await model_file.read())
        
        # Import model
        model_service = get_model_service()
        model_id = model_service.import_model(temp_file_path, name)
        
        # Clean up in background
        background_tasks.add_task(os.remove, temp_file_path)
        
        if not model_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to import model"
            )
        
        return {
            "status": "success",
            "model_id": model_id,
            "message": f"Model uploaded successfully: {model_file.filename}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to upload model"
        )


@router.delete(
    "/models/{model_id}",
    tags=["Model Management"],
    summary="Delete a model",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def delete_model(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a model file and its metadata.
    """
    try:
        model_service = get_model_service()
        success = model_service.delete_model(model_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_id}"
            )
        
        return {"status": "success", "message": f"Deleted model: {model_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {model_id}"
        )


# --- Configuration endpoints ---

@router.get(
    "/config",
    tags=["Configuration"],
    summary="Get system configuration",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def get_config(db: Session = Depends(get_db)):
    """
    Get current system configuration.
    """
    try:
        config_service = get_config_service(db)
        return config_service.export_config()
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve configuration"
        )


@router.put(
    "/config/detection",
    tags=["Configuration"],
    summary="Update detection configuration",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def update_detection_config(
    config: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Update detection configuration.
    """
    try:
        config_service = get_config_service(db)
        success = config_service.update_detection_config(config)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Invalid configuration"
            )
        
        return {"status": "success", "message": "Detection configuration updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating detection configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update detection configuration"
        )


@router.put(
    "/config/camera/{camera_id}",
    tags=["Configuration"],
    summary="Update camera configuration",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def update_camera_config(
    camera_id: int,
    config: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Update configuration for a specific camera.
    """
    try:
        # Check if camera exists
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera not found: {camera_id}"
            )
        
        config_service = get_config_service(db)
        success = config_service.update_camera_config(camera_id, config)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Invalid configuration"
            )
        
        return {"status": "success", "message": f"Configuration updated for camera {camera_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating camera configuration for {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration for camera {camera_id}"
        )


# --- Zone management endpoints ---

@router.get(
    "/zones/camera/{camera_id}",
    tags=["Zones"],
    summary="Get zones for camera",
    response_model=List[Dict[str, Any]],
    responses=STANDARD_RESPONSES
)
def get_camera_zones(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all monitoring zones for a specific camera.
    """
    try:
        # Check if camera exists
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera not found: {camera_id}"
            )
        
        zone_service = get_zone_service(db)
        zones = zone_service.get_zones(camera_id)
        
        return zones
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting zones for camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve zones for camera {camera_id}"
        )


@router.post(
    "/zones",
    tags=["Zones"],
    summary="Create a new zone",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def create_zone(
    zone_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Create a new monitoring zone for a camera.
    """
    try:
        # Check if camera exists
        camera_id = zone_data.get("camera_id")
        if not camera_id:
            raise HTTPException(
                status_code=400,
                detail="camera_id is required"
            )
        
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera not found: {camera_id}"
            )
        
        # Validate polygon
        polygon = zone_data.get("polygon")
        if not polygon or len(polygon) < 3:
            raise HTTPException(
                status_code=400,
                detail="Valid polygon with at least 3 points is required"
            )
        
        # Create zone
        zone_service = get_zone_service(db)
        zone = zone_service.create_zone(zone_data)
        
        if not zone:
            raise HTTPException(
                status_code=500,
                detail="Failed to create zone"
            )
        
        return {
            "status": "success",
            "message": "Zone created successfully",
            "zone": zone
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating zone: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create zone"
        )


@router.put(
    "/zones/{zone_id}",
    tags=["Zones"],
    summary="Update a zone",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def update_zone(
    zone_id: int,
    zone_data: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Update an existing monitoring zone.
    """
    try:
        zone_service = get_zone_service(db)
        zone = zone_service.update_zone(zone_id, zone_data)
        
        if not zone:
            raise HTTPException(
                status_code=404,
                detail=f"Zone not found: {zone_id}"
            )
        
        return {
            "status": "success",
            "message": "Zone updated successfully",
            "zone": zone
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating zone {zone_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update zone {zone_id}"
        )


@router.delete(
    "/zones/{zone_id}",
    tags=["Zones"],
    summary="Delete a zone",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def delete_zone(
    zone_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a monitoring zone.
    """
    try:
        zone_service = get_zone_service(db)
        success = zone_service.delete_zone(zone_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Zone not found: {zone_id}"
            )
        
        return {
            "status": "success",
            "message": f"Zone {zone_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting zone {zone_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete zone {zone_id}"
        )


# --- Notification endpoints ---

@router.get(
    "/notifications/channels",
    tags=["Notifications"],
    summary="Get notification channels",
    response_model=Dict[str, Dict[str, Any]],
    responses=STANDARD_RESPONSES
)
def get_notification_channels(db: Session = Depends(get_db)):
    """
    Get all configured notification channels and their status.
    """
    try:
        notification_service = get_notification_service(db)
        return notification_service.get_channels()
    except Exception as e:
        logger.error(f"Error getting notification channels: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve notification channels"
        )


@router.get(
    "/notifications/config/{channel}",
    tags=["Notifications"],
    summary="Get notification channel configuration",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def get_notification_config(
    channel: str,
    db: Session = Depends(get_db)
):
    """
    Get configuration for a specific notification channel.
    """
    try:
        notification_service = get_notification_service(db)
        config = notification_service.get_channel_config(channel)
        
        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"Channel not found: {channel}"
            )
        
        # Remove sensitive information
        if "smtp_password" in config:
            config["smtp_password"] = "********"
        
        if "twilio_auth_token" in config:
            config["twilio_auth_token"] = "********"
        
        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting notification config for {channel}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve configuration for {channel}"
        )


@router.put(
    "/notifications/config/{channel}",
    tags=["Notifications"],
    summary="Update notification channel configuration",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def update_notification_config(
    channel: str,
    config: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Update configuration for a notification channel.
    """
    try:
        notification_service = get_notification_service(db)
        success = notification_service.update_channel_config(channel, config)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to update configuration for {channel}"
            )
        
        return {
            "status": "success",
            "message": f"Configuration updated for {channel} channel"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating notification config for {channel}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update configuration for {channel}"
        )


@router.post(
    "/notifications/test/{channel}",
    tags=["Notifications"],
    summary="Test notification channel",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def test_notification_channel(
    channel: str,
    db: Session = Depends(get_db)
):
    """
    Send a test notification to a channel.
    """
    try:
        notification_service = get_notification_service(db)
        success = notification_service.test_channel(channel)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to send test notification to {channel}"
            )
        
        return {
            "status": "success",
            "message": f"Test notification sent to {channel} channel"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing notification channel {channel}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test {channel} channel"
        )


@router.post(
    "/notifications/alert/{alert_id}",
    tags=["Notifications"],
    summary="Send notification for alert",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def send_alert_notification(
    alert_id: int,
    channels: Optional[List[str]] = Body(None, description="Specific channels to notify"),
    db: Session = Depends(get_db)
):
    """
    Send notification for a specific alert.
    """
    try:
        # Check if alert exists
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=404,
                detail=f"Alert not found: {alert_id}"
            )
        
        notification_service = get_notification_service(db)
        success = notification_service.notify(alert_id, channels)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send notification for alert {alert_id}"
            )
        
        return {
            "status": "success",
            "message": f"Notification queued for alert {alert_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending notification for alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send notification for alert {alert_id}"
        )


# --- Camera monitoring endpoints ---

@router.get(
    "/monitoring/status",
    tags=["Monitoring"],
    summary="Get monitoring status for all cameras",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def get_monitoring_status(db: Session = Depends(get_db)):
    """
    Get monitoring status for all cameras.
    """
    try:
        from backend.services.video import get_all_statuses
        
        statuses = get_all_statuses()
        cameras = db.query(Camera).all()
        
        # Camera status map
        camera_statuses = {}
        for camera in cameras:
            status = statuses.get(camera.id, {})
            
            camera_statuses[camera.id] = {
                "id": camera.id,
                "name": camera.name,
                "location": camera.location,
                "is_active": camera.is_active,
                "is_monitoring": camera.id in statuses,
                "stats": status
            }
        
        return {
            "active_cameras": len(statuses),
            "total_cameras": len(cameras),
            "cameras": camera_statuses
        }
    except Exception as e:
        logger.error(f"Error getting monitoring status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve monitoring status"
        )


@router.get(
    "/monitoring/camera/{camera_id}/latest",
    tags=["Monitoring"],
    summary="Get latest frame from camera",
    responses={
        200: {
            "content": {"image/jpeg": {}}
        },
        **STANDARD_RESPONSES
    }
)
def get_latest_frame(
    camera_id: int,
    format: str = Query("jpeg", description="Output format (jpeg, png, base64)"),
    width: Optional[int] = Query(None, description="Resize width"),
    height: Optional[int] = Query(None, description="Resize height"),
    quality: int = Query(90, description="JPEG quality (1-100)"),
    db: Session = Depends(get_db)
):
    """
    Get the latest processed frame from a camera.
    """
    try:
        # Check if camera exists
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera not found: {camera_id}"
            )
        
        # Get processor
        processor = get_processor(camera_id)
        if not processor:
            raise HTTPException(
                status_code=404,
                detail=f"Camera {camera_id} not currently monitoring"
            )
        
        # Get current frame
        frame = processor.get_current_frame()
        if frame is None:
            raise HTTPException(
                status_code=404,
                detail="No frame available"
            )
        
        # Resize if requested
        if width and height:
            frame = cv2.resize(frame, (width, height))
        
        # Convert to requested format
        if format.lower() == "jpeg":
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            io_buf = io.BytesIO(buffer)
            return StreamingResponse(io_buf, media_type="image/jpeg")
        elif format.lower() == "png":
            # Encode as PNG
            ret, buffer = cv2.imencode('.png', frame)
            io_buf = io.BytesIO(buffer)
            return StreamingResponse(io_buf, media_type="image/png")
        elif format.lower() == "base64":
            # Encode as base64
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img_str = base64.b64encode(buffer).decode('utf-8')
            return {"image": f"data:image/jpeg;base64,{img_str}"}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest frame: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get latest frame"
        )


@router.post(
    "/monitoring/camera/{camera_id}/start",
    tags=["Monitoring"],
    summary="Start monitoring camera",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def start_camera_monitoring(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """
    Start monitoring a camera for safety violations.
    """
    try:
        # Check if camera exists
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera not found: {camera_id}"
            )
        
        # Check if already monitoring
        if get_processor(camera_id):
            return {"status": "already_running", "camera_id": camera_id}
        
        # Get services
        detection_service = get_detection_service()
        alert_service = get_alert_service(db)
        
        # Get configuration
        config_service = get_config_service(db)
        frame_sample_rate = config_service.get_frame_sample_rate(camera_id)
        
        # Start processor
        success = start_processor(
            camera_id=camera.id,
            camera_url=camera.url,
            detection_service=detection_service,
            alert_service=alert_service,
            frame_sample_rate=frame_sample_rate
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start monitoring camera {camera_id}"
            )
        
        return {"status": "started", "camera_id": camera_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting monitoring for camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start monitoring camera {camera_id}"
        )


@router.post(
    "/monitoring/camera/{camera_id}/stop",
    tags=["Monitoring"],
    summary="Stop monitoring camera",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def stop_camera_monitoring(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """
    Stop monitoring a camera.
    """
    try:
        # Check if camera exists
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(
                status_code=404,
                detail=f"Camera not found: {camera_id}"
            )
        
        # Stop processor
        success = stop_processor(camera_id)
        
        if not success:
            return {"status": "not_running", "camera_id": camera_id}
        
        return {"status": "stopped", "camera_id": camera_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping monitoring for camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop monitoring camera {camera_id}"
        )


@router.post(
    "/monitoring/camera/{camera_id}/restart",
    tags=["Monitoring"],
    summary="Restart camera monitoring",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def restart_camera_monitoring(
    camera_id: int,
    db: Session = Depends(get_db)
):
    """
    Restart monitoring for a camera.
    """
    try:
        # First stop
        stop_camera_monitoring(camera_id, db)
        
        # Then start
        return start_camera_monitoring(camera_id, db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting monitoring for camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restart monitoring camera {camera_id}"
        )


@router.post(
    "/monitoring/start-all",
    tags=["Monitoring"],
    summary="Start monitoring all active cameras",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def start_all_cameras(db: Session = Depends(get_db)):
    """
    Start monitoring all active cameras.
    """
    try:
        # Get active cameras
        cameras = db.query(Camera).filter(Camera.is_active == True).all()
        
        results = {
            "started": [],
            "failed": [],
            "already_running": []
        }
        
        # Start each camera
        for camera in cameras:
            try:
                response = start_camera_monitoring(camera.id, db)
                if response["status"] == "started":
                    results["started"].append(camera.id)
                elif response["status"] == "already_running":
                    results["already_running"].append(camera.id)
            except Exception:
                results["failed"].append(camera.id)
        
        return results
    except Exception as e:
        logger.error(f"Error starting all cameras: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start all cameras"
        )


@router.post(
    "/monitoring/stop-all",
    tags=["Monitoring"],
    summary="Stop monitoring all cameras",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
def stop_all_cameras(db: Session = Depends(get_db)):
    """
    Stop monitoring all cameras.
    """
    try:
        # Get all running processors
        from backend.services.video import get_all_processors
        processors = get_all_processors()
        
        results = {
            "stopped": [],
            "failed": []
        }
        
        # Stop each processor
        for camera_id in list(processors.keys()):
            try:
                stop_camera_monitoring(camera_id, db)
                results["stopped"].append(camera_id)
            except Exception:
                results["failed"].append(camera_id)
        
        return results
    except Exception as e:
        logger.error(f"Error stopping all cameras: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to stop all cameras"
        )


# --- Test endpoints for debugging ---

@router.post(
    "/test/detection",
    tags=["Testing"],
    summary="Test object detection on image",
    response_model=Dict[str, Any],
    responses=STANDARD_RESPONSES
)
async def test_detection(
    image: UploadFile = File(...),
    confidence: float = Query(0.25, description="Detection confidence threshold"),
    db: Session = Depends(get_db)
):
    """
    Test object detection on an uploaded image.
    """
    try:
        # Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format"
            )
        
        # Get detection service
        detection_service = get_detection_service()
        
        # Set confidence threshold
        original_confidence = detection_service.confidence
        detection_service.confidence = confidence
        
        # Run detection
        start_time = time.time()
        detections = detection_service.detect(img)
        
        # Restore original confidence threshold
        detection_service.confidence = original_confidence
        
        # Calculate detection time
        detection_time = time.time() - start_time
        
        # Draw detections on image
        annotated = img.copy()
        
        for detection in detections:
            bbox = detection["bbox"]
            label = detection["class"]
            conf = detection["confidence"]
            is_violation = detection.get("violation", False)
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = (0, 0, 255) if is_violation else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{label}: {conf:.2f}"
            if is_violation:
                label_text += f" - {detection.get('violation_type', '')}"
                
            cv2.putText(
                annotated,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Encode result image
        _, img_encoded = cv2.imencode(".jpg", annotated)
        
        # Convert to base64
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")
        
        return {
            "detection_time_ms": detection_time * 1000,
            "detections": detections,
            "image_base64": img_base64,
            "num_detections": len(detections),
            "violations": sum(1 for d in detections if d.get("violation", False))
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing detection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to test detection"
        )


@router.get(
    "/test/screenshot/{alert_id}",
    tags=["Testing"],
    summary="View alert screenshot",
    responses={
        200: {
            "content": {"image/jpeg": {}}
        },
        **STANDARD_RESPONSES
    }
)
def view_alert_screenshot(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """
    View screenshot for a specific alert.
    """
    try:
        # Get alert
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert or not alert.screenshot_path:
            raise HTTPException(
                status_code=404,
                detail=f"Screenshot not found for alert {alert_id}"
            )
        
        # Check if path is relative to SCREENSHOTS_DIR
        if alert.screenshot_path.startswith("/screenshots/"):
            screenshot_path = os.path.join(
                BASE_DIR, "data", alert.screenshot_path.lstrip("/")
            )
        else:
            screenshot_path = alert.screenshot_path
        
        # Check if file exists
        if not os.path.exists(screenshot_path):
            raise HTTPException(
                status_code=404,
                detail=f"Screenshot file not found for alert {alert_id}"
            )
        
        return FileResponse(
            screenshot_path,
            media_type="image/jpeg",
            filename=f"alert_{alert_id}.jpg"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting screenshot for alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get screenshot for alert {alert_id}"
        )