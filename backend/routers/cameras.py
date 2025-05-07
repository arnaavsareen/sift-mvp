from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from backend.database import get_db
from backend.models import Camera
from backend.services.detection import get_detection_service
from backend.services.alert import get_alert_service
from backend.services.video import start_processor, stop_processor, get_processor, get_all_processors

# Pydantic models for request/response
from pydantic import BaseModel
from datetime import datetime

class CameraBase(BaseModel):
    name: str
    url: str
    location: Optional[str] = None
    is_active: bool = True

class CameraCreate(CameraBase):
    pass

class CameraResponse(CameraBase):
    id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    location: Optional[str] = None
    is_active: Optional[bool] = None


# Create router
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(camera: CameraCreate, db: Session = Depends(get_db)):
    """Create a new camera."""
    db_camera = Camera(**camera.dict())
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    return db_camera

@router.get("/", response_model=List[CameraResponse])
def get_cameras(
    skip: int = 0, 
    limit: int = 100, 
    active_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get all cameras."""
    query = db.query(Camera)
    
    if active_only:
        query = query.filter(Camera.is_active == True)
    
    cameras = query.offset(skip).limit(limit).all()
    return cameras

@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: int, db: Session = Depends(get_db)):
    """Get camera by ID."""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera

@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(
    camera_id: int, 
    camera_update: CameraUpdate, 
    db: Session = Depends(get_db)
):
    """Update camera."""
    db_camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Update fields
    update_data = camera_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_camera, key, value)
    
    db.commit()
    db.refresh(db_camera)
    return db_camera

@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    """Delete camera."""
    # First stop processing if running
    processor = get_processor(camera_id)
    if processor:
        stop_processor(camera_id)
    
    # Then delete from database
    db_camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    db.delete(db_camera)
    db.commit()
    return None

@router.post("/{camera_id}/start", response_model=dict)
def start_camera(camera_id: int, db: Session = Depends(get_db)):
    """Start processing camera feed."""
    # Get camera from database
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Check if already processing
    if get_processor(camera_id):
        return {"status": "already_running", "camera_id": camera_id}
    
    # Get services
    detection_service = get_detection_service()
    alert_service = get_alert_service(db)
    
    # Start processor
    success = start_processor(
        camera_id=camera.id,
        camera_url=camera.url,
        detection_service=detection_service,
        alert_service=alert_service
    )
    
    if not success:
        raise HTTPException(
            status_code=500, 
            detail="Failed to start camera processing"
        )
    
    return {"status": "started", "camera_id": camera_id}

@router.post("/{camera_id}/stop", response_model=dict)
def stop_camera(camera_id: int, db: Session = Depends(get_db)):
    """Stop processing camera feed."""
    # Check if camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Stop processor
    success = stop_processor(camera_id)
    
    if not success:
        return {"status": "not_running", "camera_id": camera_id}
    
    return {"status": "stopped", "camera_id": camera_id}

@router.get("/{camera_id}/status", response_model=dict)
def get_camera_status(camera_id: int, db: Session = Depends(get_db)):
    """Get camera processing status."""
    # Check if camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Get processor
    processor = get_processor(camera_id)
    
    return {
        "camera_id": camera_id,
        "is_processing": processor is not None,
        "frame_count": getattr(processor, "frame_count", 0) if processor else 0,
        "last_frame_time": getattr(processor, "last_frame_time", None)
    }