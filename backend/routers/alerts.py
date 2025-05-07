from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
from datetime import datetime, timedelta

from backend.database import get_db
from backend.models import Alert, Camera

# Pydantic models for request/response
from pydantic import BaseModel


class AlertResponse(BaseModel):
    id: int
    camera_id: int
    violation_type: str
    confidence: float
    screenshot_path: Optional[str] = None
    created_at: datetime
    resolved: bool
    resolved_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class AlertResolve(BaseModel):
    resolved: bool = True


# Create router
router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=List[AlertResponse])
def get_alerts(
    skip: int = 0, 
    limit: int = 100,
    camera_id: Optional[int] = None,
    violation_type: Optional[str] = None,
    unresolved_only: bool = False,
    hours: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get alerts with filtering options."""
    query = db.query(Alert)
    
    # Apply filters
    if camera_id is not None:
        query = query.filter(Alert.camera_id == camera_id)
    
    if violation_type:
        query = query.filter(Alert.violation_type == violation_type)
    
    if unresolved_only:
        query = query.filter(Alert.resolved == False)
    
    if hours is not None:
        time_threshold = datetime.now() - timedelta(hours=hours)
        query = query.filter(Alert.created_at >= time_threshold)
    
    # Order by creation time (newest first)
    query = query.order_by(Alert.created_at.desc())
    
    alerts = query.offset(skip).limit(limit).all()
    return alerts


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    """Get alert by ID."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.put("/{alert_id}/resolve", response_model=AlertResponse)
def resolve_alert(
    alert_id: int, 
    alert_resolve: AlertResolve,
    db: Session = Depends(get_db)
):
    """Mark alert as resolved."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.resolved = alert_resolve.resolved
    
    if alert_resolve.resolved:
        alert.resolved_at = datetime.now()
    else:
        alert.resolved_at = None
    
    db.commit()
    db.refresh(alert)
    return alert


@router.get("/stats/summary", response_model=dict)
def get_alert_stats(
    hours: int = 24,
    camera_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get alert statistics."""
    time_threshold = datetime.now() - timedelta(hours=hours)
    
    # Base query
    query = db.query(Alert).filter(Alert.created_at >= time_threshold)
    
    if camera_id is not None:
        query = query.filter(Alert.camera_id == camera_id)
    
    # Total alerts
    total_alerts = query.count()
    
    # Alerts by type
    alerts_by_type = {}
    violation_types = db.query(Alert.violation_type).distinct().all()
    
    for type_row in violation_types:
        violation_type = type_row[0]
        count = query.filter(Alert.violation_type == violation_type).count()
        alerts_by_type[violation_type] = count
    
    # Alerts by camera
    alerts_by_camera = {}
    
    if camera_id is None:
        cameras = db.query(Camera).all()
        
        for camera in cameras:
            count = query.filter(Alert.camera_id == camera.id).count()
            alerts_by_camera[camera.id] = {
                "camera_id": camera.id,
                "name": camera.name,
                "count": count
            }
    
    return {
        "time_range_hours": hours,
        "total_alerts": total_alerts,
        "by_type": alerts_by_type,
        "by_camera": alerts_by_camera if camera_id is None else None
    }