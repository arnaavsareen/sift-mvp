from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
from datetime import datetime, timedelta
import cv2
import base64
import numpy as np

from backend.database import get_db
from backend.models import Alert, Camera
from backend.services.video import get_processor

# Create router
router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/overview")
def get_dashboard_overview(
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """Get overview data for dashboard."""
    time_threshold = datetime.now() - timedelta(hours=hours)
    
    # Get cameras
    cameras = db.query(Camera).all()
    camera_count = len(cameras)
    active_cameras = len([c for c in cameras if c.is_active])
    
    # Get processors
    from backend.services.video import get_all_processors
    processors = get_all_processors()
    monitoring_count = len(processors)
    
    # Get alerts
    alerts = db.query(Alert).filter(Alert.created_at >= time_threshold).all()
    alert_count = len(alerts)
    
    # Get violation breakdown
    violation_types = {}
    for alert in alerts:
        vtype = alert.violation_type
        if vtype not in violation_types:
            violation_types[vtype] = 0
        violation_types[vtype] += 1
    
    # Get compliance score (simple metric)
    compliance_score = 100.0
    if alert_count > 0 and camera_count > 0:
        # Rough estimate: reduce score based on alerts per camera
        alerts_per_camera = alert_count / camera_count
        compliance_score = max(0, 100 - (alerts_per_camera * 5))
    
    return {
        "time_range_hours": hours,
        "cameras": {
            "total": camera_count,
            "active": active_cameras,
            "monitoring": monitoring_count
        },
        "alerts": {
            "total": alert_count,
            "by_type": violation_types
        },
        "compliance_score": round(compliance_score, 1)
    }


@router.get("/cameras/{camera_id}/latest-frame")
def get_latest_frame(
    camera_id: int,
    format: str = "jpeg",
    quality: int = 90,
    width: Optional[int] = None,
    height: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get latest processed frame from camera."""
    # Check if camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Get processor
    processor = get_processor(camera_id)
    if not processor:
        raise HTTPException(
            status_code=404, 
            detail="Camera not currently processing"
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
        _, img_encoded = cv2.imencode(
            ".jpg", 
            frame, 
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        response = Response(content=img_encoded.tobytes())
        response.headers["Content-Type"] = "image/jpeg"
        return response
        
    elif format.lower() == "png":
        # Encode as PNG
        _, img_encoded = cv2.imencode(".png", frame)
        response = Response(content=img_encoded.tobytes())
        response.headers["Content-Type"] = "image/png"
        return response
        
    elif format.lower() == "base64":
        # Encode as base64
        _, img_encoded = cv2.imencode(
            ".jpg", 
            frame, 
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        base64_str = base64.b64encode(img_encoded).decode("utf-8")
        return {"image": base64_str, "format": "jpeg"}
    
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format: {format}"
        )


@router.get("/timeline")
def get_alert_timeline(
    hours: int = 24,
    interval_minutes: int = 60,
    camera_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get alert timeline data for charts."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    # Generate time intervals
    intervals = []
    current_time = start_time
    
    while current_time < end_time:
        interval_end = current_time + timedelta(minutes=interval_minutes)
        if interval_end > end_time:
            interval_end = end_time
            
        intervals.append({
            "start": current_time,
            "end": interval_end,
            "label": current_time.strftime("%H:%M")
        })
        
        current_time = interval_end
    
    # Query alerts for each interval
    timeline_data = []
    
    for interval in intervals:
        # Base query for this interval
        query = db.query(Alert).filter(
            Alert.created_at >= interval["start"],
            Alert.created_at < interval["end"]
        )
        
        if camera_id is not None:
            query = query.filter(Alert.camera_id == camera_id)
        
        # Count total alerts
        count = query.count()
        
        # Count by type
        types = {}
        violation_types = db.query(Alert.violation_type).distinct().all()
        
        for type_row in violation_types:
            vtype = type_row[0]
            type_count = query.filter(Alert.violation_type == vtype).count()
            types[vtype] = type_count
        
        timeline_data.append({
            "time": interval["label"],
            "timestamp": interval["start"].isoformat(),
            "total": count,
            "by_type": types
        })
    
    return {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "interval_minutes": interval_minutes,
        "timeline": timeline_data
    }