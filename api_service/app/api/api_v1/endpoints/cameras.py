"""
Camera management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.core.auth import get_current_active_user, get_current_user
from app.core.exceptions import NotFoundError
from app.db.database import get_db
from app.models.schemas.camera import (
    CameraCreate, 
    CameraResponse, 
    CameraWithStats,
    CameraFilterParams,
    CameraStatusUpdate,
    CameraViolationTimeline
)
from app.models.camera import Camera
from app.models.detection import Detection
from app.models.user import User


router = APIRouter()


@router.post(
    "",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Camera",
    description="Create a new camera."
)
async def create_camera(
    camera_in: CameraCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new camera in the system.
    """
    # Create new camera object
    new_camera = Camera(
        name=camera_in.name,
        description=camera_in.description,
        location=camera_in.location,
        area_id=camera_in.area_id,
        stream_url=camera_in.stream_url,
        position_x=camera_in.position_x,
        position_y=camera_in.position_y,
        floor=camera_in.floor,
        coverage_radius=camera_in.coverage_radius,
        coverage_angle=camera_in.coverage_angle,
        connection_details=camera_in.connection_details,
        is_active=True,
        status="offline"
    )
    
    db.add(new_camera)
    db.commit()
    db.refresh(new_camera)
    
    return new_camera


@router.get(
    "",
    response_model=List[CameraResponse],
    status_code=status.HTTP_200_OK,
    summary="List Cameras",
    description="Get a paginated list of cameras with filtering options."
)
async def list_cameras(
    params: CameraFilterParams = Depends(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a paginated list of cameras with filtering options.
    """
    # Start with base query
    query = db.query(Camera)
    
    # Apply filters
    if params.area_id:
        query = query.filter(Camera.area_id == params.area_id)
    
    if params.is_active is not None:
        query = query.filter(Camera.is_active == params.is_active)
    
    if params.status:
        query = query.filter(Camera.status == params.status)
    
    if params.floor:
        query = query.filter(Camera.floor == params.floor)
    
    # Apply search if provided (search in name, description, or location)
    if params.search:
        search_term = f"%{params.search}%"
        query = query.filter(
            (Camera.name.ilike(search_term)) | 
            (Camera.description.ilike(search_term)) |
            (Camera.location.ilike(search_term))
        )
    
    # Calculate total items for pagination info
    total = query.count()
    
    # Apply sorting
    if params.sort_by:
        sort_column = getattr(Camera, params.sort_by, Camera.name)
        if params.sort_desc:
            sort_column = desc(sort_column)
        query = query.order_by(sort_column)
    else:
        # Default sort by name
        query = query.order_by(Camera.name)
    
    # Apply pagination
    query = query.offset((params.page - 1) * params.page_size).limit(params.page_size)
    
    # Execute query
    cameras = query.all()
    
    return cameras


@router.get(
    "/stats",
    response_model=List[CameraWithStats],
    status_code=status.HTTP_200_OK,
    summary="Camera Statistics",
    description="Get list of cameras with detection statistics."
)
async def get_camera_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    area_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get list of cameras with detection statistics for a given time period.
    """
    # Default to last 30 days if no date range provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    # Get all cameras first
    query = db.query(Camera)
    
    # Apply area filter if provided
    if area_id:
        query = query.filter(Camera.area_id == area_id)
    
    # Get cameras
    cameras = query.all()
    
    # Calculate stats for each camera
    results = []
    for camera in cameras:
        # Get detections for this camera
        detection_query = (db.query(Detection)
                           .filter(Detection.camera_id == camera.id)
                           .filter(Detection.timestamp >= start_date)
                           .filter(Detection.timestamp <= end_date))
        
        detection_count = detection_query.count()
        violation_count = detection_query.filter(Detection.violations_detected == True).count()
        
        # Calculate compliance rate
        compliance_rate = 1.0 - (violation_count / detection_count if detection_count > 0 else 0)
        
        # Get last detection and violation
        last_detection = (detection_query.order_by(desc(Detection.timestamp)).first())
        last_violation = (detection_query.filter(Detection.violations_detected == True)
                          .order_by(desc(Detection.timestamp)).first())
                          
        # Convert to CameraWithStats
        camera_stats = CameraWithStats(
            **camera.__dict__,
            detection_count=detection_count,
            violation_count=violation_count,
            compliance_rate=compliance_rate,
            last_detection=last_detection.timestamp if last_detection else None,
            last_violation=last_violation.timestamp if last_violation else None
        )
        
        results.append(camera_stats)
    
    return results


@router.get(
    "/{camera_id}",
    response_model=CameraWithStats,
    status_code=status.HTTP_200_OK,
    summary="Get Camera",
    description="Get detailed information about a specific camera with statistics."
)
async def get_camera(
    camera_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get detailed information about a specific camera with statistics.
    """
    # Default to last 30 days for stats if no date range provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    # Get camera
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise NotFoundError("Camera not found")
    
    # Get detections for this camera
    detection_query = (db.query(Detection)
                       .filter(Detection.camera_id == camera_id)
                       .filter(Detection.timestamp >= start_date)
                       .filter(Detection.timestamp <= end_date))
    
    detection_count = detection_query.count()
    violation_count = detection_query.filter(Detection.violations_detected == True).count()
    
    # Calculate compliance rate
    compliance_rate = 1.0 - (violation_count / detection_count if detection_count > 0 else 0)
    
    # Get last detection and violation
    last_detection = (detection_query.order_by(desc(Detection.timestamp)).first())
    last_violation = (detection_query.filter(Detection.violations_detected == True)
                      .order_by(desc(Detection.timestamp)).first())
                      
    # Convert to CameraWithStats
    camera_stats = CameraWithStats(
        **camera.__dict__,
        detection_count=detection_count,
        violation_count=violation_count,
        compliance_rate=compliance_rate,
        last_detection=last_detection.timestamp if last_detection else None,
        last_violation=last_violation.timestamp if last_violation else None
    )
    
    return camera_stats


@router.put(
    "/{camera_id}",
    response_model=CameraResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Camera",
    description="Update an existing camera."
)
async def update_camera(
    camera_id: str,
    camera_in: CameraCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update an existing camera.
    """
    # Get camera
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise NotFoundError("Camera not found")
    
    # Update camera fields
    camera.name = camera_in.name
    camera.description = camera_in.description
    camera.location = camera_in.location
    camera.area_id = camera_in.area_id
    camera.stream_url = camera_in.stream_url
    camera.position_x = camera_in.position_x
    camera.position_y = camera_in.position_y
    camera.floor = camera_in.floor
    camera.coverage_radius = camera_in.coverage_radius
    camera.coverage_angle = camera_in.coverage_angle
    
    if camera_in.connection_details:
        camera.connection_details = camera_in.connection_details
    
    db.add(camera)
    db.commit()
    db.refresh(camera)
    
    return camera


@router.patch(
    "/{camera_id}/status",
    response_model=CameraResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Camera Status",
    description="Update the status of a camera."
)
async def update_camera_status(
    camera_id: str,
    status_update: CameraStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update the status of a camera.
    """
    # Get camera
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise NotFoundError("Camera not found")
    
    # Update status fields
    if status_update.is_active is not None:
        camera.is_active = status_update.is_active
    
    if status_update.status is not None:
        camera.status = status_update.status
    
    if status_update.last_seen is not None:
        camera.last_seen = status_update.last_seen
    
    db.add(camera)
    db.commit()
    db.refresh(camera)
    
    return camera


@router.delete(
    "/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Camera",
    description="Delete a specific camera."
)
async def delete_camera(
    camera_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a camera.
    Note: This doesn't delete associated detection records.
    """
    # Get camera
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise NotFoundError("Camera not found")
    
    db.delete(camera)
    db.commit()
    return None


@router.get(
    "/{camera_id}/timeline",
    response_model=CameraViolationTimeline,
    status_code=status.HTTP_200_OK,
    summary="Camera Violation Timeline",
    description="Get violation timeline data for a specific camera."
)
async def get_camera_timeline(
    camera_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    time_unit: str = Query("hour", description="Time unit for aggregation: hour, day, week, month"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get violation timeline data for a specific camera.
    """
    # Default to last 7 days if no date range provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)
    if end_date is None:
        end_date = datetime.now()
    
    # Get camera
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise NotFoundError("Camera not found")
    
    # This query requires database-specific date truncation functions
    # Here's a simplified approach for PostgreSQL
    # For other databases, adjust the query accordingly
    
    time_periods = []
    
    # For simplicity, we'll use a Python approach rather than complex SQL
    # In production, you'd want to optimize this with proper SQL aggregation
    
    # Get all detections for this camera in the time range
    detections = (db.query(Detection)
                  .filter(Detection.camera_id == camera_id)
                  .filter(Detection.timestamp >= start_date)
                  .filter(Detection.timestamp <= end_date)
                  .order_by(Detection.timestamp)
                  .all())
    
    # Group by time period
    period_data = {}
    
    for detection in detections:
        # Format period based on time_unit
        if time_unit == "hour":
            period = detection.timestamp.strftime("%Y-%m-%d-%H:00")
        elif time_unit == "day":
            period = detection.timestamp.strftime("%Y-%m-%d")
        elif time_unit == "week":
            # ISO week format (year-week)
            period = detection.timestamp.strftime("%Y-W%W")
        elif time_unit == "month":
            period = detection.timestamp.strftime("%Y-%m")
        else:
            # Default to day
            period = detection.timestamp.strftime("%Y-%m-%d")
        
        if period not in period_data:
            period_data[period] = {
                "period": period,
                "detection_count": 0,
                "violation_count": 0,
                "compliance_rate": 0
            }
        
        period_data[period]["detection_count"] += 1
        if detection.violations_detected:
            period_data[period]["violation_count"] += 1
    
    # Calculate compliance rate for each period
    for period, data in period_data.items():
        detection_count = data["detection_count"]
        violation_count = data["violation_count"]
        if detection_count > 0:
            data["compliance_rate"] = round(1.0 - (violation_count / detection_count), 2)
    
    # Convert to list sorted by period
    time_periods = list(period_data.values())
    time_periods.sort(key=lambda x: x["period"])
    
    return CameraViolationTimeline(
        camera_id=camera_id,
        camera_name=camera.name,
        time_periods=time_periods
    )
