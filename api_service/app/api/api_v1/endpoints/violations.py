"""
Safety violations API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional, Dict, Any
import io
import csv
import json
from datetime import datetime, timedelta

from app.core.auth import get_current_active_user
from app.core.exceptions import NotFoundError
from app.db.database import get_db
from app.models.schemas.violation import (
    ViolationCreate,
    ViolationResponse,
    ViolationDetail,
    ViolationFilterParams,
    ViolationStatusUpdate,
    ViolationTrendsResponse,
    ViolationHotspotsResponse
)
from app.models.violation import Violation
from app.models.detection import Detection
from app.models.camera import Camera
from app.models.user import User


router = APIRouter()


@router.post(
    "",
    response_model=ViolationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Violation",
    description="Create a new safety violation record."
)
async def create_violation(
    violation_in: ViolationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new safety violation record.
    
    This endpoint is typically called by the detection service
    when a safety violation is detected.
    """
    # First, check if the detection exists
    detection = db.query(Detection).filter(Detection.id == violation_in.detection_id).first()
    if not detection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Detection not found"
        )
    
    # Create new violation
    new_violation = Violation(
        detection_id=violation_in.detection_id,
        violation_type=violation_in.violation_type,
        confidence=violation_in.confidence,
        bounding_box=violation_in.bounding_box,
        severity=violation_in.severity,
        status="open",
        timestamp=detection.timestamp  # Use the detection timestamp
    )
    
    db.add(new_violation)
    db.commit()
    db.refresh(new_violation)
    
    return new_violation


@router.get(
    "",
    response_model=List[ViolationResponse],
    status_code=status.HTTP_200_OK,
    summary="List Violations",
    description="Get a paginated list of safety violations with filtering options."
)
async def list_violations(
    params: ViolationFilterParams = Depends(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a paginated list of safety violations with filtering options.
    """
    # Start with base query
    query = db.query(Violation)
    
    # Apply filters
    if params.camera_id:
        # This requires a join with Detection table
        query = query.join(Detection, Violation.detection_id == Detection.id)
        query = query.filter(Detection.camera_id == params.camera_id)
    
    if params.violation_type:
        query = query.filter(Violation.violation_type == params.violation_type)
    
    if params.status:
        query = query.filter(Violation.status == params.status)
    
    if params.severity:
        query = query.filter(Violation.severity == params.severity)
    
    if params.min_confidence is not None:
        query = query.filter(Violation.confidence >= params.min_confidence)
    
    # Apply time range filter if provided
    if params.time_range:
        if params.time_range.start_date:
            query = query.filter(Violation.timestamp >= params.time_range.start_date)
        if params.time_range.end_date:
            query = query.filter(Violation.timestamp <= params.time_range.end_date)
    
    # Calculate total items for pagination info
    total = query.count()
    
    # Apply sorting
    if params.sort_by:
        sort_column = getattr(Violation, params.sort_by, Violation.timestamp)
        if params.sort_desc:
            sort_column = desc(sort_column)
        query = query.order_by(sort_column)
    else:
        # Default sort by timestamp descending (newest first)
        query = query.order_by(desc(Violation.timestamp))
    
    # Apply pagination
    query = query.offset((params.page - 1) * params.page_size).limit(params.page_size)
    
    # Execute query
    violations = query.all()
    
    return violations


@router.get(
    "/{violation_id}",
    response_model=ViolationDetail,
    status_code=status.HTTP_200_OK,
    summary="Get Violation",
    description="Get detailed information about a specific safety violation."
)
async def get_violation(
    violation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get detailed information about a specific safety violation.
    """
    # Get violation with join to Detection
    violation = db.query(Violation).filter(Violation.id == violation_id).first()
    if not violation:
        raise NotFoundError("Violation not found")
    
    # Get associated detection
    detection = db.query(Detection).filter(Detection.id == violation.detection_id).first()
    if not detection:
        raise NotFoundError("Associated detection not found")
    
    # Get camera info if available
    camera = None
    if detection.camera_id:
        camera = db.query(Camera).filter(Camera.id == detection.camera_id).first()
    
    # Create violation detail response
    result = ViolationDetail.from_orm(violation)
    
    # Generate S3 pre-signed URL for the detection image if S3 URL
    if detection.image_path and detection.image_path.startswith('s3://'):
        from app.services.aws import get_s3_presigned_url
        try:
            bucket, key = detection.image_path[5:].split('/', 1)
            result.detection_image_url = get_s3_presigned_url(bucket, key)
            
            # Generate cropped violation image if bounding box is available
            # This would require additional image processing service
            # We'll skip this for now and implement it later
        except Exception as e:
            # Log error but continue without image URL
            print(f"Error generating pre-signed URL: {str(e)}")
    
    # Add camera information if available
    if detection.camera_id:
        result.camera_id = detection.camera_id
        if camera:
            result.camera_name = camera.name
    
    return result


@router.patch(
    "/{violation_id}/status",
    response_model=ViolationResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Violation Status",
    description="Update the status of a safety violation."
)
async def update_violation_status(
    violation_id: str,
    status_update: ViolationStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update the status of a safety violation.
    
    This endpoint allows marking violations as acknowledged or resolved.
    """
    # Get violation
    violation = db.query(Violation).filter(Violation.id == violation_id).first()
    if not violation:
        raise NotFoundError("Violation not found")
    
    # Update status
    violation.status = status_update.status
    
    # Add resolution notes if provided
    if status_update.resolution_notes:
        violation.resolution_notes = status_update.resolution_notes
    
    # Set resolution time if resolving
    if status_update.status == "resolved":
        violation.resolution_time = datetime.now()
    
    db.add(violation)
    db.commit()
    db.refresh(violation)
    
    return violation


@router.get(
    "/trends",
    response_model=ViolationTrendsResponse,
    status_code=status.HTTP_200_OK,
    summary="Violation Trends",
    description="Get trends of violations over time."
)
async def get_violation_trends(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    camera_id: Optional[str] = None,
    time_unit: str = Query("day", description="Time unit for aggregation: day, week, month"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get trends of violations over time.
    """
    # Default to last 30 days if no date range provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    # Base query
    query = db.query(Violation).filter(
        Violation.timestamp >= start_date,
        Violation.timestamp <= end_date
    )
    
    # Apply camera filter if provided
    if camera_id:
        query = query.join(Detection, Violation.detection_id == Detection.id)
        query = query.filter(Detection.camera_id == camera_id)
    
    # Get all violations in the time range
    violations = query.all()
    
    # Group by time period and violation type
    period_data = {}
    violation_types = {}
    
    for violation in violations:
        # Format period based on time_unit
        if time_unit == "day":
            period = violation.timestamp.strftime("%Y-%m-%d")
        elif time_unit == "week":
            # ISO week format (year-week)
            period = violation.timestamp.strftime("%Y-W%W")
        elif time_unit == "month":
            period = violation.timestamp.strftime("%Y-%m")
        else:
            # Default to day
            period = violation.timestamp.strftime("%Y-%m-%d")
        
        # Initialize period data if not exists
        if period not in period_data:
            period_data[period] = {
                "period": period,
                "count": 0,
                "by_type": {}
            }
        
        # Increment period count
        period_data[period]["count"] += 1
        
        # Increment by violation type
        v_type = violation.violation_type
        if v_type not in period_data[period]["by_type"]:
            period_data[period]["by_type"][v_type] = 0
        period_data[period]["by_type"][v_type] += 1
        
        # Track violation types for most common
        if v_type not in violation_types:
            violation_types[v_type] = 0
        violation_types[v_type] += 1
    
    # Convert to list sorted by period
    trends = list(period_data.values())
    trends.sort(key=lambda x: x["period"])
    
    # Calculate most common violation type
    most_common_type = max(violation_types.items(), key=lambda x: x[1])[0] if violation_types else None
    
    return {
        "trends": trends,
        "total_violations": len(violations),
        "most_common_type": most_common_type,
        "start_date": start_date,
        "end_date": end_date
    }


@router.get(
    "/hotspots",
    response_model=ViolationHotspotsResponse,
    status_code=status.HTTP_200_OK,
    summary="Violation Hotspots",
    description="Get violation hotspots by camera location."
)
async def get_violation_hotspots(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    area_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get violation hotspots by camera location.
    """
    # Default to last 30 days if no date range provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    # This query requires joins between Detection and Violation tables
    # We'll do it in Python for clarity, but in production this should be a SQL query
    
    # Get all cameras
    camera_query = db.query(Camera)
    if area_id:
        camera_query = camera_query.filter(Camera.area_id == area_id)
    cameras = camera_query.all()
    
    # Process each camera
    hotspots = []
    total_cameras = len(cameras)
    total_violations = 0
    
    for camera in cameras:
        # Get detections for this camera
        detections = (db.query(Detection)
                     .filter(Detection.camera_id == camera.id)
                     .filter(Detection.timestamp >= start_date)
                     .filter(Detection.timestamp <= end_date)
                     .all())
        
        # Get detection IDs
        detection_ids = [d.id for d in detections]
        
        # If no detections, skip this camera
        if not detection_ids:
            continue
        
        # Get violations for these detections
        violations = (db.query(Violation)
                     .filter(Violation.detection_id.in_(detection_ids))
                     .all())
        
        # If no violations, skip this camera
        violation_count = len(violations)
        if violation_count == 0:
            continue
        
        # Count violations by type
        violation_types = {}
        severity_distribution = {"low": 0, "medium": 0, "high": 0}
        
        for v in violations:
            if v.violation_type not in violation_types:
                violation_types[v.violation_type] = 0
            violation_types[v.violation_type] += 1
            
            severity_distribution[v.severity] += 1
        
        # Find most common violation type
        most_common_type = max(violation_types.items(), key=lambda x: x[1])[0] if violation_types else None
        
        # Calculate violation rate
        detection_count = len(detections)
        violation_rate = violation_count / detection_count if detection_count > 0 else 0
        
        # Add to hotspots
        hotspots.append({
            "camera_id": camera.id,
            "camera_name": camera.name,
            "count": violation_count,
            "most_common_type": most_common_type,
            "severity_distribution": severity_distribution,
            "violation_rate": violation_rate
        })
        
        total_violations += violation_count
    
    # Sort hotspots by violation count, descending
    hotspots.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "hotspots": hotspots,
        "total_cameras": total_cameras,
        "total_violations": total_violations
    }


@router.get(
    "/export",
    status_code=status.HTTP_200_OK,
    summary="Export Violations",
    description="Export violation data in CSV or JSON format."
)
async def export_violations(
    format: str = Query("csv", description="Export format (csv or json)"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    status: Optional[str] = None,
    camera_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Export violation data in CSV or JSON format.
    """
    # Default to last 30 days if no date range provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    # Base query with date range filter
    query = db.query(Violation).filter(
        Violation.timestamp >= start_date,
        Violation.timestamp <= end_date
    )
    
    # Apply status filter if provided
    if status:
        query = query.filter(Violation.status == status)
    
    # Apply camera filter if provided
    if camera_id:
        query = query.join(Detection, Violation.detection_id == Detection.id)
        query = query.filter(Detection.camera_id == camera_id)
    
    # Order by timestamp
    query = query.order_by(Violation.timestamp)
    
    # Execute query
    violations = query.all()
    
    # Get detection ids to fetch camera information
    detection_ids = [v.detection_id for v in violations]
    
    # Get detections with camera info
    detections = (db.query(Detection)
                 .filter(Detection.id.in_(detection_ids))
                 .all())
    
    # Create lookup dict for detections
    detection_map = {d.id: d for d in detections}
    
    # Get camera ids
    camera_ids = {d.camera_id for d in detections if d.camera_id}
    
    # Get cameras
    cameras = db.query(Camera).filter(Camera.id.in_(camera_ids)).all()
    
    # Create lookup dict for cameras
    camera_map = {c.id: c for c in cameras}
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sift_violations_{timestamp}"
    
    if format.lower() == "csv":
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        header = [
            "id", "violation_type", "timestamp", "status", "severity", 
            "confidence", "camera_id", "camera_name", "detection_id"
        ]
        
        writer.writerow(header)
        
        # Write data
        for violation in violations:
            detection = detection_map.get(violation.detection_id)
            camera_id = detection.camera_id if detection else None
            camera_name = camera_map.get(camera_id).name if camera_id in camera_map else None
            
            row = [
                violation.id, 
                violation.violation_type,
                violation.timestamp.isoformat(),
                violation.status,
                violation.severity,
                violation.confidence,
                camera_id,
                camera_name,
                violation.detection_id
            ]
            
            writer.writerow(row)
        
        # Create response
        output.seek(0)
        
        # Convert StringIO to BytesIO for response
        bytes_output = io.BytesIO(output.getvalue().encode('utf-8'))
        
        response = StreamingResponse(
            bytes_output,
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = f"attachment; filename={filename}.csv"
        return response
    
    elif format.lower() == "json":
        # Convert to JSON
        result = []
        for violation in violations:
            detection = detection_map.get(violation.detection_id)
            camera_id = detection.camera_id if detection else None
            camera_name = camera_map.get(camera_id).name if camera_id in camera_map else None
            
            item = {
                "id": violation.id,
                "violation_type": violation.violation_type,
                "timestamp": violation.timestamp.isoformat(),
                "status": violation.status,
                "severity": violation.severity,
                "confidence": violation.confidence,
                "bounding_box": violation.bounding_box,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "detection_id": violation.detection_id,
                "resolution_notes": violation.resolution_notes,
                "resolution_time": violation.resolution_time.isoformat() if violation.resolution_time else None
            }
            
            result.append(item)
        
        # Convert to JSON string
        json_data = json.dumps(result)
        
        # Create response
        response = StreamingResponse(
            io.BytesIO(json_data.encode('utf-8')),
            media_type="application/json"
        )
        response.headers["Content-Disposition"] = f"attachment; filename={filename}.json"
        return response
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported format. Use 'csv' or 'json'."
        )
