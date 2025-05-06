"""
Detection API endpoints for PPE detection results.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional
import io
import csv
import json
from datetime import datetime, timedelta

from app.core.auth import get_current_active_user
from app.core.exceptions import NotFoundError
from app.db.database import get_db
from app.models.schemas.detection import (
    DetectionCreate, 
    DetectionResponse, 
    DetectionDetail,
    DetectionFilterParams,
    DetectionStatistics
)
from app.models.detection import Detection
from app.models.user import User


router = APIRouter()


@router.post(
    "",
    response_model=DetectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Detection",
    description="Create a new detection record."
)
async def create_detection(
    detection_in: DetectionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new detection record.
    
    This endpoint is typically called by the detection service
    after processing images from the SQS queue.
    """
    # Convert Pydantic model to database model
    new_detection = Detection(
        image_id=detection_in.image_id,
        image_path=detection_in.image_path,
        timestamp=detection_in.timestamp,
        camera_id=detection_in.camera_id,
        num_detections=detection_in.num_detections,
        ppe_detected=detection_in.ppe_detected,
        violations_detected=detection_in.violations_detected,
        confidence_threshold=detection_in.confidence_threshold,
        model_version=detection_in.model_version,
        processing_time=detection_in.processing_time,
        detection_results=detection_in.detection_results
    )
    
    db.add(new_detection)
    db.commit()
    db.refresh(new_detection)
    
    return new_detection


@router.get(
    "",
    response_model=List[DetectionResponse],
    status_code=status.HTTP_200_OK,
    summary="List Detections",
    description="Get a paginated list of detection records with filtering options."
)
async def list_detections(
    params: DetectionFilterParams = Depends(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a paginated list of detection records with filtering options.
    """
    # Start with base query
    query = db.query(Detection)
    
    # Apply filters
    if params.camera_id:
        query = query.filter(Detection.camera_id == params.camera_id)
    
    if params.ppe_detected is not None:
        query = query.filter(Detection.ppe_detected == params.ppe_detected)
    
    if params.violations_detected is not None:
        query = query.filter(Detection.violations_detected == params.violations_detected)
    
    if params.min_confidence is not None:
        query = query.filter(Detection.confidence_threshold >= params.min_confidence)
    
    # Apply time range filter if provided
    if params.time_range:
        if params.time_range.start_date:
            query = query.filter(Detection.timestamp >= params.time_range.start_date)
        if params.time_range.end_date:
            query = query.filter(Detection.timestamp <= params.time_range.end_date)
    
    # Apply search if provided (search in image_id or camera_id)
    if params.search:
        search_term = f"%{params.search}%"
        query = query.filter(
            (Detection.image_id.ilike(search_term)) | 
            (Detection.camera_id.ilike(search_term))
        )
    
    # Calculate total items for pagination info
    total = query.count()
    
    # Apply sorting
    if params.sort_by:
        sort_column = getattr(Detection, params.sort_by, Detection.timestamp)
        if params.sort_desc:
            sort_column = desc(sort_column)
        query = query.order_by(sort_column)
    else:
        # Default sort by timestamp descending (newest first)
        query = query.order_by(desc(Detection.timestamp))
    
    # Apply pagination
    query = query.offset((params.page - 1) * params.page_size).limit(params.page_size)
    
    # Execute query
    detections = query.all()
    
    return detections


@router.get(
    "/{detection_id}",
    response_model=DetectionDetail,
    status_code=status.HTTP_200_OK,
    summary="Get Detection",
    description="Get detailed information about a specific detection."
)
async def get_detection(
    detection_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get detailed information about a specific detection.
    """
    detection = db.query(Detection).filter(Detection.id == detection_id).first()
    if not detection:
        raise NotFoundError("Detection not found")
    
    # Convert to DetectionDetail response
    result = DetectionDetail.from_orm(detection)
    
    # Generate S3 pre-signed URL for the image if S3 URL
    if detection.image_path and detection.image_path.startswith('s3://'):
        from app.services.aws import get_s3_presigned_url
        try:
            bucket, key = detection.image_path[5:].split('/', 1)
            result.image_url = get_s3_presigned_url(bucket, key)
        except Exception as e:
            # Log error but continue without image URL
            print(f"Error generating pre-signed URL: {str(e)}")
    
    # Get associated violations if any
    # This would require a join with violations table
    # We'll implement this after creating the violations endpoints
    
    return result


@router.get(
    "/statistics",
    response_model=DetectionStatistics,
    status_code=status.HTTP_200_OK,
    summary="Detection Statistics",
    description="Get aggregated statistics for detections."
)
async def get_detection_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    camera_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get aggregated statistics for detections.
    """
    # Default to last 30 days if no date range provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    # Base query with date range filter
    query = db.query(Detection).filter(
        Detection.timestamp >= start_date,
        Detection.timestamp <= end_date
    )
    
    # Apply camera filter if provided
    if camera_id:
        query = query.filter(Detection.camera_id == camera_id)
    
    # Calculate basic metrics
    total_detections = query.count()
    violations_query = query.filter(Detection.violations_detected == True)
    total_violations = violations_query.count()
    
    # Calculate compliance rate
    compliance_rate = 1.0 - (total_violations / total_detections if total_detections > 0 else 0)
    
    # Initialize statistics object
    statistics = {
        "total_detections": total_detections,
        "total_violations": total_violations,
        "compliance_rate": compliance_rate,
        "by_type": [],
        "by_camera": [],
        "by_time": [],
        "start_date": start_date,
        "end_date": end_date
    }
    
    # Get statistics by detection type
    # We need to unpack the detection_results JSON and group by class_name
    # This is a complex DB query that might be database-specific
    # For now, we'll skip this and implement it later
    
    # Get statistics by camera
    camera_stats = (
        db.query(
            Detection.camera_id,
            func.count().label('count'),
            func.sum(Detection.violations_detected.cast(int)).label('violation_count')
        )
        .filter(
            Detection.timestamp >= start_date,
            Detection.timestamp <= end_date
        )
        .group_by(Detection.camera_id)
        .all()
    )
    
    for camera in camera_stats:
        camera_compliance = 1.0 - (camera.violation_count / camera.count if camera.count > 0 else 0)
        statistics["by_camera"].append({
            "camera_id": camera.camera_id,
            "camera_name": camera.camera_id,  # We'll replace with actual name when we have camera data
            "count": camera.count,
            "violation_count": camera.violation_count,
            "compliance_rate": camera_compliance
        })
    
    # Get statistics by time (daily)
    # This query is also database-specific and might need customization
    # We'll skip this for now and implement it later
    
    return DetectionStatistics(**statistics)


@router.get(
    "/export",
    status_code=status.HTTP_200_OK,
    summary="Export Detections",
    description="Export detection data in CSV or JSON format."
)
async def export_detections(
    format: str = Query("csv", description="Export format (csv or json)"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    camera_id: Optional[str] = None,
    include_results: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Export detection data in CSV or JSON format.
    """
    # Default to last 30 days if no date range provided
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    # Base query with date range filter
    query = db.query(Detection).filter(
        Detection.timestamp >= start_date,
        Detection.timestamp <= end_date
    )
    
    # Apply camera filter if provided
    if camera_id:
        query = query.filter(Detection.camera_id == camera_id)
    
    # Order by timestamp
    query = query.order_by(Detection.timestamp)
    
    # Execute query
    detections = query.all()
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sift_detections_{timestamp}"
    
    if format.lower() == "csv":
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        header = [
            "id", "image_id", "timestamp", "camera_id",
            "num_detections", "ppe_detected", "violations_detected",
            "confidence_threshold", "model_version", "processing_time"
        ]
        if include_results:
            header.append("detection_results")
        
        writer.writerow(header)
        
        # Write data
        for detection in detections:
            row = [
                detection.id, detection.image_id, detection.timestamp.isoformat(),
                detection.camera_id, detection.num_detections, detection.ppe_detected,
                detection.violations_detected, detection.confidence_threshold,
                detection.model_version, detection.processing_time
            ]
            
            if include_results:
                row.append(json.dumps(detection.detection_results))
            
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
        for detection in detections:
            item = {
                "id": detection.id,
                "image_id": detection.image_id,
                "timestamp": detection.timestamp.isoformat(),
                "camera_id": detection.camera_id,
                "num_detections": detection.num_detections,
                "ppe_detected": detection.ppe_detected,
                "violations_detected": detection.violations_detected,
                "confidence_threshold": detection.confidence_threshold,
                "model_version": detection.model_version,
                "processing_time": detection.processing_time
            }
            
            if include_results:
                item["detection_results"] = detection.detection_results
            
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
