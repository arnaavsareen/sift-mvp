"""
Schemas for PPE detection endpoints.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from app.models.schemas.common import FilterParams


class DetectionResult(BaseModel):
    """Schema for individual detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in normalized coordinates
    
    class Config:
        schema_extra = {
            "example": {
                "class_id": 0,
                "class_name": "helmet",
                "confidence": 0.92,
                "bbox": [0.214, 0.342, 0.278, 0.462]
            }
        }


class DetectionBase(BaseModel):
    """Base detection schema."""
    image_id: str
    image_path: str
    timestamp: datetime
    camera_id: Optional[str] = None
    num_detections: int
    ppe_detected: bool
    violations_detected: bool
    confidence_threshold: float
    model_version: str
    processing_time: float


class DetectionCreate(DetectionBase):
    """Schema for creating a new detection."""
    detection_results: List[DetectionResult]
    
    class Config:
        schema_extra = {
            "example": {
                "image_id": "frame_20250506_010825_camera1",
                "image_path": "s3://osha-mvp-bucket/frames/frame_20250506_010825_camera1.jpg",
                "timestamp": "2025-05-06T01:08:25.123Z",
                "camera_id": "camera1",
                "num_detections": 2,
                "ppe_detected": True,
                "violations_detected": False,
                "confidence_threshold": 0.5,
                "model_version": "yolov8n.pt",
                "processing_time": 0.354,
                "detection_results": [
                    {
                        "class_id": 0,
                        "class_name": "helmet",
                        "confidence": 0.92,
                        "bbox": [0.214, 0.342, 0.278, 0.462]
                    },
                    {
                        "class_id": 1,
                        "class_name": "vest",
                        "confidence": 0.88,
                        "bbox": [0.187, 0.385, 0.342, 0.591]
                    }
                ]
            }
        }


class DetectionResponse(DetectionBase):
    """Schema for detection API response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    detection_results: List[DetectionResult]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class DetectionDetail(DetectionResponse):
    """Schema for detailed detection response with S3 URLs."""
    image_url: Optional[str] = None
    camera_name: Optional[str] = None
    violations: List[Dict[str, Any]] = []
    
    class Config:
        orm_mode = True


class DetectionFilterParams(FilterParams):
    """Filter parameters for detections."""
    camera_id: Optional[str] = None
    ppe_detected: Optional[bool] = None
    violations_detected: Optional[bool] = None
    detection_type: Optional[str] = None
    min_confidence: Optional[float] = None


class DetectionStatsByType(BaseModel):
    """Statistics of detections by type."""
    type: str
    count: int
    percentage: float


class DetectionStatsByCamera(BaseModel):
    """Statistics of detections by camera."""
    camera_id: str
    camera_name: str
    count: int
    violation_count: int
    compliance_rate: float


class DetectionStatsByTimeRange(BaseModel):
    """Statistics of detections by time period."""
    period: str  # e.g., "2025-05-06" for daily, "2025-05" for monthly
    count: int
    violation_count: int


class DetectionStatistics(BaseModel):
    """Aggregated statistics for detections."""
    total_detections: int
    total_violations: int
    compliance_rate: float
    by_type: List[DetectionStatsByType] = []
    by_camera: List[DetectionStatsByCamera] = []
    by_time: List[DetectionStatsByTimeRange] = []
    start_date: datetime
    end_date: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "total_detections": 1250,
                "total_violations": 78,
                "compliance_rate": 0.9376,
                "by_type": [
                    {"type": "helmet", "count": 950, "percentage": 76.0},
                    {"type": "vest", "count": 750, "percentage": 60.0}
                ],
                "by_camera": [
                    {
                        "camera_id": "camera1",
                        "camera_name": "Building A Entrance",
                        "count": 450,
                        "violation_count": 28,
                        "compliance_rate": 0.9378
                    }
                ],
                "by_time": [
                    {
                        "period": "2025-05-05",
                        "count": 250,
                        "violation_count": 15
                    }
                ],
                "start_date": "2025-05-01T00:00:00Z",
                "end_date": "2025-05-06T23:59:59Z"
            }
        }
