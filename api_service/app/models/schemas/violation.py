"""
Schemas for safety violation endpoints.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

from app.models.schemas.common import FilterParams


class BoundingBox(BaseModel):
    """Schema for bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def as_list(self) -> List[float]:
        """Return bounding box as a list."""
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_list(cls, bbox: List[float]):
        """Create a BoundingBox from a list of coordinates."""
        if len(bbox) != 4:
            raise ValueError("Bounding box must have exactly 4 coordinates")
        return cls(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])


class ViolationBase(BaseModel):
    """Base violation schema."""
    detection_id: str
    violation_type: str
    confidence: float
    bounding_box: List[float]  # [x1, y1, x2, y2] normalized
    severity: str = "medium"
    status: str = "open"
    timestamp: datetime


class ViolationCreate(BaseModel):
    """Schema for creating a new violation."""
    detection_id: str
    violation_type: str
    confidence: float
    bounding_box: List[float]
    severity: str = "medium"
    
    @validator('violation_type')
    def validate_violation_type(cls, v):
        valid_types = ["no_helmet", "no_vest", "no_gloves", "no_goggles", "no_mask", "other"]
        if v not in valid_types:
            raise ValueError(f"Violation type must be one of: {', '.join(valid_types)}")
        return v
    
    @validator('severity')
    def validate_severity(cls, v):
        valid_severity = ["low", "medium", "high"]
        if v not in valid_severity:
            raise ValueError(f"Severity must be one of: {', '.join(valid_severity)}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "detection_id": "123e4567-e89b-12d3-a456-426614174000",
                "violation_type": "no_helmet",
                "confidence": 0.92,
                "bounding_box": [0.214, 0.342, 0.278, 0.462],
                "severity": "high"
            }
        }


class ViolationResponse(ViolationBase):
    """Schema for violation API response."""
    id: str
    status: str
    resolution_notes: Optional[str] = None
    resolution_time: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class ViolationDetail(ViolationResponse):
    """Schema for detailed violation response with image."""
    detection_image_url: Optional[str] = None
    camera_id: Optional[str] = None
    camera_name: Optional[str] = None
    violation_image_url: Optional[str] = None  # Cropped to violation area
    
    class Config:
        orm_mode = True


class ViolationFilterParams(FilterParams):
    """Filter parameters for violations."""
    camera_id: Optional[str] = None
    violation_type: Optional[str] = None
    status: Optional[str] = None
    severity: Optional[str] = None
    min_confidence: Optional[float] = None


class ViolationStatusUpdate(BaseModel):
    """Schema for updating violation status."""
    status: str
    resolution_notes: Optional[str] = None
    
    @validator('status')
    def validate_status(cls, v):
        valid_status = ["open", "acknowledged", "resolved"]
        if v not in valid_status:
            raise ValueError(f"Status must be one of: {', '.join(valid_status)}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "status": "resolved",
                "resolution_notes": "Employee was provided with a helmet and trained on safety protocols"
            }
        }


class ViolationTrend(BaseModel):
    """Schema for violation trends over time."""
    period: str
    count: int
    by_type: Dict[str, int]


class ViolationTrendsResponse(BaseModel):
    """Response schema for violation trends endpoint."""
    trends: List[ViolationTrend]
    total_violations: int
    most_common_type: str
    start_date: datetime
    end_date: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "trends": [
                    {
                        "period": "2025-05-01",
                        "count": 12,
                        "by_type": {"no_helmet": 8, "no_vest": 4}
                    },
                    {
                        "period": "2025-05-02",
                        "count": 9,
                        "by_type": {"no_helmet": 5, "no_vest": 4}
                    }
                ],
                "total_violations": 21,
                "most_common_type": "no_helmet",
                "start_date": "2025-05-01T00:00:00Z",
                "end_date": "2025-05-02T23:59:59Z"
            }
        }


class ViolationHotspot(BaseModel):
    """Schema for violation hotspots by camera."""
    camera_id: str
    camera_name: str
    count: int
    most_common_type: str
    severity_distribution: Dict[str, int]
    violation_rate: float  # violations / total detections
    
    class Config:
        schema_extra = {
            "example": {
                "camera_id": "camera1",
                "camera_name": "Building A Entrance",
                "count": 28,
                "most_common_type": "no_helmet",
                "severity_distribution": {"low": 5, "medium": 15, "high": 8},
                "violation_rate": 0.112
            }
        }


class ViolationHotspotsResponse(BaseModel):
    """Response schema for violation hotspots endpoint."""
    hotspots: List[ViolationHotspot]
    total_cameras: int
    total_violations: int
    
    class Config:
        orm_mode = True
