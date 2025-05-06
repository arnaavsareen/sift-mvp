"""
Schemas for camera management endpoints.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

from app.models.schemas.common import FilterParams


class CameraBase(BaseModel):
    """Base camera schema."""
    name: str
    description: Optional[str] = None
    location: Optional[str] = None
    area_id: Optional[str] = None
    stream_url: Optional[str] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    floor: Optional[str] = None
    coverage_radius: Optional[float] = None
    coverage_angle: Optional[float] = None


class CameraCreate(CameraBase):
    """Schema for creating a new camera."""
    connection_details: Optional[Dict[str, Any]] = None
    
    @validator('stream_url')
    def validate_stream_url(cls, v):
        if v and not (v.startswith('rtsp://') or v.startswith('http://') or v.startswith('https://')):
            raise ValueError('Stream URL must start with rtsp://, http://, or https://')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Building A Entrance",
                "description": "Main entrance camera covering the north entrance",
                "location": "North Wing, 1st Floor",
                "area_id": "building_a",
                "stream_url": "rtsp://192.168.1.100:554/main",
                "position_x": 120.5,
                "position_y": 45.2,
                "floor": "1",
                "coverage_radius": 10.5,
                "coverage_angle": 120.0,
                "connection_details": {
                    "username": "admin",
                    "password": "secure_password",
                    "fps": 10,
                    "resolution": "1280x720"
                }
            }
        }


class CameraResponse(CameraBase):
    """Schema for camera API response."""
    id: str
    is_active: bool = True
    status: str
    last_seen: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True


class CameraWithStats(CameraResponse):
    """Camera response with detection statistics."""
    detection_count: int
    violation_count: int
    compliance_rate: float
    last_detection: Optional[datetime] = None
    last_violation: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class CameraFilterParams(FilterParams):
    """Filter parameters for cameras."""
    area_id: Optional[str] = None
    is_active: Optional[bool] = None
    status: Optional[str] = None
    floor: Optional[str] = None


class CameraViolationTimeline(BaseModel):
    """Timeline data for violations by camera."""
    camera_id: str
    camera_name: str
    time_periods: List[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "camera_id": "camera1",
                "camera_name": "Building A Entrance",
                "time_periods": [
                    {
                        "period": "2025-05-06-08:00",
                        "violation_count": 3,
                        "detection_count": 25,
                        "compliance_rate": 0.88
                    },
                    {
                        "period": "2025-05-06-09:00",
                        "violation_count": 1,
                        "detection_count": 28,
                        "compliance_rate": 0.96
                    }
                ]
            }
        }


class CameraStatusUpdate(BaseModel):
    """Schema for updating camera status."""
    is_active: Optional[bool] = None
    status: Optional[str] = None
    last_seen: Optional[str] = None
    
    @validator('status')
    def validate_status(cls, v):
        if v and v not in ['online', 'offline', 'error']:
            raise ValueError("Status must be one of 'online', 'offline', or 'error'")
        return v
