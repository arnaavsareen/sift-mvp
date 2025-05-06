"""
Common schemas used across the API.
"""
from typing import Generic, List, Optional, TypeVar, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ResponseStatus(BaseModel):
    """Base schema for API response status."""
    success: bool
    message: Optional[str] = None


T = TypeVar('T')


class PaginatedResponse(BaseModel, Generic[T]):
    """Schema for paginated responses."""
    items: List[T]
    total: int
    page: int
    page_size: int
    pages: int
    has_next: bool
    has_prev: bool


class TimeRangeParams(BaseModel):
    """Time range parameters for filtering."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class FilterParams(BaseModel):
    """Common filter parameters."""
    page: int = 1
    page_size: int = 50
    sort_by: Optional[str] = None
    sort_desc: bool = False
    search: Optional[str] = None
    time_range: Optional[TimeRangeParams] = None


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "timestamp": "2025-05-06T01:08:25.123Z"
            }
        }


class ExportFormat(BaseModel):
    """Export format options."""
    format: str = Field(..., description="Export format (csv, json)")
    include_images: bool = Field(False, description="Include image URLs in export")
    
    class Config:
        schema_extra = {
            "example": {
                "format": "csv",
                "include_images": False
            }
        }


class WebSocketMessage(BaseModel):
    """Base schema for WebSocket messages."""
    event: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "event": "new_violation",
                "data": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "violation_type": "no_helmet",
                    "camera_id": "camera1"
                },
                "timestamp": "2025-05-06T01:08:25.123Z"
            }
        }
