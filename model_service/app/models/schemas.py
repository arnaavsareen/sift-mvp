"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from enum import Enum

class DetectionResult(BaseModel):
    """Schema for detection results."""
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

class DetectionResponse(BaseModel):
    """Schema for detection API response."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    ppe_detected: bool
    violations_detected: bool
    num_detections: int
    detections: List[DetectionResult]
    processing_time: float
    
    class Config:
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "image_id": "frame_20250505_123045_camera1",
                "timestamp": "2025-05-05T12:30:45.123Z",
                "ppe_detected": True,
                "violations_detected": False,
                "num_detections": 2,
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "helmet",
                        "confidence": 0.92,
                        "bbox": [0.214, 0.342, 0.278, 0.462]
                    }
                ],
                "processing_time": 0.145
            }
        }

class ProcessImageRequest(BaseModel):
    """Schema for manual image processing request."""
    image_id: str
    s3_path: str
    source_id: Optional[str] = None
    source_type: Optional[str] = "manual"
    
    class Config:
        schema_extra = {
            "example": {
                "image_id": "custom_image_123",
                "s3_path": "s3://sift-images-bucket/uploads/image123.jpg",
                "source_id": "manual_upload",
                "source_type": "manual"
            }
        }

class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "timestamp": "2025-05-05T12:30:45.123Z"
            }
        }


class ProcessorStatus(str, Enum):
    """Enum for processor status."""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    NOT_INITIALIZED = "not_initialized"


class ProcessorMetrics(BaseModel):
    """Schema for processor metrics."""
    frames_processed: int
    frames_failed: int
    frames_retried: int
    total_detections: int
    total_violations: int
    poison_messages: int
    avg_processing_time: float
    last_processed: Optional[datetime] = None

    class Config:
        schema_extra = {
            "example": {
                "frames_processed": 1250,
                "frames_failed": 12,
                "frames_retried": 25,
                "total_detections": 3218,
                "total_violations": 78,
                "poison_messages": 3,
                "avg_processing_time": 0.354,
                "last_processed": "2025-05-05T12:30:45.123Z"
            }
        }


class ProcessorConfig(BaseModel):
    """Schema for processor configuration."""
    batch_size: int
    wait_time: int
    visibility_timeout: int
    max_retries: int
    retry_delay: int
    has_dead_letter_queue: bool
    model_path: str
    confidence_threshold: float

    class Config:
        schema_extra = {
            "example": {
                "batch_size": 10,
                "wait_time": 5,
                "visibility_timeout": 300,
                "max_retries": 3,
                "retry_delay": 30,
                "has_dead_letter_queue": True,
                "model_path": "./models/yolov8n.pt",
                "confidence_threshold": 0.5
            }
        }


class QueueAttributes(BaseModel):
    """Schema for SQS queue attributes."""
    available_messages: int
    in_flight_messages: int
    delayed_messages: int

    class Config:
        schema_extra = {
            "example": {
                "available_messages": 24,
                "in_flight_messages": 10,
                "delayed_messages": 0
            }
        }


class ProcessorStatusResponse(BaseModel):
    """Schema for processor status response."""
    running: bool
    metrics: ProcessorMetrics
    config: ProcessorConfig
    in_flight_messages: int
    queue: Optional[QueueAttributes] = None

    class Config:
        schema_extra = {
            "example": {
                "running": True,
                "metrics": {
                    "frames_processed": 1250,
                    "frames_failed": 12,
                    "frames_retried": 25,
                    "total_detections": 3218,
                    "total_violations": 78,
                    "poison_messages": 3,
                    "avg_processing_time": 0.354,
                    "last_processed": "2025-05-05T12:30:45.123Z"
                },
                "config": {
                    "batch_size": 10,
                    "wait_time": 5,
                    "visibility_timeout": 300,
                    "max_retries": 3,
                    "retry_delay": 30,
                    "has_dead_letter_queue": True,
                    "model_path": "./models/yolov8n.pt",
                    "confidence_threshold": 0.5
                },
                "in_flight_messages": 5,
                "queue": {
                    "available_messages": 24,
                    "in_flight_messages": 10,
                    "delayed_messages": 0
                }
            }
        }


class QueueStatusResponse(BaseModel):
    """Schema for queue status response."""
    queue_url: str
    attributes: QueueAttributes
    last_checked: datetime

    class Config:
        schema_extra = {
            "example": {
                "queue_url": "https://sqs.us-east-1.amazonaws.com/123456789012/sift-image-queue",
                "attributes": {
                    "available_messages": 24,
                    "in_flight_messages": 10,
                    "delayed_messages": 0
                },
                "last_checked": "2025-05-05T12:30:45.123Z"
            }
        }


class ProcessorHealthCheck(BaseModel):
    """Schema for processor health check."""
    status: str  # "healthy", "degraded", "unhealthy"
    checks: Dict[str, Dict[str, Any]]
    metrics: Optional[Dict[str, Any]] = None
    timestamp: datetime

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "checks": {
                    "manager_initialized": {
                        "status": "ok",
                        "message": "Manager is initialized"
                    },
                    "processor": {
                        "status": "ok",
                        "message": "Processor is running"
                    },
                    "sqs_queue": {
                        "status": "ok",
                        "message": "Queue is accessible: 24 available, 10 in flight",
                        "details": {
                            "available_messages": 24,
                            "in_flight_messages": 10
                        }
                    },
                    "model": {
                        "status": "ok",
                        "message": "Model is loaded: yolov8n.pt",
                        "details": {
                            "model_path": "./models/yolov8n.pt",
                            "confidence_threshold": 0.5
                        }
                    }
                },
                "metrics": {
                    "frames_processed": 1250,
                    "frames_failed": 12,
                    "total_detections": 3218,
                    "total_violations": 78
                },
                "timestamp": "2025-05-05T12:30:45.123Z"
            }
        }
