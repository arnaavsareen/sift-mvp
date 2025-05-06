"""
Database models for storing PPE detection results.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.sql import func
from app.db.database import Base
import uuid

class Detection(Base):
    """
    Model for storing PPE detection results.
    """
    __tablename__ = "detections"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String, index=True, nullable=False)
    image_path = Column(String, nullable=False)  # S3 path
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Source information
    source_id = Column(String, index=True)  # Camera ID or video source
    source_type = Column(String)  # 'camera', 'video', etc.
    frame_number = Column(Integer, nullable=True)
    
    # Detection results
    num_detections = Column(Integer, default=0)
    ppe_detected = Column(Boolean, default=False)
    violations_detected = Column(Boolean, default=False)
    detection_results = Column(JSON)  # Detailed results including bounding boxes
    
    # Processing metadata
    confidence_threshold = Column(Float)
    model_version = Column(String)
    processing_time = Column(Float)  # in seconds
    
    def __repr__(self):
        return f"<Detection(id={self.id}, image_id={self.image_id}, ppe_detected={self.ppe_detected})>"
