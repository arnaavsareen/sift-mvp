"""
Detection model for storing PPE detection results.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, ForeignKey, Index, func
from sqlalchemy.orm import relationship
import uuid

from app.db.database import Base
from app.models.base import TimestampMixin


class Detection(Base, TimestampMixin):
    """
    Model for storing PPE detection results.
    """
    __tablename__ = "detections"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String, index=True, nullable=False)
    image_path = Column(String, nullable=False)  # S3 path
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Source information
    camera_id = Column(String, ForeignKey("cameras.id"), index=True)
    camera = relationship("Camera", back_populates="detections")
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
    
    # Relationships
    violations = relationship("Violation", back_populates="detection", cascade="all, delete-orphan")
    
    # Indexes for efficient filtering and analytics
    __table_args__ = (
        Index("ix_detections_camera_timestamp", "camera_id", "timestamp"),
        Index("ix_detections_violations", "violations_detected"),
        Index("ix_detections_ppe", "ppe_detected"),
    )
    
    def __repr__(self):
        return f"<Detection(id={self.id}, image_id={self.image_id}, violations={self.violations_detected})>"
