"""
Camera model for managing camera data sources.
"""
from sqlalchemy import Column, String, Boolean, Float, Text, JSON, func, select
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
import uuid

from app.db.database import Base
from app.models.base import TimestampMixin


class Camera(Base, TimestampMixin):
    """
    Model for storing camera information and settings.
    """
    __tablename__ = "cameras"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    area_id = Column(String, nullable=True, index=True)  # For grouping cameras by area
    stream_url = Column(String, nullable=True)  # RTSP or other stream URL
    
    # Status and metadata
    is_active = Column(Boolean, default=True, index=True)
    last_seen = Column(String, nullable=True)
    status = Column(String, default="offline")  # online, offline, error
    connection_details = Column(JSON, nullable=True)  # Additional connection details
    
    # Camera positioning and coverage
    position_x = Column(Float, nullable=True)  # For mapping visualization
    position_y = Column(Float, nullable=True)
    floor = Column(String, nullable=True)  # For multi-floor setups
    coverage_radius = Column(Float, nullable=True)  # In meters
    coverage_angle = Column(Float, nullable=True)  # Field of view in degrees
    
    # Relationships
    detections = relationship("Detection", back_populates="camera")
    
    @hybrid_property
    def detection_count(self):
        """Get the number of detections for this camera"""
        return len(self.detections)
    
    @detection_count.expression
    def detection_count(cls):
        from app.models.detection import Detection
        return select(func.count(Detection.id)).where(Detection.camera_id == cls.id).scalar_subquery()
    
    @hybrid_property
    def violation_count(self):
        """Get the number of violations for this camera"""
        return sum(1 for detection in self.detections if detection.violations_detected)
    
    @violation_count.expression
    def violation_count(cls):
        from app.models.detection import Detection
        return select(func.count(Detection.id)).where(
            Detection.camera_id == cls.id,
            Detection.violations_detected == True
        ).scalar_subquery()
    
    def __repr__(self):
        return f"<Camera(id={self.id}, name={self.name}, status={self.status})>"
