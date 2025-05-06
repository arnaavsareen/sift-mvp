"""
Violation model for tracking safety violations.
"""
from sqlalchemy import Column, String, DateTime, Float, JSON, Text, ForeignKey, Index, func
from sqlalchemy.orm import relationship
import uuid

from app.db.database import Base
from app.models.base import TimestampMixin


class Violation(Base, TimestampMixin):
    """
    Model for storing safety violations detected by the system.
    """
    __tablename__ = "violations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    detection_id = Column(String, ForeignKey("detections.id"), nullable=False, index=True)
    detection = relationship("Detection", back_populates="violations")
    
    # Violation details
    violation_type = Column(String, nullable=False, index=True)  # no_helmet, no_vest, etc.
    confidence = Column(Float, nullable=False)
    bounding_box = Column(JSON, nullable=False)  # [x1, y1, x2, y2] normalized
    
    # Additional metadata
    severity = Column(String, default="medium", index=True)  # low, medium, high
    status = Column(String, default="open", index=True)  # open, acknowledged, resolved
    resolution_notes = Column(Text, nullable=True)
    resolution_time = Column(DateTime(timezone=True), nullable=True)
    
    # The timestamp when the violation was detected (inherited from detection)
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    
    # Efficient indexing for analytics and reporting
    __table_args__ = (
        Index("ix_violations_type_timestamp", "violation_type", "timestamp"),
        Index("ix_violations_status_timestamp", "status", "timestamp"),
        Index("ix_violations_severity_timestamp", "severity", "timestamp"),
    )
    
    def __repr__(self):
        return f"<Violation(id={self.id}, type={self.violation_type}, status={self.status})>"
