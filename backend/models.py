from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from datetime import datetime

from backend.database import Base

class Camera(Base):
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    url = Column(String, nullable=False)
    location = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Optional: Camera configuration
    config = Column(JSON, nullable=True)

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"))
    violation_type = Column(String, nullable=False)  # E.g., "no_hardhat", "no_vest", etc.
    confidence = Column(Float, nullable=False)
    bbox = Column(JSON, nullable=True)  # Bounding box coordinates [x1, y1, x2, y2]
    screenshot_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

class Zone(Base):
    __tablename__ = "zones"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"))
    name = Column(String, nullable=False)
    polygon = Column(JSON, nullable=False)  # Array of [x, y] coordinates defining zone
    rule_type = Column(String, nullable=False)  # Rule type for this zone
    created_at = Column(DateTime(timezone=True), default=func.now())