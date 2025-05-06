"""
Base models for the SIFT API Service.
"""
from sqlalchemy import Column, Integer, String, DateTime, func
from app.db.database import Base


class TimestampMixin:
    """
    Mixin that adds created_at and updated_at columns to models.
    """
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
