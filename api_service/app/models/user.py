"""
User model for authentication and authorization.
"""
from sqlalchemy import Column, String, Boolean, DateTime, func
import uuid

from app.db.database import Base
from app.models.base import TimestampMixin


class User(Base, TimestampMixin):
    """
    Model for storing user information.
    """
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, index=True, unique=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, admin={self.is_admin})>"
