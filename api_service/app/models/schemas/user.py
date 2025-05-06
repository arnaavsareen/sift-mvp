"""
Schemas for user authentication and management.
"""
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field, validator
import re
from datetime import datetime


class UserBase(BaseModel):
    """Base schema for user data."""
    email: EmailStr
    full_name: str
    is_active: bool = True
    is_admin: bool = False


class UserCreate(UserBase):
    """Schema for user creation with password."""
    password: str
    
    @validator('password')
    def password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r'[0-9]', v):
            raise ValueError("Password must contain at least one number")
        if not re.search(r'[^A-Za-z0-9]', v):
            raise ValueError("Password must contain at least one special character")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "full_name": "John Doe",
                "password": "Secure!Password123",
                "is_active": True,
                "is_admin": False
            }
        }


class UserUpdate(BaseModel):
    """Schema for updating user information."""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None


class UserInDB(UserBase):
    """Schema for user as stored in the database."""
    id: str
    hashed_password: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None


class UserResponse(UserBase):
    """Schema for user API response."""
    id: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class Token(BaseModel):
    """Schema for JWT token."""
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_at": "2025-05-06T02:08:25.123Z"
            }
        }


class TokenPayload(BaseModel):
    """Schema for JWT token payload."""
    sub: str  # User ID
    exp: int  # Expiration timestamp
    iat: int  # Issued at timestamp
    admin: bool = False
    scopes: List[str] = []


class LoginRequest(BaseModel):
    """Schema for login request."""
    email: EmailStr
    password: str
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "Secure!Password123"
            }
        }


class PasswordReset(BaseModel):
    """Schema for password reset request."""
    email: EmailStr
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com"
            }
        }


class PasswordChange(BaseModel):
    """Schema for password change."""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def password_strength(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r'[0-9]', v):
            raise ValueError("Password must contain at least one number")
        if not re.search(r'[^A-Za-z0-9]', v):
            raise ValueError("Password must contain at least one special character")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "current_password": "OldPassword123!",
                "new_password": "NewSecurePassword456!"
            }
        }
