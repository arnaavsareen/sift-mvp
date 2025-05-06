"""
Security utilities for password hashing and JWT token handling.
"""
from datetime import datetime, timedelta
from jose import jwt
from passlib.context import CryptContext
from typing import Any, Dict, Optional

from app.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
ALGORITHM = "HS256"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify that a plain password matches a hashed password.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Generate a bcrypt hash of the password.
    """
    return pwd_context.hash(password)


def create_access_token(
    subject: str,
    scopes: list = None,
    is_admin: bool = False,
    expires_delta: Optional[timedelta] = None
) -> Dict[str, str]:
    """
    Create a new JWT access token.
    
    Args:
        subject: Subject of the token (typically user ID)
        scopes: Permission scopes to include in the token
        is_admin: Whether the user is an admin
        expires_delta: Token expiration time delta
        
    Returns:
        JWT token data with token and expiration time
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    expire = datetime.utcnow() + expires_delta
    
    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.utcnow(),
        "admin": is_admin,
    }
    
    # Add scopes if provided
    if scopes:
        to_encode["scopes"] = scopes
        
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.JWT_SECRET_KEY, 
        algorithm=ALGORITHM
    )
    
    return {
        "access_token": encoded_jwt,
        "token_type": "bearer",
        "expires_at": expire
    }
