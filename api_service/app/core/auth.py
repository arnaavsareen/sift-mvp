"""
Authentication middleware and dependencies for securing API endpoints.
"""
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import ValidationError

from app.core.config import settings
from app.core.security import ALGORITHM
from app.db.database import get_db
from app.models.schemas.user import TokenPayload
from app.models.user import User

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    scopes={
        "detections:read": "Read detections",
        "detections:write": "Write detections",
        "cameras:read": "Read camera data",
        "cameras:write": "Manage cameras",
        "violations:read": "Read violations",
        "violations:write": "Manage violations",
        "analytics:read": "View analytics",
        "users:admin": "User administration",
    },
)


async def get_current_user(
    security_scopes: SecurityScopes,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Get the current authenticated user based on the JWT token.
    
    Args:
        security_scopes: Security scopes required for the endpoint
        db: Database session
        token: JWT token from request
    
    Returns:
        User: The current authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise credentials_exception
        
        token_scopes = payload.get("scopes", [])
        token_data = TokenPayload(
            sub=user_id,
            exp=payload.get("exp"),
            iat=payload.get("iat"),
            admin=payload.get("admin", False),
            scopes=token_scopes
        )
    except (JWTError, ValidationError):
        raise credentials_exception
    
    # Check if the token has required scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            # Admin users have all scopes
            if not token_data.admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not enough permissions. Required: {scope}",
                    headers={"WWW-Authenticate": authenticate_value},
                )
    
    # Get user from database
    user = db.query(User).filter(User.id == token_data.sub).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user


def get_current_active_user(current_user: User = Security(get_current_user)) -> User:
    """
    Dependency to get the current active user.
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def get_current_admin_user(
    current_user: User = Security(
        get_current_user,
        scopes=["users:admin"]
    )
) -> User:
    """
    Dependency to get the current admin user.
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Admin privileges required.",
        )
    return current_user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """
    Get a user by email from the database.
    """
    return db.query(User).filter(User.email == email).first()
