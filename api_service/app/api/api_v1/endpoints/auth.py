"""
Authentication API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.auth import get_user_by_email
from app.core.security import verify_password, create_access_token
from app.db.database import get_db
from app.models.schemas.user import Token, LoginRequest
from app.models.user import User


router = APIRouter()


@router.post(
    "/login",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticate a user and generate an access token."
)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    user = get_user_by_email(db, email=form_data.username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Define scopes based on user roles
    scopes = [
        "detections:read",
        "cameras:read",
        "violations:read",
        "analytics:read",
    ]
    
    # Add write scopes for admin users
    if user.is_admin:
        scopes.extend([
            "detections:write",
            "cameras:write",
            "violations:write",
            "users:admin",
        ])
    
    # Update last login time
    user.last_login = datetime.now()
    db.add(user)
    db.commit()
    
    # Generate access token
    return create_access_token(
        subject=user.id,
        scopes=scopes,
        is_admin=user.is_admin
    )


@router.post(
    "/login/json",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="JSON Login",
    description="Login with JSON credentials instead of form data."
)
async def login_json(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    JSON compatible login endpoint that doesn't require form data.
    This is useful for programmatic API access.
    """
    user = get_user_by_email(db, email=login_data.email)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Define scopes based on user roles
    scopes = [
        "detections:read",
        "cameras:read",
        "violations:read",
        "analytics:read",
    ]
    
    # Add write scopes for admin users
    if user.is_admin:
        scopes.extend([
            "detections:write",
            "cameras:write",
            "violations:write",
            "users:admin",
        ])
    
    # Update last login time
    user.last_login = datetime.now()
    db.add(user)
    db.commit()
    
    # Generate access token
    return create_access_token(
        subject=user.id,
        scopes=scopes,
        is_admin=user.is_admin
    )
