"""
User management API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.core.auth import get_current_admin_user, get_current_active_user
from app.core.security import get_password_hash, verify_password
from app.db.database import get_db
from app.models.schemas.user import UserCreate, UserUpdate, UserResponse, PasswordChange
from app.models.user import User


router = APIRouter()


@router.get(
    "",
    response_model=List[UserResponse],
    status_code=status.HTTP_200_OK,
    summary="List Users",
    description="Get a list of all users (admin only)."
)
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get a list of all users in the system.
    Requires admin privileges.
    """
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create User",
    description="Create a new user (admin only)."
)
async def create_user(
    user_in: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Create a new user.
    Requires admin privileges.
    """
    # Check if user already exists
    db_user = db.query(User).filter(User.email == user_in.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )
    
    # Create new user
    new_user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(user_in.password),
        is_active=user_in.is_active,
        is_admin=user_in.is_admin
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Current User",
    description="Get information about the currently authenticated user."
)
async def read_user_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information.
    """
    return current_user


@router.put(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Current User",
    description="Update information for the currently authenticated user."
)
async def update_user_me(
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update current user information.
    """
    # Only update provided fields
    if user_in.full_name is not None:
        current_user.full_name = user_in.full_name
    if user_in.email is not None:
        # Check if email is already in use
        if current_user.email != user_in.email:
            existing_user = db.query(User).filter(User.email == user_in.email).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered"
                )
        current_user.email = user_in.email
    
    # Don't allow users to modify their own admin status
    
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return current_user


@router.put(
    "/me/password",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Change Password",
    description="Change the password for the currently authenticated user."
)
async def change_password(
    password_in: PasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Change user password.
    """
    # Verify current password
    if not verify_password(password_in.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    current_user.hashed_password = get_password_hash(password_in.new_password)
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return current_user


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get User by ID",
    description="Get a specific user by ID (admin only)."
)
async def get_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get a specific user by ID.
    Requires admin privileges.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return user


@router.put(
    "/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Update User",
    description="Update a specific user (admin only)."
)
async def update_user(
    user_id: str,
    user_in: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Update a user.
    Requires admin privileges.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Only update provided fields
    if user_in.full_name is not None:
        user.full_name = user_in.full_name
    if user_in.email is not None:
        # Check if email is already in use by another user
        if user.email != user_in.email:
            existing_user = db.query(User).filter(User.email == user_in.email).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered"
                )
        user.email = user_in.email
    if user_in.is_active is not None:
        user.is_active = user_in.is_active
    if user_in.is_admin is not None:
        user.is_admin = user_in.is_admin
    
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete User",
    description="Delete a specific user (admin only)."
)
async def delete_user(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Delete a user.
    Requires admin privileges.
    """
    # Prevent self-deletion
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own user account"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    db.delete(user)
    db.commit()
    return None
