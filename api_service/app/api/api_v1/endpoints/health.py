"""
Health check API endpoints.
"""
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from datetime import datetime

from app.db.database import get_db
from app.models.schemas.common import HealthResponse
from app.core.config import settings


router = APIRouter()


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the health and status of the API service."
)
async def health_check(db: Session = Depends(get_db)):
    """
    Check the health and status of the API service.
    
    - **Checks database connectivity**
    - **Returns service version**
    - **Includes current timestamp**
    """
    # Check database connectivity
    try:
        # Simple query to check database connection
        db.execute("SELECT 1")
        db_status = "ok"
    except Exception:
        db_status = "error"
        
    # Overall status is "ok" if database is okay
    overall_status = "ok" if db_status == "ok" else "error"
    
    return {
        "status": overall_status,
        "version": settings.API_VERSION,
        "timestamp": datetime.now()
    }
