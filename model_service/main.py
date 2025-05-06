"""
SIFT Model Service - PPE Detection Service
Main application entry point
"""
import uvicorn
from app.api.api import app
from app.core.config import settings
from app.core.logging import setup_logging

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Run the application
    uvicorn.run(
        "app.api.api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG_MODE
    )
