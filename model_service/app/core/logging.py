"""
Logging configuration for the SIFT Model Service.
"""
import os
import sys
from loguru import logger
from app.core.config import settings

def setup_logging():
    """Configure logging for the application."""
    # Remove default handler
    logger.remove()
    
    # Add stderr handler with appropriate log level
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler for production
    if settings.LOG_LEVEL.upper() != "DEBUG":
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, "app.log")
        
        logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            level=settings.LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    
    # Set up SQLAlchemy logging if debug mode
    if settings.DEBUG_MODE:
        logging_level = "INFO"
        logger.debug(f"Setting SQLAlchemy logging level to {logging_level}")
    
    return logger
