"""
Logging configuration for the SIFT API Service.
"""
import os
import sys
from loguru import logger
from pathlib import Path


def setup_logging():
    """
    Configure logging for the application.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        diagnose=True
    )
    logger.add(
        "logs/sift_api_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # new file at midnight
        retention="30 days",  # keep logs for 30 days
        compression="zip",
        format=log_format,
        level=log_level,
        diagnose=True
    )
    
    return logger
