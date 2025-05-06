"""
Configuration settings for the SIFT Model Service.
Loads settings from environment variables.
"""
import os
from pydantic import BaseSettings, Field, PostgresDsn
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "SIFT PPE Detection Service"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG_MODE: bool = Field(default=False)
    
    # AWS Settings
    AWS_REGION: str = Field(...)
    AWS_ACCESS_KEY_ID: str = Field(...)
    AWS_SECRET_ACCESS_KEY: str = Field(...)
    
    # SQS Settings
    SQS_QUEUE_URL: str = Field(...)
    SQS_MAX_MESSAGES: int = Field(default=10)
    SQS_WAIT_TIME: int = Field(default=5)
    
    # S3 Settings
    S3_BUCKET_NAME: str = Field(...)
    
    # Database Settings
    DATABASE_URL: PostgresDsn = Field(...)
    
    # Model Settings
    MODEL_PATH: str = Field(...)
    CONFIDENCE_THRESHOLD: float = Field(default=0.5)
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()
