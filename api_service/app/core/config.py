"""
Configuration settings for the SIFT API Service.
"""
import os
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, field_validator, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.
    """
    PROJECT_NAME: str = "SIFT API Service"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    PROJECT_DESCRIPTION: str = "API service for the SIFT PPE detection platform"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Authentication settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "supersecretkey")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost/sift_db")
    
    # AWS settings
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    
    # S3 settings
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "sift-images-bucket")
    
    # SQS settings
    SQS_QUEUE_URL: str = os.getenv("SQS_QUEUE_URL", "")
    
    # Cache settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # WebSocket settings
    WS_NOTIFICATIONS_ENABLED: bool = os.getenv("WS_NOTIFICATIONS_ENABLED", "True").lower() == "true"
    
    # Pagination defaults
    DEFAULT_PAGE_SIZE: int = int(os.getenv("DEFAULT_PAGE_SIZE", "50"))
    MAX_PAGE_SIZE: int = int(os.getenv("MAX_PAGE_SIZE", "100"))
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"  # Allow extra fields in environment variables
    )


# Create settings instance
settings = Settings()
