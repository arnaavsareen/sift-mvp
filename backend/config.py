import os
import torch
import logging
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# API settings
API_PREFIX = os.getenv("API_PREFIX", "/api")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sift.db")

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Model settings
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODELS_DIR, "yolo11m.pt"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.10))

# Paths
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "data", "screenshots")

# Ensure directories exist
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Performance settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
FRAME_SAMPLE_RATE = int(os.getenv("FRAME_SAMPLE_RATE", 5))
MAX_DETECTION_TIME_MS = int(os.getenv("MAX_DETECTION_TIME_MS", 500))

# PPE Detection settings
DEFAULT_REQUIRED_PPE = os.getenv("DEFAULT_REQUIRED_PPE", "hardhat,vest").split(",")
PPE_ASSOCIATION_THRESHOLD = float(os.getenv("PPE_ASSOCIATION_THRESHOLD", 0.3))

# Device settings - use CPU by default
DEVICE = "cpu"

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)