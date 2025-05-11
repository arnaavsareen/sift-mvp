import os
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
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.15))

# Paths
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "data", "screenshots")

# Ensure directories exist
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Log the model path being used
print(f"Using model path: {MODEL_PATH}")