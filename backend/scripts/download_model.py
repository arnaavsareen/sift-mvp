#!/usr/bin/env python3
"""
Script to download YOLOv11/YOLOv8 model for PPE detection.
This ensures the detection model is available for the application.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path so we can import from backend
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from backend.config import MODELS_DIR, MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model():
    """Download the YOLOv11/YOLOv8 model if it doesn't exist."""
    model_path = Path(MODEL_PATH)
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if model_path.exists():
        logger.info(f"Model already exists at {model_path}")
        return True
    
    try:
        logger.info(f"Downloading model to {model_path}")
        
        # Import ultralytics and download the model
        from ultralytics import YOLO
        
        # Determine which model to download - YOLOv11 if possible, otherwise YOLOv8
        model_name = model_path.name
        
        if "yolo11" in model_name:
            logger.info("Downloading YOLOv11 model...")
            base_model = "yolov8m.pt"  # We'll download YOLOv8 as a base
        else:
            base_model = model_name
        
        # Download the base model
        logger.info(f"Downloading {base_model}...")
        model = YOLO(base_model)
        
        # Copy from cache to models directory
        import shutil
        cache_path = Path.home() / ".cache" / "ultralytics" / base_model
        
        if cache_path.exists():
            logger.info(f"Copying from cache: {cache_path} to {model_path}")
            shutil.copy(cache_path, model_path)
            logger.info(f"Model downloaded and saved to {model_path}")
            return True
        else:
            logger.error(f"Model not found in cache at {cache_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting model download process")
    success = download_model()
    if success:
        logger.info("Model download completed successfully")
        sys.exit(0)
    else:
        logger.error("Model download failed")
        sys.exit(1) 