"""
PPE detection service using YOLOv8 model.
"""
import os
import time
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from loguru import logger
from ultralytics import YOLO

from app.core.config import settings

class PPEDetectionService:
    """Service for detecting PPE in images using YOLOv8."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the PPE detection service.
        
        Args:
            model_path: Path to the YOLO model file (.pt)
        """
        if model_path is None:
            model_path = settings.MODEL_PATH
            
        self.model_path = model_path
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
        # Load model
        logger.info(f"Loading YOLOv8 model from {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            # Get model version from filename
            self.model_version = os.path.basename(self.model_path)
            logger.info(f"Successfully loaded model: {self.model_version}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image and detect PPE.
        
        Args:
            image_path: Path to the local image file
            
        Returns:
            Dict containing detection results
        """
        start_time = time.time()
        
        # Load image
        try:
            # Use OpenCV to load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image at {image_path}")
            
            # Convert from BGR to RGB (YOLO expects RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return {
                "error": f"Failed to load image: {str(e)}",
                "ppe_detected": False,
                "violations_detected": False,
                "num_detections": 0,
                "detections": [],
                "processing_time": 0
            }
        
        # Run inference
        try:
            results = self.model(image, conf=self.confidence_threshold)
            processing_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            return {
                "error": f"Model inference failed: {str(e)}",
                "ppe_detected": False,
                "violations_detected": False,
                "num_detections": 0,
                "detections": [],
                "processing_time": time.time() - start_time
            }
        
        # Process results
        detections = []
        ppe_detected = False
        violations_detected = False
        
        # Get class names mapping
        class_names = self.model.names
        
        try:
            # Go through each detection
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get box data
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # get normalized coords
                    
                    class_id = int(box.cls[0].item())
                    class_name = class_names[class_id]
                    confidence = float(box.conf[0].item())
                    
                    # Determine PPE detection and violations based on class
                    # This is a placeholder - you would customize this for your specific PPE classes
                    if class_name in ['helmet', 'vest', 'gloves', 'goggles', 'mask']:
                        ppe_detected = True
                    elif class_name in ['no_helmet', 'no_vest', 'no_protection']:
                        violations_detected = True
                    
                    # Add detection to results
                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)]
                    })
        except Exception as e:
            logger.error(f"Error processing detection results: {e}")
            return {
                "error": f"Error processing results: {str(e)}",
                "ppe_detected": False,
                "violations_detected": False,
                "num_detections": 0,
                "detections": [],
                "processing_time": time.time() - start_time
            }
        
        # Prepare result
        result = {
            "ppe_detected": ppe_detected,
            "violations_detected": violations_detected,
            "num_detections": len(detections),
            "detections": detections,
            "processing_time": processing_time,
            "model_version": self.model_version,
            "confidence_threshold": self.confidence_threshold
        }
        
        logger.info(f"Processed image in {processing_time:.3f}s - " 
                   f"Found {len(detections)} objects, PPE: {ppe_detected}, Violations: {violations_detected}")
        
        return result
