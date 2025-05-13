import os
import numpy as np
import logging
import time
import cv2
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Import ultralytics
from ultralytics import YOLO

from backend.config import MODEL_PATH, CONFIDENCE_THRESHOLD, MODELS_DIR, DEVICE

logger = logging.getLogger(__name__)

class DetectionService:
    """
    Enhanced detection service with real PPE detection using YOLOv11.
    Detects persons and their PPE items (hardhat, vest, etc.) and determines violations.
    """
    
    def __init__(
        self, 
        model_path: str = MODEL_PATH, 
        confidence: float = CONFIDENCE_THRESHOLD,
        device: Optional[str] = DEVICE  # Auto-select device (CPU/GPU) from config
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = None
        self.class_names = {}
        self.last_inference_time = 0
        
        # PPE class mapping - maps YOLO class names to standard categories
        self.ppe_class_mapping = {
            'person': 'person',
            'hardhat': 'hardhat',
            'helmet': 'hardhat',
            'hard_hat': 'hardhat',
            'safety_helmet': 'hardhat',
            'vest': 'vest',
            'safety_vest': 'vest',
            'high_vis_vest': 'vest',
            'reflective_vest': 'vest',
            'mask': 'mask',
            'face_mask': 'mask',
            'safety_mask': 'mask',
            'goggles': 'goggles',
            'safety_goggles': 'goggles',
            'gloves': 'gloves',
            'safety_gloves': 'gloves',
            'boots': 'boots',
            'safety_boots': 'boots',
            'no_hardhat': 'no_hardhat',
            'no_helmet': 'no_hardhat',
            'no_vest': 'no_vest',
            'no_safety_vest': 'no_vest'
        }
        
        # Required PPE by default (can be overridden by zone configuration)
        self.default_required_ppe = ['hardhat', 'vest']
        
        # Initialize model
        self.load_model()
    
    def load_model(self) -> bool:
        """Load YOLOv11 model with proper error handling and optimization."""
        try:
            model_path = Path(self.model_path)
            
            # Check if model exists
            if not model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                
                # Look for alternative models
                alternatives = [
                    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
                    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"
                ]
                
                for alt in alternatives:
                    alt_path = Path(MODELS_DIR) / alt
                    if alt_path.exists():
                        logger.info(f"Using alternative model: {alt_path}")
                        self.model_path = str(alt_path)
                        model_path = alt_path
                        break
                else:
                    logger.error("No valid model files found!")
                    return False
            
            logger.info(f"Loading YOLO model from: {model_path}")
            start_time = time.time()
            
            # Log device information
            logger.info(f"Using device: {self.device}")
            
            # Load the model with specified device
            self.model = YOLO(self.model_path)
            
            # Force model to specified device
            if self.device:
                try:
                    self.model.to(self.device)
                except Exception as e:
                    logger.warning(f"Failed to set device to {self.device}: {str(e)}")
                    logger.warning("Continuing with default device")
            
            # Store class names
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}
            
            # Log available PPE classes in the model
            ppe_classes = []
            for class_id, class_name in self.class_names.items():
                if self._standardize_class(class_name) in ['hardhat', 'vest', 'mask', 'goggles', 'gloves', 'boots'] or class_name.startswith('no_'):
                    ppe_classes.append(f"{class_id}: {class_name}")
            
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            logger.info(f"Model has {len(self.class_names)} classes")
            logger.info(f"PPE classes available: {', '.join(ppe_classes) if ppe_classes else 'None'}")
            logger.info(f"Device: {self.model.device}")
            logger.info(f"Confidence threshold: {self.confidence}")
            
            # Run warmup inference
            self._warmup_model()
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _warmup_model(self) -> None:
        """Run model on dummy image to initialize weights and optimize performance."""
        try:
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            logger.info("Warming up model...")
            start_time = time.time()
            _ = self.model(dummy_img, verbose=False)
            logger.info(f"Model warmup completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")
    
    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Run PPE detection on a frame.
        
        Args:
            frame: Image frame (numpy array)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] coordinates
            - class: Class name
            - confidence: Detection confidence
            - violation: Whether this detection represents a violation
            - violation_type: Type of violation detected
            - detected_ppe: List of PPE items detected for this person
        """
        if self.model is None:
            logger.warning("Model not loaded, attempting to reload")
            success = self.load_model()
            if not success:
                logger.error("Failed to load model")
                return []
        
        try:
            # Validate frame
            if frame is None:
                logger.error("Received None frame for detection")
                return []
                
            # Check frame shape and type
            if not isinstance(frame, np.ndarray):
                logger.error(f"Frame is not a numpy array: {type(frame)}")
                return []
            
            # Ensure frame is in correct format
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                logger.warning(f"Unexpected frame shape: {frame.shape}, attempting to convert")
                if len(frame.shape) == 2:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    logger.error(f"Cannot process frame with shape: {frame.shape}")
                    return []
            
            # Measure inference time
            start_time = time.time()
            
            # Run YOLO detection
            results = self.model(
                frame,
                conf=self.confidence,
                verbose=False,
                agnostic_nms=True,  # Class-agnostic NMS
                max_det=100  # Maximum detections per image
            )
            
            inference_time = time.time() - start_time
            self.last_inference_time = inference_time
            logger.debug(f"Detection inference took {inference_time:.3f} seconds")
            
            # Process results
            detections = []
            
            if len(results) == 0:
                logger.debug("No results from model inference")
                return []
                
            # Get the first result
            result = results[0]
            
            # Check if result has boxes
            if not hasattr(result, 'boxes') or result.boxes is None:
                logger.warning("Invalid result format from model inference")
                return []
                
            # Extract detections
            boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []
            confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else []
            class_ids = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes, 'cls') else []
            
            logger.debug(f"Detection found {len(boxes)} objects")
            
            if len(boxes) == 0:
                return []
            
            # Separate persons and PPE items
            persons = []
            ppe_items = []
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confs, class_ids)):
                x1, y1, x2, y2 = box
                
                # Get class name
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                standardized_class = self._standardize_class(class_name)
                
                detection = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "class": class_name,
                    "std_class": standardized_class,
                    "confidence": float(conf),
                    "center": [(x1 + x2) / 2, (y1 + y2) / 2]
                }
                
                if standardized_class == "person":
                    detection["detected_ppe"] = []
                    detection["violations"] = []
                    persons.append(detection)
                elif standardized_class in ['hardhat', 'vest', 'mask', 'goggles', 'gloves', 'boots']:
                    ppe_items.append(detection)
                
                detections.append(detection)
            
            # Associate PPE with persons
            if persons:
                detections = self._associate_ppe_with_persons(persons, ppe_items)
                detections = self._check_ppe_violations(detections)
            
            logger.debug(f"Processed {len(detections)} detections, {len(persons)} persons")
            return detections
        
        except Exception as e:
            logger.error(f"Error in detection: {str(e)}")
            logger.exception(e)
            return []
    
    def _standardize_class(self, class_name: str) -> str:
        """Map class name to a standard category for consistent processing."""
        class_lower = class_name.lower().replace('-', '_').replace(' ', '_')
        
        # Check for exact match first
        if class_lower in self.ppe_class_mapping:
            return self.ppe_class_mapping[class_lower]
        
        # Check for partial matches
        for key, value in self.ppe_class_mapping.items():
            if key in class_lower or class_lower in key:
                return value
        
        # Return original if no match found
        return class_name
    
    def _associate_ppe_with_persons(self, persons: List[Dict], ppe_items: List[Dict]) -> List[Dict]:
        """
        Associate detected PPE items with nearby persons using spatial proximity.
        """
        for person in persons:
            person_bbox = person["bbox"]
            person_center = person["center"]
            
            # Create expanded search area around person
            search_area = self._expand_bbox(person_bbox, factor=1.5)  # Increased from 1.3 to 1.5
            
            for ppe in ppe_items:
                ppe_center = ppe["center"]
                
                # Check if PPE item is within search area
                if self._point_in_bbox(ppe_center, search_area):
                    # Calculate distance for weight consideration
                    distance = self._calculate_distance(person_center, ppe_center)
                    ppe_type = ppe["std_class"]
                    
                    # Add PPE to person's detected items
                    if ppe_type not in person["detected_ppe"]:
                        person["detected_ppe"].append(ppe_type)
                        logger.debug(f"Associated {ppe_type} with person at {person_center}")
        
        return persons + ppe_items
    
    def _expand_bbox(self, bbox: List[float], factor: float = 1.3) -> List[float]:
        """Expand bounding box by given factor."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Calculate expansion
        expand_w = w * (factor - 1) / 2
        expand_h = h * (factor - 1) / 2
        
        return [
            x1 - expand_w,
            y1 - expand_h,
            x2 + expand_w,
            y2 + expand_h
        ]
    
    def _point_in_bbox(self, point: List[float], bbox: List[float]) -> bool:
        """Check if a point is inside a bounding box."""
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points."""
        x1, y1 = point1
        x2, y2 = point2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def _check_ppe_violations(self, detections: List[Dict]) -> List[Dict]:
        """
        Check each person for PPE violations based on required equipment.
        """
        required_ppe = self.default_required_ppe  # Can be overridden by zone rules
        
        for detection in detections:
            if detection["std_class"] != "person":
                continue
            
            # Check for missing PPE
            missing_ppe = []
            for required in required_ppe:
                if required not in detection["detected_ppe"]:
                    missing_ppe.append(required)
            
            # Set violation flags
            if missing_ppe:
                # Generate appropriate violation message
                violation_types = []
                for item in missing_ppe:
                    violation_types.append(f"no_{item}")
                
                # Add violation details
                detection["violation"] = True
                detection["violation_type"] = ",".join(violation_types)
                detection["violations"] = missing_ppe
                detection["missing_ppe"] = missing_ppe
                
                logger.debug(f"Violation detected: {detection['violation_type']}")
            else:
                detection["violation"] = False
                detection["violation_type"] = ""
        
        return detections
    
    def set_required_ppe(self, required_ppe: List[str]) -> None:
        """Set the list of required PPE items for violation checking."""
        self.default_required_ppe = required_ppe
        logger.info(f"Updated required PPE: {required_ppe}")


# Singleton instance for global access
_detection_service = None

def get_detection_service() -> DetectionService:
    """Get or create the detection service singleton."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service