import os
import numpy as np
import logging
import time
import cv2
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Conditionally import ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.config import MODEL_PATH, CONFIDENCE_THRESHOLD, MODELS_DIR

logger = logging.getLogger(__name__)

class DetectionService:
    """
    Service for object detection using YOLO models with optimized implementation
    for safety violation detection in manufacturing environments.
    """
    
    def __init__(
        self, 
        model_path: str = MODEL_PATH, 
        confidence: float = CONFIDENCE_THRESHOLD,
        device: Optional[str] = None  # Auto-select device (CPU/GPU)
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = None
        self.class_names = {}
        self.last_inference_time = 0
        
        # PPE class mapping - maps different class name variations to standardized categories
        self.ppe_class_mapping = {
            'person': ['person', 'people', 'human'],
            'hardhat': ['hardhat', 'helmet', 'hard hat', 'safety helmet', 'construction helmet'],
            'vest': ['vest', 'safety vest', 'high-vis vest', 'high-visibility vest', 'reflective vest'],
            'mask': ['mask', 'face mask', 'safety mask', 'respirator'],
            'goggles': ['goggles', 'safety goggles', 'eye protection', 'safety glasses'],
            'gloves': ['gloves', 'safety gloves', 'hand protection', 'work gloves'],
            'boots': ['boots', 'safety boots', 'foot protection', 'work boots']
        }
        
        # Violation rules - defines which PPE items are required
        self.violation_rules = {
            'factory_floor': ['hardhat', 'vest'],
            'chemical_area': ['hardhat', 'vest', 'mask', 'goggles', 'gloves'],
            'construction': ['hardhat', 'vest', 'boots'],
            'default': ['hardhat', 'vest']
        }
        
        # Initialize model
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load YOLO model with proper error handling and logging.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Ultralytics not available. Install with 'pip install ultralytics'")
            return False
        
        try:
            model_path = Path(self.model_path)
            
            # Check if model exists
            if not model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                
                # Check if we have the default model
                default_model_path = Path(MODELS_DIR) / "yolov8s.pt"
                if default_model_path.exists():
                    logger.info(f"Using default model: {default_model_path}")
                    self.model_path = str(default_model_path)
                    model_path = default_model_path
                else:
                    # Try to download the model
                    try:
                        logger.info("Attempting to download YOLOv8s model...")
                        os.makedirs(MODELS_DIR, exist_ok=True)
                        # Use ultralytics YOLO to download model
                        self.model = YOLO("yolov8s.pt")
                        self.model_path = os.path.join(MODELS_DIR, "yolov8s.pt")
                        logger.info(f"Model downloaded to {self.model_path}")
                        return True
                    except Exception as download_error:
                        logger.error(f"Error downloading model: {str(download_error)}")
                        return False
            
            start_time = time.time()
            
            # Load the model with proper device selection
            device_args = {}
            if self.device:
                device_args = {"device": self.device}
            
            self.model = YOLO(self.model_path, **device_args)
            
            # Store class names
            self.class_names = self.model.names if hasattr(self.model, 'names') else {}
            
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            logger.info(f"Model has {len(self.class_names)} classes")
            logger.info(f"Running on device: {self.model.device}")
            
            # Run a warmup inference to initialize the model
            self._warmup_model()
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _warmup_model(self) -> None:
        """Run model on a dummy image to initialize weights and optimize performance."""
        try:
            # Create a dummy image (black 640x640 frame)
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            logger.info("Warming up model with dummy inference...")
            start_time = time.time()
            _ = self.model(dummy_img, verbose=False)
            logger.info(f"Model warmup completed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")
    
    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Run object detection on a frame with improved PPE violation logic.
        
        Args:
            frame: Image frame (numpy array)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] coordinates
            - class: Class name
            - confidence: Detection confidence
            - violation: Whether this detection represents a violation
            - violation_type: Type of violation detected
        """
        if self.model is None:
            logger.warning("Model not loaded, returning mock detections")
            return self._mock_detections(frame)
        
        try:
            # Measure inference time
            start_time = time.time()
            
            # Run YOLO detection with optimized parameters
            results = self.model(
                frame, 
                conf=self.confidence,
                verbose=False,  # Reduce console output
                augment=False,  # No TTA for inference
                iou=0.45,  # NMS IoU threshold
            )[0]  # Get first (and only) result
            
            inference_time = time.time() - start_time
            self.last_inference_time = inference_time
            
            # Process results
            detections = []
            
            # Extract boxes, confidences, and class IDs
            boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results, 'boxes') else []
            confs = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, 'conf') else []
            class_ids = results.boxes.cls.cpu().numpy().astype(int) if hasattr(results.boxes, 'cls') else []
            
            # Extract all persons first for associating PPE
            persons = []
            ppe_items = []
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confs, class_ids)):
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                
                # Standardize the class name using our mapping
                std_class = self._standardize_class(class_name)
                
                # Create base detection dict
                detection = {
                    "bbox": box.tolist(),
                    "class": class_name,
                    "std_class": std_class,  # Standardized class
                    "confidence": float(conf),
                    "id": i,  # For tracking
                }
                
                # Separate persons and PPE items
                if std_class == 'person':
                    # Add person-specific fields
                    detection.update({
                        "has_ppe": {},  # Will track which PPE items are found
                        "violations": [],  # Will hold violation types
                    })
                    persons.append(detection)
                elif std_class in self.ppe_class_mapping and std_class != 'person':
                    ppe_items.append(detection)
                else:
                    # Other objects (non-person, non-PPE)
                    detections.append(detection)
            
            # Associate PPE with persons and detect violations
            processed_detections = self._process_safety_violations(persons, ppe_items, frame)
            detections.extend(processed_detections)
            
            # Log performance metrics periodically
            if len(detections) > 0 and random.random() < 0.05:  # Log ~5% of inferences
                logger.info(f"Inference time: {inference_time:.3f}s, Detected: {len(detections)} objects")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in detection: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _standardize_class(self, class_name: str) -> str:
        """Map class name to a standard category for consistent processing."""
        class_lower = class_name.lower()
        
        # Check each category
        for category, variants in self.ppe_class_mapping.items():
            if class_lower in variants or any(variant in class_lower for variant in variants):
                return category
        
        # If no match found, return original
        return class_name
    
    def _process_safety_violations(self, persons: List[Dict], ppe_items: List[Dict], frame) -> List[Dict]:
        """
        Process detections to determine safety violations.
        Associates PPE items with detected persons and identifies violations.
        
        Args:
            persons: List of person detections
            ppe_items: List of PPE item detections
            frame: Original image frame
            
        Returns:
            List of processed detections with violation flags
        """
        # Get image dimensions for relative positioning
        frame_h, frame_w = frame.shape[:2]
        
        # For each person, check for required PPE
        for person in persons:
            p_bbox = person["bbox"]
            px1, py1, px2, py2 = p_bbox
            p_width = px2 - px1
            p_height = py2 - py1
            
            # Initialize PPE tracking for this person
            for ppe_type in self.ppe_class_mapping.keys():
                if ppe_type != 'person':
                    person["has_ppe"][ppe_type] = False
            
            # Define body zones for this person
            person_zones = {
                "head": {
                    "y1": py1,
                    "y2": py1 + p_height * 0.2,
                    "x1": px1 + p_width * 0.25,
                    "x2": px2 - p_width * 0.25
                },
                "torso": {
                    "y1": py1 + p_height * 0.2,
                    "y2": py1 + p_height * 0.6,
                    "x1": px1,
                    "x2": px2
                },
                "legs": {
                    "y1": py1 + p_height * 0.6,
                    "y2": py2,
                    "x1": px1,
                    "x2": px2
                }
            }
            
            # For each PPE item, check if it's associated with this person
            for ppe in ppe_items:
                ppe_bbox = ppe["bbox"]
                ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe_bbox
                ppe_type = ppe["std_class"]
                
                # Skip non-PPE items
                if ppe_type not in self.ppe_class_mapping or ppe_type == 'person':
                    continue
                
                # Calculate center point of PPE 
                ppe_center_x = (ppe_x1 + ppe_x2) / 2
                ppe_center_y = (ppe_y1 + ppe_y2) / 2
                
                # Check association based on PPE type
                if ppe_type == 'hardhat':
                    # Check if hardhat is near the head zone
                    head_zone = person_zones["head"]
                    if (ppe_center_y < head_zone["y2"] and
                        ppe_center_x >= head_zone["x1"] and 
                        ppe_center_x <= head_zone["x2"]):
                        person["has_ppe"]["hardhat"] = True
                
                elif ppe_type == 'vest':
                    # Check if vest overlaps with torso
                    torso_zone = person_zones["torso"]
                    if (ppe_center_y >= torso_zone["y1"] and 
                        ppe_center_y <= torso_zone["y2"] and
                        ppe_x1 <= torso_zone["x2"] and
                        ppe_x2 >= torso_zone["x1"]):
                        person["has_ppe"]["vest"] = True
                
                elif ppe_type in ['mask', 'goggles']:
                    # Check if face protection is near the head
                    head_zone = person_zones["head"]
                    if (ppe_center_y >= head_zone["y1"] and 
                        ppe_center_y <= head_zone["y2"] and
                        ppe_center_x >= head_zone["x1"] and 
                        ppe_center_x <= head_zone["x2"]):
                        person["has_ppe"][ppe_type] = True
                
                elif ppe_type == 'gloves':
                    # For gloves, simplistic check for now
                    if (ppe_center_x >= px1 and 
                        ppe_center_x <= px2 and 
                        ppe_center_y >= py1 + p_height * 0.4 and 
                        ppe_center_y <= py2):
                        person["has_ppe"]["gloves"] = True
                
                elif ppe_type == 'boots':
                    # Check if boots are near the bottom of person
                    legs_zone = person_zones["legs"]
                    if (ppe_center_y >= legs_zone["y1"] and 
                        ppe_center_x >= legs_zone["x1"] and 
                        ppe_center_x <= legs_zone["x2"]):
                        person["has_ppe"]["boots"] = True
            
            # Apply safety rule for this location (using default rule for now)
            rule_name = "default"
            required_ppe = self.violation_rules.get(rule_name, ["hardhat", "vest"])
            
            # Check for violations
            for ppe_type in required_ppe:
                if not person["has_ppe"].get(ppe_type, False):
                    person["violations"].append(f"no_{ppe_type}")
            
            # Set overall violation flag and type
            person["violation"] = len(person["violations"]) > 0
            person["violation_type"] = ",".join(person["violations"])
            
            # Clean up internal tracking fields
            person.pop("has_ppe", None)
            person.pop("id", None)
            person.pop("std_class", None)
            person.pop("violations", None)
        
        # Add PPE items to results
        for ppe in ppe_items:
            ppe.pop("id", None)
            ppe.pop("std_class", None)
            ppe["violation"] = False
            ppe["violation_type"] = ""
            persons.append(ppe)
        
        return persons
    
    def _mock_detections(self, frame) -> List[Dict]:
        """Generate mock detections for testing without model."""
        height, width = frame.shape[:2]
        
        # Create a person detection with a safety violation
        person = {
            "bbox": [width * 0.3, height * 0.2, width * 0.7, height * 0.9],
            "class": "person",
            "confidence": 0.92,
            "violation": True,
            "violation_type": "no_hardhat,no_vest"
        }
        
        return [person]


# Singleton instance for global access
_detection_service = None

def get_detection_service() -> DetectionService:
    """Get or create the detection service singleton."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service

# Allow direct imports
import random  # Used for random logging