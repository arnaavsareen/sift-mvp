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
    Production-ready enhanced detection service with improved PPE detection.
    Detects persons and their PPE items (hardhat, vest, etc.) and accurately determines violations.
    """
    
    def __init__(
        self, 
        model_path: str = MODEL_PATH, 
        confidence: float = CONFIDENCE_THRESHOLD,
        device: Optional[str] = DEVICE
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = None
        self.class_names = {}
        self.last_inference_time = 0
        
        # Set a higher default threshold for more accurate detections
        self.detection_threshold = max(0.45, self.confidence)
        
        # Enhanced PPE class mapping for YOLO models - maps model class names to standard categories
        self.ppe_class_mapping = {
            # Person classes
            'person': 'person',
            'worker': 'person',
            'human': 'person',
            
            # Hard hat / Helmet classes
            'hardhat': 'hardhat',
            'hard_hat': 'hardhat',
            'helmet': 'hardhat',
            'safety_helmet': 'hardhat',
            'construction_helmet': 'hardhat',
            'hard-hat': 'hardhat',
            
            # Safety vest classes
            'vest': 'vest',
            'safety_vest': 'vest',
            'high_vis_vest': 'vest',
            'reflective_vest': 'vest',
            'visibility_vest': 'vest',
            'hi_vis_vest': 'vest',
            'safety-vest': 'vest',
            'high-vis': 'vest',
            
            # Mask classes
            'mask': 'mask',
            'face_mask': 'mask',
            'safety_mask': 'mask',
            'respirator': 'mask',
            
            # Goggles classes
            'goggles': 'goggles',
            'safety_goggles': 'goggles',
            'protective_goggles': 'goggles',
            'eye_protection': 'goggles',
            
            # Gloves classes
            'gloves': 'gloves',
            'safety_gloves': 'gloves',
            'work_gloves': 'gloves',
            'protective_gloves': 'gloves',
            
            # Boots classes
            'boots': 'boots',
            'safety_boots': 'boots',
            'work_boots': 'boots',
            'steel_toe_boots': 'boots',
            'protective_boots': 'boots',
        }
        
        # Enhanced PPE association rules - precision-tuned for accuracy
        self.ppe_association_rules = {
            'hardhat': {
                'distance_threshold': 0.7,   # How close PPE must be to person (as ratio of person box)
                'vertical_bias': 0.3,       # Bias upward for head area (0.3 = upper third)
                'area_threshold_min': 0.02,  # Minimum area relative to person
                'area_threshold_max': 0.3,   # Maximum area relative to person
                'position_check': 'top',     # Must be at the top of person
                'tolerance': 0.2,           # Acceptable deviation from position
            },
            'vest': {
                'distance_threshold': 0.6,
                'vertical_bias': 0.0,       # Center of person
                'area_threshold_min': 0.15,  # Minimum area (vests are larger)
                'area_threshold_max': 0.7,
                'position_check': 'middle', 
                'tolerance': 0.3,
            },
            'mask': {
                'distance_threshold': 0.6,
                'vertical_bias': 0.25,      # Face area
                'area_threshold_min': 0.01,
                'area_threshold_max': 0.15,
                'position_check': 'top',
                'tolerance': 0.2,
            },
            'goggles': {
                'distance_threshold': 0.6,
                'vertical_bias': 0.3,       # Face area
                'area_threshold_min': 0.01,
                'area_threshold_max': 0.2,
                'position_check': 'top',
                'tolerance': 0.2,
            },
            'gloves': {
                'distance_threshold': 0.5,
                'vertical_bias': -0.1,      # Lower body area
                'area_threshold_min': 0.01,
                'area_threshold_max': 0.25,
                'position_check': 'sides',
                'tolerance': 0.3,
            },
            'boots': {
                'distance_threshold': 0.5,
                'vertical_bias': -0.4,      # Bottom of person
                'area_threshold_min': 0.01,
                'area_threshold_max': 0.25,
                'position_check': 'bottom',
                'tolerance': 0.2,
            }
        }
        
        # Required PPE by default (can be overridden by zone configuration)
        self.default_required_ppe = ['hardhat', 'vest']
        
        # Confidence thresholds for violation reporting - higher confidence needed to report violations
        self.violation_confidence_threshold = 0.6  # Minimum confidence to report a violation
        
        # Initialize model
        self.load_model()
        
        logger.info("Production DetectionService initialized with enhanced settings")
    
    def load_model(self) -> bool:
        """Load YOLOv11 model with proper error handling and optimization."""
        try:
            model_path = Path(self.model_path)
            
            # Check if model exists
            if not model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                
                # Look for alternative models with YOLO11 priority
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
            person_classes = []
            for class_id, class_name in self.class_names.items():
                std_class = self._standardize_class(class_name)
                if std_class == 'person':
                    person_classes.append(f"{class_id}: {class_name}")
                elif std_class in ['hardhat', 'vest', 'mask', 'goggles', 'gloves', 'boots'] or class_name.startswith('no_'):
                    ppe_classes.append(f"{class_id}: {class_name}")
            
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            logger.info(f"Model has {len(self.class_names)} classes")
            logger.info(f"Person classes available: {', '.join(person_classes)}")
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
        Production-quality PPE detection with enhanced accuracy and robustness.
        
        Args:
            frame: Image frame (numpy array)
            
        Returns:
            List of detection dictionaries with enhanced violation detection
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
                
            # Ensure frame is properly formatted
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
            
            # Run YOLO detection with enhanced settings
            results = self.model(
                frame,
                conf=self.detection_threshold,  # Use higher threshold for cleaner results
                verbose=False,
                agnostic_nms=True,             # Class-agnostic NMS for better detection
                max_det=100,                   # Maximum detections per image
                iou=0.45                       # Adjust IoU threshold for NMS
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
            
            # Separate persons and PPE items with careful filtering
            persons = []
            ppe_items = []
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confs, class_ids)):
                # Skip very low confidence detections
                if conf < self.confidence:
                    continue
                    
                x1, y1, x2, y2 = box
                
                # Skip invalid bounding boxes
                if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0:
                    continue
                
                # Get class name
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                standardized_class = self._standardize_class(class_name)
                
                # Calculate area and center
                area = (x2 - x1) * (y2 - y1)
                center = [(x1 + x2) / 2, (y1 + y2) / 2]
                
                detection = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "class": class_name,
                    "std_class": standardized_class,
                    "confidence": float(conf),
                    "center": center,
                    "area": area,
                    "class_id": class_id
                }
                
                if standardized_class == "person":
                    detection["detected_ppe"] = []  # Initialize as empty list
                    detection["missing_ppe"] = []   # Initialize missing items list
                    detection["violations"] = []    # Initialize violations list
                    detection["violation"] = False  # Default: no violation
                    persons.append(detection)
                elif standardized_class in ['hardhat', 'vest', 'mask', 'goggles', 'gloves', 'boots']:
                    ppe_items.append(detection)
                
                # Add all detections to the main list
                detections.append(detection)
            
            # Only proceed with PPE association if we have persons
            if persons:
                # Associate PPE with persons using enhanced method
                detections = self._associate_ppe_with_persons_enhanced(persons, ppe_items)
                
                # Check for violations only AFTER all PPE associations are complete
                detections = self._check_ppe_violations_enhanced(detections)
            
            logger.debug(f"Processed {len(detections)} detections, {len(persons)} persons")
            return detections
        
        except Exception as e:
            logger.error(f"Error in detection: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
        
        # Check for combined class names (e.g., "person_with_hardhat")
        if 'person' in class_lower:
            return 'person'
        
        # Return original if no match found
        return class_name
    
    def _associate_ppe_with_persons_enhanced(self, persons: List[Dict], ppe_items: List[Dict]) -> List[Dict]:
        """
        Enhanced PPE association using spatial analysis and object area filtering.
        """
        for person in persons:
            person_bbox = person["bbox"]
            person_center = person["center"]
            person_area = person["area"]
            
            # Calculate expanded search area for different PPE types
            for ppe in ppe_items:
                ppe_type = ppe["std_class"]
                
                # Skip if this is a violation detection
                if ppe_type == "violation":
                    continue
                
                # Get association rules for this PPE type
                rules = self.ppe_association_rules.get(ppe_type, {
                    'distance_threshold': 0.6,
                    'vertical_bias': 0.0,
                    'area_threshold_min': 0.01,
                    'area_threshold_max': 0.5
                })
                
                # Check spatial relationship
                if self._is_ppe_associated_with_person(person_bbox, ppe, rules):
                    # Add PPE to person's detected items
                    if ppe_type not in person["detected_ppe"]:
                        person["detected_ppe"].append(ppe_type)
                        logger.debug(f"Associated {ppe_type} with person at {person_center}")
        
        return persons + ppe_items
    
    def _is_ppe_associated_with_person(self, person_bbox: List[float], ppe: Dict, rules: Dict) -> bool:
        """
        Enhanced PPE association with more accurate spatial reasoning and position verification.
        """
        person_x1, person_y1, person_x2, person_y2 = person_bbox
        ppe_bbox = ppe["bbox"]
        ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe_bbox
        
        # Calculate person and PPE properties
        person_width = person_x2 - person_x1
        person_height = person_y2 - person_y1
        person_area = person_width * person_height
        ppe_area = ppe["area"]
        
        # Skip if the PPE is too large or too small relative to the person
        area_ratio = ppe_area / person_area
        if area_ratio < rules['area_threshold_min'] or area_ratio > rules['area_threshold_max']:
            return False
        
        # Calculate centers
        ppe_center_x = (ppe_x1 + ppe_x2) / 2
        ppe_center_y = (ppe_y1 + ppe_y2) / 2
        
        # Calculate adjusted person center based on PPE type vertical bias
        vertical_bias = rules['vertical_bias']
        person_center_x = (person_x1 + person_x2) / 2
        person_center_y = person_y1 + (person_y2 - person_y1) * (0.5 + vertical_bias)
        
        # Calculate distance relative to person size
        dx = abs(ppe_center_x - person_center_x) / person_width
        dy = abs(ppe_center_y - person_center_y) / person_height
        
        # Check if PPE is within threshold distance
        distance_threshold = rules['distance_threshold']
        if dx > distance_threshold or dy > distance_threshold:
            return False
        
        # Position-specific checks based on the type of PPE
        position_check = rules.get('position_check', None)
        tolerance = rules.get('tolerance', 0.2)
        
        if position_check == 'top':
            # Check if PPE is in the top portion of the person
            ppeCenterYRatio = (ppe_center_y - person_y1) / person_height
            if ppeCenterYRatio > 0.4:  # Should be in top 40% of person
                return False
                
        elif position_check == 'middle':
            # Check if PPE is in the middle portion of the person
            ppeCenterYRatio = (ppe_center_y - person_y1) / person_height
            if ppeCenterYRatio < 0.2 or ppeCenterYRatio > 0.8:
                return False
                
        elif position_check == 'bottom':
            # Check if PPE is in the bottom portion of the person
            ppeCenterYRatio = (ppe_center_y - person_y1) / person_height
            if ppeCenterYRatio < 0.6:  # Should be in bottom 40% of person
                return False
                
        elif position_check == 'sides':
            # Check if PPE is on the sides (e.g., for gloves)
            ppeCenterXRatio = (ppe_center_x - person_x1) / person_width
            # Should be on the left side or right side
            if ppeCenterXRatio > 0.3 and ppeCenterXRatio < 0.7:
                return False
        
        # If we've passed all checks, the PPE is associated with this person
        return True
    
    def _check_ppe_violations_enhanced(self, detections: List[Dict]) -> List[Dict]:
        """
        Production-quality PPE violation checking with confidence scoring and false positive reduction.
        """
        for detection in detections:
            # Only check violations for person detections
            if detection["std_class"] != "person":
                continue
            
            # Get required PPE (in production this could be overridden by zone rules)
            required_ppe = self.default_required_ppe.copy()
            
            # Skip low-confidence person detections for violation reporting
            # This reduces false positive violations
            if detection["confidence"] < self.violation_confidence_threshold:
                detection["violation"] = False
                detection["violation_type"] = ""
                detection["violation_description"] = "Low confidence detection - not evaluated"
                continue
            
            # Check for missing PPE
            missing_ppe = []
            for required in required_ppe:
                if required not in detection["detected_ppe"]:
                    missing_ppe.append(required)
            
            # Set violation flags only if items are missing
            if missing_ppe:
                # Generate violation types
                violation_types = []
                for item in missing_ppe:
                    violation_types.append(f"no_{item}")
                
                # Calculate violation confidence based on detection confidence
                # Higher person confidence = more confident in the violation
                violation_confidence = detection["confidence"] * 0.95
                
                # Add violation details
                detection["violation"] = True
                detection["violation_type"] = ",".join(violation_types)
                detection["violations"] = missing_ppe
                detection["missing_ppe"] = missing_ppe
                detection["violation_confidence"] = violation_confidence
                
                # Create human-readable description
                if len(missing_ppe) == 1:
                    detection["violation_description"] = f"Missing {missing_ppe[0].replace('_', ' ')}"
                else:
                    detection["violation_description"] = f"Missing PPE: {', '.join([item.replace('_', ' ') for item in missing_ppe])}"
                
                logger.debug(f"Violation detected: {detection['violation_type']} (confidence: {violation_confidence:.2f})")
            else:
                # Person has all required PPE - no violation
                detection["violation"] = False
                detection["violation_type"] = ""
                detection["missing_ppe"] = []
                detection["violation_description"] = "All required PPE detected"
        
        return detections
    
    def set_required_ppe(self, required_ppe: List[str]) -> None:
        """Set the list of required PPE items for violation checking."""
        self.default_required_ppe = required_ppe
        logger.info(f"Updated required PPE: {required_ppe}")
    
    def get_model_classes(self) -> Dict[int, str]:
        """Get all model classes."""
        return self.class_names.copy() if self.class_names else {}
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection service statistics."""
        return {
            "model_path": self.model_path,
            "confidence_threshold": self.confidence,
            "device": str(self.device),
            "last_inference_time_ms": self.last_inference_time * 1000,
            "model_classes": len(self.class_names),
            "required_ppe": self.default_required_ppe,
            "is_loaded": self.model is not None
        }


# Singleton instance for global access
_detection_service = None

def get_detection_service() -> DetectionService:
    """Get or create the detection service singleton."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service