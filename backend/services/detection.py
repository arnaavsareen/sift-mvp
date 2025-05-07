import os
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import time

# Conditionally import ultralytics to handle environments where it might not be available
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.config import MODEL_PATH, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

class DetectionService:
    """
    Service for object detection using YOLO models.
    """
    
    def __init__(self, model_path: str = MODEL_PATH, confidence: float = CONFIDENCE_THRESHOLD):
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        
        # Initialize model
        self.load_model()
    
    def load_model(self) -> bool:
        """Load YOLO model."""
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Ultralytics not available. Using mock detection.")
            return False
        
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            start_time = time.time()
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def detect(self, frame) -> List[Dict[str, Any]]:
        """
        Run object detection on a frame.
        
        Args:
            frame: Image frame (numpy array)
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] coordinates
            - class: Class name
            - confidence: Detection confidence
            - violation: Whether this detection represents a violation
        """
        if self.model is None:
            # Return mock detections for testing
            return self._mock_detections(frame)
        
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence)[0]
            
            # Process results
            detections = []
            
            # Convert tensor results to numpy arrays
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confs, class_ids)):
                class_name = results.names[class_id]
                
                detection = {
                    "bbox": box.tolist(),
                    "class": class_name,
                    "confidence": float(conf),
                    "violation": False  # Will be determined by rules
                }
                
                detections.append(detection)
            
            # Apply rules to determine violations
            return self._apply_rules(detections, frame)
            
        except Exception as e:
            logger.error(f"Error in detection: {str(e)}")
            return []
    
    def _apply_rules(self, detections: List[Dict], frame) -> List[Dict]:
        """
        Apply safety rules to detections to determine violations.
        
        Simple rule for MVP: Each person should have PPE (hardhat, vest) nearby.
        """
        # Find persons and PPE items
        persons = [d for d in detections if d["class"] == "person"]
        hardhats = [d for d in detections if d["class"] in ["hardhat", "helmet", "hard hat"]]
        vests = [d for d in detections if d["class"] in ["vest", "safety vest"]]
        
        # For each person, check if they have required PPE
        for person in persons:
            person_box = person["bbox"]
            has_hardhat = False
            has_vest = False
            
            # Check for nearby hardhat
            for hardhat in hardhats:
                if self._is_ppe_associated(person_box, hardhat["bbox"]):
                    has_hardhat = True
                    break
            
            # Check for nearby vest
            for vest in vests:
                if self._is_ppe_associated(person_box, vest["bbox"]):
                    has_vest = True
                    break
            
            # Mark violation if missing PPE
            person["violation"] = not (has_hardhat and has_vest)
            person["violation_type"] = []
            
            if not has_hardhat:
                person["violation_type"].append("no_hardhat")
            
            if not has_vest:
                person["violation_type"].append("no_vest")
            
            person["violation_type"] = ",".join(person["violation_type"]) if person["violation_type"] else ""
        
        return detections
    
    def _is_ppe_associated(self, person_box, ppe_box) -> bool:
        """
        Determine if PPE is associated with a person.
        
        Simple heuristic for MVP:
        - Hardhat should be above and overlapping with person horizontally
        - Vest should overlap with person's torso
        """
        # Unpack boxes
        p_x1, p_y1, p_x2, p_y2 = person_box
        ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe_box
        
        # Check horizontal overlap
        horizontal_overlap = max(0, min(p_x2, ppe_x2) - max(p_x1, ppe_x1))
        
        # Check if PPE is inside or near person box
        is_inside_x = horizontal_overlap > 0
        is_above = ppe_y2 < p_y1 + (p_y2 - p_y1) * 0.3  # Above head/shoulders
        
        return is_inside_x and is_above
    
    def _mock_detections(self, frame) -> List[Dict]:
        """Generate mock detections for testing without model."""
        height, width = frame.shape[:2]
        
        # Create a person detection in center
        person = {
            "bbox": [width * 0.3, height * 0.2, width * 0.7, height * 0.9],
            "class": "person",
            "confidence": 0.92,
            "violation": True,
            "violation_type": "no_hardhat"
        }
        
        return [person]


# Singleton instance
_detection_service = None

def get_detection_service() -> DetectionService:
    """Get or create the detection service singleton."""
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service