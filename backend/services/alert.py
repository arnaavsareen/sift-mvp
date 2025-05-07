import os
import cv2
import logging
import uuid
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from backend.models import Alert
from backend.config import SCREENSHOTS_DIR

logger = logging.getLogger(__name__)

class AlertService:
    """
    Service for generating alerts based on detected violations.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def process_alerts(self, camera_id: int, detections: List[Dict], frame) -> List[Dict]:
        """
        Process detections and generate alerts for violations.
        
        Args:
            camera_id: Camera ID
            detections: List of detection dictionaries
            frame: Original image frame
            
        Returns:
            List of alert dictionaries
        """
        # Filter for violations
        violations = [d for d in detections if d.get("violation", False)]
        
        if not violations:
            return []
        
        alerts = []
        
        for violation in violations:
            # Save screenshot with annotation
            screenshot_path = self._save_screenshot(frame, violation, camera_id)
            
            # Get violation type
            violation_type = violation.get("violation_type", "unknown")
            if not violation_type:
                violation_type = "ppe_violation"  # Default type
            
            # Create alert record
            alert = Alert(
                camera_id=camera_id,
                violation_type=violation_type,
                confidence=violation.get("confidence", 0.0),
                bbox=violation.get("bbox"),
                screenshot_path=screenshot_path,
                created_at=datetime.now()
            )
            
            # Save to database
            self.db.add(alert)
            self.db.commit()
            self.db.refresh(alert)
            
            # Add to result list
            alerts.append({
                "id": alert.id,
                "camera_id": alert.camera_id,
                "violation_type": alert.violation_type,
                "confidence": alert.confidence,
                "screenshot_path": alert.screenshot_path,
                "created_at": alert.created_at.isoformat() if alert.created_at else None
            })
        
        return alerts
    
    def _save_screenshot(self, frame, violation, camera_id) -> str:
        """
        Save screenshot of violation.
        
        Args:
            frame: Original image frame
            violation: Detection dictionary
            camera_id: Camera ID
            
        Returns:
            Path to saved screenshot
        """
        try:
            # Create a copy of the frame
            screenshot = frame.copy()
            
            # Draw violation bounding box
            if "bbox" in violation:
                bbox = violation["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw red box around violation
                cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add label
                label = f"{violation.get('class', 'Unknown')}: {violation.get('confidence', 0.0):.2f}"
                cv2.putText(
                    screenshot,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cam{camera_id}_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
            filepath = os.path.join(SCREENSHOTS_DIR, filename)
            
            # Ensure directory exists
            os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
            
            # Save image
            cv2.imwrite(filepath, screenshot)
            
            # Return relative path for storage in database
            return f"/screenshots/{filename}"
            
        except Exception as e:
            logger.error(f"Error saving screenshot: {str(e)}")
            return ""


# Factory function to create alert service
def get_alert_service(db: Session) -> AlertService:
    """Create alert service with database session."""
    return AlertService(db)