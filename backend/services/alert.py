import os
import cv2
import numpy as np
import logging
import uuid
import json
import time
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func
import asyncio

from backend.models import Alert, Camera
from backend.config import SCREENSHOTS_DIR

logger = logging.getLogger(__name__)

class AlertService:
    """
    Enhanced alert service with improved violation tracking and smart alerting.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.alert_cache = {}
        self.cache_ttl = 60
        self.min_alert_interval = 5.0
        self.last_alert_times = {}
        self.consecutive_violation_threshold = 2  # Require 2 consecutive violations
        self.violation_counters = {}  # Track consecutive violations
        
        # Ensure screenshots directory exists
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        
        # Enhanced color mapping for violations
        self.violation_colors = {
            "no_hardhat": (0, 0, 255),      # Red
            "no_vest": (0, 165, 255),       # Orange  
            "no_mask": (128, 0, 255),       # Purple
            "no_goggles": (255, 0, 128),    # Pink
            "no_gloves": (128, 128, 0),     # Olive
            "no_boots": (0, 128, 128),      # Teal
            "restricted_area": (255, 0, 0), # Bright red
            "default": (0, 0, 255)          # Default red
        }
        
        logger.info("Enhanced Alert Service initialized")
    
    def process_alerts(self, camera_id: int, detections: List[Dict], frame) -> List[Dict]:
        """
        Enhanced alert processing with smart duplicate detection and false positive reduction.
        """
        violations = [d for d in detections if d.get("violation", False)]
        
        if not violations:
            # Reset consecutive violation counters for this camera
            self._reset_violation_counters(camera_id)
            return []
        
        camera = self._get_camera(camera_id)
        alerts = []
        now = datetime.now()
        
        # Clean expired entries
        self._clean_alert_cache()
        
        for violation in violations:
            violation_type = violation.get("violation_type", "unknown")
            if not violation_type:
                violation_type = "ppe_violation"
            
            bbox = violation.get("bbox", [0, 0, 0, 0])
            confidence = violation.get("confidence", 0.0)
            detected_ppe = violation.get("detected_ppe", [])
            
            # Generate cache key for deduplication
            cache_key = self._generate_cache_key(camera_id, bbox, violation_type)
            
            # Check for duplicate alerts
            if cache_key in self.alert_cache:
                continue
            
            # Track consecutive violations for this position
            violation_counter_key = f"{camera_id}_{cache_key}"
            if violation_counter_key not in self.violation_counters:
                self.violation_counters[violation_counter_key] = {
                    'count': 0,
                    'first_seen': now,
                    'last_seen': now
                }
            
            # Increment counter
            self.violation_counters[violation_counter_key]['count'] += 1
            self.violation_counters[violation_counter_key]['last_seen'] = now
            
            # Check if we have enough consecutive violations
            if self.violation_counters[violation_counter_key]['count'] < self.consecutive_violation_threshold:
                logger.debug(f"Waiting for more consecutive violations: {self.violation_counters[violation_counter_key]['count']}/{self.consecutive_violation_threshold}")
                continue
            
            # Add to cache to prevent duplicates
            self.alert_cache[cache_key] = {
                "timestamp": time.time(),
                "alert_count": 1
            }
            
            # Generate enhanced screenshot
            screenshot_path = self._save_enhanced_screenshot(
                frame, camera_id, violation, violation_type, detected_ppe
            )
            
            # Create alert record with enhanced metadata
            alert = Alert(
                camera_id=camera_id,
                violation_type=violation_type,
                confidence=confidence,
                bbox=bbox,
                screenshot_path=screenshot_path,
                created_at=now
            )
            
            # Save to database
            try:
                self.db.add(alert)
                self.db.commit()
                self.db.refresh(alert)
                
                # Build enhanced alert response
                alert_dict = {
                    "id": alert.id,
                    "camera_id": alert.camera_id,
                    "violation_type": alert.violation_type,
                    "confidence": alert.confidence,
                    "bbox": alert.bbox,
                    "screenshot_path": alert.screenshot_path,
                    "created_at": alert.created_at.isoformat() if alert.created_at else None,
                    "detected_ppe": detected_ppe,
                    "violation_description": self._get_violation_description(violation_type, detected_ppe),
                    "severity": self._calculate_violation_severity(violation_type),
                    "consecutive_count": self.violation_counters[violation_counter_key]['count']
                }
                
                alerts.append(alert_dict)
                
                # Update last alert time
                self.last_alert_times[camera_id] = time.time()
                
                # Reset counter after successful alert
                self.violation_counters[violation_counter_key]['count'] = 0
                
                # Queue for WebSocket broadcast
                try:
                    from backend.main import get_alert_queue
                    get_alert_queue().put((camera_id, alert_dict))
                    logger.info(f"Enhanced alert created for camera {camera_id}: {violation_type} (confidence: {confidence:.2f})")
                except Exception as e:
                    logger.error(f"Error queueing alert for broadcast: {str(e)}")
                
            except Exception as e:
                logger.error(f"Database error when creating alert: {str(e)}")
                self.db.rollback()
        
        if alerts:
            logger.info(f"Generated {len(alerts)} enhanced alerts for camera {camera_id}")
        
        return alerts
    
    def _save_enhanced_screenshot(
        self,
        frame,
        camera_id: int,
        violation: Dict,
        violation_type: str,
        detected_ppe: List[str]
    ) -> str:
        """Save enhanced screenshot with detailed annotations."""
        try:
            # Create copy with proper color conversion
            if len(frame.shape) == 2:
                screenshot = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                screenshot = frame.copy()
            
            # Add dark overlay for better text visibility
            h, w = screenshot.shape[:2]
            overlay = screenshot.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            screenshot = cv2.addWeighted(overlay, 0.2, screenshot, 0.8, 0)
            
            # Draw violation bounding box
            if "bbox" in violation:
                bbox = violation["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Get violation-specific color
                color = self.violation_colors.get(violation_type.split(',')[0], self.violation_colors["default"])
                
                # Draw main bounding box
                cv2.rectangle(screenshot, (x1, y1), (x2, y2), color, 3)
                
                # Draw alert indicator
                cv2.circle(screenshot, (x1 + 20, y1 + 20), 15, color, -1)
                cv2.putText(screenshot, "!", (x1 + 13, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                
                # Create comprehensive label
                label_lines = []
                label_lines.append(f"VIOLATION: {violation_type.replace('_', ' ').upper()}")
                label_lines.append(f"Confidence: {violation.get('confidence', 0.0):.2f}")
                
                if detected_ppe:
                    label_lines.append(f"Detected PPE: {', '.join(detected_ppe)}")
                else:
                    label_lines.append("No PPE detected")
                
                # Draw text background
                text_height = 25
                total_height = len(label_lines) * text_height + 20
                cv2.rectangle(
                    screenshot,
                    (x1, max(0, y1 - total_height)),
                    (min(w, x1 + 400), y1),
                    color,
                    -1
                )
                
                # Draw text lines
                for i, line in enumerate(label_lines):
                    cv2.putText(
                        screenshot,
                        line,
                        (x1 + 10, max(20, y1 - total_height + (i + 1) * text_height)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
            
            # Add comprehensive metadata
            metadata_lines = [
                f"Camera: {self._get_camera_name(camera_id)}",
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Alert ID: {uuid.uuid4().hex[:8]}",
                f"Violation: {violation_type}",
                f"Severity: {self._calculate_violation_severity(violation_type)}"
            ]
            
            # Draw metadata background
            metadata_height = len(metadata_lines) * 25 + 20
            cv2.rectangle(
                screenshot,
                (0, h - metadata_height),
                (w, h),
                (0, 0, 0),
                -1
            )
            
            # Draw metadata text
            for i, line in enumerate(metadata_lines):
                cv2.putText(
                    screenshot,
                    line,
                    (10, h - metadata_height + (i + 1) * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            # Generate unique filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{camera_id}_{violation_type.split(',')[0]}_{timestamp_str}_{uuid.uuid4().hex[:6]}.jpg"
            filepath = os.path.join(SCREENSHOTS_DIR, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save with high quality
            cv2.imwrite(filepath, screenshot, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return f"/screenshots/{filename}"
            
        except Exception as e:
            logger.error(f"Error saving enhanced screenshot: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def _get_violation_description(self, violation_type: str, detected_ppe: List[str] = None) -> str:
        """Generate descriptive text for violation type."""
        descriptions = {
            "no_hardhat": "Worker detected without required hard hat protection",
            "no_vest": "Worker detected without required high-visibility safety vest",
            "no_mask": "Worker detected without required face mask protection",
            "no_goggles": "Worker detected without required eye protection",
            "no_gloves": "Worker detected without required hand protection",
            "no_boots": "Worker detected without required safety footwear"
        }
        
        if "," in violation_type:
            # Multiple violations
            violations = violation_type.split(",")
            desc = "Multiple violations detected: "
            desc += ", ".join([descriptions.get(v.strip(), v.strip()) for v in violations])
        else:
            desc = descriptions.get(violation_type, f"Safety violation: {violation_type}")
        
        if detected_ppe:
            desc += f". Currently wearing: {', '.join(detected_ppe)}"
        
        return desc
    
    def _calculate_violation_severity(self, violation_type: str) -> str:
        """Calculate violation severity level."""
        if not violation_type:
            return "LOW"
        
        severity_mapping = {
            "no_hardhat": "HIGH",
            "no_vest": "HIGH", 
            "no_mask": "MEDIUM",
            "no_goggles": "MEDIUM",
            "no_gloves": "LOW",
            "no_boots": "LOW"
        }
        
        # Check for multiple violations
        if "," in violation_type:
            return "CRITICAL"
        
        return severity_mapping.get(violation_type, "MEDIUM")
    
    def _get_camera_name(self, camera_id: int) -> str:
        """Get camera name or return default."""
        camera = self._get_camera(camera_id)
        return camera.name if camera else f"Camera {camera_id}"
    
    def _reset_violation_counters(self, camera_id: int) -> None:
        """Reset violation counters for a specific camera."""
        keys_to_remove = [key for key in self.violation_counters.keys() if key.startswith(f"{camera_id}_")]
        for key in keys_to_remove:
            del self.violation_counters[key]
    
    def _generate_cache_key(self, camera_id: int, bbox: List[float], violation_type: str) -> str:
        """Generate a unique cache key for alert deduplication."""
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Quantize position to grid to handle small movements
            center_x = int((x1 + x2) / 2 / 30) * 30  # 30-pixel grid
            center_y = int((y1 + y2) / 2 / 30) * 30
            width = int((x2 - x1) / 20) * 20  # 20-pixel grid for size
            height = int((y2 - y1) / 20) * 20
            
            return f"{camera_id}_{center_x}_{center_y}_{width}_{height}_{violation_type}"
        
        # Fallback if bbox is invalid
        return f"{camera_id}_{violation_type}_{time.time()}"
    
    def _clean_alert_cache(self) -> None:
        """Remove expired entries from the alert cache."""
        now = time.time()
        expired_keys = []
        
        for key, data in self.alert_cache.items():
            if now - data["timestamp"] > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.alert_cache[key]
    
    def _get_camera(self, camera_id: int) -> Optional[Camera]:
        """Get camera data from database."""
        try:
            return self.db.query(Camera).filter(Camera.id == camera_id).first()
        except Exception as e:
            logger.error(f"Error fetching camera {camera_id}: {str(e)}")
            return None

    def create_alert(
        self, 
        camera_id: int, 
        violation_type: str, 
        confidence: float,
        frame,
        bbox: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new safety violation alert with rate limiting.
        
        Args:
            camera_id: ID of the camera that detected the violation
            violation_type: Type of violation detected
            confidence: Confidence score of the detection
            frame: The frame image where the violation was detected
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            metadata: Additional metadata about the violation
            
        Returns:
            Alert data dictionary if created, None otherwise
        """
        try:
            # Check if we should create a new alert (rate limiting)
            current_time = time.time()
            if camera_id in self.last_alert_times:
                time_since_last = current_time - self.last_alert_times[camera_id]
                if time_since_last < self.min_alert_interval:
                    logger.debug(f"Rate limiting alert for camera {camera_id} ({time_since_last:.1f}s < {self.min_alert_interval:.1f}s)")
                    return None
            
            # Get camera details
            camera = self.db.query(Camera).filter(Camera.id == camera_id).first()
            if not camera:
                logger.error(f"Camera not found: {camera_id}")
                return None
                
            # Generate a unique screenshot filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"violation_{camera_id}_{timestamp}_{unique_id}.jpg"
            filepath = os.path.join(SCREENSHOTS_DIR, filename)
            
            # Save the screenshot with enhanced annotations
            if frame is not None:
                # Make a copy to avoid modifying the original frame
                alert_frame = frame.copy()
                
                # Draw bounding box on the frame if coordinates are provided
                if bbox:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    cv2.rectangle(alert_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Add violation type and confidence as text
                    text = f"{violation_type}: {confidence:.2f}"
                    cv2.putText(alert_frame, text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Save the frame as an image
                cv2.imwrite(filepath, alert_frame)
                logger.info(f"Saved violation screenshot to {filepath}")
            else:
                logger.warning("No frame provided for screenshot")
                
            # Create alert record
            alert = Alert(
                camera_id=camera_id,
                violation_type=violation_type,
                confidence=confidence,
                screenshot_path=filename,
                bbox=bbox
            )
            
            self.db.add(alert)
            self.db.commit()
            self.db.refresh(alert)
            
            # Update last alert time
            self.last_alert_times[camera_id] = current_time
            
            # Convert alert to dict for return and WebSocket broadcast
            alert_dict = {
                "id": alert.id,
                "camera_id": alert.camera_id,
                "violation_type": alert.violation_type,
                "confidence": alert.confidence,
                "timestamp": alert.created_at.isoformat(),
                "screenshot_url": f"/screenshots/{alert.screenshot_path}",
                "resolved": alert.resolved
            }
            
            # Broadcast alert via WebSocket if available
            try:
                from backend.main import get_alert_queue
                
                # Add the alert to the queue for broadcasting from the main thread
                get_alert_queue().put((camera_id, alert_dict))
                logger.info(f"Added alert for camera {camera_id} to broadcast queue (violation: {violation_type}, confidence: {confidence:.2f})")
                
            except ImportError:
                logger.debug("WebSocket broadcast not available")
            except Exception as e:
                logger.error(f"Error queueing alert for broadcast: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
            return alert_dict
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Rollback in case of error
            self.db.rollback()
            return None

    def get_recent_alerts(self, camera_id: Optional[int] = None, limit: int = 10) -> List[Dict]:
        """
        Get recent alerts with optional camera filtering.
        
        Args:
            camera_id: Optional camera ID filter
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        query = self.db.query(Alert).order_by(Alert.created_at.desc())
        
        if camera_id is not None:
            query = query.filter(Alert.camera_id == camera_id)
        
        alerts = query.limit(limit).all()
        
        return [
            {
                "id": alert.id,
                "camera_id": alert.camera_id,
                "violation_type": alert.violation_type,
                "confidence": alert.confidence,
                "screenshot_path": alert.screenshot_path,
                "created_at": alert.created_at.isoformat() if alert.created_at else None,
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            for alert in alerts
        ]

    def get_alert_stats(self, camera_id: Optional[int] = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get alert statistics for dashboard displays.
        
        Args:
            camera_id: Optional camera ID filter
            hours: Time window in hours
            
        Returns:
            Dictionary of alert statistics
        """
        # Import here to avoid circular import
        from datetime import timedelta
        
        # Calculate time threshold
        time_threshold = datetime.now() - timedelta(hours=hours)
        
        # Base query
        query = self.db.query(Alert).filter(Alert.created_at >= time_threshold)
        
        if camera_id is not None:
            query = query.filter(Alert.camera_id == camera_id)
        
        # Total alerts
        total_alerts = query.count()
        active_alerts = query.filter(Alert.resolved == False).count()
        
        # Alerts by type
        alerts_by_type = {}
        violation_types = self.db.query(Alert.violation_type).distinct().all()
        
        for type_row in violation_types:
            violation_type = type_row[0]
            count = query.filter(Alert.violation_type == violation_type).count()
            alerts_by_type[violation_type] = count
        
        # Alerts by camera (if not filtered by camera)
        alerts_by_camera = {}
        
        if camera_id is None:
            cameras = self.db.query(Camera).all()
            
            for camera in cameras:
                count = query.filter(Alert.camera_id == camera.id).count()
                if count > 0:  # Only include cameras with alerts
                    alerts_by_camera[camera.id] = {
                        "camera_id": camera.id,
                        "name": camera.name,
                        "location": camera.location,
                        "count": count
                    }
        
        # Get hourly distribution
        hourly_counts = []
        for hour in range(hours):
            hour_start = datetime.now() - timedelta(hours=hours-hour)
            hour_end = datetime.now() - timedelta(hours=hours-hour-1)
            
            hour_count = query.filter(
                Alert.created_at >= hour_start,
                Alert.created_at < hour_end
            ).count()
            
            hourly_counts.append({
                "hour": hour_start.strftime("%H:%M"),
                "count": hour_count
            })
        
        return {
            "time_range_hours": hours,
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "by_type": alerts_by_type,
            "by_camera": alerts_by_camera if camera_id is None else None,
            "hourly_distribution": hourly_counts
        }


# Factory function to create alert service
def get_alert_service(db: Session) -> AlertService:
   """Create alert service with database session."""
   return AlertService(db)