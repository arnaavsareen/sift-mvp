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
    Service for generating and managing alerts based on detected safety violations.
    Includes enhanced visualization, alert deduplication, and persistent storage.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.alert_cache = {}  # Cache of recent alerts to prevent duplicates
        self.cache_ttl = 60  # Time to live for cached alerts (seconds)
        self.min_alert_interval = 5.0  # Minimum time between alerts for the same camera
        self.last_alert_times = {}  # Track last alert time per camera
        
        # Ensure screenshots directory exists
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        
        # Color mapping for different violation types
        self.violation_colors = {
            "no_hardhat": (0, 0, 255),    # Red for missing hardhat
            "no_vest": (0, 165, 255),     # Orange for missing vest
            "no_mask": (0, 0, 128),       # Dark red for missing mask
            "no_goggles": (128, 0, 128),  # Purple for missing goggles
            "no_gloves": (128, 0, 0),     # Dark blue for missing gloves
            "no_boots": (0, 128, 128),    # Teal for missing boots
            "default": (0, 0, 255)        # Default red
        }
    
    def process_alerts(self, camera_id: int, detections: List[Dict], frame) -> List[Dict]:
        """
        Process detections and generate alerts for violations.
        Includes deduplication, annotation, and persistent storage.
        
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
        
        # Get camera details
        camera = self._get_camera(camera_id)
        
        alerts = []
        now = datetime.now()
        
        # Clear expired entries from alert cache
        self._clean_alert_cache()
        
        for violation in violations:
            # Extract violation info
            violation_type = violation.get("violation_type", "unknown")
            if not violation_type:
                violation_type = "ppe_violation"  # Default type
            
            bbox = violation.get("bbox", [0, 0, 0, 0])
            confidence = violation.get("confidence", 0.0)
            
            # Check if this is a duplicate alert (same location, type, and recent)
            cache_key = self._generate_cache_key(camera_id, bbox, violation_type)
            
            if cache_key in self.alert_cache:
                # Skip duplicate alert
                continue
            
            # Add to cache to prevent duplicates
            self.alert_cache[cache_key] = {
                "timestamp": time.time(),
                "alert_count": 1
            }
            
            # Generate alert screenshot with annotations
            screenshot_path = self._save_screenshot(
                frame, camera_id, violation, violation_type
            )
            
            # Create alert record
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
                
                # Build alert response dictionary
                alert_dict = {
                    "id": alert.id,
                    "camera_id": alert.camera_id,
                    "violation_type": alert.violation_type,
                    "confidence": alert.confidence,
                    "bbox": alert.bbox,
                    "screenshot_path": alert.screenshot_path,
                    "created_at": alert.created_at.isoformat() if alert.created_at else None
                }
                
                alerts.append(alert_dict)
                
                # Update last alert time
                self.last_alert_times[camera_id] = time.time()
                
                # Broadcast alert via WebSocket if available
                try:
                    from backend.main import broadcast_alert
                    
                    # Schedule the WebSocket broadcast in the background
                    asyncio.create_task(broadcast_alert(camera_id, alert_dict))
                    
                except ImportError:
                    logger.debug("WebSocket broadcast not available")
                except Exception as e:
                    logger.error(f"Error broadcasting alert: {str(e)}")
                
            except Exception as e:
                logger.error(f"Database error when creating alert: {str(e)}")
                self.db.rollback()
        
        if alerts:
            logger.info(f"Generated {len(alerts)} new alerts for camera {camera_id}")
        
        return alerts
    
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
    
    def _save_screenshot(
        self, 
        frame, 
        camera_id: int, 
        violation: Dict, 
        violation_type: str
    ) -> str:
        """
        Save enhanced screenshot of violation with annotations.
        
        Args:
            frame: Original image frame
            camera_id: Camera ID
            violation: Detection dictionary
            violation_type: Type of violation detected
            
        Returns:
            Path to saved screenshot
        """
        try:
            # Create a copy of the frame with proper color conversion
            if len(frame.shape) == 2:  # Grayscale
                screenshot = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                screenshot = frame.copy()
            
            # Draw semi-transparent overlay on the entire image
            overlay = screenshot.copy()
            cv2.rectangle(
                overlay, 
                (0, 0), 
                (screenshot.shape[1], screenshot.shape[0]), 
                (0, 0, 0), 
                -1
            )
            # Apply semi-transparent overlay (darken the image)
            screenshot = cv2.addWeighted(overlay, 0.3, screenshot, 0.7, 0)
            
            # Draw violation bounding box
            if "bbox" in violation:
                bbox = violation["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Get appropriate color for this violation type
                color = self.violation_colors.get(violation_type.split(',')[0], self.violation_colors["default"])
                
                # Draw box around violation (thick enough to be visible)
                cv2.rectangle(screenshot, (x1, y1), (x2, y2), color, 3)
                
                # Draw filled rectangle for text background
                label = f"{violation.get('class', 'Person')}: {violation_type.replace('_', ' ')}"
                confidence_text = f"Confidence: {violation.get('confidence', 0.0):.2f}"
                
                # Get text sizes
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                conf_size, _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Text background height based on text
                text_bg_height = label_size[1] + conf_size[1] + 20
                text_bg_width = max(label_size[0], conf_size[0]) + 20
                
                # Draw text background
                cv2.rectangle(
                    screenshot, 
                    (x1, y1 - text_bg_height), 
                    (x1 + text_bg_width, y1), 
                    color, 
                    -1
                )
                
                # Draw main violation text
                cv2.putText(
                    screenshot,
                    label,
                    (x1 + 10, y1 - text_bg_height + label_size[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Draw confidence value
                cv2.putText(
                    screenshot,
                    confidence_text,
                    (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            # Add alert metadata at bottom of image
            h, w = screenshot.shape[:2]
            
            # Draw metadata background
            metadata_height = 60
            cv2.rectangle(
                screenshot, 
                (0, h - metadata_height), 
                (w, h), 
                (0, 0, 0), 
                -1
            )
            
            # Draw alert timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                screenshot,
                f"Time: {timestamp}",
                (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Draw camera identifier
            camera = self._get_camera(camera_id)
            camera_text = f"Camera: {camera.name if camera else f'ID {camera_id}'}"
            if camera and camera.location:
                camera_text += f" | Location: {camera.location}"
            
            cv2.putText(
                screenshot,
                camera_text,
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Draw violation type
            cv2.putText(
                screenshot,
                f"Violation: {violation_type.replace('_', ' ').upper()}",
                (w // 2, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,  # Use violation color
                2
            )
            
            # Generate unique filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{camera_id}_{violation_type.split(',')[0]}_{timestamp_str}_{uuid.uuid4().hex[:6]}.jpg"
            filepath = os.path.join(SCREENSHOTS_DIR, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save image with high quality
            cv2.imwrite(filepath, screenshot, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Return relative path for storage in database
            return f"/screenshots/{filename}"
            
        except Exception as e:
            logger.error(f"Error saving screenshot: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def _generate_cache_key(self, camera_id: int, bbox: List[float], violation_type: str) -> str:
        """
        Generate a unique cache key for alert deduplication.
        
        Args:
            camera_id: Camera ID
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            violation_type: Type of violation
            
        Returns:
            String cache key
        """
        # Round bbox coordinates to reduce small movements triggering new alerts
        # Convert to integers to ensure consistent keys
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Calculate center point and round to nearest 5% of image
            center_x = int((x1 + x2) / 2 / 20) * 20
            center_y = int((y1 + y2) / 2 / 20) * 20
            
            # Calculate size and round to nearest 10%
            width = int((x2 - x1) / 10) * 10
            height = int((y2 - y1) / 10) * 10
            
            # Create key using center and size
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
                    # Too soon for a new alert
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
            
            # Save the screenshot
            import cv2
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
                from backend.main import broadcast_alert
                
                # Use asyncio to run the coroutine in the background
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, create a task
                    asyncio.create_task(broadcast_alert(camera_id, alert_dict))
                else:
                    # Otherwise, run the coroutine in a new thread
                    import threading
                    def run_async():
                        asyncio.run(broadcast_alert(camera_id, alert_dict))
                    threading.Thread(target=run_async).start()
                
            except ImportError:
                logger.debug("WebSocket broadcast not available")
            except Exception as e:
                logger.error(f"Error broadcasting alert: {str(e)}")
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


# Factory function to create alert service
def get_alert_service(db: Session) -> AlertService:
    """Create alert service with database session."""
    return AlertService(db)

# Import for timedelta
from datetime import timedelta