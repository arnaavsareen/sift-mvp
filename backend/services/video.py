import cv2
import numpy as np
import threading
import time
import os
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from backend.config import SCREENSHOTS_DIR

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Simple video processing service that captures frames from 
    camera streams and processes them for detection.
    """
    
    def __init__(
        self, 
        camera_id: int, 
        camera_url: str,
        detection_service,
        alert_service,
        frame_sample_rate: int = 10,
    ):
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.detection_service = detection_service
        self.alert_service = alert_service
        self.frame_sample_rate = frame_sample_rate
        
        # Processing state
        self.is_running = False
        self.current_frame = None
        self.last_frame_time = None
        
        # Processing thread
        self.thread = None
    
    def start(self):
        """Start video processing in a background thread."""
        if self.is_running:
            logger.info(f"Camera {self.camera_id} already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._process_stream)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Started processing camera {self.camera_id}")
        return True
    
    def stop(self):
        """Stop video processing."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        logger.info(f"Stopped processing camera {self.camera_id}")
        return True
    
    def get_current_frame(self):
        """Get the current processed frame."""
        return self.current_frame
    
    def _process_stream(self):
        """Main processing loop for the video stream."""
        cap = cv2.VideoCapture(self.camera_url)
        
        if not cap.isOpened():
            logger.error(f"Could not open video stream at {self.camera_url}")
            self.is_running = False
            return
        
        frame_count = 0
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning(f"Could not read frame from {self.camera_url}")
                    # Try to reconnect
                    time.sleep(1.0)
                    cap.release()
                    cap = cv2.VideoCapture(self.camera_url)
                    continue
                
                # Process only every nth frame
                if frame_count % self.frame_sample_rate == 0:
                    # Run detection
                    detections = self.detection_service.detect(frame)
                    
                    # Process alerts
                    if detections:
                        alerts = self.alert_service.process_alerts(
                            self.camera_id, 
                            detections, 
                            frame
                        )
                    
                    # Save the processed frame
                    self.current_frame = self._annotate_frame(frame, detections)
                    self.last_frame_time = datetime.now()
                
                frame_count += 1
                
                # Avoid 100% CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
        finally:
            cap.release()
            self.is_running = False
    
    def _annotate_frame(self, frame, detections):
        """Draw detection boxes and labels on the frame."""
        annotated = frame.copy()
        
        for detection in detections:
            bbox = detection["bbox"]
            label = detection["class"]
            confidence = detection["confidence"]
            is_violation = detection.get("violation", False)
            
            # Draw bounding box
            color = (0, 0, 255) if is_violation else (0, 255, 0)
            cv2.rectangle(
                annotated,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2
            )
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(
                annotated,
                label_text,
                (int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return annotated


# Global storage for active processors
active_processors = {}

def get_processor(camera_id: int) -> Optional[VideoProcessor]:
    """Get active video processor for camera."""
    return active_processors.get(camera_id)

def start_processor(
    camera_id: int, 
    camera_url: str,
    detection_service,
    alert_service
) -> bool:
    """Start a new video processor for camera."""
    if camera_id in active_processors:
        return False
    
    processor = VideoProcessor(
        camera_id=camera_id,
        camera_url=camera_url,
        detection_service=detection_service,
        alert_service=alert_service
    )
    
    if processor.start():
        active_processors[camera_id] = processor
        return True
    
    return False

def stop_processor(camera_id: int) -> bool:
    """Stop video processor for camera."""
    processor = active_processors.get(camera_id)
    if not processor:
        return False
    
    processor.stop()
    del active_processors[camera_id]
    return True

def get_all_processors() -> Dict[int, VideoProcessor]:
    """Get all active video processors."""
    return active_processors