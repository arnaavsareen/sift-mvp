import cv2
import numpy as np
import threading
import time
import os
import queue
import uuid
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import traceback

from backend.config import SCREENSHOTS_DIR

logger = logging.getLogger(__name__)

class FrameBuffer:
    """
    Thread-safe circular buffer for frames to decouple frame capture from processing.
    This reduces frame drops and allows smoother processing.
    """
    def __init__(self, maxsize=30):
        self.buffer = queue.Queue(maxsize=maxsize)
        self.last_frame = None
        self.last_put_time = time.time()
        self.skip_count = 0  # Track skipped frames for rate control
    
    def put(self, frame):
        """Add a frame to the buffer, dropping oldest frame if buffer is full."""
        if frame is None:
            return
        
        current_time = time.time()
        time_since_last = current_time - self.last_put_time
            
        # Save this frame regardless of whether it goes into the buffer
        self.last_frame = frame.copy()
        
        # Only add every 2nd frame to buffer to prevent processing backlog
        # This helps keep real-time detection more current with the video
        self.skip_count += 1
        if self.skip_count < 2:  # Skip this frame
            return
        
        # Reset counter
        self.skip_count = 0
        self.last_put_time = current_time
        
        try:
            # Try to add to buffer, dropping oldest frame if needed
            if self.buffer.full():
                try:
                    # Remove oldest frame
                    self.buffer.get_nowait()
                except queue.Empty:
                    pass
            
            self.buffer.put_nowait(frame)
        except:
            # If any error occurs, just continue
            pass
    
    def get(self):
        """Get a frame from the buffer, returning last frame if buffer is empty."""
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            # Return the last frame we saw if buffer is empty
            return self.last_frame
    
    def clear(self):
        """Clear all frames from buffer."""
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break


class VideoProcessor:
    """
    Advanced video processing service that captures frames from 
    camera streams and processes them for safety violation detection.
    Features include multi-threaded processing, frame buffering,
    and performance monitoring.
    """
    
    def __init__(
        self, 
        camera_id: int, 
        camera_url: str,
        detection_service,
        alert_service,
        frame_sample_rate: int = 5,
        reconnect_delay: float = 3.0,
        max_reconnect_attempts: int = 10,
        buffer_size: int = 60
    ):
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.detection_service = detection_service
        self.alert_service = alert_service
        self.frame_sample_rate = frame_sample_rate
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Processing state
        self.is_running = False
        self.current_frame = None
        self.annotated_frame = None
        self.last_frame_time = None
        self.last_alert_time = 0
        self.last_detection_time = 0
        self.frame_count = 0
        self.processed_count = 0
        self.alert_count = 0
        self.start_time = None
        self.reconnect_attempts = 0
        
        # Performance metrics
        self.fps = 0
        self.processing_fps = 0
        self.detection_times = []  # Keep last 50 detection times
        self.frame_times = []      # Keep last 50 frame capture times
        
        # Frame buffer for async processing
        self.frame_buffer = FrameBuffer(maxsize=buffer_size)
        
        # Processing threads
        self.capture_thread = None
        self.process_thread = None
        
        # Session ID for logging & tracking
        self.session_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Initialized video processor for camera {camera_id} (session: {self.session_id})")
    
    def start(self) -> bool:
        """
        Start video processing in background threads.
        Returns True if started successfully.
        """
        if self.is_running:
            logger.info(f"Camera {self.camera_id} already running")
            return True
        
        try:
            # Reset state
            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.processed_count = 0
            self.alert_count = 0
            self.reconnect_attempts = 0
            self.frame_buffer.clear()
            
            # Start capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_frames,
                daemon=True,
                name=f"capture-{self.camera_id}"
            )
            self.capture_thread.start()
            
            # Start processing thread
            self.process_thread = threading.Thread(
                target=self._process_frames,
                daemon=True,
                name=f"process-{self.camera_id}"
            )
            self.process_thread.start()
            
            logger.info(f"Started processing for camera {self.camera_id} (session: {self.session_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera {self.camera_id}: {str(e)}")
            logger.error(traceback.format_exc())
            self.is_running = False
            return False
    
    def stop(self) -> bool:
        """Stop video processing and release resources."""
        if not self.is_running:
            return True
        
        logger.info(f"Stopping processing for camera {self.camera_id}")
        self.is_running = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
            self.process_thread = None
        
        # Clear buffer
        self.frame_buffer.clear()
        
        # Calculate final stats
        duration = time.time() - self.start_time if self.start_time else 0
        if duration > 0:
            avg_fps = self.frame_count / duration
            avg_proc_fps = self.processed_count / duration
            logger.info(f"Camera {self.camera_id} stats: Duration={duration:.1f}s, "
                       f"Frames={self.frame_count}, Processed={self.processed_count}, "
                       f"Alerts={self.alert_count}, Avg FPS={avg_fps:.2f}, Avg Proc FPS={avg_proc_fps:.2f}")
        
        return True
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the latest annotated frame."""
        if self.annotated_frame is not None:
            return self.annotated_frame
        elif self.current_frame is not None:
            # If we have a current frame but no annotated frame yet,
            # quickly create a basic annotated version
            return self._add_frame_metadata(self.current_frame.copy())
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status and performance metrics."""
        duration = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / duration if duration > 0 else 0
        avg_processing_fps = self.processed_count / duration if duration > 0 else 0
        
        # Calculate averages if we have data
        avg_detection_time = (sum(self.detection_times) / len(self.detection_times) 
                             if self.detection_times else 0)
        
        return {
            "camera_id": self.camera_id,
            "is_running": self.is_running,
            "uptime_seconds": duration,
            "frame_count": self.frame_count,
            "processed_count": self.processed_count,
            "alert_count": self.alert_count,
            "fps": self.fps,
            "processing_fps": self.processing_fps,
            "avg_fps": avg_fps,
            "avg_processing_fps": avg_processing_fps,
            "avg_detection_time_ms": avg_detection_time * 1000,
            "last_frame_time": self.last_frame_time.isoformat() if self.last_frame_time else None,
            "session_id": self.session_id
        }
    
    def _capture_frames(self) -> None:
        """
        Background thread to continuously capture frames from the video source.
        Handles connection issues and maintains frame capture state.
        """
        cap = None
        last_frame_time = time.time()
        frames_since_log = 0
        
        try:
            while self.is_running:
                # (Re)open capture if needed
                if cap is None or not cap.isOpened():
                    if self.reconnect_attempts >= self.max_reconnect_attempts:
                        logger.error(
                            f"Failed to connect to camera {self.camera_id} after "
                            f"{self.reconnect_attempts} attempts. Stopping processor."
                        )
                        self.is_running = False
                        break
                    
                    try:
                        # Try to open the video source
                        if cap is not None:
                            cap.release()  # Release any existing capture
                        
                        logger.info(f"Connecting to camera {self.camera_id} (attempt {self.reconnect_attempts + 1})")
                        cap = cv2.VideoCapture(self.camera_url)
                        
                        # Set capture properties if possible
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for realtime
                        
                        if not cap.isOpened():
                            logger.warning(f"Could not open video stream at {self.camera_url}")
                            self.reconnect_attempts += 1
                            time.sleep(self.reconnect_delay)
                            continue
                        else:
                            self.reconnect_attempts = 0  # Reset after successful connection
                            logger.info(f"Successfully connected to camera {self.camera_id}")
                    except Exception as e:
                        logger.error(f"Error opening camera {self.camera_id}: {str(e)}")
                        self.reconnect_attempts += 1
                        time.sleep(self.reconnect_delay)
                        continue
                
                # Capture frame
                try:
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        logger.warning(f"Failed to read frame from camera {self.camera_id}")
                        time.sleep(0.1)  # Small delay to avoid busy wait
                        continue
                    
                    # Update frame count and timestamp
                    self.frame_count += 1
                    now = time.time()
                    frame_time = now - last_frame_time
                    last_frame_time = now
                    
                    # Calculate FPS over the last second
                    self.frame_times.append(frame_time)
                    if len(self.frame_times) > 50:  # Keep size manageable
                        self.frame_times.pop(0)
                    
                    self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                    
                    # Log status periodically
                    frames_since_log += 1
                    if frames_since_log >= 100:  # Log every 100 frames
                        frames_since_log = 0
                        logger.debug(
                            f"Camera {self.camera_id}: Capturing at {self.fps:.1f} FPS, "
                            f"processing at {self.processing_fps:.1f} FPS, "
                            f"total frames: {self.frame_count}"
                        )
                    
                    # Add to frame buffer for processing
                    self.frame_buffer.put(frame)
                    
                except Exception as e:
                    logger.error(f"Error reading from camera {self.camera_id}: {str(e)}")
                    # If we have capture errors, wait briefly to avoid rapid reconnection
                    time.sleep(0.5)
                    continue
                
                # Prevent 100% CPU usage
                time.sleep(0.001)
        
        except Exception as e:
            logger.error(f"Capture thread error for camera {self.camera_id}: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            # Clean up resources
            if cap is not None:
                cap.release()
            logger.info(f"Capture thread for camera {self.camera_id} terminated")
    
    def _process_frames(self) -> None:
        """
        Background thread to process frames from the buffer.
        Performs detection and handles safety violations.
        """
        frames_since_detection = 0
        accumulated_detection_time = 0
        detection_count = 0
        avg_detection_time = 0
        adaptive_sample_rate = self.frame_sample_rate  # Start with the configured sample rate
        
        try:
            while self.is_running:
                start_time = time.time()
                
                # Get a frame from the buffer
                frame = self.frame_buffer.get()
                
                # Skip if no frame is available
                if frame is None:
                    time.sleep(0.01)  # Short sleep to prevent CPU spinning
                    continue
                
                # Store the raw frame
                self.current_frame = frame.copy()
                self.last_frame_time = datetime.now()
                
                # Count processed frames
                self.processed_count += 1
                
                # Check if we should run detection on this frame
                # Use adaptive sampling based on performance
                should_detect = frames_since_detection >= adaptive_sample_rate
                
                if should_detect:
                    # Run object detection
                    detection_start = time.time()
                    try:
                        logger.debug(f"Running detection on frame for camera {self.camera_id}")
                        detections = self.detection_service.detect(frame)
                        detection_time = time.time() - detection_start
                        
                        # Log detection count
                        logger.debug(f"Camera {self.camera_id}: Found {len(detections)} detections in {detection_time:.3f}s")
                        
                        # Update detection performance metrics
                        self.detection_times.append(detection_time)
                        if len(self.detection_times) > 50:
                            self.detection_times.pop(0)
                        
                        # Keep track of accumulated detection time for adaptive sampling
                        accumulated_detection_time += detection_time
                        detection_count += 1
                        if detection_count >= 10:  # Recalculate after 10 detections
                            avg_detection_time = accumulated_detection_time / detection_count
                            
                            # Adjust sample rate based on performance
                            # Target: Keep detection overhead under 40% of total processing time
                            if avg_detection_time > 0.2:  # More than 200ms per detection
                                adaptive_sample_rate = min(adaptive_sample_rate + 1, 15)  # Increase, max 15
                                logger.debug(f"Camera {self.camera_id}: Increasing sample rate to {adaptive_sample_rate}")
                            elif avg_detection_time < 0.05:  # Less than 50ms per detection
                                adaptive_sample_rate = max(adaptive_sample_rate - 1, 1)  # Decrease, min 1
                                logger.debug(f"Camera {self.camera_id}: Decreasing sample rate to {adaptive_sample_rate}")
                            
                            # Reset accumulators
                            accumulated_detection_time = 0
                            detection_count = 0
                        
                        # Record when we last ran detection
                        self.last_detection_time = time.time()
                        frames_since_detection = 0
                        
                        # If detections found, process for safety violations
                        if detections:
                            # Process each detection for potential violations
                            self._process_detections(frame, detections)
                            
                            # Create annotated frame with all detections
                            self.annotated_frame = self._annotate_frame(frame, detections)
                        else:
                            # Just add metadata overlay if no detections
                            self.annotated_frame = self._add_frame_metadata(frame)
                    except Exception as e:
                        logger.error(f"Detection error for camera {self.camera_id}: {str(e)}")
                        logger.error(traceback.format_exc())
                        # Still create an annotated frame with error indication
                        self.annotated_frame = self._add_frame_metadata(frame)
                        frames_since_detection = 0  # Reset to force another attempt soon
                else:
                    # For frames where we skip detection, still annotate with basic info
                    self.annotated_frame = self._add_frame_metadata(frame.copy())
                    frames_since_detection += 1
                
                # Calculate processing frame rate
                processing_time = time.time() - start_time
                if processing_time > 0:
                    self.processing_fps = 1.0 / processing_time
                
                # If processing is very fast, add a small delay to prevent CPU spinning
                if processing_time < 0.01:
                    time.sleep(0.01 - processing_time)
                
        except Exception as e:
            logger.error(f"Error in processing thread for camera {self.camera_id}: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _annotate_frame(self, frame, detections) -> np.ndarray:
        """
        Draw detection boxes, labels, and metadata on the frame.
        
        Args:
            frame: Original video frame
            detections: List of detection dictionaries
            
        Returns:
            Annotated frame with boxes, labels, and metadata
        """
        try:
            # Make a copy to avoid modifying the original
            annotated = frame.copy()
            
            # Define colors for different classes and violations
            color_map = {
                "person": (0, 255, 0),  # Green for persons without violations
                "hardhat": (0, 255, 255),  # Yellow for hardhat
                "vest": (255, 128, 0),  # Orange for vest
                "violation": (0, 0, 255),  # Red for violations
                "mask": (255, 0, 255),  # Purple for mask
                "goggles": (255, 255, 0),  # Cyan for goggles
                "gloves": (128, 0, 255),  # Purple for gloves
                "boots": (0, 128, 255),  # Light blue for boots
            }
            
            # Sort to draw violations on top
            detections_sorted = sorted(
                detections, 
                key=lambda d: 1 if d.get("violation", False) else 0
            )
            
            # Log detection count for debugging
            logger.info(f"Drawing {len(detections)} detections on frame (violations: {sum(1 for d in detections if d.get('violation', False))})")
            
            for detection in detections_sorted:
                bbox = detection["bbox"]
                label = detection["class"]
                confidence = detection["confidence"]
                is_violation = detection.get("violation", False)
                violation_type = detection.get("violation_type", "")
                
                # Convert box coordinates to integers
                x1, y1, x2, y2 = map(int, bbox)
                
                # Ensure coordinates are within frame
                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(0, min(x2, frame.shape[1] - 1))
                y2 = max(0, min(y2, frame.shape[0] - 1))
                
                # Pick color based on class/violation
                if is_violation:
                    color = color_map.get("violation", (0, 0, 255))
                else:
                    color = color_map.get(label.lower(), (255, 255, 255))
                
                # Draw thicker bounding box (increased from 2 to 3 pixels)
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )
                
                # Create label text
                if is_violation:
                    label_text = f"{label}: {violation_type} ({confidence:.2f})"
                else:
                    label_text = f"{label}: {confidence:.2f}"
                
                # Get text size for background
                text_size, baseline = cv2.getTextSize(
                    label_text, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6,  # Increased from 0.5 
                    2
                )
                
                # Draw filled background for text (more visible)
                cv2.rectangle(
                    annotated,
                    (x1, max(0, y1 - text_size[1] - 10)),
                    (x1 + text_size[0] + 10, y1),
                    color,
                    -1  # Filled rectangle
                )
                
                # Draw label with increased size
                cv2.putText(
                    annotated,
                    label_text,
                    (x1 + 5, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,  # Increased from 0.5
                    (255, 255, 255),
                    2
                )
                
                # For violations, add extra highlight
                if is_violation:
                    # Draw a second, outer box for emphasis
                    cv2.rectangle(
                        annotated,
                        (x1-5, y1-5),
                        (x2+5, y2+5),
                        (0, 0, 255),
                        1
                    )
            
            # Add frame metadata (timestamp, FPS, etc.)
            return self._add_frame_metadata(annotated)
            
        except Exception as e:
            logger.error(f"Error annotating frame: {str(e)}")
            # If annotation fails, return original frame with minimal metadata
            return self._add_frame_metadata(frame.copy())
    
    def _add_frame_metadata(self, frame) -> np.ndarray:
        """
        Add timestamp, camera info, and performance metrics to frame.
        
        Args:
            frame: Video frame to annotate
            
        Returns:
            Frame with added metadata overlay
        """
        try:
            # Add transparent overlay at the bottom
            h, w = frame.shape[:2]
            overlay_h = 60  # Height of the overlay area
            overlay = frame[h - overlay_h:h, 0:w].copy()
            overlay = cv2.addWeighted(
                overlay, 0.5, 
                np.zeros(overlay.shape, dtype=overlay.dtype) + np.array([0, 0, 50], dtype=np.uint8), 
                0.5, 0
            )
            frame[h - overlay_h:h, 0:w] = overlay
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                frame,
                f"Time: {timestamp}",
                (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Add camera info
            cv2.putText(
                frame,
                f"Camera: {self.camera_id} (Session: {self.session_id})",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Add performance metrics
            metrics_text = f"FPS: {self.fps:.1f} | Processing: {self.processing_fps:.1f} FPS | Alerts: {self.alert_count}"
            text_size = cv2.getTextSize(metrics_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(
                frame,
                metrics_text,
                (w - text_size[0] - 10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Error adding frame metadata: {str(e)}")
            return frame  # Return unmodified frame if error occurs

    def _process_detections(self, frame, detections) -> None:
        """
        Process detections to identify safety violations and generate alerts.
        
        Args:
            frame: The original frame
            detections: List of detection dictionaries
        """
        try:
            # Filter for violations
            violations = [d for d in detections if d.get("violation", False)]
            
            if not violations:
                return
                
            # Process each violation to generate alerts
            current_time = time.time()
                
            # Only generate alerts if enough time has passed since the last one
            if (current_time - self.last_alert_time) < 5.0:  # 5 second minimum between alerts
                return
                
            for violation in violations:
                # Extract violation info
                violation_type = violation.get("violation_type", "Unknown violation")
                confidence = violation.get("confidence", 0.0)
                bbox = violation.get("bbox")
                
                # Additional metadata to include with the alert
                metadata = {
                    "session_id": self.session_id,
                    "frame_count": self.frame_count,
                    "detection_class": violation.get("class", ""),
                    "bbox": bbox
                }
                
                # Create alert through alert service
                alert = self.alert_service.create_alert(
                    camera_id=self.camera_id,
                    violation_type=violation_type,
                    confidence=confidence,
                    frame=frame,
                    bbox=bbox,
                    metadata=metadata
                )
                
                if alert:
                    self.last_alert_time = current_time
                    self.alert_count += 1
                    logger.info(f"Generated alert for camera {self.camera_id}: {violation_type} ({confidence:.2f})")
                    
                    # Only create one alert at a time to prevent alert flooding
                    break
                    
        except Exception as e:
            logger.error(f"Error processing detections: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())


# Global storage for active processors
_active_processors = {}

def get_processor(camera_id: int) -> Optional[VideoProcessor]:
    """Get active video processor for camera."""
    return _active_processors.get(camera_id)

def start_processor(
    camera_id: int, 
    camera_url: str,
    detection_service,
    alert_service,
    frame_sample_rate: int = 5
) -> bool:
    """
    Start a new video processor for camera.
    
    Args:
        camera_id: Camera identifier
        camera_url: URL or path to video source
        detection_service: Object detection service
        alert_service: Alert generation service
        frame_sample_rate: Process 1 out of N frames
        
    Returns:
        True if processor started successfully
    """
    # Check if already running
    if camera_id in _active_processors:
        logger.info(f"Camera {camera_id} already has an active processor")
        return True
    
    try:
        # Create processor
        processor = VideoProcessor(
            camera_id=camera_id,
            camera_url=camera_url,
            detection_service=detection_service,
            alert_service=alert_service,
            frame_sample_rate=frame_sample_rate
        )
        
        # Start processor
        if processor.start():
            _active_processors[camera_id] = processor
            logger.info(f"Started processor for camera {camera_id}")
            return True
        else:
            logger.error(f"Failed to start processor for camera {camera_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating processor for camera {camera_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def stop_processor(camera_id: int) -> bool:
    """
    Stop video processor for camera.
    
    Args:
        camera_id: Camera identifier
        
    Returns:
        True if processor stopped successfully
    """
    processor = _active_processors.get(camera_id)
    if not processor:
        logger.warning(f"No active processor for camera {camera_id}")
        return False
    
    try:
        processor.stop()
        del _active_processors[camera_id]
        logger.info(f"Stopped processor for camera {camera_id}")
        return True
    except Exception as e:
        logger.error(f"Error stopping processor for camera {camera_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_all_processors() -> Dict[int, VideoProcessor]:
    """Get all active video processors."""
    return _active_processors

def get_all_statuses() -> Dict[int, Dict]:
    """Get status information for all active processors."""
    return {
        camera_id: processor.get_status()
        for camera_id, processor in _active_processors.items()
    }