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
    Thread-safe circular buffer for frames with improved efficiency.
    """
    def __init__(self, maxsize=30):
        self.buffer = queue.Queue(maxsize=maxsize)
        self.last_frame = None
        self.last_put_time = time.time()
        self.skip_count = 0
    
    def put(self, frame):
        """Add a frame to the buffer, dropping oldest frame if buffer is full."""
        if frame is None:
            return
        
        current_time = time.time()
        self.last_frame = frame.copy()
        
        # Frame rate control
        self.skip_count += 1
        if self.skip_count < 2:  # Skip every other frame
            return
        
        self.skip_count = 0
        self.last_put_time = current_time
        
        try:
            if self.buffer.full():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    pass
            
            self.buffer.put_nowait(frame)
        except:
            pass
    
    def get(self):
        """Get a frame from the buffer."""
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
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
    Enhanced video processing service with batch processing capabilities.
    """
    
    def __init__(
        self, 
        camera_id: int, 
        camera_url: str,
        detection_service,
        alert_service,
        frame_sample_rate: int = 5,
        batch_size: int = 4,
        reconnect_delay: float = 3.0,
        max_reconnect_attempts: int = 10,
        buffer_size: int = 60
    ):
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.detection_service = detection_service
        self.alert_service = alert_service
        self.frame_sample_rate = frame_sample_rate
        self.batch_size = batch_size
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
        self.detection_times = []
        self.frame_times = []
        self.batch_processing_times = []
        
        # Batch processing
        self.batch_buffer = []
        self.batch_results_queue = queue.Queue()
        
        # Frame buffer
        self.frame_buffer = FrameBuffer(maxsize=buffer_size)
        
        # Processing threads
        self.capture_thread = None
        self.process_thread = None
        self.batch_thread = None
        
        # Session ID
        self.session_id = str(uuid.uuid4())[:8]
        
        # Violation tracking
        self.violation_history = {}
        self.violation_timeout = 600  # 10 minutes
        
        logger.info(f"Initialized enhanced video processor for camera {camera_id} (session: {self.session_id})")
    
    def start(self) -> bool:
        """Start video processing with batch capabilities."""
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
            
            # Start threads
            self.capture_thread = threading.Thread(
                target=self._capture_frames,
                daemon=True,
                name=f"capture-{self.camera_id}"
            )
            self.capture_thread.start()
            
            self.batch_thread = threading.Thread(
                target=self._batch_processor,
                daemon=True,
                name=f"batch-{self.camera_id}"
            )
            self.batch_thread.start()
            
            self.process_thread = threading.Thread(
                target=self._process_frames,
                daemon=True,
                name=f"process-{self.camera_id}"
            )
            self.process_thread.start()
            
            logger.info(f"Started enhanced processing for camera {self.camera_id}")
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
        threads = [self.capture_thread, self.process_thread, self.batch_thread]
        for thread in threads:
            if thread:
                thread.join(timeout=2.0)
        
        # Clear buffers
        self.frame_buffer.clear()
        self.batch_buffer.clear()
        
        # Log final stats
        duration = time.time() - self.start_time if self.start_time else 0
        if duration > 0:
            avg_fps = self.frame_count / duration
            avg_proc_fps = self.processed_count / duration
            logger.info(f"Camera {self.camera_id} final stats: "
                       f"Duration={duration:.1f}s, Frames={self.frame_count}, "
                       f"Processed={self.processed_count}, Alerts={self.alert_count}, "
                       f"Avg FPS={avg_fps:.2f}, Avg Proc FPS={avg_proc_fps:.2f}")
        
        return True
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the latest annotated frame."""
        if self.annotated_frame is not None:
            return self.annotated_frame
        elif self.current_frame is not None:
            return self._add_frame_metadata(self.current_frame.copy())
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status and performance metrics."""
        duration = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / duration if duration > 0 else 0
        avg_processing_fps = self.processed_count / duration if duration > 0 else 0
        
        # Calculate average detection times
        avg_detection_time = (sum(self.detection_times) / len(self.detection_times) 
                             if self.detection_times else 0)
        avg_batch_time = (sum(self.batch_processing_times) / len(self.batch_processing_times)
                         if self.batch_processing_times else 0)
        
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
            "avg_batch_time_ms": avg_batch_time * 1000,
            "last_frame_time": self.last_frame_time.isoformat() if self.last_frame_time else None,
            "session_id": self.session_id,
            "batch_size": self.batch_size
        }
    
    def _capture_frames(self) -> None:
        """Background thread to capture frames from the camera stream."""
        cap = None
        last_frame_time = time.time()
        frames_since_log = 0
        
        try:
            while self.is_running:
                # Handle connection logic
                if cap is None or (hasattr(cap, 'isOpened') and not cap.isOpened()):
                    if self.reconnect_attempts >= self.max_reconnect_attempts:
                        logger.error(f"Max reconnect attempts reached for camera {self.camera_id}")
                        self.is_running = False
                        break
                    
                    try:
                        if cap is not None:
                            cap.release()
                        
                        camera_url = self.camera_url
                        if camera_url.startswith("file:///"):
                            camera_url = camera_url.replace("file:///", "/")
                        
                        logger.info(f"Connecting to camera {self.camera_id} (attempt {self.reconnect_attempts + 1})")
                        
                        # Configure based on stream type
                        if camera_url.startswith("rtsp://"):
                            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TRANSPORT_TCP)
                            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
                            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
                        else:
                            cap = cv2.VideoCapture(camera_url)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        if not cap.isOpened():
                            logger.warning(f"Could not open video stream at {camera_url}")
                            self.reconnect_attempts += 1
                            time.sleep(self.reconnect_delay)
                            continue
                        else:
                            self.reconnect_attempts = 0
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
                        
                        # If this is a file source, try to reopen it (looping video)
                        if self.camera_url.startswith("file:///") or not self.camera_url.startswith(("rtsp://", "http://", "https://")):
                            logger.info(f"Reopening video file for camera {self.camera_id}")
                            if cap is not None:
                                cap.release()
                            
                            camera_url = self.camera_url
                            if camera_url.startswith("file:///"):
                                camera_url = camera_url.replace("file:///", "/")
                            
                            cap = cv2.VideoCapture(camera_url)
                            ret, frame = cap.read()
                            
                            if not ret or frame is None:
                                logger.error(f"Could not reopen video file for camera {self.camera_id}")
                                time.sleep(0.5)
                                continue
                            
                            logger.info(f"Successfully reopened video file for camera {self.camera_id}")
                        else:
                            # For streaming sources, just wait a bit and retry
                            time.sleep(0.1)
                            continue
                    
                    # Update metrics
                    self.frame_count += 1
                    now = time.time()
                    frame_time = now - last_frame_time
                    last_frame_time = now
                    
                    # Calculate FPS
                    self.frame_times.append(frame_time)
                    if len(self.frame_times) > 50:
                        self.frame_times.pop(0)
                    
                    self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                    
                    # Log periodically
                    frames_since_log += 1
                    if frames_since_log >= 100:
                        frames_since_log = 0
                        logger.debug(f"Camera {self.camera_id}: Capturing at {self.fps:.1f} FPS")
                    
                    # Add to frame buffer
                    self.frame_buffer.put(frame)
                    
                except Exception as e:
                    logger.error(f"Error reading from camera {self.camera_id}: {str(e)}")
                    time.sleep(0.5)
                    continue
                
                time.sleep(0.001)  # Prevent 100% CPU usage
        
        except Exception as e:
            logger.error(f"Capture thread error for camera {self.camera_id}: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            if cap is not None:
                cap.release()
            logger.info(f"Capture thread for camera {self.camera_id} terminated")
    
    def _batch_processor(self) -> None:
        """Background thread for batch processing frames."""
        batch_buffer = []
        last_batch_time = time.time()
        
        try:
            while self.is_running:
                # Collect frames for batch
                frame = self.frame_buffer.get()
                if frame is not None:
                    batch_buffer.append(frame)
                
                # Process batch when full or timeout
                current_time = time.time()
                should_process = (len(batch_buffer) >= self.batch_size or 
                                (len(batch_buffer) > 0 and current_time - last_batch_time > 0.1))
                
                if should_process:
                    start_time = time.time()
                    
                    try:
                        # Process each frame individually since batch processing isn't implemented yet
                        batch_results = []
                        for frame in batch_buffer:
                            results = self.detection_service.detect(frame)
                            batch_results.append(results)
                        
                        # Store results with frames
                        for frame, results in zip(batch_buffer, batch_results):
                            self.batch_results_queue.put((frame, results))
                        
                        batch_time = time.time() - start_time
                        self.batch_processing_times.append(batch_time)
                        if len(self.batch_processing_times) > 50:
                            self.batch_processing_times.pop(0)
                        
                        logger.debug(f"Processed batch of {len(batch_buffer)} frames in {batch_time:.3f}s")
                        
                    except Exception as e:
                        logger.error(f"Batch processing error: {str(e)}")
                        # Add frames without results to prevent blocking
                        for frame in batch_buffer:
                            self.batch_results_queue.put((frame, []))
                    
                    batch_buffer.clear()
                    last_batch_time = current_time
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
        
        except Exception as e:
            logger.error(f"Batch processor error for camera {self.camera_id}: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _process_frames(self) -> None:
        """Background thread to process batch results and generate alerts."""
        try:
            while self.is_running:
                try:
                    # Get frame and results from batch processor
                    frame, detections = self.batch_results_queue.get(timeout=0.1)
                    
                    # Store frame and update metrics
                    self.current_frame = frame.copy()
                    self.last_frame_time = datetime.now()
                    self.processed_count += 1
                    
                    # Calculate processing FPS
                    start_time = time.time()
                    
                    # Process detections for violations
                    if detections:
                        self._process_detections(frame, detections)
                        self.annotated_frame = self._annotate_frame(frame, detections)
                    else:
                        self.annotated_frame = self._add_frame_metadata(frame)
                    
                    processing_time = time.time() - start_time
                    if processing_time > 0:
                        self.processing_fps = 1.0 / processing_time
                    
                except queue.Empty:
                    if self.current_frame is not None:
                        # If no new frames, just update metadata overlay
                        self.annotated_frame = self._add_frame_metadata(self.current_frame.copy())
                    continue
                except Exception as e:
                    logger.error(f"Frame processing error: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Process thread error for camera {self.camera_id}: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _annotate_frame(self, frame, detections) -> np.ndarray:
        """Draw detection boxes, labels, and metadata on the frame."""
        try:
            annotated = frame.copy()
            
            # Enhanced color mapping
            color_map = {
                "person": (0, 255, 0),      # Green for safe persons
                "hardhat": (0, 255, 255),   # Yellow for hardhat
                "vest": (255, 128, 0),      # Orange for vest
                "violation": (0, 0, 255),   # Red for violations
                "mask": (255, 0, 255),      # Purple for mask
                "goggles": (255, 255, 0),   # Cyan for goggles
                "gloves": (128, 0, 255),    # Purple for gloves
                "boots": (0, 128, 255),     # Light blue for boots
            }
            
            # Sort detections to draw violations on top
            detections_sorted = sorted(
                detections,
                key=lambda d: 1 if d.get("violation", False) else 0
            )
            
            logger.debug(f"Drawing {len(detections)} detections")
            
            for detection in detections_sorted:
                bbox = detection["bbox"]
                label = detection["class"]
                confidence = detection["confidence"]
                is_violation = detection.get("violation", False)
                violation_type = detection.get("violation_type", "")
                
                # Convert to integers and ensure within frame bounds
                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                # Select color
                if is_violation:
                    color = color_map.get("violation", (0, 0, 255))
                    thickness = 4  # Thicker for violations
                else:
                    color = color_map.get(label.lower(), (255, 255, 255))
                    thickness = 2
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                # Create label text
                if is_violation and violation_type:
                    label_text = f"{label}: {violation_type.replace(',', ', ')} ({confidence:.2f})"
                    # Add detected PPE if it's a person
                    if detection.get("detected_ppe"):
                        ppe_list = ", ".join(detection["detected_ppe"])
                        label_text += f" | Has: {ppe_list}"
                else:
                    label_text = f"{label}: {confidence:.2f}"
                
                # Get text dimensions
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness_text = 2
                text_size, baseline = cv2.getTextSize(label_text, font, font_scale, thickness_text)
                
                # Draw text background
                text_background = (0, 0, 0) if is_violation else color
                cv2.rectangle(
                    annotated,
                    (x1, max(0, y1 - text_size[1] - 10)),
                    (x1 + text_size[0] + 10, y1),
                    text_background,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated,
                    label_text,
                    (x1 + 5, max(15, y1 - 5)),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness_text
                )
                
                # Draw violation indicator
                if is_violation:
                    # Add warning triangle
                    triangle_size = 20
                    triangle_points = np.array([
                        [x2 - triangle_size, y1],
                        [x2, y1],
                        [x2, y1 + triangle_size]
                    ], np.int32)
                    cv2.fillPoly(annotated, [triangle_points], (0, 0, 255))
                    cv2.putText(
                        annotated,
                        "!",
                        (x2 - 15, y1 + 15),
                        font,
                        0.7,
                        (255, 255, 255),
                        2
                    )
            
            # Add comprehensive frame metadata
            return self._add_frame_metadata(annotated)
            
        except Exception as e:
            logger.error(f"Error annotating frame: {str(e)}")
            return self._add_frame_metadata(frame.copy())
    
    def _add_frame_metadata(self, frame) -> np.ndarray:
        """Add comprehensive metadata overlay to frame."""
        try:
            h, w = frame.shape[:2]
            overlay_h = 80  # Increased height for more info
            
            # Create dark overlay for text
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
                (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Add camera info
            cv2.putText(
                frame,
                f"Camera: {self.camera_id} | Session: {self.session_id}",
                (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Add performance metrics (left side)
            metrics_text = f"FPS: {self.fps:.1f} | Processing: {self.processing_fps:.1f}"
            cv2.putText(
                frame,
                metrics_text,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Add statistics (right side)
            stats_text = f"Alerts: {self.alert_count} | Frames: {self.frame_count}"
            text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(
                frame,
                stats_text,
                (w - text_size[0] - 10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Add system status indicator
            status_color = (0, 255, 0) if self.is_running else (0, 0, 255)
            cv2.circle(frame, (w - 20, 20), 8, status_color, -1)
            cv2.circle(frame, (w - 20, 20), 8, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error adding frame metadata: {str(e)}")
            return frame
    
    def _process_detections(self, frame, detections) -> None:
        """Process detections to identify violations and generate alerts."""
        try:
            # Filter for violations
            violations = [d for d in detections if d.get("violation", False)]
            
            if not violations:
                return
            
            current_time = time.time()
            
            # Rate limiting for alerts
            if (current_time - self.last_alert_time) < 5.0:
                return
            
            for violation in violations:
                violation_type = violation.get("violation_type", "Unknown violation")
                confidence = violation.get("confidence", 0.0)
                bbox = violation.get("bbox")
                detected_ppe = violation.get("detected_ppe", [])
                
                # Skip if already alerted recently for this position
                if bbox:
                    person_id = f"{int(bbox[0]/10)},{int(bbox[1]/10)},{int(bbox[2]/10)},{int(bbox[3]/10)}"
                    
                    if person_id in self.violation_history:
                        last_violation_time = self.violation_history[person_id]
                        if (current_time - last_violation_time) < self.violation_timeout:
                            logger.debug(f"Skipping repeated violation for same person: {violation_type}")
                            continue
                
                # Enhanced metadata
                metadata = {
                    "session_id": self.session_id,
                    "frame_count": self.frame_count,
                    "detection_class": violation.get("class", ""),
                    "bbox": bbox,
                    "detected_ppe": detected_ppe,
                    "fps": self.fps,
                    "processing_fps": self.processing_fps
                }
                
                # Create alert
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
                    
                    # Update violation history
                    if bbox:
                        person_id = f"{int(bbox[0]/10)},{int(bbox[1]/10)},{int(bbox[2]/10)},{int(bbox[3]/10)}"
                        self.violation_history[person_id] = current_time
                        
                        # Cleanup old entries periodically
                        if self.alert_count % 10 == 0:
                            for p_id in list(self.violation_history.keys()):
                                if (current_time - self.violation_history[p_id]) > self.violation_timeout:
                                    del self.violation_history[p_id]
                    
                    # Only create one alert at a time
                    break
                    
        except Exception as e:
            logger.error(f"Error processing detections: {str(e)}")
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
    """Start a new enhanced video processor for camera."""
    if camera_id in _active_processors:
        logger.info(f"Camera {camera_id} already has an active processor")
        return True
    
    try:
        processor = VideoProcessor(
            camera_id=camera_id,
            camera_url=camera_url,
            detection_service=detection_service,
            alert_service=alert_service,
            frame_sample_rate=frame_sample_rate,
            batch_size=4  # Optimized batch size
        )
        
        if processor.start():
            _active_processors[camera_id] = processor
            logger.info(f"Started enhanced processor for camera {camera_id}")
            return True
        else:
            logger.error(f"Failed to start enhanced processor for camera {camera_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating enhanced processor for camera {camera_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def stop_processor(camera_id: int) -> bool:
    """Stop video processor for camera."""
    processor = _active_processors.get(camera_id)
    if not processor:
        logger.warning(f"No active processor for camera {camera_id}")
        return False
    
    try:
        processor.stop()
        del _active_processors[camera_id]
        logger.info(f"Stopped enhanced processor for camera {camera_id}")
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