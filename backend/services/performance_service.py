import os
import time
import logging
import threading
import json
import psutil
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from collections import deque
import numpy as np
import traceback

from backend.config import BASE_DIR

# Initialize logger
logger = logging.getLogger(__name__)

class PerformanceMetric:
    """Base class for performance metrics"""
    def __init__(self, name: str, window_size: int = 100):
        self.name = name
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.last_update = time.time()
    
    def add_value(self, value: float) -> None:
        """Add a value to the metric"""
        self.values.append(value)
        self.last_update = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this metric"""
        if not self.values:
            return {
                "name": self.name,
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "p95": None,
                "last": None,
                "last_update": self.last_update
            }
        
        values_array = np.array(self.values)
        return {
            "name": self.name,
            "count": len(self.values),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "p95": float(np.percentile(values_array, 95)),
            "last": float(self.values[-1]),
            "last_update": self.last_update
        }


class PerformanceService:
    """
    Service for monitoring system performance metrics.
    Tracks CPU, memory, GPU (if available), detection times, and more.
    """
    
    def __init__(
        self, 
        metrics_dir: str = os.path.join(BASE_DIR, "data", "metrics"),
        snapshot_interval: int = 300,  # 5 minutes
        max_snapshots: int = 288  # 1 day at 5 minute intervals
    ):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        
        # Performance metrics
        self.metrics: Dict[str, PerformanceMetric] = {}
        
        # System info
        self.system_info = self._get_system_info()
        
        # Aggregated statistics
        self.stats_snapshot: Dict[str, Any] = {}
        self.last_snapshot_time = 0
        
        # Background monitoring thread
        self.monitor_thread = None
        self.is_monitoring = False
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Start background monitoring
        self.start_monitoring()
        
        logger.info("Performance monitoring service initialized")
    
    def _initialize_metrics(self) -> None:
        """Initialize performance metrics"""
        # System metrics
        self.metrics["cpu_usage"] = PerformanceMetric("CPU Usage (%)")
        self.metrics["memory_usage"] = PerformanceMetric("Memory Usage (%)")
        self.metrics["disk_usage"] = PerformanceMetric("Disk Usage (%)")
        
        # Detection metrics
        self.metrics["detection_time"] = PerformanceMetric("Detection Time (ms)")
        self.metrics["frame_processing_time"] = PerformanceMetric("Frame Processing Time (ms)")
        self.metrics["frames_per_second"] = PerformanceMetric("Frames Per Second")
        
        # Alert metrics
        self.metrics["alerts_per_minute"] = PerformanceMetric("Alerts Per Minute")
        self.metrics["alert_processing_time"] = PerformanceMetric("Alert Processing Time (ms)")
        
        # Model metrics
        self.metrics["model_loading_time"] = PerformanceMetric("Model Loading Time (ms)")
        self.metrics["inference_time"] = PerformanceMetric("Inference Time (ms)")
        
        # Try to initialize GPU metrics if available
        try:
            import torch
            if torch.cuda.is_available():
                self.metrics["gpu_usage"] = PerformanceMetric("GPU Usage (%)")
                self.metrics["gpu_memory"] = PerformanceMetric("GPU Memory Usage (%)")
        except (ImportError, Exception):
            pass
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_memory": psutil.virtual_memory().total,
            "hostname": platform.node(),
            "architecture": platform.machine()
        }
        
        # Try to get GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu_available"] = True
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["cuda_version"] = torch.version.cuda
            else:
                info["gpu_available"] = False
        except (ImportError, Exception):
            info["gpu_available"] = False
        
        return info
    
    def start_monitoring(self) -> bool:
        """Start the background monitoring thread"""
        if self.is_monitoring:
            return True
        
        try:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="performance-monitor"
            )
            self.monitor_thread.start()
            logger.info("Started performance monitoring thread")
            return True
        except Exception as e:
            logger.error(f"Error starting performance monitor: {str(e)}")
            self.is_monitoring = False
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop the background monitoring thread"""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
            logger.info("Stopped performance monitoring thread")
        
        return True
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        try:
            while self.is_monitoring:
                # Update system metrics
                self._update_system_metrics()
                
                # Take periodic snapshots
                now = time.time()
                if now - self.last_snapshot_time >= self.snapshot_interval:
                    self._take_snapshot()
                    self.last_snapshot_time = now
                
                # Sleep to avoid high CPU usage
                time.sleep(5.0)  # Update every 5 seconds
                
        except Exception as e:
            logger.error(f"Error in performance monitor loop: {str(e)}")
            logger.error(traceback.format_exc())
            self.is_monitoring = False
    
    def _update_system_metrics(self) -> None:
        """Update system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics["cpu_usage"].add_value(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics["memory_usage"].add_value(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics["disk_usage"].add_value(disk.percent)
            
            # GPU metrics if available
            if "gpu_usage" in self.metrics:
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Note: Getting accurate GPU usage requires additional tools
                        # This is a placeholder - in production, use nvidia-smi or similar
                        self.metrics["gpu_usage"].add_value(50.0)  # Placeholder
                        
                        # GPU memory usage
                        # This requires nvidia-smi or similar tools in production
                        self.metrics["gpu_memory"].add_value(50.0)  # Placeholder
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {str(e)}")
    
    def _take_snapshot(self) -> None:
        """Take a snapshot of current metrics and save to disk"""
        try:
            # Create snapshot
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    name: metric.get_stats()
                    for name, metric in self.metrics.items()
                },
                "system_info": self.system_info
            }
            
            # Save snapshot to memory
            self.stats_snapshot = snapshot
            
            # Save snapshot to disk
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_file = self.metrics_dir / f"snapshot_{timestamp}.json"
            
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f, indent=2)
            
            # Clean up old snapshots
            self._cleanup_old_snapshots()
            
            logger.debug(f"Saved performance snapshot to {snapshot_file}")
            
        except Exception as e:
            logger.error(f"Error taking performance snapshot: {str(e)}")
    
    def _cleanup_old_snapshots(self) -> None:
        """Clean up old snapshot files"""
        try:
            # Get all snapshot files
            snapshot_files = sorted(
                self.metrics_dir.glob("snapshot_*.json"),
                key=lambda p: p.stat().st_mtime
            )
            
            # Delete oldest files if we have too many
            if len(snapshot_files) > self.max_snapshots:
                for file in snapshot_files[:-self.max_snapshots]:
                    file.unlink()
                    logger.debug(f"Deleted old snapshot: {file}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old snapshots: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary with metrics statistics
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                name: metric.get_stats()
                for name, metric in self.metrics.items()
            },
            "system_info": self.system_info
        }
    
    def get_latest_snapshot(self) -> Dict[str, Any]:
        """
        Get latest metrics snapshot.
        
        Returns:
            Dictionary with snapshot data
        """
        # If we have a recent in-memory snapshot, return it
        if self.stats_snapshot:
            return self.stats_snapshot
        
        # Otherwise try to load latest from disk
        try:
            snapshot_files = sorted(
                self.metrics_dir.glob("snapshot_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if snapshot_files:
                with open(snapshot_files[0], 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading latest snapshot: {str(e)}")
        
        # If no snapshot available, return current metrics
        return self.get_metrics()
    
    def get_historical_metrics(
        self, 
        metric_name: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get historical metrics data.
        
        Args:
            metric_name: Optional specific metric name
            hours: Number of hours to look back
            
        Returns:
            Dictionary with historical data
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        try:
            # Get all snapshot files within time range
            snapshot_files = []
            for file in self.metrics_dir.glob("snapshot_*.json"):
                # Parse timestamp from filename
                try:
                    file_timestamp = datetime.strptime(
                        file.stem.replace("snapshot_", ""),
                        "%Y%m%d_%H%M%S"
                    )
                    if file_timestamp >= cutoff_time:
                        snapshot_files.append((file, file_timestamp))
                except ValueError:
                    continue
            
            # Sort by timestamp
            snapshot_files.sort(key=lambda x: x[1])
            
            # Load snapshots
            snapshots = []
            for file, _ in snapshot_files:
                try:
                    with open(file, 'r') as f:
                        snapshots.append(json.load(f))
                except Exception as e:
                    logger.error(f"Error loading snapshot {file}: {str(e)}")
            
            # If specific metric requested, extract just that data
            if metric_name and metric_name in self.metrics:
                return {
                    "metric": metric_name,
                    "data": [
                        {
                            "timestamp": s["timestamp"],
                            "value": s["metrics"].get(metric_name, {}).get("mean")
                        }
                        for s in snapshots
                        if metric_name in s.get("metrics", {})
                    ]
                }
            
            # Otherwise return all metrics
            return {
                "start_time": cutoff_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "snapshots": snapshots
            }
            
        except Exception as e:
            logger.error(f"Error getting historical metrics: {str(e)}")
            return {
                "error": str(e),
                "start_time": cutoff_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "snapshots": []
            }
    
    def record_detection_time(self, detection_time_ms: float) -> None:
        """
        Record detection time.
        
        Args:
            detection_time_ms: Detection time in milliseconds
        """
        if "detection_time" in self.metrics:
            self.metrics["detection_time"].add_value(detection_time_ms)
    
    def record_frame_processing_time(self, processing_time_ms: float) -> None:
        """
        Record frame processing time.
        
        Args:
            processing_time_ms: Processing time in milliseconds
        """
        if "frame_processing_time" in self.metrics:
            self.metrics["frame_processing_time"].add_value(processing_time_ms)
    
    def record_fps(self, fps: float) -> None:
        """
        Record frames per second.
        
        Args:
            fps: Frames per second
        """
        if "frames_per_second" in self.metrics:
            self.metrics["frames_per_second"].add_value(fps)
    
    def record_alert(self, processing_time_ms: float) -> None:
        """
        Record alert generation.
        
        Args:
            processing_time_ms: Alert processing time in milliseconds
        """
        if "alert_processing_time" in self.metrics:
            self.metrics["alert_processing_time"].add_value(processing_time_ms)
        
        # Update alerts per minute
        if "alerts_per_minute" in self.metrics:
            # Get current value or default to 0
            current = self.metrics["alerts_per_minute"].values[-1] if self.metrics["alerts_per_minute"].values else 0
            # Increment by 1/60 (assuming we aggregate over 60 seconds)
            self.metrics["alerts_per_minute"].add_value(current + 1/60)
    
    def record_model_loading_time(self, loading_time_ms: float) -> None:
        """
        Record model loading time.
        
        Args:
            loading_time_ms: Loading time in milliseconds
        """
        if "model_loading_time" in self.metrics:
            self.metrics["model_loading_time"].add_value(loading_time_ms)
    
    def record_inference_time(self, inference_time_ms: float) -> None:
        """
        Record model inference time.
        
        Args:
            inference_time_ms: Inference time in milliseconds
        """
        if "inference_time" in self.metrics:
            self.metrics["inference_time"].add_value(inference_time_ms)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status.
        
        Returns:
            Dictionary with system status information
        """
        # Get latest metrics
        metrics = self.get_metrics()
        
        # Extract key metrics
        cpu_usage = metrics["metrics"].get("cpu_usage", {}).get("last", 0)
        memory_usage = metrics["metrics"].get("memory_usage", {}).get("last", 0)
        disk_usage = metrics["metrics"].get("disk_usage", {}).get("last", 0)
        
        # Determine overall status
        status = "healthy"
        issues = []
        
        if cpu_usage > 90:
            status = "warning"
            issues.append("High CPU usage")
        
        if memory_usage > 90:
            status = "warning"
            issues.append("High memory usage")
        
        if disk_usage > 90:
            status = "warning"
            issues.append("High disk usage")
        
        if "gpu_usage" in metrics["metrics"]:
            gpu_usage = metrics["metrics"]["gpu_usage"].get("last", 0)
            if gpu_usage > 90:
                status = "warning"
                issues.append("High GPU usage")
        
        # Check FPS
        if "frames_per_second" in metrics["metrics"]:
            fps = metrics["metrics"]["frames_per_second"].get("last", 0)
            if fps < 1:
                status = "critical"
                issues.append("Very low FPS")
            elif fps < 5:
                status = "warning"
                issues.append("Low FPS")
        
        # Check detection time
        if "detection_time" in metrics["metrics"]:
            detection_time = metrics["metrics"]["detection_time"].get("last", 0)
            if detection_time > 5000:  # More than 5 seconds
                status = "critical"
                issues.append("Very slow detection")
            elif detection_time > 1000:  # More than 1 second
                status = "warning"
                issues.append("Slow detection")
        
        return {
            "status": status,
            "issues": issues,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "fps": metrics["metrics"].get("frames_per_second", {}).get("last"),
            "detection_time": metrics["metrics"].get("detection_time", {}).get("mean"),
            "gpu_available": metrics["system_info"].get("gpu_available", False),
            "gpu_usage": metrics["metrics"].get("gpu_usage", {}).get("last") if metrics["system_info"].get("gpu_available", False) else None,
            "timestamp": datetime.now().isoformat()
        }


# Singleton instance
_performance_service = None

def get_performance_service() -> PerformanceService:
    """Get or create performance service singleton."""
    global _performance_service
    if _performance_service is None:
        _performance_service = PerformanceService()
    return _performance_service

# Import optimization flags
try:
    import torch
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
        # Enable cuDNN auto-tuner for better performance
        torch.backends.cudnn.benchmark = True
except ImportError:
    pass