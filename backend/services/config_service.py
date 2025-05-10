import os
import json
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator

from backend.config import BASE_DIR

logger = logging.getLogger(__name__)

# Define configuration models with validation
class PPERequirements(BaseModel):
    """PPE requirements configuration with validation"""
    hardhat: bool = True
    vest: bool = True
    mask: bool = False
    goggles: bool = False
    gloves: bool = False
    boots: bool = False

    def to_list(self) -> List[str]:
        """Convert to list of required PPE items"""
        return [item for item, required in self.dict().items() if required]
    
    @classmethod
    def from_list(cls, ppe_list: List[str]) -> 'PPERequirements':
        """Create from list of required PPE items"""
        return cls(**{item: item in ppe_list for item in cls.__fields__})


class ZoneConfig(BaseModel):
    """Configuration for a detection zone"""
    id: int
    name: str
    polygon: List[List[int]]  # [[x1,y1], [x2,y2], ...]
    ppe_requirements: PPERequirements = Field(default_factory=PPERequirements)
    enabled: bool = True


class DetectionConfig(BaseModel):
    """Detection configuration with validation"""
    confidence_threshold: float = Field(0.25, ge=0.05, le=0.95)
    frame_sample_rate: int = Field(10, ge=1, le=30)
    alert_throttle_seconds: float = Field(5.0, ge=1.0, le=60.0)
    default_ppe_requirements: PPERequirements = Field(default_factory=PPERequirements)
    use_cuda: bool = True


class CameraConfig(BaseModel):
    """Camera-specific configuration"""
    id: int
    name: str = "Camera"
    enabled: bool = True
    ppe_requirements: Optional[PPERequirements] = None
    zones: List[ZoneConfig] = []
    detection_config: Optional[DetectionConfig] = None


class SystemConfig(BaseModel):
    """Global system configuration"""
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    cameras: Dict[str, CameraConfig] = {}
    last_updated: datetime = Field(default_factory=datetime.now)
    version: str = "1.0"


class ConfigService:
    """
    Service for managing system configuration.
    Provides centralized access to all configurable parameters.
    """
    
    def __init__(
        self, 
        db: Optional[Session] = None,
        config_dir: str = os.path.join(BASE_DIR, "data", "config"),
        config_file: str = "system_config.json"
    ):
        self.db = db
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / config_file
        
        # Load or create configuration
        self.config = self._load_config()
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Cache of camera configs with last update time
        self._camera_config_cache = {}
        
        logger.info(f"Configuration service initialized with {len(self.config.cameras)} cameras")
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from file or create default if not exists."""
        if not self.config_file.exists():
            # Create default configuration
            config = SystemConfig()
            self._save_config(config)
            return config
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Convert string camera IDs back to integers
            if "cameras" in config_data:
                cameras = {}
                for cam_id_str, cam_config in config_data["cameras"].items():
                    try:
                        cam_id = int(cam_id_str)
                        cam_config["id"] = cam_id
                        cameras[cam_id_str] = cam_config
                    except ValueError:
                        logger.error(f"Invalid camera ID: {cam_id_str}")
                
                config_data["cameras"] = cameras
            
            # Parse with validation
            return SystemConfig(**config_data)
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Create and return default configuration
            return SystemConfig()
    
    def _save_config(self, config: SystemConfig) -> bool:
        """Save configuration to file."""
        with self._lock:
            try:
                # Ensure config directory exists
                self.config_dir.mkdir(parents=True, exist_ok=True)
                
                # Update timestamp
                config.last_updated = datetime.now()
                
                # Save to file
                with open(self.config_file, 'w') as f:
                    f.write(config.json(indent=2))
                
                return True
            except Exception as e:
                logger.error(f"Error saving configuration: {str(e)}")
                return False
    
    def get_detection_config(self) -> DetectionConfig:
        """Get global detection configuration."""
        with self._lock:
            return self.config.detection
    
    def update_detection_config(self, config: Dict[str, Any]) -> bool:
        """
        Update global detection configuration.
        
        Args:
            config: Dictionary with configuration values to update
            
        Returns:
            True if update successful
        """
        with self._lock:
            # Update only provided fields
            detection_config = self.config.detection.dict()
            detection_config.update(config)
            
            # Validate and apply
            try:
                self.config.detection = DetectionConfig(**detection_config)
                logger.info(f"Updated detection configuration: {config}")
                return self._save_config(self.config)
            except Exception as e:
                logger.error(f"Invalid detection configuration: {str(e)}")
                return False
    
    def get_camera_config(self, camera_id: int) -> Optional[CameraConfig]:
        """
        Get camera-specific configuration.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Camera configuration or None if not found
        """
        with self._lock:
            camera_id_str = str(camera_id)
            
            if camera_id_str in self.config.cameras:
                return self.config.cameras[camera_id_str]
            
            # Return None if camera not found
            return None
    
    def get_camera_configs(self) -> Dict[int, CameraConfig]:
        """
        Get all camera configurations.
        
        Returns:
            Dictionary of camera configurations with integer keys
        """
        with self._lock:
            result = {}
            for camera_id_str, config in self.config.cameras.items():
                try:
                    camera_id = int(camera_id_str)
                    result[camera_id] = config
                except ValueError:
                    continue
            return result
    
    def update_camera_config(self, camera_id: int, config: Dict[str, Any]) -> bool:
        """
        Update camera-specific configuration.
        
        Args:
            camera_id: Camera identifier
            config: Dictionary with configuration values to update
            
        Returns:
            True if update successful
        """
        with self._lock:
            camera_id_str = str(camera_id)
            
            # Get existing config or create new one
            if camera_id_str in self.config.cameras:
                camera_config = self.config.cameras[camera_id_str].dict()
            else:
                camera_config = {"id": camera_id, "name": f"Camera {camera_id}"}
            
            # Update with new values
            camera_config.update(config)
            
            # Validate and apply
            try:
                self.config.cameras[camera_id_str] = CameraConfig(**camera_config)
                logger.info(f"Updated configuration for camera {camera_id}: {config}")
                
                # Update cache timestamp
                self._camera_config_cache[camera_id] = time.time()
                
                return self._save_config(self.config)
            except Exception as e:
                logger.error(f"Invalid camera configuration for camera {camera_id}: {str(e)}")
                return False
    
    def delete_camera_config(self, camera_id: int) -> bool:
        """
        Delete camera configuration.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            True if deletion successful
        """
        with self._lock:
            camera_id_str = str(camera_id)
            
            if camera_id_str in self.config.cameras:
                del self.config.cameras[camera_id_str]
                
                # Remove from cache
                if camera_id in self._camera_config_cache:
                    del self._camera_config_cache[camera_id]
                
                logger.info(f"Deleted configuration for camera {camera_id}")
                return self._save_config(self.config)
            
            # Camera not found, consider it a success
            return True
    
    def get_ppe_requirements(self, camera_id: Optional[int] = None, zone_id: Optional[int] = None) -> List[str]:
        """
        Get required PPE items for a specific camera and zone.
        Falls back to defaults if specific config not found.
        
        Args:
            camera_id: Optional camera ID
            zone_id: Optional zone ID (requires camera_id)
            
        Returns:
            List of required PPE items
        """
        with self._lock:
            # Start with system default
            requirements = self.config.detection.default_ppe_requirements
            
            # If no camera specified, return system default
            if camera_id is None:
                return requirements.to_list()
            
            # Try to get camera config
            camera_config = self.get_camera_config(camera_id)
            if camera_config is None:
                return requirements.to_list()
            
            # Use camera-specific requirements if defined
            if camera_config.ppe_requirements is not None:
                requirements = camera_config.ppe_requirements
            
            # If no zone specified, return camera-level requirements
            if zone_id is None:
                return requirements.to_list()
            
            # Try to find zone-specific requirements
            for zone in camera_config.zones:
                if zone.id == zone_id and zone.enabled:
                    return zone.ppe_requirements.to_list()
            
            # Fall back to camera-level requirements
            return requirements.to_list()
    
    def get_confidence_threshold(self, camera_id: Optional[int] = None) -> float:
        """
        Get confidence threshold for a specific camera or system default.
        
        Args:
            camera_id: Optional camera ID
            
        Returns:
            Confidence threshold value
        """
        with self._lock:
            # Start with system default
            threshold = self.config.detection.confidence_threshold
            
            # If no camera specified, return system default
            if camera_id is None:
                return threshold
            
            # Try to get camera config
            camera_config = self.get_camera_config(camera_id)
            if camera_config is None or camera_config.detection_config is None:
                return threshold
            
            # Use camera-specific threshold if defined
            return camera_config.detection_config.confidence_threshold
    
    def get_frame_sample_rate(self, camera_id: Optional[int] = None) -> int:
        """
        Get frame sample rate for a specific camera or system default.
        
        Args:
            camera_id: Optional camera ID
            
        Returns:
            Frame sample rate value
        """
        with self._lock:
            # Start with system default
            rate = self.config.detection.frame_sample_rate
            
            # If no camera specified, return system default
            if camera_id is None:
                return rate
            
            # Try to get camera config
            camera_config = self.get_camera_config(camera_id)
            if camera_config is None or camera_config.detection_config is None:
                return rate
            
            # Use camera-specific rate if defined
            return camera_config.detection_config.frame_sample_rate
    
    def get_alert_throttle_seconds(self, camera_id: Optional[int] = None) -> float:
        """
        Get alert throttling period for a specific camera or system default.
        
        Args:
            camera_id: Optional camera ID
            
        Returns:
            Alert throttle period in seconds
        """
        with self._lock:
            # Start with system default
            seconds = self.config.detection.alert_throttle_seconds
            
            # If no camera specified, return system default
            if camera_id is None:
                return seconds
            
            # Try to get camera config
            camera_config = self.get_camera_config(camera_id)
            if camera_config is None or camera_config.detection_config is None:
                return seconds
            
            # Use camera-specific period if defined
            return camera_config.detection_config.alert_throttle_seconds
    
    def get_zones(self, camera_id: int) -> List[ZoneConfig]:
        """
        Get detection zones for a specific camera.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            List of zone configurations
        """
        with self._lock:
            # Try to get camera config
            camera_config = self.get_camera_config(camera_id)
            if camera_config is None:
                return []
            
            # Return zones (filtering disabled ones)
            return [zone for zone in camera_config.zones if zone.enabled]
    
    def is_cuda_enabled(self) -> bool:
        """
        Check if CUDA (GPU acceleration) is enabled.
        
        Returns:
            True if CUDA is enabled
        """
        with self._lock:
            return self.config.detection.use_cuda
    
    def set_cuda_enabled(self, enabled: bool) -> bool:
        """
        Enable or disable CUDA (GPU acceleration).
        
        Args:
            enabled: Whether to enable CUDA
            
        Returns:
            True if update successful
        """
        with self._lock:
            self.config.detection.use_cuda = enabled
            logger.info(f"CUDA {'enabled' if enabled else 'disabled'}")
            return self._save_config(self.config)
    
    def export_config(self) -> Dict[str, Any]:
        """
        Export configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        with self._lock:
            return json.loads(self.config.json())
    
    def import_config(self, config_data: Dict[str, Any]) -> bool:
        """
        Import configuration from dictionary.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            True if import successful
        """
        with self._lock:
            try:
                # Parse and validate
                new_config = SystemConfig(**config_data)
                
                # Apply and save
                self.config = new_config
                logger.info("Imported new configuration")
                
                # Clear cache
                self._camera_config_cache.clear()
                
                return self._save_config(self.config)
            except Exception as e:
                logger.error(f"Invalid configuration import: {str(e)}")
                return False
    
    def get_config_last_updated(self) -> datetime:
        """
        Get timestamp of last configuration update.
        
        Returns:
            Last update timestamp
        """
        with self._lock:
            return self.config.last_updated
    
    def reset_to_defaults(self) -> bool:
        """
        Reset configuration to defaults.
        
        Returns:
            True if reset successful
        """
        with self._lock:
            try:
                # Create default configuration
                self.config = SystemConfig()
                logger.warning("Reset configuration to defaults")
                
                # Clear cache
                self._camera_config_cache.clear()
                
                return self._save_config(self.config)
            except Exception as e:
                logger.error(f"Error resetting configuration: {str(e)}")
                return False


# Singleton instance
_config_service = None

def get_config_service(db: Optional[Session] = None) -> ConfigService:
    """Get or create configuration service singleton."""
    global _config_service
    if _config_service is None:
        _config_service = ConfigService(db)
    return _config_service