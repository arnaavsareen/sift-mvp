import logging
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from shapely.geometry import Point, Polygon
from sqlalchemy.orm import Session

from backend.models import Zone, Camera
from backend.services.config_service import get_config_service

logger = logging.getLogger(__name__)

class ZoneService:
    """
    Service for managing and processing camera zones.
    Zones are defined areas within a camera view with specific safety rules.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.zone_cache: Dict[int, Dict[int, Dict[str, Any]]] = {}  # camera_id -> zone_id -> zone_data
        self.polygon_cache: Dict[str, Polygon] = {}  # zone_key -> shapely.Polygon
        self.config_service = get_config_service(db)
    
    def get_zones(self, camera_id: int) -> List[Dict[str, Any]]:
        """
        Get all zones for a camera.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            List of zone dictionaries
        """
        try:
            # Try to get from cache first
            if camera_id in self.zone_cache:
                return list(self.zone_cache[camera_id].values())
            
            # Query database
            zones = self.db.query(Zone).filter(Zone.camera_id == camera_id).all()
            
            # Initialize cache for this camera
            self.zone_cache[camera_id] = {}
            
            # Convert to dictionaries and cache
            result = []
            for zone in zones:
                zone_dict = {
                    "id": zone.id,
                    "camera_id": zone.camera_id,
                    "name": zone.name,
                    "polygon": zone.polygon,
                    "rule_type": zone.rule_type,
                    "created_at": zone.created_at.isoformat() if zone.created_at else None
                }
                
                # Cache zone
                self.zone_cache[camera_id][zone.id] = zone_dict
                
                # Cache polygon
                self._cache_polygon(camera_id, zone.id, zone.polygon)
                
                result.append(zone_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting zones for camera {camera_id}: {str(e)}")
            return []
    
    def get_zone(self, zone_id: int) -> Optional[Dict[str, Any]]:
        """
        Get zone by ID.
        
        Args:
            zone_id: Zone ID
            
        Returns:
            Zone dictionary or None if not found
        """
        try:
            # Check all camera caches
            for camera_zones in self.zone_cache.values():
                if zone_id in camera_zones:
                    return camera_zones[zone_id]
            
            # Query database
            zone = self.db.query(Zone).filter(Zone.id == zone_id).first()
            if not zone:
                return None
            
            # Convert to dictionary
            zone_dict = {
                "id": zone.id,
                "camera_id": zone.camera_id,
                "name": zone.name,
                "polygon": zone.polygon,
                "rule_type": zone.rule_type,
                "created_at": zone.created_at.isoformat() if zone.created_at else None
            }
            
            # Cache zone
            if zone.camera_id not in self.zone_cache:
                self.zone_cache[zone.camera_id] = {}
            
            self.zone_cache[zone.camera_id][zone.id] = zone_dict
            
            # Cache polygon
            self._cache_polygon(zone.camera_id, zone.id, zone.polygon)
            
            return zone_dict
            
        except Exception as e:
            logger.error(f"Error getting zone {zone_id}: {str(e)}")
            return None
    
    def create_zone(self, zone_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new zone.
        
        Args:
            zone_data: Zone data dictionary
            
        Returns:
            Created zone dictionary or None if failed
        """
        try:
            # Create zone object
            zone = Zone(
                camera_id=zone_data["camera_id"],
                name=zone_data["name"],
                polygon=zone_data["polygon"],
                rule_type=zone_data["rule_type"]
            )
            
            # Save to database
            self.db.add(zone)
            self.db.commit()
            self.db.refresh(zone)
            
            # Convert to dictionary
            zone_dict = {
                "id": zone.id,
                "camera_id": zone.camera_id,
                "name": zone.name,
                "polygon": zone.polygon,
                "rule_type": zone.rule_type,
                "created_at": zone.created_at.isoformat() if zone.created_at else None
            }
            
            # Cache zone
            if zone.camera_id not in self.zone_cache:
                self.zone_cache[zone.camera_id] = {}
            
            self.zone_cache[zone.camera_id][zone.id] = zone_dict
            
            # Cache polygon
            self._cache_polygon(zone.camera_id, zone.id, zone.polygon)
            
            logger.info(f"Created zone {zone.id} for camera {zone.camera_id}")
            return zone_dict
            
        except Exception as e:
            logger.error(f"Error creating zone: {str(e)}")
            self.db.rollback()
            return None
    
    def update_zone(self, zone_id: int, zone_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a zone.
        
        Args:
            zone_id: Zone ID
            zone_data: Zone data dictionary with fields to update
            
        Returns:
            Updated zone dictionary or None if failed
        """
        try:
            # Get zone
            zone = self.db.query(Zone).filter(Zone.id == zone_id).first()
            if not zone:
                logger.warning(f"Zone {zone_id} not found")
                return None
            
            # Update fields
            if "name" in zone_data:
                zone.name = zone_data["name"]
            
            if "polygon" in zone_data:
                zone.polygon = zone_data["polygon"]
            
            if "rule_type" in zone_data:
                zone.rule_type = zone_data["rule_type"]
            
            # Save to database
            self.db.commit()
            self.db.refresh(zone)
            
            # Convert to dictionary
            zone_dict = {
                "id": zone.id,
                "camera_id": zone.camera_id,
                "name": zone.name,
                "polygon": zone.polygon,
                "rule_type": zone.rule_type,
                "created_at": zone.created_at.isoformat() if zone.created_at else None
            }
            
            # Update cache
            if zone.camera_id in self.zone_cache:
                self.zone_cache[zone.camera_id][zone.id] = zone_dict
            
            # Update polygon cache
            if "polygon" in zone_data:
                self._cache_polygon(zone.camera_id, zone.id, zone.polygon)
            
            logger.info(f"Updated zone {zone.id}")
            return zone_dict
            
        except Exception as e:
            logger.error(f"Error updating zone {zone_id}: {str(e)}")
            self.db.rollback()
            return None
    
    def delete_zone(self, zone_id: int) -> bool:
        """
        Delete a zone.
        
        Args:
            zone_id: Zone ID
            
        Returns:
            True if successful
        """
        try:
            # Get zone
            zone = self.db.query(Zone).filter(Zone.id == zone_id).first()
            if not zone:
                logger.warning(f"Zone {zone_id} not found")
                return False
            
            camera_id = zone.camera_id
            
            # Delete from database
            self.db.delete(zone)
            self.db.commit()
            
            # Remove from cache
            if camera_id in self.zone_cache and zone_id in self.zone_cache[camera_id]:
                del self.zone_cache[camera_id][zone_id]
            
            # Remove from polygon cache
            zone_key = f"{camera_id}_{zone_id}"
            if zone_key in self.polygon_cache:
                del self.polygon_cache[zone_key]
            
            logger.info(f"Deleted zone {zone_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting zone {zone_id}: {str(e)}")
            self.db.rollback()
            return False
    
    def _cache_polygon(self, camera_id: int, zone_id: int, polygon_points: List[List[int]]) -> None:
        """
        Cache a shapely.Polygon object for a zone.
        
        Args:
            camera_id: Camera ID
            zone_id: Zone ID
            polygon_points: List of [x, y] coordinates
        """
        try:
            # Create key
            zone_key = f"{camera_id}_{zone_id}"
            
            # Validate polygon
            if not polygon_points or len(polygon_points) < 3:
                logger.warning(f"Invalid polygon for zone {zone_id}: requires at least 3 points")
                if zone_key in self.polygon_cache:
                    del self.polygon_cache[zone_key]
                return
            
            # Create shapely polygon
            polygon = Polygon(polygon_points)
            
            # Validate polygon
            if not polygon.is_valid:
                logger.warning(f"Invalid polygon for zone {zone_id}: geometry is not valid")
                if zone_key in self.polygon_cache:
                    del self.polygon_cache[zone_key]
                return
            
            # Cache polygon
            self.polygon_cache[zone_key] = polygon
            
        except Exception as e:
            logger.error(f"Error caching polygon for zone {zone_id}: {str(e)}")
    
    def point_in_zone(self, camera_id: int, zone_id: int, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside a zone.
        
        Args:
            camera_id: Camera ID
            zone_id: Zone ID
            point: (x, y) coordinates
            
        Returns:
            True if point is inside zone
        """
        try:
            # Get polygon
            zone_key = f"{camera_id}_{zone_id}"
            polygon = self.polygon_cache.get(zone_key)
            
            # If polygon not cached, get zone and cache polygon
            if polygon is None:
                zone = self.get_zone(zone_id)
                if not zone or zone["camera_id"] != camera_id:
                    return False
                
                # Try again
                polygon = self.polygon_cache.get(zone_key)
                if polygon is None:
                    return False
            
            # Check if point is in polygon
            return polygon.contains(Point(point))
            
        except Exception as e:
            logger.error(f"Error checking point in zone {zone_id}: {str(e)}")
            return False
    
    def detect_zones(self, camera_id: int, detection: Dict[str, Any]) -> List[int]:
        """
        Detect which zones a detection is in based on its bounding box.
        
        Args:
            camera_id: Camera ID
            detection: Detection dictionary with bbox field
            
        Returns:
            List of zone IDs that the detection is in
        """
        try:
            if "bbox" not in detection:
                return []
            
            # Get zones for camera
            zones = self.get_zones(camera_id)
            if not zones:
                return []
            
            # Extract bounding box
            bbox = detection["bbox"]
            if len(bbox) != 4:
                return []
            
            x1, y1, x2, y2 = bbox
            
            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Check which zones the center point is in
            result = []
            for zone in zones:
                zone_id = zone["id"]
                if self.point_in_zone(camera_id, zone_id, (center_x, center_y)):
                    result.append(zone_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting zones for detection in camera {camera_id}: {str(e)}")
            return []
    
    def get_zone_ppe_requirements(self, camera_id: int, zone_ids: List[int]) -> Dict[str, bool]:
        """
        Get combined PPE requirements for a list of zones.
        
        Args:
            camera_id: Camera ID
            zone_ids: List of zone IDs
            
        Returns:
            Dictionary mapping PPE item names to boolean requirement
        """
        try:
            requirements = {}
            
            # Get default requirements from config service
            default_requirements = self.config_service.get_ppe_requirements(camera_id)
            for item in default_requirements:
                requirements[item] = True
            
            # Get requirements for each zone
            for zone_id in zone_ids:
                # Get zone
                zone = self.get_zone(zone_id)
                if not zone or zone["camera_id"] != camera_id:
                    continue
                
                # Get rule type and apply requirements
                rule_type = zone.get("rule_type")
                if not rule_type:
                    continue
                
                # Get zone-specific requirements
                zone_requirements = self.config_service.get_ppe_requirements(camera_id, zone_id)
                for item in zone_requirements:
                    requirements[item] = True
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error getting PPE requirements for camera {camera_id}, zones {zone_ids}: {str(e)}")
            return {}
    
    def draw_zones(self, frame, camera_id: int, color_map: Optional[Dict[str, Tuple[int, int, int]]] = None) -> np.ndarray:
        """
        Draw zones on frame for visualization.
        
        Args:
            frame: Image frame
            camera_id: Camera ID
            color_map: Optional mapping of rule_type to RGB color
            
        Returns:
            Frame with zones drawn
        """
        try:
            if color_map is None:
                # Default colors for different rule types
                color_map = {
                    "ppe_required": (0, 255, 0),    # Green
                    "restricted_area": (0, 0, 255), # Red
                    "hazmat_area": (0, 165, 255),   # Orange
                    "fall_protection": (255, 0, 0), # Blue
                    "default": (255, 255, 255)      # White
                }
            
            # Get zones for camera
            zones = self.get_zones(camera_id)
            if not zones:
                return frame
            
            # Create a copy of the frame to avoid modifying the original
            result = frame.copy()
            
            # Draw each zone
            for zone in zones:
                # Get polygon points
                polygon = zone["polygon"]
                if not polygon or len(polygon) < 3:
                    continue
                
                # Get color for rule type
                rule_type = zone.get("rule_type", "default")
                color = color_map.get(rule_type, color_map["default"])
                
                # Convert polygon to numpy array
                points = np.array(polygon, dtype=np.int32)
                
                # Draw filled polygon with transparency
                overlay = result.copy()
                cv2.fillPoly(overlay, [points], color)
                cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
                
                # Draw polygon outline
                cv2.polylines(result, [points], True, color, 2)
                
                # Draw zone name
                name = zone.get("name", f"Zone {zone['id']}")
                
                # Calculate centroid for text placement
                centroid_x = int(np.mean(points[:, 0]))
                centroid_y = int(np.mean(points[:, 1]))
                
                # Draw text background
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(
                    result,
                    (centroid_x - text_size[0] // 2 - 5, centroid_y - text_size[1] // 2 - 5),
                    (centroid_x + text_size[0] // 2 + 5, centroid_y + text_size[1] // 2 + 5),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    result,
                    name,
                    (centroid_x - text_size[0] // 2, centroid_y + text_size[1] // 2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0) if sum(color) > 384 else (255, 255, 255),
                    2
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error drawing zones for camera {camera_id}: {str(e)}")
            return frame
    
    def process_detection_zones(self, camera_id: int, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process detections to add zone information and apply zone-specific rules.
        
        Args:
            camera_id: Camera ID
            detections: List of detection dictionaries
            
        Returns:
            Updated list of detection dictionaries with zone information
        """
        try:
            # Get zones for camera
            zones = self.get_zones(camera_id)
            if not zones:
                return detections
            
            result = []
            
            for detection in detections:
                # Create a copy of the detection
                updated = detection.copy()
                
                # Detect which zones the detection is in
                zone_ids = self.detect_zones(camera_id, detection)
                updated["zones"] = zone_ids
                
                # If detection is a person, apply PPE requirements
                if detection.get("class") == "person":
                    # Get combined PPE requirements for all zones
                    ppe_requirements = self.get_zone_ppe_requirements(camera_id, zone_ids)
                    
                    # Check if any required PPE is missing
                    violations = []
                    
                    for ppe_item, required in ppe_requirements.items():
                        if required and not detection.get(f"has_{ppe_item}", False):
                            violations.append(f"no_{ppe_item}")
                    
                    # Update detection with violations
                    if violations:
                        updated["violation"] = True
                        updated["violation_type"] = ",".join(violations)
                    
                    # Add rule_types for all zones
                    rule_types = []
                    for zone_id in zone_ids:
                        zone = next((z for z in zones if z["id"] == zone_id), None)
                        if zone and zone.get("rule_type"):
                            rule_types.append(zone["rule_type"])
                    
                    updated["rule_types"] = list(set(rule_types))
                
                result.append(updated)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing detection zones for camera {camera_id}: {str(e)}")
            return detections


# Factory function to create zone service
def get_zone_service(db: Session) -> ZoneService:
    """Create zone service with database session."""
    return ZoneService(db)