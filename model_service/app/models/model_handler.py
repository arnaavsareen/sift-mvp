"""
YOLO Model Handler for PPE Detection

This module provides a robust implementation for loading and using YOLOv8 models
for Personal Protective Equipment (PPE) detection and violation identification.
It includes functionality for model loading, inference, result processing, and
image annotation.

Author: SIFT Development Team
Date: May 5, 2025
"""

import os
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set, NamedTuple

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Type aliases for clarity
BoundingBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalized)
ImageType = Union[np.ndarray, str, Path]  # OpenCV image, file path, or Path object


class ViolationType(str, Enum):
    """Enumeration of PPE violation types."""
    NO_HARDHAT = "no_hardhat"
    NO_SAFETY_VEST = "no_safety_vest"
    NO_GLOVES = "no_gloves"
    NO_SAFETY_GOGGLES = "no_safety_goggles"
    NO_MASK = "no_mask"
    NO_SAFETY_BOOTS = "no_safety_boots"
    NO_FALL_PROTECTION = "no_fall_protection"
    UNKNOWN = "unknown_violation"


class PPEClass(str, Enum):
    """Enumeration of PPE equipment class names."""
    HARDHAT = "hardhat"
    SAFETY_VEST = "safety_vest"
    GLOVES = "gloves"
    SAFETY_GOGGLES = "safety_goggles"
    MASK = "mask"
    SAFETY_BOOTS = "safety_boots"
    FALL_PROTECTION = "fall_protection"
    PERSON = "person"  # For standard YOLO models


class PPEViolation(NamedTuple):
    """Named tuple representing a PPE violation."""
    violation_type: ViolationType
    confidence: float
    bbox: BoundingBox
    person_bbox: Optional[BoundingBox] = None


class DetectionResult(NamedTuple):
    """Named tuple representing a detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox  # Normalized coordinates [x1, y1, x2, y2]


class PPEDetector:
    """
    YOLOv8-based detector for Personal Protective Equipment (PPE) violations.
    
    This class provides methods for loading YOLO models, performing inference,
    processing results, and annotating images with detection results.
    
    Attributes:
        model_path (str): Path to the YOLO model file
        model (YOLO): Loaded YOLO model instance
        confidence_threshold (float): Confidence threshold for detections
        iou_threshold (float): IoU threshold for non-maximum suppression
        device (str): Device to run inference on ('cpu', 'cuda', etc.)
        class_mapping (Dict[int, str]): Mapping from class IDs to class names
        violation_mapping (Dict[str, ViolationType]): Mapping from class names to violation types
        is_custom_model (bool): Whether using a custom PPE model or standard YOLO
        person_class_id (Optional[int]): Class ID for 'person' if using standard YOLO
    """

    # Standard YOLO COCO class mapping (subset)
    COCO_CLASS_MAPPING = {
        0: "person",
        1: "bicycle",
        2: "car",
        # ... (other classes not relevant for PPE detection)
    }

    # Default custom PPE model class mapping
    DEFAULT_PPE_CLASS_MAPPING = {
        0: "person",
        1: "hardhat",
        2: "safety_vest",
        3: "no_hardhat",
        4: "no_safety_vest",
        5: "gloves",
        6: "no_gloves",
        7: "safety_goggles",
        8: "no_safety_goggles",
        9: "mask",
        10: "no_mask",
        11: "safety_boots",
        12: "no_safety_boots",
        13: "fall_protection",
        14: "no_fall_protection"
    }

    # Mapping from class names to violation types
    DEFAULT_VIOLATION_MAPPING = {
        "no_hardhat": ViolationType.NO_HARDHAT,
        "no_safety_vest": ViolationType.NO_SAFETY_VEST,
        "no_gloves": ViolationType.NO_GLOVES,
        "no_safety_goggles": ViolationType.NO_SAFETY_GOGGLES,
        "no_mask": ViolationType.NO_MASK,
        "no_safety_boots": ViolationType.NO_SAFETY_BOOTS,
        "no_fall_protection": ViolationType.NO_FALL_PROTECTION
    }

    # Color mapping for visualization
    VIOLATION_COLORS = {
        ViolationType.NO_HARDHAT: (255, 0, 0),       # Red
        ViolationType.NO_SAFETY_VEST: (255, 165, 0), # Orange
        ViolationType.NO_GLOVES: (255, 255, 0),      # Yellow
        ViolationType.NO_SAFETY_GOGGLES: (0, 255, 0), # Green
        ViolationType.NO_MASK: (0, 255, 255),        # Cyan
        ViolationType.NO_SAFETY_BOOTS: (0, 0, 255),  # Blue
        ViolationType.NO_FALL_PROTECTION: (255, 0, 255), # Magenta
        ViolationType.UNKNOWN: (128, 128, 128)       # Gray
    }

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        class_mapping: Optional[Dict[int, str]] = None,
        violation_mapping: Optional[Dict[str, ViolationType]] = None,
        is_custom_model: bool = True,
    ):
        """
        Initialize the PPE detector with the specified model and parameters.
        
        Args:
            model_path: Path to the YOLO model file (.pt)
            confidence_threshold: Confidence threshold for detections (0.0-1.0)
            iou_threshold: IoU threshold for non-maximum suppression (0.0-1.0)
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
            class_mapping: Custom mapping from class IDs to class names
            violation_mapping: Custom mapping from class names to violation types
            is_custom_model: Whether using a custom PPE model or standard YOLO
        
        Raises:
            FileNotFoundError: If the model file does not exist
            RuntimeError: If the model fails to load
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.is_custom_model = is_custom_model
        
        # Validate model path
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Set class mapping based on model type
        if class_mapping is not None:
            self.class_mapping = class_mapping
        elif is_custom_model:
            self.class_mapping = self.DEFAULT_PPE_CLASS_MAPPING
        else:
            self.class_mapping = self.COCO_CLASS_MAPPING
        
        # Set violation mapping
        self.violation_mapping = violation_mapping or self.DEFAULT_VIOLATION_MAPPING
        
        # Set person class ID for standard YOLO
        self.person_class_id = 0 if is_custom_model else 0  # Person is class 0 in both
        
        # Load the model
        self._load_model()
        
        # Update class mapping with actual model classes if available
        self._update_class_mapping()
        
        logger.info(f"Initialized PPE detector with model: {os.path.basename(model_path)}")
        logger.info(f"Confidence threshold: {confidence_threshold}, IoU threshold: {iou_threshold}")
        logger.info(f"Device: {self.model.device}")
        logger.info(f"Model type: {'Custom PPE' if is_custom_model else 'Standard YOLO'}")

    def _load_model(self) -> None:
        """
        Load the YOLO model from the specified path.
        
        Raises:
            RuntimeError: If the model fails to load
        """
        try:
            logger.info(f"Loading YOLOv8 model from: {self.model_path}")
            start_time = time.time()
            
            # Load the model with specified parameters
            self.model = YOLO(self.model_path)
            
            # Configure model parameters
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            
            # Move model to specified device if not auto
            if self.device != "auto":
                self.model.to(self.device)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"Model version: {self.get_model_version()}")
            
        except Exception as e:
            error_msg = f"Failed to load model from {self.model_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _update_class_mapping(self) -> None:
        """
        Update the class mapping with actual model classes if available.
        """
        try:
            # Get class names from the model if available
            if hasattr(self.model, "names") and self.model.names:
                # Create new mapping from model's names
                model_classes = self.model.names
                logger.info(f"Model has {len(model_classes)} classes")
                
                # Update class mapping with model classes
                for class_id, class_name in model_classes.items():
                    # Store original class name as defined in the model
                    self.class_mapping[class_id] = class_name
                    logger.debug(f"Class {class_id}: {class_name}")
        except Exception as e:
            logger.warning(f"Could not update class mapping from model: {e}")
            logger.warning("Using default class mapping instead")

    def get_model_version(self) -> str:
        """
        Get the model version based on the filename.
        
        Returns:
            str: Model version string
        """
        return os.path.basename(self.model_path)

    def detect(self, image: ImageType) -> List[DetectionResult]:
        """
        Run inference on an image and return all detections.
        
        Args:
            image: Image for inference (numpy array, file path, or Path object)
            
        Returns:
            List[DetectionResult]: List of all detection results
            
        Raises:
            ValueError: If the image cannot be loaded or processed
            RuntimeError: If inference fails
        """
        try:
            start_time = time.time()
            
            # Ensure image is properly loaded
            if isinstance(image, (str, Path)):
                # Check if the file exists
                if not os.path.exists(image):
                    error_msg = f"Image file not found: {image}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                # Load image from file path
                img_path = str(image)
                logger.debug(f"Loading image from path: {img_path}")
                
                # YOLO can process the image path directly
                img_for_inference = img_path
            else:
                # Assume image is already a numpy array
                img_for_inference = image
            
            # Run inference
            logger.debug("Running inference on image")
            results = self.model(
                img_for_inference,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Parse results into DetectionResult objects
            detections = self._parse_detections(results)
            
            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.3f} seconds")
            logger.info(f"Found {len(detections)} detections above confidence threshold")
            
            return detections
            
        except FileNotFoundError:
            # Re-raise file not found error
            raise
        except Exception as e:
            error_msg = f"Error during inference: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _parse_detections(self, results) -> List[DetectionResult]:
        """
        Parse YOLO results into a list of DetectionResult objects.
        
        Args:
            results: Results from YOLO inference
            
        Returns:
            List[DetectionResult]: List of detection results
        """
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get coordinates (pixels)
                
                # Convert to normalized coordinates (0-1)
                if hasattr(result, 'orig_shape'):
                    height, width = result.orig_shape
                    x1, x2 = x1 / width, x2 / width
                    y1, y2 = y1 / height, y2 / height
                
                # Get class information
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Get class name from mapping
                class_name = self.class_mapping.get(class_id, f"unknown_{class_id}")
                
                # Create DetectionResult object
                detection = DetectionResult(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(float(x1), float(y1), float(x2), float(y2))
                )
                
                detections.append(detection)
        
        return detections

    def detect_violations(self, image: ImageType) -> Tuple[List[PPEViolation], List[DetectionResult]]:
        """
        Detect PPE violations in an image.
        
        For custom PPE models, directly identify violation classes.
        For standard YOLO models, detect people and infer violations based on absence of PPE.
        
        Args:
            image: Image for inference (numpy array, file path, or Path object)
            
        Returns:
            Tuple containing:
            - List[PPEViolation]: List of identified PPE violations
            - List[DetectionResult]: All detections (for reference)
            
        Raises:
            ValueError: If the image cannot be loaded or processed
            RuntimeError: If inference fails
        """
        # Run detection
        detections = self.detect(image)
        
        if self.is_custom_model:
            # For custom model, filter for explicit violation classes
            return self._process_custom_model_detections(detections), detections
        else:
            # For standard YOLO, we need logic to infer violations from people detections
            return self._process_standard_model_detections(detections), detections

    def _process_custom_model_detections(self, detections: List[DetectionResult]) -> List[PPEViolation]:
        """
        Process detections from a custom PPE model to identify violations.
        
        Args:
            detections: List of detection results
            
        Returns:
            List[PPEViolation]: List of PPE violations
        """
        violations = []
        
        for detection in detections:
            # Check if class name is in violation mapping
            if detection.class_name in self.violation_mapping:
                violation_type = self.violation_mapping[detection.class_name]
                
                # Create PPEViolation object
                violation = PPEViolation(
                    violation_type=violation_type,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    person_bbox=None  # No associated person bbox for direct violations
                )
                
                violations.append(violation)
                logger.debug(f"Found violation: {violation_type.value} with confidence {detection.confidence:.2f}")
        
        return violations

    def _process_standard_model_detections(self, detections: List[DetectionResult]) -> List[PPEViolation]:
        """
        Process detections from a standard YOLO model to identify PPE violations.
        
        This is a placeholder implementation since standard YOLO models don't have
        PPE-specific classes. In a real implementation, this would use additional
        logic to determine if a person is missing PPE.
        
        Args:
            detections: List of detection results
            
        Returns:
            List[PPEViolation]: List of PPE violations (empty for standard models)
        """
        logger.warning("Standard YOLO model detection used. Cannot identify specific PPE violations.")
        logger.warning("For PPE violation detection, use a custom-trained PPE model.")
        
        # For standard models, this would require additional logic or a second model
        # This is just a placeholder implementation
        return []

    def annotate_image(
        self, 
        image: ImageType,
        detections: Optional[List[DetectionResult]] = None,
        violations: Optional[List[PPEViolation]] = None,
        show_all_detections: bool = False,
        line_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 2
    ) -> np.ndarray:
        """
        Annotate an image with detection results and violations.
        
        Args:
            image: Image to annotate (numpy array, file path, or Path object)
            detections: List of all detections (optional if violations provided)
            violations: List of PPE violations (optional)
            show_all_detections: Whether to show all detections or just violations
            line_thickness: Thickness of bounding box lines
            font_scale: Scale factor for font size
            font_thickness: Thickness of font strokes
            
        Returns:
            np.ndarray: Annotated image
            
        Raises:
            ValueError: If the image cannot be loaded
        """
        # Load image if needed
        if isinstance(image, (str, Path)):
            if not os.path.exists(image):
                raise ValueError(f"Image file not found: {image}")
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Failed to load image: {image}")
        else:
            # Clone the array to avoid modifying the original
            img = image.copy()
        
        # Convert from BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img.copy()
        
        height, width = img_rgb.shape[:2]
        
        # Draw violations if provided
        if violations:
            for violation in violations:
                # Get bounding box in pixel coordinates
                x1, y1, x2, y2 = violation.bbox
                x1, x2 = int(x1 * width), int(x2 * width)
                y1, y2 = int(y1 * height), int(y2 * height)
                
                # Get color for this violation type
                color = self.VIOLATION_COLORS.get(violation.violation_type, self.VIOLATION_COLORS[ViolationType.UNKNOWN])
                
                # Draw bounding box
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, line_thickness)
                
                # Prepare label
                label = f"{violation.violation_type.value}: {violation.confidence:.2f}"
                
                # Draw label background
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(img_rgb, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale, (255, 255, 255), font_thickness)
        
        # Draw all detections if requested and provided
        if show_all_detections and detections:
            # Skip detections that are already drawn as violations
            violation_boxes = {v.bbox for v in violations} if violations else set()
            
            for detection in detections:
                # Skip if this box is already drawn as a violation
                if detection.bbox in violation_boxes:
                    continue
                
                # Get bounding box in pixel coordinates
                x1, y1, x2, y2 = detection.bbox
                x1, x2 = int(x1 * width), int(x2 * width)
                y1, y2 = int(y1 * height), int(y2 * height)
                
                # Use gray color for regular detections
                color = (128, 128, 128)  # Gray
                
                # Draw bounding box
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, line_thickness)
                
                # Prepare label
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                
                # Draw label background
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(img_rgb, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale, (255, 255, 255), font_thickness)
        
        # Convert back to BGR for OpenCV compatibility
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        return img_bgr

    def save_annotated_image(
        self,
        image: ImageType,
        output_path: str,
        detections: Optional[List[DetectionResult]] = None,
        violations: Optional[List[PPEViolation]] = None,
        show_all_detections: bool = False
    ) -> str:
        """
        Annotate an image and save it to disk.
        
        Args:
            image: Image to annotate
            output_path: Path to save the annotated image
            detections: List of all detections (optional if violations provided)
            violations: List of PPE violations (optional)
            show_all_detections: Whether to show all detections or just violations
            
        Returns:
            str: Path to the saved annotated image
            
        Raises:
            ValueError: If the image cannot be loaded or saved
        """
        # Get annotated image
        annotated_img = self.annotate_image(
            image=image,
            detections=detections,
            violations=violations,
            show_all_detections=show_all_detections
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save image
        try:
            cv2.imwrite(output_path, annotated_img)
            logger.info(f"Saved annotated image to: {output_path}")
            return output_path
        except Exception as e:
            error_msg = f"Failed to save annotated image to {output_path}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def get_summary(self, violations: List[PPEViolation]) -> Dict[str, Any]:
        """
        Generate a summary of detected violations.
        
        Args:
            violations: List of PPE violations
            
        Returns:
            Dict[str, Any]: Summary of violations
        """
        violation_counts = {vtype.value: 0 for vtype in ViolationType}
        violation_confidences = {vtype.value: [] for vtype in ViolationType}
        
        # Count violations by type
        for violation in violations:
            vtype = violation.violation_type.value
            violation_counts[vtype] += 1
            violation_confidences[vtype].append(violation.confidence)
        
        # Calculate average confidence for each violation type
        avg_confidences = {}
        for vtype, confidences in violation_confidences.items():
            if confidences:
                avg_confidences[vtype] = sum(confidences) / len(confidences)
            else:
                avg_confidences[vtype] = 0
        
        # Remove violation types with zero counts
        for vtype in list(violation_counts.keys()):
            if violation_counts[vtype] == 0:
                del violation_counts[vtype]
                del avg_confidences[vtype]
        
        summary = {
            "total_violations": len(violations),
            "violation_counts": violation_counts,
            "avg_confidences": avg_confidences,
            "has_violations": len(violations) > 0,
        }
        
        return summary

    def update_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold for detections.
        
        Args:
            threshold: New confidence threshold (0.0-1.0)
            
        Raises:
            ValueError: If threshold is outside valid range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {threshold}")
        
        self.confidence_threshold = threshold
        self.model.conf = threshold
        logger.info(f"Updated confidence threshold to {threshold}")

    def update_iou_threshold(self, threshold: float) -> None:
        """
        Update the IoU threshold for non-maximum suppression.
        
        Args:
            threshold: New IoU threshold (0.0-1.0)
            
        Raises:
            ValueError: If threshold is outside valid range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"IoU threshold must be between 0.0 and 1.0, got {threshold}")
        
        self.iou_threshold = threshold
        self.model.iou = threshold
        logger.info(f"Updated IoU threshold to {threshold}")


# Example usage
if __name__ == "__main__":
    # Initialize detector with default settings
    detector = PPEDetector(
        model_path="./models/ppe_model.pt",
        confidence_threshold=0.5,
        is_custom_model=True
    )
    
    # Process an image
    image_path = "./test_images/construction_site.jpg"
    
    # Detect violations
    violations, all_detections = detector.detect_violations(image_path)
    
    # Annotate and save image
    detector.save_annotated_image(
        image=image_path,
        output_path="./output/annotated_image.jpg",
        detections=all_detections,
        violations=violations,
        show_all_detections=True
    )
    
    # Get summary
    summary = detector.get_summary(violations)
    print(f"Violation summary: {summary}")
