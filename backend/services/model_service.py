import os
import json
import logging
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import threading
import time
from urllib.request import urlretrieve
from urllib.error import URLError

# Import ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from backend.config import MODELS_DIR

logger = logging.getLogger(__name__)

class ModelMetadata:
    """Metadata for a YOLO model"""
    def __init__(
        self,
        id: str,
        name: str,
        file_path: str,
        task_type: str = "object_detection",
        description: str = "",
        version: str = "1.0",
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        classes: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.name = name
        self.file_path = file_path
        self.task_type = task_type
        self.description = description
        self.version = version
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.classes = classes or []
        self.metrics = metrics or {}
        self.custom_metadata = custom_metadata or {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        # Parse dates from strings if needed
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            file_path=data.get('file_path', ''),
            task_type=data.get('task_type', 'object_detection'),
            description=data.get('description', ''),
            version=data.get('version', '1.0'),
            created_at=created_at,
            updated_at=updated_at,
            classes=data.get('classes', []),
            metrics=data.get('metrics', {}),
            custom_metadata=data.get('custom_metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'file_path': self.file_path,
            'task_type': self.task_type,
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'classes': self.classes,
            'metrics': self.metrics,
            'custom_metadata': self.custom_metadata
        }


class ModelService:
    """
    Service for managing YOLO models, including:
    - Model discovery and metadata tracking
    - Loading and validating models
    - Model selection based on task requirements
    - Model performance metrics
    """
    
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file path
        self.metadata_file = self.models_dir / "models_metadata.json"
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Model metadata cache
        self.models_metadata: Dict[str, ModelMetadata] = {}
        
        # Default model IDs by task
        self.default_models: Dict[str, str] = {}
        
        # Cache of loaded models
        self.loaded_models: Dict[str, Any] = {}
        self.model_last_used: Dict[str, float] = {}
        
        # Maximum number of models to keep in memory
        self.max_loaded_models = 3
        
        # Initialize
        self._load_metadata()
        self._scan_models_directory()
        
        logger.info(f"Model service initialized with {len(self.models_metadata)} models")
    
    def _load_metadata(self) -> None:
        """Load model metadata from file"""
        with self._lock:
            if not self.metadata_file.exists():
                self.models_metadata = {}
                self.default_models = {}
                return
            
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Load model metadata
                self.models_metadata = {
                    model_id: ModelMetadata.from_dict(model_data)
                    for model_id, model_data in data.get('models', {}).items()
                }
                
                # Load default model selections
                self.default_models = data.get('default_models', {})
                
                logger.info(f"Loaded metadata for {len(self.models_metadata)} models")
            except Exception as e:
                logger.error(f"Error loading model metadata: {str(e)}")
                self.models_metadata = {}
                self.default_models = {}
    
    def _save_metadata(self) -> bool:
        """Save model metadata to file"""
        with self._lock:
            try:
                data = {
                    'models': {
                        model_id: metadata.to_dict()
                        for model_id, metadata in self.models_metadata.items()
                    },
                    'default_models': self.default_models,
                    'last_updated': datetime.now().isoformat()
                }
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                return True
            except Exception as e:
                logger.error(f"Error saving model metadata: {str(e)}")
                return False
    
    def _scan_models_directory(self) -> None:
        """Scan models directory for new models"""
        with self._lock:
            # Track existing files
            existing_files = set()
            
            # Scan for model files
            for file_path in self.models_dir.glob("*.pt"):
                if file_path.name == "models_metadata.json":
                    continue
                
                file_str = str(file_path)
                existing_files.add(file_str)
                
                # Check if we already have metadata for this file
                found = False
                for metadata in self.models_metadata.values():
                    if metadata.file_path == file_str:
                        found = True
                        break
                
                if not found:
                    # New model file, create metadata
                    model_id = self._generate_model_id(file_path)
                    model_name = file_path.stem  # Use filename without extension
                    
                    metadata = ModelMetadata(
                        id=model_id,
                        name=model_name,
                        file_path=file_str,
                        created_at=datetime.fromtimestamp(file_path.stat().st_mtime),
                        updated_at=datetime.now()
                    )
                    
                    # Extract model info if possible
                    if ULTRALYTICS_AVAILABLE:
                        try:
                            model_info = self._extract_model_info(file_path)
                            if model_info:
                                metadata.classes = model_info.get('classes', [])
                                metadata.metrics = model_info.get('metrics', {})
                                metadata.custom_metadata = model_info.get('extra', {})
                        except Exception as e:
                            logger.warning(f"Error extracting model info for {file_path.name}: {str(e)}")
                    
                    # Add to metadata
                    self.models_metadata[model_id] = metadata
                    logger.info(f"Added new model: {model_name} (ID: {model_id})")
            
            # Check specifically for yolo11m.pt
            yolo_path = self.models_dir / "yolo11m.pt"
            if yolo_path.exists() and str(yolo_path) not in existing_files:
                model_id = self._generate_model_id(yolo_path)
                model_name = "YOLO11-M"
                
                metadata = ModelMetadata(
                    id=model_id,
                    name=model_name,
                    file_path=str(yolo_path),
                    description="YOLO model for object detection",
                    created_at=datetime.fromtimestamp(yolo_path.stat().st_mtime),
                    updated_at=datetime.now()
                )
                
                # Try to extract model info
                if ULTRALYTICS_AVAILABLE:
                    try:
                        model_info = self._extract_model_info(yolo_path)
                        if model_info:
                            metadata.classes = model_info.get('classes', [])
                            metadata.metrics = model_info.get('metrics', {})
                            metadata.custom_metadata = model_info.get('extra', {})
                    except Exception as e:
                        logger.warning(f"Error extracting info for YOLO model: {str(e)}")
                
                # Add to metadata
                self.models_metadata[model_id] = metadata
                
                # Set as default model for object detection
                self.default_models['object_detection'] = model_id
                
                logger.info(f"Added YOLO model (ID: {model_id}) and set as default")
                existing_files.add(str(yolo_path))
            
            # Remove metadata for files that no longer exist
            removed_ids = []
            for model_id, metadata in self.models_metadata.items():
                if metadata.file_path not in existing_files:
                    removed_ids.append(model_id)
            
            for model_id in removed_ids:
                # Remove from metadata
                del self.models_metadata[model_id]
                
                # Remove from default models if it was selected
                for task, default_id in list(self.default_models.items()):
                    if default_id == model_id:
                        del self.default_models[task]
                
                # Remove from loaded models
                if model_id in self.loaded_models:
                    del self.loaded_models[model_id]
                    
                if model_id in self.model_last_used:
                    del self.model_last_used[model_id]
                
                logger.info(f"Removed model with ID: {model_id} (file no longer exists)")
            
            # If we have no default model for object detection, set one if available
            if 'object_detection' not in self.default_models or self.default_models['object_detection'] not in self.models_metadata:
                # First, check for yolo11m.pt
                for model_id, metadata in self.models_metadata.items():
                    if "yolo11m" in metadata.file_path.lower():
                        self.default_models['object_detection'] = model_id
                        logger.info(f"Set default object detection model to YOLO (ID: {model_id})")
                        break
                else:
                    # If no YOLO model found, use any available object detection model
                    for model_id, metadata in self.models_metadata.items():
                        if metadata.task_type == 'object_detection':
                            self.default_models['object_detection'] = model_id
                            logger.info(f"Set default object detection model to: {metadata.name} (ID: {model_id})")
                            break
            
            # Save metadata if changes were made
            if removed_ids or (len(existing_files) > 0 and len(self.models_metadata) > 0):
                self._save_metadata()
    
    def _generate_model_id(self, file_path: Path) -> str:
        """Generate a unique ID for a model file"""
        # Use first 8 chars of file hash as ID
        hash_obj = hashlib.sha256()
        hash_obj.update(file_path.name.encode())
        hash_obj.update(str(file_path.stat().st_size).encode())
        hash_obj.update(str(file_path.stat().st_mtime).encode())
        return hash_obj.hexdigest()[:8]
    
    def _extract_model_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract information from a YOLO model file"""
        if not ULTRALYTICS_AVAILABLE:
            return {}
        
        try:
            # Load model without initializing
            model = YOLO(str(file_path))
            
            # Extract class names
            class_names = []
            if hasattr(model, 'names'):
                class_names = list(model.names.values())
            
            # Extract metrics if available
            metrics = {}
            if hasattr(model, 'metrics') and model.metrics is not None:
                metrics_dict = model.metrics.results_dict if hasattr(model.metrics, 'results_dict') else {}
                # Convert tensors to Python types
                metrics = {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in metrics_dict.items()
                }
            
            # Extract additional information
            extra = {}
            for attr in ['task', 'stride', 'pt_path']:
                if hasattr(model, attr):
                    extra[attr] = str(getattr(model, attr))
            
            return {
                'classes': class_names,
                'metrics': metrics,
                'extra': extra
            }
        except Exception as e:
            logger.warning(f"Error analyzing model {file_path.name}: {str(e)}")
            return {}
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of all available models with metadata.
        
        Returns:
            List of model metadata dictionaries
        """
        with self._lock:
            # Rescan to make sure we have the latest
            self._scan_models_directory()
            
            return [
                {
                    'id': metadata.id,
                    'name': metadata.name,
                    'task_type': metadata.task_type,
                    'description': metadata.description,
                    'file_path': metadata.file_path,
                    'version': metadata.version,
                    'created_at': metadata.created_at.isoformat() if metadata.created_at else None,
                    'updated_at': metadata.updated_at.isoformat() if metadata.updated_at else None,
                    'classes': metadata.classes,
                    'is_default': self.default_models.get(metadata.task_type) == metadata.id,
                    'is_loaded': metadata.id in self.loaded_models
                }
                for metadata in self.models_metadata.values()
            ]
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model metadata by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model metadata dictionary or None if not found
        """
        with self._lock:
            metadata = self.models_metadata.get(model_id)
            if not metadata:
                return None
            
            return {
                'id': metadata.id,
                'name': metadata.name,
                'task_type': metadata.task_type,
                'description': metadata.description,
                'file_path': metadata.file_path,
                'version': metadata.version,
                'created_at': metadata.created_at.isoformat() if metadata.created_at else None,
                'updated_at': metadata.updated_at.isoformat() if metadata.updated_at else None,
                'classes': metadata.classes,
                'metrics': metadata.metrics,
                'custom_metadata': metadata.custom_metadata,
                'is_default': self.default_models.get(metadata.task_type) == metadata.id,
                'is_loaded': metadata.id in self.loaded_models
            }
    
    def get_default_model(self, task_type: str = "object_detection") -> Optional[Dict[str, Any]]:
        """
        Get default model for a specific task.
        
        Args:
            task_type: Task type (e.g., "object_detection")
            
        Returns:
            Model metadata dictionary or None if no default set
        """
        with self._lock:
            model_id = self.default_models.get(task_type)
            if not model_id or model_id not in self.models_metadata:
                return None
            
            return self.get_model_by_id(model_id)
    
    def set_default_model(self, model_id: str, task_type: Optional[str] = None) -> bool:
        """
        Set default model for a task.
        
        Args:
            model_id: Model ID
            task_type: Task type (uses model's task_type if None)
            
        Returns:
            True if successful
        """
        with self._lock:
            if model_id not in self.models_metadata:
                logger.warning(f"Cannot set default: Model ID {model_id} not found")
                return False
            
            metadata = self.models_metadata[model_id]
            task = task_type or metadata.task_type
            
            self.default_models[task] = model_id
            logger.info(f"Set default model for {task} to: {metadata.name} (ID: {model_id})")
            
            return self._save_metadata()
    
    def load_model(self, model_id: Optional[str] = None, task_type: str = "object_detection") -> Tuple[bool, Optional[Any]]:
        """
        Load a model by ID or the default model for a task.
        
        Args:
            model_id: Model ID (uses default for task_type if None)
            task_type: Task type (e.g., "object_detection")
            
        Returns:
            Tuple of (success, model_object)
        """
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Cannot load model: Ultralytics not available")
            return False, None
        
        with self._lock:
            # Use default model if ID not provided
            if model_id is None:
                if task_type in self.default_models:
                    model_id = self.default_models[task_type]
                    if model_id not in self.models_metadata:
                        logger.warning(f"Default model for {task_type} (ID: {model_id}) not found")
                        model_id = None
            
            # If still no model ID, try to find any model for this task
            if model_id is None:
                # First try to find YOLO11m model
                for m_id, metadata in self.models_metadata.items():
                    if "yolo11m" in metadata.file_path.lower():
                        model_id = m_id
                        logger.info(f"Selected YOLO11m model for {task_type}")
                        break
                # If not found, use any model for the task
                if model_id is None:
                    for m_id, metadata in self.models_metadata.items():
                        if metadata.task_type == task_type:
                            model_id = m_id
                            break
            
            # If we still don't have a model, give up
            if model_id is None or model_id not in self.models_metadata:
                logger.error(f"No model available for {task_type}")
                return False, None
            
            # Check if already loaded
            if model_id in self.loaded_models:
                # Update last used time
                self.model_last_used[model_id] = time.time()
                return True, self.loaded_models[model_id]
            
            # Get model file path
            metadata = self.models_metadata[model_id]
            
            try:
                # Load model using ultralytics YOLO
                logger.info(f"Loading model from {metadata.file_path}")
                model = YOLO(metadata.file_path)
                
                # Update model classes if not already set
                if not metadata.classes and hasattr(model, 'names'):
                    metadata.classes = list(model.names.values())
                    self._save_metadata()
                
                # Add to loaded models
                self.loaded_models[model_id] = model
                self.model_last_used[model_id] = time.time()
                
                # Clean up old models if needed
                self._clean_loaded_models()
                
                logger.info(f"Loaded model: {metadata.name} (ID: {model_id})")
                return True, model
                
            except Exception as e:
                logger.error(f"Error loading model {metadata.name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False, None
    
    def _clean_loaded_models(self) -> None:
        """Unload least recently used models if too many are loaded"""
        if len(self.loaded_models) <= self.max_loaded_models:
            return
        
        # Sort models by last used time
        sorted_models = sorted(
            self.model_last_used.items(),
            key=lambda x: x[1]
        )
        
        # Unload oldest models
        for model_id, _ in sorted_models[:len(sorted_models) - self.max_loaded_models]:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                del self.model_last_used[model_id]
                logger.info(f"Unloaded model {model_id} due to memory constraints")
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if successful
        """
        with self._lock:
            if model_id not in self.loaded_models:
                return False
            
            del self.loaded_models[model_id]
            
            if model_id in self.model_last_used:
                del self.model_last_used[model_id]
            
            logger.info(f"Unloaded model {model_id}")
            return True
    
    def unload_all_models(self) -> bool:
        """
        Unload all models from memory.
        
        Returns:
            True if successful
        """
        with self._lock:
            self.loaded_models.clear()
            self.model_last_used.clear()
            logger.info("Unloaded all models")
            return True
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model file and its metadata.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if successful
        """
        with self._lock:
            if model_id not in self.models_metadata:
                logger.warning(f"Cannot delete: Model ID {model_id} not found")
                return False
            
            metadata = self.models_metadata[model_id]
            file_path = Path(metadata.file_path)
            
            # Unload if loaded
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                
                if model_id in self.model_last_used:
                    del self.model_last_used[model_id]
            
            # Remove from default models if selected
            for task, default_id in list(self.default_models.items()):
                if default_id == model_id:
                    del self.default_models[task]
            
            # Delete file
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted model file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting model file {file_path}: {str(e)}")
                return False
            
            # Remove metadata
            del self.models_metadata[model_id]
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Deleted model: {metadata.name} (ID: {model_id})")
            return True
    
    def import_model(self, source_path: str, new_name: Optional[str] = None) -> Optional[str]:
        """
        Import a model file into the models directory.
        
        Args:
            source_path: Source file path
            new_name: New name for the model (uses original filename if None)
            
        Returns:
            Model ID if successful, None otherwise
        """
        source = Path(source_path)
        if not source.exists() or not source.is_file():
            logger.warning(f"Source file not found: {source_path}")
            return None
        
        # Determine new file path
        if new_name:
            new_filename = f"{new_name}.pt"
        else:
            new_filename = source.name
        
        # Ensure .pt extension
        if not new_filename.endswith('.pt'):
            new_filename += '.pt'
        
        # Copy file
        try:
            target = self.models_dir / new_filename
            
            # Check for duplicate
            if target.exists():
                logger.warning(f"Target file already exists: {target}")
                # Use a numbered version
                base_name = target.stem
                for i in range(1, 100):
                    target = self.models_dir / f"{base_name}_{i}.pt"
                    if not target.exists():
                        break
                
                if target.exists():
                    logger.error(f"Cannot find unique filename for {new_filename}")
                    return None
            
            # Copy file
            shutil.copy2(source, target)
            logger.info(f"Imported model from {source_path} to {target}")
            
            # Rescan to pick up the new model
            self._scan_models_directory()
            
            # Find the newly imported model
            for model_id, metadata in self.models_metadata.items():
                if str(target) == metadata.file_path:
                    return model_id
            
            logger.warning(f"Imported model file but couldn't find it in metadata")
            return None
            
        except Exception as e:
            logger.error(f"Error importing model from {source_path}: {str(e)}")
            return None
    
    def get_model_classes(self, model_id: Optional[str] = None) -> List[str]:
        """
        Get classes detected by a model.
        
        Args:
            model_id: Model ID (uses default object detection model if None)
            
        Returns:
            List of class names
        """
        with self._lock:
            # Use default model if ID not provided
            if model_id is None:
                if 'object_detection' in self.default_models:
                    model_id = self.default_models['object_detection']
            
            if model_id is None or model_id not in self.models_metadata:
                return []
            
            metadata = self.models_metadata[model_id]
            
            # If classes already cached, return them
            if metadata.classes:
                return metadata.classes
            
            # Otherwise try to load model to get classes
            success, model = self.load_model(model_id)
            if not success or not model:
                return []
            
            # Extract class names
            class_names = []
            if hasattr(model, 'names'):
                class_names = list(model.names.values())
            
            # Update metadata
            metadata.classes = class_names
            self._save_metadata()
            
            return class_names
    
    def update_model_metadata(self, model_id: str, data: Dict[str, Any]) -> bool:
        """
        Update model metadata.
        
        Args:
            model_id: Model ID
            data: Data to update
            
        Returns:
            True if successful
        """
        with self._lock:
            if model_id not in self.models_metadata:
                logger.warning(f"Cannot update: Model ID {model_id} not found")
                return False
            
            metadata = self.models_metadata[model_id]
            
            # Update fields
            if 'name' in data:
                metadata.name = data['name']
            
            if 'description' in data:
                metadata.description = data['description']
            
            if 'task_type' in data:
                metadata.task_type = data['task_type']
            
            if 'version' in data:
                metadata.version = data['version']
            
            if 'classes' in data:
                metadata.classes = data['classes']
            
            if 'metrics' in data:
                metadata.metrics = data['metrics']
            
            if 'custom_metadata' in data:
                metadata.custom_metadata = data['custom_metadata']
            
            # Update timestamp
            metadata.updated_at = datetime.now()
            
            # Save metadata
            return self._save_metadata()


# Singleton instance
_model_service = None

def get_model_service() -> ModelService:
    """Get or create model service singleton."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service