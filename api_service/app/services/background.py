"""
Background task service for executing long-running operations.
"""
import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast
from uuid import uuid4
import traceback
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.core.logging import logger


T = TypeVar('T')
TaskFunc = Callable[..., T]


class TaskStatus:
    """Status of a background task."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResult:
    """Result of a background task."""
    
    def __init__(
        self,
        task_id: str,
        status: str = TaskStatus.QUEUED,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        progress: float = 0.0,
        created_at: float = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None
    ):
        self.task_id = task_id
        self.status = status
        self.result = result
        self.error = error
        self.progress = progress
        self.created_at = created_at or time.time()
        self.started_at = started_at
        self.completed_at = completed_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class BackgroundTaskManager:
    """Manager for background tasks."""
    
    def __init__(self, max_workers: int = None):
        """
        Initialize the background task manager.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self._max_workers = max_workers or settings.MAX_WORKERS
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._tasks: Dict[str, TaskResult] = {}
        self._running_tasks = 0
        self._lock = asyncio.Lock()
        
        logger.info(f"Background task manager initialized with {self._max_workers} workers")
    
    async def enqueue_task(
        self,
        func: TaskFunc,
        *args: Any,
        **kwargs: Any
    ) -> str:
        """
        Enqueue a task to be executed in the background.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Task ID
        """
        task_id = str(uuid4())
        
        # Create initial task result
        task_result = TaskResult(task_id)
        
        async with self._lock:
            self._tasks[task_id] = task_result
        
        # Create a callback to update the task result
        def done_callback(future):
            loop = asyncio.get_event_loop()
            
            try:
                result = future.result()
                
                async def update_result():
                    async with self._lock:
                        task_result = self._tasks.get(task_id)
                        if task_result:
                            task_result.status = TaskStatus.COMPLETED
                            task_result.result = result
                            task_result.completed_at = time.time()
                            task_result.progress = 1.0
                            self._running_tasks -= 1
                
                loop.create_task(update_result())
            except Exception as e:
                error = str(e)
                tb = traceback.format_exc()
                logger.error(f"Error in background task {task_id}: {error}\n{tb}")
                
                async def update_error():
                    async with self._lock:
                        task_result = self._tasks.get(task_id)
                        if task_result:
                            task_result.status = TaskStatus.FAILED
                            task_result.error = f"{error}\n{tb}"
                            task_result.completed_at = time.time()
                            self._running_tasks -= 1
                
                loop.create_task(update_error())
        
        # Wrap the function to update the task status when it starts
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                async def update_started():
                    async with self._lock:
                        task_result = self._tasks.get(task_id)
                        if task_result:
                            task_result.status = TaskStatus.RUNNING
                            task_result.started_at = time.time()
                
                loop.run_until_complete(update_started())
                
                return func(*args, **kwargs)
            finally:
                loop.close()
        
        # Check if we can start the task or if we need to wait
        can_start = True
        async with self._lock:
            if self._running_tasks >= self._max_workers:
                can_start = False
        
        if can_start:
            # Submit task for execution and increment running tasks
            async with self._lock:
                self._running_tasks += 1
            
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(self._executor, wrapper, *args, **kwargs)
            future.add_done_callback(done_callback)
        else:
            # Log that the task is queued but not executing yet
            logger.info(f"Task {task_id} queued but not executing yet (max workers reached)")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status or None if the task doesn't exist
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict() if task else None
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if it's still queued.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if the task was canceled, False otherwise
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.QUEUED:
                task.status = TaskStatus.FAILED
                task.error = "Task canceled"
                task.completed_at = time.time()
                return True
            return False
    
    async def clean_old_tasks(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old completed or failed tasks.
        
        Args:
            max_age_seconds: Maximum age of tasks to keep (default: 1 hour)
            
        Returns:
            Number of tasks cleaned up
        """
        now = time.time()
        tasks_to_remove = []
        
        async with self._lock:
            for task_id, task in self._tasks.items():
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    if task.completed_at and now - task.completed_at > max_age_seconds:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self._tasks[task_id]
        
        return len(tasks_to_remove)
    
    async def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all tasks.
        
        Returns:
            List of all tasks
        """
        async with self._lock:
            return [task.to_dict() for task in self._tasks.values()]


# Singleton instance
task_manager = BackgroundTaskManager()
