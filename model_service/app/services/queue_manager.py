"""
Queue Manager for SIFT Model Service

This module provides functions to manage the SQS processing background tasks,
monitor queue depth and processor status, and perform health checks
on the processing pipeline.

Author: SIFT Development Team
Date: May 5, 2025
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import boto3
from botocore.exceptions import ClientError
from fastapi import BackgroundTasks
from loguru import logger

from app.core.config import settings
from app.services.sqs_processor import SQSProcessor


class QueueManager:
    """
    Manager for SQS processing tasks.
    
    This class provides methods to start and stop SQS processing tasks,
    monitor queue depth and processor status, and perform health checks
    on the processing pipeline.
    
    Attributes:
        processor: The SQS processor instance
        is_initialized: Flag indicating if the manager is initialized
        background_tasks: List of background tasks
        monitoring_task: Asyncio task for background monitoring
        last_queue_check: Timestamp of last queue depth check
        queue_attributes: Latest queue attributes
    """
    
    def __init__(self):
        """Initialize the queue manager."""
        self.processor: Optional[SQSProcessor] = None
        self.is_initialized = False
        self.background_tasks: List[asyncio.Task] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_queue_check = datetime.min
        self.queue_attributes: Dict[str, Any] = {}
        self.shutdown_event = asyncio.Event()
        
        logger.info("Initialized queue manager")
    
    async def initialize(
        self,
        model_path: str = None,
        confidence_threshold: float = None,
        batch_size: int = None,
        wait_time: int = None,
        visibility_timeout: int = None,
        max_retries: int = None,
        retry_delay: int = None,
        dead_letter_queue_url: str = None,
        is_custom_model: bool = True
    ) -> None:
        """
        Initialize the queue manager and SQS processor.
        
        Args:
            model_path: Path to the YOLO model file
            confidence_threshold: Confidence threshold for detections
            batch_size: Number of messages to process in each batch
            wait_time: SQS long-polling wait time in seconds
            visibility_timeout: SQS message visibility timeout in seconds
            max_retries: Maximum number of times to retry processing a message
            retry_delay: Delay between retries in seconds
            dead_letter_queue_url: URL of the dead-letter queue for failed messages
            is_custom_model: Whether using a custom PPE model or standard YOLO
        """
        if self.is_initialized:
            logger.warning("Queue manager is already initialized")
            return
        
        # Initialize SQS processor
        self.processor = SQSProcessor(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
            wait_time=wait_time,
            visibility_timeout=visibility_timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            dead_letter_queue_url=dead_letter_queue_url,
            is_custom_model=is_custom_model
        )
        
        self.is_initialized = True
        self.shutdown_event.clear()
        
        logger.info("Queue manager initialized")
    
    async def start_processing(self, start_monitoring: bool = True) -> bool:
        """
        Start the SQS processor and optionally start monitoring.
        
        Args:
            start_monitoring: Whether to start the monitoring task
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.is_initialized or not self.processor:
            logger.error("Queue manager not initialized")
            return False
        
        if self.processor.running:
            logger.warning("Processor is already running")
            return True
        
        try:
            # Start the processor
            await self.processor.start()
            
            # Start monitoring if requested
            if start_monitoring:
                await self.start_monitoring()
            
            logger.info("Started SQS processing")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start SQS processing: {str(e)}")
            return False
    
    async def stop_processing(self, timeout: int = 60) -> bool:
        """
        Stop the SQS processor and monitoring tasks.
        
        Args:
            timeout: Maximum time to wait for shutdown in seconds
            
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_initialized or not self.processor:
            logger.error("Queue manager not initialized")
            return False
        
        if not self.processor.running:
            logger.warning("Processor is not running")
            return True
        
        try:
            # Signal shutdown for monitoring
            self.shutdown_event.set()
            
            # Stop monitoring task if running
            if self.monitoring_task and not self.monitoring_task.done():
                try:
                    logger.info("Stopping monitoring task")
                    self.monitoring_task.cancel()
                    await asyncio.gather(self.monitoring_task, return_exceptions=True)
                except Exception as e:
                    logger.warning(f"Error stopping monitoring task: {str(e)}")
            
            # Stop the processor
            await self.processor.stop(timeout=timeout)
            
            # Cancel all background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks = []
            self.monitoring_task = None
            
            logger.info("Stopped SQS processing")
            return True
        
        except Exception as e:
            logger.error(f"Failed to stop SQS processing: {str(e)}")
            return False
    
    async def start_monitoring(self, interval: int = 60) -> None:
        """
        Start monitoring queue depth and processor status.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Monitoring task is already running")
            return
        
        self.monitoring_task = asyncio.create_task(
            self._monitor_continuously(interval=interval)
        )
        
        logger.info(f"Started queue monitoring (interval: {interval}s)")
    
    async def _monitor_continuously(self, interval: int = 60) -> None:
        """
        Continuously monitor queue depth and processor status.
        
        Args:
            interval: Monitoring interval in seconds
        """
        logger.info("Starting continuous queue monitoring")
        
        while not self.shutdown_event.is_set():
            try:
                # Check queue depth
                attributes = await self.get_queue_attributes()
                
                # Log queue status
                available = int(attributes.get('ApproximateNumberOfMessages', '0'))
                in_flight = int(attributes.get('ApproximateNumberOfMessagesNotVisible', '0'))
                
                logger.info(
                    f"Queue status: {available} available, {in_flight} in flight"
                )
                
                # Check processor status if running
                if self.processor and self.processor.running:
                    status = self.processor.get_status()
                    metrics = status['metrics']
                    
                    logger.info(
                        f"Processor metrics: {metrics['frames_processed']} processed, "
                        f"{metrics['frames_failed']} failed, "
                        f"{metrics['total_violations']} violations"
                    )
            
            except Exception as e:
                logger.error(f"Error in monitoring task: {str(e)}")
            
            # Wait for next interval or shutdown
            try:
                await asyncio.wait_for(
                    self.shutdown_event.wait(),
                    timeout=interval
                )
            except asyncio.TimeoutError:
                # Timeout is expected, continue monitoring
                pass
        
        logger.info("Stopped continuous queue monitoring")
    
    async def get_queue_attributes(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Get attributes of the SQS queue, including queue depth.
        
        Args:
            force_refresh: Whether to force a refresh of queue attributes
            
        Returns:
            Dict containing queue attributes
        """
        # Use cached attributes if available and not forcing refresh
        if (
            not force_refresh and
            self.last_queue_check > datetime.now() - timedelta(minutes=1) and
            self.queue_attributes
        ):
            return self.queue_attributes
        
        try:
            # Initialize SQS client
            sqs_client = boto3.client(
                'sqs',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            
            # Get queue attributes
            response = sqs_client.get_queue_attributes(
                QueueUrl=settings.SQS_QUEUE_URL,
                AttributeNames=[
                    'ApproximateNumberOfMessages',
                    'ApproximateNumberOfMessagesNotVisible',
                    'ApproximateNumberOfMessagesDelayed',
                    'CreatedTimestamp',
                    'LastModifiedTimestamp'
                ]
            )
            
            self.queue_attributes = response.get('Attributes', {})
            self.last_queue_check = datetime.now()
            
            return self.queue_attributes
        
        except ClientError as e:
            logger.error(f"Error getting queue attributes: {str(e)}")
            return {}
    
    async def run_processor_once(
        self, 
        max_messages: int = 10,
        wait_time: int = 5
    ) -> Dict[str, Any]:
        """
        Run the processor once to process a batch of messages.
        
        This method is useful for processing a single batch of messages
        on demand, without starting the continuous processing loop.
        
        Args:
            max_messages: Maximum number of messages to process
            wait_time: SQS long-polling wait time in seconds
            
        Returns:
            Dict containing processing results
        """
        if not self.is_initialized or not self.processor:
            logger.error("Queue manager not initialized")
            return {"error": "Queue manager not initialized"}
        
        try:
            # Save current batch size and wait time
            original_batch_size = self.processor.batch_size
            original_wait_time = self.processor.wait_time
            
            # Set temporary batch size and wait time
            self.processor.batch_size = max_messages
            self.processor.wait_time = wait_time
            
            # Process a batch of messages
            batch_start_time = time.time()
            batch_result = await self.processor._process_batch()
            processing_time = time.time() - batch_start_time
            
            # Restore original batch size and wait time
            self.processor.batch_size = original_batch_size
            self.processor.wait_time = original_wait_time
            
            if not batch_result:
                return {
                    "success": True,
                    "message": "No messages available for processing",
                    "processed": 0,
                    "processing_time": processing_time
                }
            
            return {
                "success": True,
                "processed": len(batch_result.successful),
                "failed": len(batch_result.failed),
                "retried": len(batch_result.retried),
                "invalid": len(batch_result.invalid),
                "processing_time": processing_time
            }
        
        except Exception as e:
            logger.error(f"Error running processor once: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def purge_queue(self) -> bool:
        """
        Purge all messages from the SQS queue.
        
        WARNING: This will delete all messages from the queue.
        
        Returns:
            bool: True if purged successfully, False otherwise
        """
        try:
            logger.warning("Purging SQS queue")
            
            # Initialize SQS client
            sqs_client = boto3.client(
                'sqs',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            
            # Purge the queue
            sqs_client.purge_queue(QueueUrl=settings.SQS_QUEUE_URL)
            
            logger.info("SQS queue purged successfully")
            return True
        
        except ClientError as e:
            logger.error(f"Error purging SQS queue: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the processing pipeline.
        
        Returns:
            Dict containing health check results
        """
        health_result = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check manager initialization
        health_result["checks"]["manager_initialized"] = {
            "status": "ok" if self.is_initialized else "error",
            "message": "Manager is initialized" if self.is_initialized 
                     else "Manager is not initialized"
        }
        
        # Check processor status
        processor_status = "not_initialized"
        processor_message = "Processor is not initialized"
        
        if self.processor:
            if self.processor.running:
                processor_status = "ok"
                processor_message = "Processor is running"
            else:
                processor_status = "warning"
                processor_message = "Processor is initialized but not running"
        
        health_result["checks"]["processor"] = {
            "status": processor_status,
            "message": processor_message
        }
        
        # Check SQS queue
        try:
            queue_attributes = await self.get_queue_attributes(force_refresh=True)
            
            if queue_attributes:
                available = int(queue_attributes.get('ApproximateNumberOfMessages', '0'))
                in_flight = int(queue_attributes.get('ApproximateNumberOfMessagesNotVisible', '0'))
                
                health_result["checks"]["sqs_queue"] = {
                    "status": "ok",
                    "message": f"Queue is accessible: {available} available, {in_flight} in flight",
                    "details": {
                        "available_messages": available,
                        "in_flight_messages": in_flight
                    }
                }
            else:
                health_result["checks"]["sqs_queue"] = {
                    "status": "error",
                    "message": "Failed to get queue attributes"
                }
                health_result["status"] = "unhealthy"
        except Exception as e:
            health_result["checks"]["sqs_queue"] = {
                "status": "error",
                "message": f"Error checking SQS queue: {str(e)}"
            }
            health_result["status"] = "unhealthy"
        
        # Check model detector
        if self.processor and hasattr(self.processor, 'detector'):
            try:
                model_version = self.processor.detector.get_model_version()
                health_result["checks"]["model"] = {
                    "status": "ok",
                    "message": f"Model is loaded: {model_version}",
                    "details": {
                        "model_path": self.processor.detector.model_path,
                        "confidence_threshold": self.processor.detector.confidence_threshold
                    }
                }
            except Exception as e:
                health_result["checks"]["model"] = {
                    "status": "error",
                    "message": f"Error checking model: {str(e)}"
                }
                health_result["status"] = "unhealthy"
        else:
            health_result["checks"]["model"] = {
                "status": "warning",
                "message": "Model detector not initialized"
            }
            if health_result["status"] == "healthy":
                health_result["status"] = "degraded"
        
        # Check processor metrics if running
        if self.processor and self.processor.running:
            status = self.processor.get_status()
            metrics = status['metrics']
            
            # Check if processor has processed any frames recently
            last_processed = metrics.get('last_processed')
            if last_processed:
                last_processed_dt = datetime.fromisoformat(last_processed) if isinstance(last_processed, str) else last_processed
                if last_processed_dt < datetime.now() - timedelta(minutes=30):
                    health_result["checks"]["processor_activity"] = {
                        "status": "warning",
                        "message": f"No frames processed in the last 30 minutes (last: {last_processed})"
                    }
                    if health_result["status"] == "healthy":
                        health_result["status"] = "degraded"
                else:
                    health_result["checks"]["processor_activity"] = {
                        "status": "ok",
                        "message": f"Processor active (last processed: {last_processed})"
                    }
            
            # Add metrics to health check
            health_result["metrics"] = {
                "frames_processed": metrics['frames_processed'],
                "frames_failed": metrics['frames_failed'],
                "frames_retried": metrics['frames_retried'],
                "total_detections": metrics['total_detections'],
                "total_violations": metrics['total_violations'],
                "poison_messages": metrics['poison_messages'],
                "avg_processing_time": metrics['avg_processing_time']
            }
        
        return health_result


# Global queue manager instance
queue_manager = QueueManager()


async def initialize_queue_manager() -> QueueManager:
    """
    Initialize the global queue manager.
    
    Returns:
        The initialized queue manager instance
    """
    if not queue_manager.is_initialized:
        await queue_manager.initialize()
    
    return queue_manager


async def start_processing(background_tasks: BackgroundTasks = None) -> Dict[str, Any]:
    """
    Start processing messages from the SQS queue.
    
    This function can be called from FastAPI endpoints to start
    the SQS processor in the background.
    
    Args:
        background_tasks: FastAPI BackgroundTasks instance
        
    Returns:
        Dict containing start result
    """
    if not queue_manager.is_initialized:
        await queue_manager.initialize()
    
    if background_tasks:
        # Start in the background using FastAPI's BackgroundTasks
        async def _start():
            await queue_manager.start_processing()
        
        background_tasks.add_task(_start)
        return {
            "status": "starting",
            "message": "Processing will start in the background"
        }
    else:
        # Start directly
        success = await queue_manager.start_processing()
        return {
            "status": "started" if success else "error",
            "message": "Processing started" if success else "Failed to start processing"
        }


async def stop_processing() -> Dict[str, Any]:
    """
    Stop processing messages from the SQS queue.
    
    Returns:
        Dict containing stop result
    """
    if not queue_manager.is_initialized:
        return {
            "status": "error",
            "message": "Queue manager not initialized"
        }
    
    success = await queue_manager.stop_processing()
    return {
        "status": "stopped" if success else "error",
        "message": "Processing stopped" if success else "Failed to stop processing"
    }


async def get_processor_status() -> Dict[str, Any]:
    """
    Get the status of the SQS processor.
    
    Returns:
        Dict containing processor status
    """
    if not queue_manager.is_initialized or not queue_manager.processor:
        return {
            "status": "not_initialized",
            "message": "Queue manager or processor not initialized"
        }
    
    # Get processor status
    processor_status = queue_manager.processor.get_status()
    
    # Get queue attributes
    try:
        queue_attributes = await queue_manager.get_queue_attributes()
        processor_status["queue"] = {
            "available_messages": int(queue_attributes.get('ApproximateNumberOfMessages', '0')),
            "in_flight_messages": int(queue_attributes.get('ApproximateNumberOfMessagesNotVisible', '0')),
            "delayed_messages": int(queue_attributes.get('ApproximateNumberOfMessagesDelayed', '0'))
        }
    except Exception as e:
        processor_status["queue"] = {
            "error": f"Failed to get queue attributes: {str(e)}"
        }
    
    return processor_status


async def run_processor_once(
    max_messages: int = 10,
    wait_time: int = 5
) -> Dict[str, Any]:
    """
    Run the processor once to process a batch of messages.
    
    Args:
        max_messages: Maximum number of messages to process
        wait_time: SQS long-polling wait time in seconds
        
    Returns:
        Dict containing processing results
    """
    if not queue_manager.is_initialized:
        await queue_manager.initialize()
    
    return await queue_manager.run_processor_once(
        max_messages=max_messages,
        wait_time=wait_time
    )


async def health_check() -> Dict[str, Any]:
    """
    Perform a health check on the processing pipeline.
    
    Returns:
        Dict containing health check results
    """
    if not queue_manager.is_initialized:
        await queue_manager.initialize()
    
    return await queue_manager.health_check()
