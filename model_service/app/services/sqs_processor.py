"""
SQS Message Processor for SIFT Model Service

This module provides a robust SQSProcessor class that continuously polls
an SQS queue for new frame messages, processes them with the PPE detection model,
stores results in PostgreSQL, and handles errors with comprehensive retry logic.

Author: SIFT Development Team
Date: May 5, 2025
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set, NamedTuple, Callable

import boto3
from botocore.exceptions import ClientError
from fastapi import Depends
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import SessionLocal
from app.models.detection import Detection
from app.models.model_handler import PPEDetector, PPEViolation, DetectionResult
from app.services.aws import S3Service, SQSService


class MessageStatus(str, Enum):
    """Enum representing possible message processing statuses."""
    SUCCESS = "success"
    RETRY = "retry"
    FAIL = "fail"
    INVALID = "invalid"


class ProcessingMetrics(NamedTuple):
    """Named tuple for tracking processing metrics."""
    frames_processed: int
    frames_failed: int
    frames_retried: int
    total_detections: int
    total_violations: int
    poison_messages: int
    avg_processing_time: float  # in seconds
    last_processed: Optional[datetime]


class ProcessedBatch(NamedTuple):
    """Results from processing a batch of messages."""
    successful: List[Dict[str, Any]]
    failed: List[Dict[str, Any]]
    retried: List[Dict[str, Any]]
    invalid: List[Dict[str, Any]]
    processing_time: float


class SQSProcessor:
    """
    Processor for SQS messages containing frame data for PPE detection.
    
    This class continuously polls an SQS queue for new messages, processes them
    with the PPE detection model, stores results in PostgreSQL, and handles
    errors with comprehensive retry logic.
    
    Attributes:
        sqs_service: Service for interacting with AWS SQS
        s3_service: Service for interacting with AWS S3
        detector: PPE detection model handler
        running: Flag indicating if the processor is running
        processing_task: Asyncio task for background processing
        metrics: Metrics for monitoring processing performance
        batch_size: Number of messages to process in each batch
        wait_time: SQS long-polling wait time in seconds
        visibility_timeout: SQS message visibility timeout in seconds
        max_retries: Maximum number of times to retry processing a message
        retry_delay: Delay between retries in seconds
        dead_letter_queue_url: URL of the dead-letter queue for failed messages
        temp_dir: Directory for temporary files
        messages_in_flight: Set of message IDs currently being processed
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = None,
        batch_size: int = None,
        wait_time: int = None,
        visibility_timeout: int = None,
        max_retries: int = None,
        retry_delay: int = None,
        dead_letter_queue_url: str = None,
        temp_dir: str = None,
        is_custom_model: bool = True,
    ):
        """
        Initialize the SQS processor with the specified parameters.
        
        Args:
            model_path: Path to the YOLO model file (default from settings)
            confidence_threshold: Confidence threshold for detections (default from settings)
            batch_size: Number of messages to process in each batch (default from settings)
            wait_time: SQS long-polling wait time in seconds (default from settings)
            visibility_timeout: SQS message visibility timeout in seconds (default from settings)
            max_retries: Maximum number of times to retry processing a message (default from settings)
            retry_delay: Delay between retries in seconds (default from settings)
            dead_letter_queue_url: URL of the dead-letter queue (default from settings)
            temp_dir: Directory for temporary files (default from settings)
            is_custom_model: Whether using a custom PPE model or standard YOLO
        """
        # Initialize services
        self.sqs_service = SQSService()
        self.s3_service = S3Service()
        
        # Initialize model detector
        self.detector = PPEDetector(
            model_path=model_path or settings.MODEL_PATH,
            confidence_threshold=confidence_threshold or settings.CONFIDENCE_THRESHOLD,
            is_custom_model=is_custom_model
        )
        
        # Processing state
        self.running = False
        self.processing_task = None
        self.shutdown_event = asyncio.Event()
        self.messages_in_flight: Set[str] = set()
        
        # Configuration
        self.batch_size = batch_size or settings.SQS_MAX_MESSAGES
        self.wait_time = wait_time or settings.SQS_WAIT_TIME
        self.visibility_timeout = visibility_timeout or getattr(settings, "SQS_VISIBILITY_TIMEOUT", 300)
        self.max_retries = max_retries or getattr(settings, "SQS_MAX_RETRIES", 3)
        self.retry_delay = retry_delay or getattr(settings, "SQS_RETRY_DELAY", 30)
        self.dead_letter_queue_url = dead_letter_queue_url or getattr(settings, "SQS_DEAD_LETTER_QUEUE_URL", None)
        
        # Temporary directory for downloaded images
        self.temp_dir = temp_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Metrics
        self.reset_metrics()
        
        logger.info(f"Initialized SQS processor with queue: {self.sqs_service.queue_url}")
        logger.info(f"Model: {os.path.basename(self.detector.model_path)}")
        logger.info(f"Batch size: {self.batch_size}, Wait time: {self.wait_time}s")
        logger.info(f"Visibility timeout: {self.visibility_timeout}s")
        logger.info(f"Max retries: {self.max_retries}, Retry delay: {self.retry_delay}s")
        if self.dead_letter_queue_url:
            logger.info(f"Dead letter queue: {self.dead_letter_queue_url}")
        else:
            logger.warning("No dead letter queue configured")
    
    def reset_metrics(self) -> None:
        """Reset processing metrics to zero."""
        self.metrics = ProcessingMetrics(
            frames_processed=0,
            frames_failed=0,
            frames_retried=0,
            total_detections=0,
            total_violations=0,
            poison_messages=0,
            avg_processing_time=0.0,
            last_processed=None
        )
    
    async def start(self) -> None:
        """
        Start the processing task in the background.
        
        This method starts an asyncio task that continuously polls the SQS queue
        for new messages and processes them in the background.
        """
        if self.running:
            logger.warning("Processor is already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        self.processing_task = asyncio.create_task(self._process_queue_continuously())
        logger.info("Started SQS processor")
    
    async def stop(self, timeout: int = 60) -> None:
        """
        Stop the processing task gracefully.
        
        This method stops the processing task and waits for it to complete
        any in-flight operations before returning.
        
        Args:
            timeout: Maximum time to wait for shutdown in seconds
        """
        if not self.running:
            logger.warning("Processor is not running")
            return
        
        logger.info(f"Stopping SQS processor (timeout: {timeout}s)")
        self.running = False
        self.shutdown_event.set()
        
        try:
            # Wait for the processing task to complete
            await asyncio.wait_for(self.processing_task, timeout=timeout)
            logger.info("Processor stopped gracefully")
        except asyncio.TimeoutError:
            logger.warning(f"Processor did not stop within {timeout}s timeout")
            # Cancel the task forcefully
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                logger.warning("Processor task cancelled")
        
        self.processing_task = None
    
    async def _process_queue_continuously(self) -> None:
        """
        Continuously process messages from the SQS queue until stopped.
        
        This method runs in a loop, polling the SQS queue for new messages
        and processing them in batches. It continues until the processor
        is stopped or an unhandled exception occurs.
        """
        logger.info("Starting continuous queue processing")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Process a batch of messages
                batch_result = await self._process_batch()
                
                # If no messages were processed, pause briefly to avoid tight polling
                if not batch_result or sum(len(getattr(batch_result, x)) for x in ['successful', 'failed', 'retried', 'invalid']) == 0:
                    await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in message processing loop: {str(e)}")
                # Pause briefly on error to avoid tight error loops
                await asyncio.sleep(5)
                # Continue with the next iteration
        
        logger.info("Stopped continuous queue processing")
    
    async def _process_batch(self) -> Optional[ProcessedBatch]:
        """
        Process a batch of messages from the SQS queue.
        
        This method receives a batch of messages from the SQS queue,
        processes each message, and returns a summary of the results.
        
        Returns:
            ProcessedBatch: Results from processing the batch, or None if no messages
        """
        batch_start_time = time.time()
        
        # Receive messages from the queue
        messages = self.sqs_service.receive_messages(
            max_messages=self.batch_size,
            wait_time=self.wait_time
        )
        
        if not messages:
            return None
        
        logger.info(f"Received {len(messages)} messages for processing")
        
        # Track results for metrics
        successful_messages = []
        failed_messages = []
        retried_messages = []
        invalid_messages = []
        
        # Process each message in the batch
        for message in messages:
            # Get message ID and receipt handle
            message_id = message.get('message_id', str(uuid.uuid4()))
            receipt_handle = message.get('receipt_handle')
            
            # Skip if already processing (handles SQS duplicate sends)
            if message_id in self.messages_in_flight:
                logger.warning(f"Message {message_id} is already being processed, skipping")
                continue
            
            try:
                # Mark message as in-flight
                self.messages_in_flight.add(message_id)
                
                # Process the message
                status, result = await self._process_message(message)
                
                # Handle message based on status
                if status == MessageStatus.SUCCESS:
                    # Delete successfully processed message
                    if receipt_handle:
                        self.sqs_service.delete_message(receipt_handle)
                    
                    successful_messages.append({
                        'message_id': message_id,
                        'result': result
                    })
                
                elif status == MessageStatus.RETRY:
                    # Message needs to be retried, don't delete
                    # The visibility timeout will expire and it will become available again
                    retried_messages.append({
                        'message_id': message_id,
                        'retry_count': message.get('retry_count', 0) + 1,
                        'error': result
                    })
                
                elif status == MessageStatus.FAIL:
                    # Message processing failed permanently
                    if self.dead_letter_queue_url and receipt_handle:
                        # Send to dead letter queue
                        await self._send_to_dead_letter_queue(message, result)
                        # Delete from original queue
                        self.sqs_service.delete_message(receipt_handle)
                    elif receipt_handle:
                        # No DLQ, just delete
                        self.sqs_service.delete_message(receipt_handle)
                    
                    failed_messages.append({
                        'message_id': message_id,
                        'error': result
                    })
                
                elif status == MessageStatus.INVALID:
                    # Message is invalid, delete it
                    if receipt_handle:
                        self.sqs_service.delete_message(receipt_handle)
                    
                    invalid_messages.append({
                        'message_id': message_id,
                        'error': result
                    })
            
            except Exception as e:
                logger.error(f"Unhandled exception processing message {message_id}: {str(e)}")
                failed_messages.append({
                    'message_id': message_id,
                    'error': f"Unhandled exception: {str(e)}"
                })
            
            finally:
                # Always remove from in-flight set
                self.messages_in_flight.discard(message_id)
        
        # Calculate batch processing time
        batch_time = time.time() - batch_start_time
        
        # Update metrics
        self._update_metrics(successful_messages, failed_messages, retried_messages, invalid_messages, batch_time)
        
        # Log batch summary
        logger.info(
            f"Batch processed in {batch_time:.2f}s: "
            f"{len(successful_messages)} successful, "
            f"{len(failed_messages)} failed, "
            f"{len(retried_messages)} retried, "
            f"{len(invalid_messages)} invalid"
        )
        
        return ProcessedBatch(
            successful=successful_messages,
            failed=failed_messages,
            retried=retried_messages,
            invalid=invalid_messages,
            processing_time=batch_time
        )
    
    async def _process_message(self, message: Dict[str, Any]) -> Tuple[MessageStatus, Any]:
        """
        Process a single message from the SQS queue.
        
        This method processes a single message from the SQS queue, including:
        - Validating the message structure
        - Downloading the image from S3
        - Running PPE detection
        - Storing results in the database
        - Uploading annotated images to S3
        
        Args:
            message: The message to process
            
        Returns:
            Tuple containing:
            - MessageStatus: The status of message processing
            - Any: Result data or error information
        """
        # Extract message data
        message_id = message.get('message_id', str(uuid.uuid4()))
        retry_count = message.get('retry_count', 0)
        
        logger.info(f"Processing message {message_id} (retry {retry_count})")
        
        # Get receipt handle for later deletion
        receipt_handle = message.get('receipt_handle')
        if not receipt_handle:
            logger.error(f"Message {message_id} is missing receipt_handle")
            return MessageStatus.INVALID, "Missing receipt_handle"
        
        # Validate required fields
        required_fields = ['image_id', 's3_path']
        for field in required_fields:
            if field not in message:
                logger.error(f"Message {message_id} is missing required field: {field}")
                return MessageStatus.INVALID, f"Missing required field: {field}"
        
        # Extract image information
        image_id = message.get('image_id')
        s3_path = message.get('s3_path')
        source_id = message.get('source_id', 'unknown')
        source_type = message.get('source_type', 'unknown')
        frame_number = message.get('frame_number')
        
        logger.info(f"Processing image {image_id} from {s3_path}")
        
        # Create temporary file path
        temp_image_path = os.path.join(self.temp_dir, f"{image_id}_{message_id}.jpg")
        
        try:
            # Download image from S3
            s3_key = self.s3_service.get_s3_path_from_full_url(s3_path)
            if not self.s3_service.download_image(s3_key, temp_image_path):
                error_msg = f"Failed to download image from S3: {s3_path}"
                logger.error(error_msg)
                
                # Check if we should retry or fail permanently
                if retry_count < self.max_retries:
                    logger.info(f"Will retry message {message_id} ({retry_count + 1}/{self.max_retries})")
                    return MessageStatus.RETRY, error_msg
                else:
                    logger.error(f"Exceeded max retries for message {message_id}")
                    return MessageStatus.FAIL, error_msg
            
            # Run PPE detection
            violations, detections = self.detector.detect_violations(temp_image_path)
            
            # Create annotated image
            annotated_path = os.path.join(self.temp_dir, f"{image_id}_{message_id}_annotated.jpg")
            self.detector.save_annotated_image(
                image=temp_image_path,
                output_path=annotated_path,
                detections=detections,
                violations=violations,
                show_all_detections=True
            )
            
            # Get detection summary
            detection_summary = self.detector.get_summary(violations)
            
            # Store results in database and upload to S3
            result = await self._store_results(
                image_id=image_id,
                s3_path=s3_path,
                temp_image_path=temp_image_path,
                annotated_path=annotated_path,
                detections=detections,
                violations=violations,
                source_id=source_id,
                source_type=source_type,
                frame_number=frame_number,
                detection_summary=detection_summary
            )
            
            return MessageStatus.SUCCESS, result
        
        except Exception as e:
            error_msg = f"Error processing message {message_id}: {str(e)}"
            logger.error(error_msg)
            
            # Check if we should retry or fail permanently
            if retry_count < self.max_retries:
                logger.info(f"Will retry message {message_id} ({retry_count + 1}/{self.max_retries})")
                return MessageStatus.RETRY, error_msg
            else:
                logger.error(f"Exceeded max retries for message {message_id}")
                return MessageStatus.FAIL, error_msg
        
        finally:
            # Clean up temporary files
            self._cleanup_temp_files([temp_image_path, annotated_path])
    
    async def _store_results(
        self,
        image_id: str,
        s3_path: str,
        temp_image_path: str,
        annotated_path: str,
        detections: List[DetectionResult],
        violations: List[PPEViolation],
        source_id: str,
        source_type: str,
        frame_number: Optional[int],
        detection_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store detection results in the database and S3.
        
        This method:
        - Uploads the annotated image to S3
        - Creates a database record for the detection
        - Uploads the detection results as JSON to S3
        
        Args:
            image_id: ID of the processed image
            s3_path: S3 path of the original image
            temp_image_path: Path to the temporary image file
            annotated_path: Path to the annotated image file
            detections: List of detection results
            violations: List of PPE violations
            source_id: ID of the source (camera, video, etc.)
            source_type: Type of source
            frame_number: Frame number for video sources
            detection_summary: Summary of detection results
            
        Returns:
            Dict containing result information
            
        Raises:
            Exception: If database or S3 operations fail
        """
        try:
            # Upload annotated image to S3
            annotated_s3_key = f"annotated/{image_id}.jpg"
            annotated_s3_path = f"s3://{self.s3_service.bucket_name}/{annotated_s3_key}"
            
            self.s3_service.s3_client.upload_file(
                annotated_path,
                self.s3_service.bucket_name,
                annotated_s3_key
            )
            
            logger.info(f"Uploaded annotated image to {annotated_s3_path}")
            
            # Format detection results
            detection_results = {
                "timestamp": datetime.now().isoformat(),
                "image_id": image_id,
                "original_path": s3_path,
                "annotated_path": annotated_s3_path,
                "source_id": source_id,
                "source_type": source_type,
                "frame_number": frame_number,
                "num_detections": len(detections),
                "num_violations": len(violations),
                "ppe_detected": detection_summary["has_violations"],
                "violations": detection_summary["violation_counts"],
                "confidence_threshold": self.detector.confidence_threshold,
                "model_version": self.detector.get_model_version(),
                "detections": [
                    {
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox": d.bbox
                    } for d in detections
                ],
                "violations": [
                    {
                        "violation_type": v.violation_type.value,
                        "confidence": v.confidence,
                        "bbox": v.bbox
                    } for v in violations
                ]
            }
            
            # Upload detection results to S3
            results_s3_key = f"results/{image_id}.json"
            self.s3_service.upload_result(results_s3_key, detection_results)
            
            logger.info(f"Uploaded detection results to s3://{self.s3_service.bucket_name}/{results_s3_key}")
            
            # Store in database
            db = SessionLocal()
            try:
                # Create database record
                detection = Detection(
                    image_id=image_id,
                    image_path=s3_path,
                    source_id=source_id,
                    source_type=source_type,
                    frame_number=frame_number,
                    num_detections=len(detections),
                    ppe_detected=detection_summary["has_violations"],
                    violations_detected=detection_summary["has_violations"],
                    detection_results=detection_results,
                    confidence_threshold=self.detector.confidence_threshold,
                    model_version=self.detector.get_model_version(),
                    processing_time=0.0  # TODO: Add processing time
                )
                
                db.add(detection)
                db.commit()
                db.refresh(detection)
                
                logger.info(f"Stored detection results in database with ID {detection.id}")
                
                return {
                    "detection_id": detection.id,
                    "image_id": image_id,
                    "annotated_path": annotated_s3_path,
                    "results_path": f"s3://{self.s3_service.bucket_name}/{results_s3_key}",
                    "num_detections": len(detections),
                    "num_violations": len(violations),
                    "ppe_detected": detection_summary["has_violations"]
                }
            
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"Database error storing detection results: {str(e)}")
                raise
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f"Error storing results for image {image_id}: {str(e)}")
            raise
    
    async def _send_to_dead_letter_queue(self, message: Dict[str, Any], error: str) -> bool:
        """
        Send a failed message to the dead letter queue.
        
        Args:
            message: The original message
            error: Error information to include
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.dead_letter_queue_url:
            logger.warning("Cannot send to dead letter queue: URL not configured")
            return False
        
        try:
            # Create dead letter message with original content and error info
            dead_letter_message = message.copy()
            dead_letter_message['processing_error'] = error
            dead_letter_message['failed_at'] = datetime.now().isoformat()
            
            # Send to dead letter queue
            sqs_client = boto3.client(
                'sqs',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            
            response = sqs_client.send_message(
                QueueUrl=self.dead_letter_queue_url,
                MessageBody=json.dumps(dead_letter_message)
            )
            
            logger.info(f"Sent message to dead letter queue: {response.get('MessageId')}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send message to dead letter queue: {str(e)}")
            return False
    
    def _cleanup_temp_files(self, file_paths: List[str]) -> None:
        """
        Clean up temporary files.
        
        Args:
            file_paths: List of file paths to delete
        """
        for path in file_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {path}: {str(e)}")
    
    def _update_metrics(
        self,
        successful: List[Dict[str, Any]],
        failed: List[Dict[str, Any]],
        retried: List[Dict[str, Any]],
        invalid: List[Dict[str, Any]],
        batch_time: float
    ) -> None:
        """
        Update processing metrics with batch results.
        
        Args:
            successful: List of successfully processed messages
            failed: List of permanently failed messages
            retried: List of messages to be retried
            invalid: List of invalid messages
            batch_time: Time taken to process the batch in seconds
        """
        # Count frames and detections
        num_successful = len(successful)
        num_failed = len(failed)
        num_retried = len(retried)
        num_invalid = len(invalid)
        
        # Calculate new metrics
        total_frames = self.metrics.frames_processed + num_successful
        total_failed = self.metrics.frames_failed + num_failed
        total_retried = self.metrics.frames_retried + num_retried
        poison_messages = self.metrics.poison_messages + num_invalid
        
        # Count detections and violations
        num_detections = 0
        num_violations = 0
        
        for msg in successful:
            result = msg.get('result', {})
            num_detections += result.get('num_detections', 0)
            num_violations += result.get('num_violations', 0)
        
        total_detections = self.metrics.total_detections + num_detections
        total_violations = self.metrics.total_violations + num_violations
        
        # Calculate average processing time
        if total_frames > 0:
            # Weighted average to include previous average
            if self.metrics.frames_processed > 0:
                avg_time = (
                    (self.metrics.avg_processing_time * self.metrics.frames_processed) + 
                    (batch_time * num_successful)
                ) / total_frames
            else:
                avg_time = batch_time if num_successful > 0 else 0.0
        else:
            avg_time = 0.0
        
        # Update metrics
        self.metrics = ProcessingMetrics(
            frames_processed=total_frames,
            frames_failed=total_failed,
            frames_retried=total_retried,
            total_detections=total_detections,
            total_violations=total_violations,
            poison_messages=poison_messages,
            avg_processing_time=avg_time,
            last_processed=datetime.now() if num_successful > 0 else self.metrics.last_processed
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the processor.
        
        Returns:
            Dict containing processor status information
        """
        return {
            "running": self.running,
            "metrics": self.metrics._asdict(),
            "config": {
                "batch_size": self.batch_size,
                "wait_time": self.wait_time,
                "visibility_timeout": self.visibility_timeout,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "has_dead_letter_queue": bool(self.dead_letter_queue_url),
                "model_path": self.detector.model_path,
                "confidence_threshold": self.detector.confidence_threshold
            },
            "in_flight_messages": len(self.messages_in_flight)
        }


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = SQSProcessor()
    
    # Define async main function
    async def main():
        # Start processor
        await processor.start()
        
        try:
            # Run for a while
            await asyncio.sleep(3600)  # 1 hour
        finally:
            # Stop processor
            await processor.stop()
    
    # Run the async main function
    asyncio.run(main())
