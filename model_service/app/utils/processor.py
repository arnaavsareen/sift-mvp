"""
Utility for processing messages from SQS queue.
"""
import os
import json
import time
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from loguru import logger

from app.services.aws import SQSService, S3Service
from app.services.detection import PPEDetectionService
from app.models.detection import Detection

async def process_sqs_messages(max_messages: int = 10, db_session: Session = None):
    """
    Process messages from SQS queue.
    
    Args:
        max_messages: Maximum number of messages to process
        db_session: Database session for storing results
    """
    logger.info(f"Starting to process up to {max_messages} messages from SQS")
    
    # Initialize services
    sqs_service = SQSService()
    s3_service = S3Service()
    detection_service = PPEDetectionService()
    
    # Receive messages from SQS
    messages = sqs_service.receive_messages(max_messages=max_messages)
    
    if not messages:
        logger.info("No messages received from SQS")
        return
    
    logger.info(f"Received {len(messages)} messages for processing")
    
    for i, message in enumerate(messages):
        logger.info(f"Processing message {i+1}/{len(messages)}")
        
        try:
            # Extract image information from message
            receipt_handle = message.get('receipt_handle')
            
            if 'image_id' not in message or 's3_path' not in message:
                logger.warning(f"Invalid message format, missing required fields: {message}")
                if receipt_handle:
                    sqs_service.delete_message(receipt_handle)
                continue
            
            image_id = message.get('image_id')
            s3_path = message.get('s3_path')
            source_id = message.get('source_id', 'unknown')
            source_type = message.get('source_type', 'unknown')
            frame_number = message.get('frame_number')
            
            # Download image from S3
            temp_dir = "./temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = f"{temp_dir}/{image_id}.jpg"
            
            s3_key = s3_service.get_s3_path_from_full_url(s3_path)
            
            if not s3_service.download_image(s3_key, temp_image_path):
                logger.error(f"Failed to download image from S3: {s3_path}")
                continue
            
            # Process image
            result = detection_service.process_image(temp_image_path)
            
            # Store result in database if session provided
            if db_session is not None:
                detection = Detection(
                    image_id=image_id,
                    image_path=s3_path,
                    source_id=source_id,
                    source_type=source_type,
                    frame_number=frame_number,
                    num_detections=result["num_detections"],
                    ppe_detected=result["ppe_detected"],
                    violations_detected=result["violations_detected"],
                    detection_results=result,
                    confidence_threshold=result["confidence_threshold"],
                    model_version=result["model_version"],
                    processing_time=result["processing_time"]
                )
                
                db_session.add(detection)
                db_session.commit()
            
            # Upload result to S3
            result_key = f"results/{image_id}_detection.json"
            s3_service.upload_result(result_key, result)
            
            # Delete message from queue
            if receipt_handle:
                sqs_service.delete_message(receipt_handle)
            
            # Clean up temporary file
            try:
                os.remove(temp_image_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary image: {e}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Continue with next message
            continue
    
    logger.info(f"Completed processing {len(messages)} messages from SQS")
