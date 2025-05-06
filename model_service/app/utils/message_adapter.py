"""
Message Adapter for SIFT PPE Detection Service

This module provides adaptation between different message formats,
allowing the processor to handle messages from various sources.
"""
import json
import logging
import uuid
from typing import Dict, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

def adapt_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt a raw SQS message to the format expected by the SQS processor.
    
    The processor expects messages with:
    - image_id: Unique identifier for the image
    - s3_path: S3 path where the image is stored
    - receipt_handle: SQS receipt handle for acknowledging the message
    
    Args:
        message: The raw SQS message
        
    Returns:
        Adapted message in the format expected by the processor
    """
    # Keep the original receipt handle
    receipt_handle = message.get('ReceiptHandle')
    
    # Try to parse the message body as JSON
    body = message.get('Body', '{}')
    try:
        body_data = json.loads(body)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse message body as JSON: {body}")
        body_data = {}
    
    # First, try to extract fields from the OSHA ingestion service format
    # This format typically has frame_path, timestamp, and camera_id fields
    if isinstance(body_data, dict):
        # Extract key fields from the OSHA format
        frame_path = body_data.get('frame_path')
        timestamp = body_data.get('timestamp')
        camera_id = body_data.get('camera_id')
        frame_number = body_data.get('frame_number')
        
        if frame_path:
            # Format appears to be from the OSHA ingestion service
            return {
                'image_id': f"{camera_id}_{timestamp}" if camera_id and timestamp else str(uuid.uuid4()),
                's3_path': frame_path,
                'source_id': camera_id or 'unknown',
                'source_type': 'camera',
                'frame_number': frame_number,
                'timestamp': timestamp,
                'receipt_handle': receipt_handle,
                'message_id': message.get('MessageId', str(uuid.uuid4())),
                'original_message': message  # Keep original for reference
            }
    
    # If we can't adapt the message, return it with a flag indicating it's invalid
    logger.warning(f"Unable to adapt message format: {message}")
    return {
        'invalid_format': True,
        'receipt_handle': receipt_handle,
        'message_id': message.get('MessageId', str(uuid.uuid4())),
        'original_message': message
    }

def is_valid_message(message: Dict[str, Any]) -> bool:
    """
    Check if a message is valid for processing.
    
    Args:
        message: The message to check
        
    Returns:
        True if valid, False otherwise
    """
    return (
        'image_id' in message and
        's3_path' in message and
        'receipt_handle' in message and
        not message.get('invalid_format', False)
    )
