"""
AWS service connectors for S3 and SQS.
"""
import boto3
import json
import os
import logging
from botocore.exceptions import ClientError
from typing import Dict, List, Any, Optional
from loguru import logger

from app.core.config import settings
from app.utils.message_adapter import adapt_message

class S3Service:
    """Service for interacting with AWS S3."""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        self.bucket_name = settings.S3_BUCKET_NAME
        logger.info(f"Initialized S3 service with bucket: {self.bucket_name}")
    
    def download_image(self, s3_key: str, local_path: str) -> bool:
        """
        Download an image from S3 to a local path.
        
        Args:
            s3_key: The S3 key of the image
            local_path: The local path to save the image to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.debug(f"Downloading {s3_key} to {local_path}")
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True
        except ClientError as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            return False
    
    def upload_result(self, result_key: str, result_data: Dict) -> bool:
        """
        Upload detection results to S3 as JSON.
        
        Args:
            result_key: The S3 key to upload results to
            result_data: The results data to upload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.debug(f"Uploading results to {result_key}")
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=result_key,
                Body=json.dumps(result_data),
                ContentType='application/json'
            )
            return True
        except ClientError as e:
            logger.error(f"Error uploading results to {result_key}: {e}")
            return False
    
    def get_s3_path_from_full_url(self, full_s3_url: str) -> str:
        """
        Extract the S3 key from a full S3 URL.
        
        Args:
            full_s3_url: Full S3 URL (e.g., s3://bucket-name/path/to/file.jpg)
            
        Returns:
            str: S3 key without the bucket name and s3:// prefix
        """
        # Remove 's3://' prefix and bucket name
        if full_s3_url.startswith(f"s3://{self.bucket_name}/"):
            return full_s3_url[len(f"s3://{self.bucket_name}/"):]
        elif full_s3_url.startswith("s3://"):
            # Extract the key without bucket
            parts = full_s3_url[5:].split('/', 1)
            if len(parts) > 1:
                return parts[1]
        
        # If doesn't match expected format, return as is (might be just the key)
        return full_s3_url


class SQSService:
    """Service for interacting with AWS SQS."""
    
    def __init__(self):
        self.sqs_client = boto3.client(
            'sqs',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        self.queue_url = settings.SQS_QUEUE_URL
        logger.info(f"Initialized SQS service with queue: {self.queue_url}")
    
    def receive_messages(self, max_messages: int = None, wait_time: int = None) -> List[Dict]:
        """
        Receive messages from SQS queue.
        
        Args:
            max_messages: Max number of messages to receive (default from settings)
            wait_time: Wait time in seconds (default from settings)
            
        Returns:
            List[Dict]: List of message dictionaries with adapted format for processing
        """
        if max_messages is None:
            max_messages = settings.SQS_MAX_MESSAGES
        
        if wait_time is None:
            wait_time = settings.SQS_WAIT_TIME
        
        try:
            logger.debug(f"Receiving up to {max_messages} messages with {wait_time}s wait time")
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time,
                AttributeNames=['All'],
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            logger.info(f"Received {len(messages)} messages from SQS")
            
            # Adapt messages to the format expected by the processor
            adapted_messages = []
            for msg in messages:
                # Use the message adapter to convert to expected format
                adapted_message = adapt_message(msg)
                adapted_messages.append(adapted_message)
                
            return adapted_messages
            
        except ClientError as e:
            logger.error(f"Error receiving messages from SQS: {e}")
            return []
    
    def delete_message(self, receipt_handle: str) -> bool:
        """
        Delete a message from the SQS queue.
        
        Args:
            receipt_handle: The receipt handle of the message to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.debug(f"Deleting message with receipt handle: {receipt_handle[:20]}...")
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            return True
        except ClientError as e:
            logger.error(f"Error deleting message from SQS: {e}")
            return False
