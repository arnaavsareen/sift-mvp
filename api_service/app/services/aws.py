"""
AWS service utilities for the SIFT API Service.
"""
import boto3
import logging
from botocore.exceptions import ClientError
from typing import Optional, Dict, Any, List
import json

from app.core.config import settings
from app.core.logging import logger


class AWSService:
    """
    Service for AWS interactions.
    """
    
    def __init__(self):
        """
        Initialize AWS service connections.
        """
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_DEFAULT_REGION
        )
        
        self.sqs_client = boto3.client(
            'sqs',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_DEFAULT_REGION
        )
    
    def get_s3_presigned_url(
        self, 
        bucket_name: str, 
        object_key: str, 
        expiration: int = 3600
    ) -> str:
        """
        Generate a pre-signed URL for an S3 object.
        
        Args:
            bucket_name: The name of the S3 bucket
            object_key: The key of the S3 object
            expiration: URL expiration time in seconds (default 1 hour)
            
        Returns:
            Pre-signed URL as string
            
        Raises:
            Exception: If there was an error generating the URL
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': bucket_name,
                    'Key': object_key
                },
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Error generating pre-signed URL: {str(e)}")
            raise
    
    def parse_s3_uri(self, s3_uri: str) -> tuple:
        """
        Parse S3 URI into bucket name and object key.
        
        Args:
            s3_uri: S3 URI in format s3://bucket/path/to/object
            
        Returns:
            Tuple of (bucket_name, object_key)
            
        Raises:
            ValueError: If the URI is not in the correct format
        """
        if not s3_uri.startswith('s3://'):
            raise ValueError("Invalid S3 URI format")
        
        # Remove 's3://' prefix
        path = s3_uri[5:]
        
        # Split into bucket and key
        parts = path.split('/', 1)
        if len(parts) < 2:
            raise ValueError("Invalid S3 URI format")
        
        bucket_name = parts[0]
        object_key = parts[1]
        
        return bucket_name, object_key
    
    def get_image_url_from_s3_uri(self, s3_uri: str) -> Optional[str]:
        """
        Convert S3 URI to a pre-signed URL.
        
        Args:
            s3_uri: S3 URI in format s3://bucket/path/to/object
            
        Returns:
            Pre-signed URL or None if invalid URI
        """
        try:
            bucket_name, object_key = self.parse_s3_uri(s3_uri)
            return self.get_s3_presigned_url(bucket_name, object_key)
        except Exception as e:
            logger.error(f"Error converting S3 URI to URL: {str(e)}")
            return None
    
    def send_sqs_message(
        self, 
        queue_url: str, 
        message_body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a message to an SQS queue.
        
        Args:
            queue_url: URL of the SQS queue
            message_body: Message body as a dictionary
            
        Returns:
            SQS response
            
        Raises:
            Exception: If there was an error sending the message
        """
        try:
            response = self.sqs_client.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(message_body)
            )
            return response
        except Exception as e:
            logger.error(f"Error sending SQS message: {str(e)}")
            raise
    
    def receive_sqs_messages(
        self, 
        queue_url: str, 
        max_messages: int = 10, 
        wait_time: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Receive messages from an SQS queue.
        
        Args:
            queue_url: URL of the SQS queue
            max_messages: Maximum number of messages to receive (1-10)
            wait_time: Long polling wait time in seconds (0-20)
            
        Returns:
            List of messages
            
        Raises:
            Exception: If there was an error receiving messages
        """
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time,
                AttributeNames=['All'],
                MessageAttributeNames=['All']
            )
            
            return response.get('Messages', [])
        except Exception as e:
            logger.error(f"Error receiving SQS messages: {str(e)}")
            raise
    
    def delete_sqs_message(self, queue_url: str, receipt_handle: str) -> None:
        """
        Delete a message from an SQS queue.
        
        Args:
            queue_url: URL of the SQS queue
            receipt_handle: Receipt handle of the message
            
        Raises:
            Exception: If there was an error deleting the message
        """
        try:
            self.sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )
        except Exception as e:
            logger.error(f"Error deleting SQS message: {str(e)}")
            raise


# Export singleton instance
aws_service = AWSService()


def get_s3_presigned_url(bucket_name: str, object_key: str, expiration: int = 3600) -> str:
    """
    Convenience function to get a pre-signed URL for an S3 object.
    """
    return aws_service.get_s3_presigned_url(bucket_name, object_key, expiration)
