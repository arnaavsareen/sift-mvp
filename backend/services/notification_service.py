import os
import logging
import json
import time
import threading
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import traceback

from backend.config import BASE_DIR
from backend.models import Alert, Camera

logger = logging.getLogger(__name__)

class NotificationChannel:
    """Base class for notification channels"""
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", False)
        self.last_notified = {}  # alert_type -> timestamp
        self.throttle_seconds = config.get("throttle_seconds", 300)  # 5 minutes default
    
    def send_notification(
        self,
        alert: Dict[str, Any],
        camera: Optional[Dict[str, Any]] = None,
        screenshot_path: Optional[str] = None
    ) -> bool:
        """
        Send notification for alert.
        
        Args:
            alert: Alert data
            camera: Camera data
            screenshot_path: Path to screenshot file
            
        Returns:
            True if notification was sent
        """
        # Check if enabled
        if not self.enabled:
            return False
        
        # Check throttling for this alert type
        alert_type = alert.get("violation_type", "unknown")
        now = time.time()
        
        if alert_type in self.last_notified:
            time_since_last = now - self.last_notified[alert_type]
            if time_since_last < self.throttle_seconds:
                logger.debug(
                    f"Throttling {alert_type} notification on {self.name} channel "
                    f"({time_since_last:.1f}s < {self.throttle_seconds}s)"
                )
                return False
        
        # Update last notified time
        self.last_notified[alert_type] = now
        
        # Notification implementation in subclasses
        return self._send(alert, camera, screenshot_path)
    
    def _send(
        self,
        alert: Dict[str, Any],
        camera: Optional[Dict[str, Any]],
        screenshot_path: Optional[str]
    ) -> bool:
        """
        Send notification for alert (to be implemented by subclasses).
        
        Args:
            alert: Alert data
            camera: Camera data
            screenshot_path: Path to screenshot file
            
        Returns:
            True if notification was sent
        """
        raise NotImplementedError("Subclasses must implement _send method")


class EmailNotification(NotificationChannel):
    """Email notification channel"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("email", config)
        self.smtp_server = config.get("smtp_server", "")
        self.smtp_port = config.get("smtp_port", 587)
        self.smtp_username = config.get("smtp_username", "")
        self.smtp_password = config.get("smtp_password", "")
        self.sender_email = config.get("sender_email", "")
        self.recipient_emails = config.get("recipient_emails", [])
        self.include_screenshot = config.get("include_screenshot", True)
        self.use_ssl = config.get("use_ssl", False)
        self.use_tls = config.get("use_tls", True)
    
    def _send(
        self,
        alert: Dict[str, Any],
        camera: Optional[Dict[str, Any]],
        screenshot_path: Optional[str]
    ) -> bool:
        """Send email notification"""
        if not self.smtp_server or not self.recipient_emails:
            logger.warning("Email notification not configured properly")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg["Subject"] = f"SIFT Safety Alert: {alert.get('violation_type', 'Violation')} Detected"
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(self.recipient_emails)
            
            # Create HTML body
            html = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .alert-box {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; }}
                    .alert-title {{ color: #d9534f; font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                    .alert-info {{ margin-bottom: 5px; }}
                    .screenshot {{ max-width: 100%; margin-top: 15px; }}
                </style>
            </head>
            <body>
                <h2>SIFT Safety Alert</h2>
                <div class="alert-box">
                    <div class="alert-title">{alert.get('violation_type', 'Violation').replace('_', ' ').title()} Detected</div>
                    <div class="alert-info"><strong>Time:</strong> {alert.get('created_at', datetime.now().isoformat())}</div>
                    <div class="alert-info"><strong>Camera:</strong> {camera.get('name', f"Camera {alert.get('camera_id', 'Unknown')}")}</div>
                    <div class="alert-info"><strong>Location:</strong> {camera.get('location', 'Unknown')}</div>
                    <div class="alert-info"><strong>Confidence:</strong> {alert.get('confidence', 0) * 100:.1f}%</div>
                    <div class="alert-info"><strong>Alert ID:</strong> {alert.get('id', 'Unknown')}</div>
                    
                    <p>A safety violation has been detected by the SIFT monitoring system. 
                    Please review the alert and take appropriate action.</p>
                    
                    {f'<p><img src="cid:screenshot" class="screenshot"></p>' if screenshot_path and self.include_screenshot else ''}
                </div>
                
                <p>This is an automated notification from SIFT Safety Monitoring System.</p>
            </body>
            </html>
            """
            
            # Attach HTML body
            msg.attach(MIMEText(html, "html"))
            
            # Attach screenshot if available
            if screenshot_path and self.include_screenshot:
                try:
                    # Check if path is relative to SCREENSHOTS_DIR
                    if screenshot_path.startswith("/screenshots/"):
                        screenshot_path = os.path.join(
                            BASE_DIR, "data", screenshot_path.lstrip("/")
                        )
                    
                    with open(screenshot_path, "rb") as img_file:
                        image = MIMEImage(img_file.read())
                        image.add_header("Content-ID", "<screenshot>")
                        image.add_header("Content-Disposition", "inline", filename="violation.jpg")
                        msg.attach(image)
                except Exception as e:
                    logger.error(f"Error attaching screenshot: {str(e)}")
            
            # Connect to SMTP server
            if self.use_ssl:
                smtp = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                smtp = smtplib.SMTP(self.smtp_server, self.smtp_port)
                
                if self.use_tls:
                    smtp.starttls()
            
            # Login and send
            if self.smtp_username and self.smtp_password:
                smtp.login(self.smtp_username, self.smtp_password)
            
            smtp.send_message(msg)
            smtp.quit()
            
            logger.info(f"Sent email notification to {len(self.recipient_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            logger.error(traceback.format_exc())
            return False


class SlackNotification(NotificationChannel):
    """Slack notification channel"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("slack", config)
        self.webhook_url = config.get("webhook_url", "")
        self.channel = config.get("channel", "")
        self.username = config.get("username", "SIFT Safety Bot")
        self.icon_emoji = config.get("icon_emoji", ":warning:")
        self.include_screenshot = config.get("include_screenshot", True)
        self.dashboard_url = config.get("dashboard_url", "")
    
    def _send(
        self,
        alert: Dict[str, Any],
        camera: Optional[Dict[str, Any]],
        screenshot_path: Optional[str]
    ) -> bool:
        """Send Slack notification"""
        if not self.webhook_url:
            logger.warning("Slack notification not configured properly")
            return False
        
        try:
            # Format violation type
            violation_type = alert.get('violation_type', 'Violation').replace('_', ' ').title()
            
            # Create message text
            text = f"*SAFETY ALERT: {violation_type} Detected*"
            
            # Create message blocks
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": text
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:*\n{datetime.fromisoformat(alert.get('created_at', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": r"*Camera:*\n" + f"{camera.get('name', 'Camera ' + str(alert.get('camera_id', 'Unknown')))}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Location:*\n{camera.get('location', 'Unknown')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Confidence:*\n{alert.get('confidence', 0) * 100:.1f}%"
                        }
                    ]
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Alert ID: {alert.get('id', 'Unknown')}"
                        }
                    ]
                }
            ]
            
            # Add screenshot if available
            if screenshot_path and self.include_screenshot:
                try:
                    # Check if path is relative to SCREENSHOTS_DIR
                    if screenshot_path.startswith("/screenshots/"):
                        public_url = f"{self.dashboard_url}{screenshot_path}" if self.dashboard_url else None
                        
                        if public_url:
                            blocks.append({
                                "type": "image",
                                "title": {
                                    "type": "plain_text",
                                    "text": "Violation Screenshot"
                                },
                                "image_url": public_url,
                                "alt_text": "Safety violation screenshot"
                            })
                except Exception as e:
                    logger.error(f"Error adding screenshot to Slack message: {str(e)}")
            
            # Add action buttons if dashboard URL is provided
            if self.dashboard_url:
                blocks.append({
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "View Alert"
                            },
                            "url": f"{self.dashboard_url}/alerts/{alert.get('id', '')}"
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "View Camera"
                            },
                            "url": f"{self.dashboard_url}/cameras/{alert.get('camera_id', '')}"
                        }
                    ]
                })
            
            # Prepare message data
            data = {
                "channel": self.channel,
                "username": self.username,
                "icon_emoji": self.icon_emoji,
                "text": text,
                "blocks": blocks
            }
            
            # Send message
            response = requests.post(
                self.webhook_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data)
            )
            
            if response.status_code != 200:
                logger.error(f"Error sending Slack notification: {response.text}")
                return False
            
            logger.info(f"Sent Slack notification to {self.channel}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            logger.error(traceback.format_exc())
            return False


class SMSNotification(NotificationChannel):
    """SMS notification channel"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("sms", config)
        self.provider = config.get("provider", "twilio")
        self.twilio_account_sid = config.get("twilio_account_sid", "")
        self.twilio_auth_token = config.get("twilio_auth_token", "")
        self.twilio_from_number = config.get("twilio_from_number", "")
        self.recipient_numbers = config.get("recipient_numbers", [])
    
    def _send(
        self,
        alert: Dict[str, Any],
        camera: Optional[Dict[str, Any]],
        screenshot_path: Optional[str]
    ) -> bool:
        """Send SMS notification"""
        if not self.twilio_account_sid or not self.twilio_auth_token or not self.recipient_numbers:
            logger.warning("SMS notification not configured properly")
            return False
        
        try:
            # Check if Twilio is available
            try:
                from twilio.rest import Client
            except ImportError:
                logger.error("Twilio library not installed. Install with 'pip install twilio'")
                return False
            
            # Format violation type
            violation_type = alert.get('violation_type', 'Violation').replace('_', ' ').title()
            
            # Create message text
            text = (
                f"SIFT SAFETY ALERT: {violation_type} detected at "
                f"{camera.get('name', 'Camera ' + str(alert.get('camera_id', 'Unknown')))}"
                f" ({camera.get('location', 'Unknown location')}). "
                f"Time: {datetime.fromisoformat(alert.get('created_at', datetime.now().isoformat())).strftime('%H:%M:%S')}. "
                f"Confidence: {alert.get('confidence', 0) * 100:.1f}%."
            )
            
            # Create Twilio client
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            # Send SMS to each recipient
            successful_sends = 0
            
            for number in self.recipient_numbers:
                try:
                    message = client.messages.create(
                        body=text,
                        from_=self.twilio_from_number,
                        to=number
                    )
                    successful_sends += 1
                except Exception as e:
                    logger.error(f"Error sending SMS to {number}: {str(e)}")
            
            logger.info(f"Sent SMS notifications to {successful_sends}/{len(self.recipient_numbers)} recipients")
            return successful_sends > 0
            
        except Exception as e:
            logger.error(f"Error sending SMS notification: {str(e)}")
            logger.error(traceback.format_exc())
            return False


class WebhookNotification(NotificationChannel):
    """Webhook notification channel"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__("webhook", config)
        self.webhook_url = config.get("webhook_url", "")
        self.webhook_method = config.get("webhook_method", "POST")
        self.webhook_headers = config.get("webhook_headers", {})
        self.include_screenshot = config.get("include_screenshot", False)
    
    def _send(
        self,
        alert: Dict[str, Any],
        camera: Optional[Dict[str, Any]],
        screenshot_path: Optional[str]
    ) -> bool:
        """Send webhook notification"""
        if not self.webhook_url:
            logger.warning("Webhook notification not configured properly")
            return False
        
        try:
            # Prepare alert data
            payload = {
                "alert": alert,
                "camera": camera,
                "timestamp": datetime.now().isoformat(),
                "system": "SIFT Safety Monitoring"
            }
            
            # Include screenshot data if requested
            if screenshot_path and self.include_screenshot:
                try:
                    # Check if path is relative to SCREENSHOTS_DIR
                    if screenshot_path.startswith("/screenshots/"):
                        screenshot_path = os.path.join(
                            BASE_DIR, "data", screenshot_path.lstrip("/")
                        )
                    
                    # Add screenshot URL
                    payload["screenshot_path"] = screenshot_path
                except Exception as e:
                    logger.error(f"Error processing screenshot for webhook: {str(e)}")
            
            # Send webhook request
            if self.webhook_method.upper() == "GET":
                response = requests.get(
                    self.webhook_url,
                    headers=self.webhook_headers,
                    params=payload
                )
            else:
                response = requests.post(
                    self.webhook_url,
                    headers=self.webhook_headers,
                    json=payload
                )
            
            if response.status_code not in [200, 201, 202, 204]:
                logger.error(f"Error sending webhook notification: {response.status_code} {response.text}")
                return False
            
            logger.info(f"Sent webhook notification to {self.webhook_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {str(e)}")
            logger.error(traceback.format_exc())
            return False


class NotificationService:
    """
    Service for sending notifications about safety alerts.
    Supports multiple notification channels: email, Slack, SMS, and webhooks.
    """
    
    def __init__(
        self, 
        db: Session,
        config_path: str = os.path.join(BASE_DIR, "data", "config", "notifications.json")
    ):
        self.db = db
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Notification channels
        self.channels: Dict[str, NotificationChannel] = {}
        
        # Initialize notification channels
        self._initialize_channels()
        
        # Background notification thread and queue
        self.notification_queue = []
        self.notification_thread = None
        self.is_running = False
        self.lock = threading.RLock()
        
        # Start background thread
        self.start()
        
        logger.info(f"Notification service initialized with {len(self.channels)} channels")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load notification configuration"""
        default_config = {
            "email": {
                "enabled": False,
                "smtp_server": "",
                "smtp_port": 587,
                "smtp_username": "",
                "smtp_password": "",
                "sender_email": "",
                "recipient_emails": [],
                "include_screenshot": True,
                "throttle_seconds": 300
            },
            "slack": {
                "enabled": False,
                "webhook_url": "",
                "channel": "",
                "username": "SIFT Safety Bot",
                "icon_emoji": ":warning:",
                "include_screenshot": True,
                "dashboard_url": "",
                "throttle_seconds": 300
            },
            "sms": {
                "enabled": False,
                "provider": "twilio",
                "twilio_account_sid": "",
                "twilio_auth_token": "",
                "twilio_from_number": "",
                "recipient_numbers": [],
                "throttle_seconds": 900
            },
            "webhook": {
                "enabled": False,
                "webhook_url": "",
                "webhook_method": "POST",
                "webhook_headers": {
                    "Content-Type": "application/json"
                },
                "include_screenshot": False,
                "throttle_seconds": 300
            }
        }
        
        # Create config directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if config file exists
        if not self.config_path.exists():
            # Create default config
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
        
        # Load config
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            
            # Merge with default config to ensure all fields exist
            for channel, default_settings in default_config.items():
                if channel not in config:
                    config[channel] = default_settings
                else:
                    # Make sure all settings exist
                    for setting, default_value in default_settings.items():
                        if setting not in config[channel]:
                            config[channel][setting] = default_value
            
            return config
        except Exception as e:
            logger.error(f"Error loading notification config: {str(e)}")
            return default_config
    
    def _save_config(self) -> bool:
        """Save notification configuration"""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving notification config: {str(e)}")
            return False
    
    def _initialize_channels(self) -> None:
        """Initialize notification channels from config"""
        with self.lock:
            # Email
            if "email" in self.config:
                self.channels["email"] = EmailNotification(self.config["email"])
            
            # Slack
            if "slack" in self.config:
                self.channels["slack"] = SlackNotification(self.config["slack"])
            
            # SMS
            if "sms" in self.config:
                self.channels["sms"] = SMSNotification(self.config["sms"])
            
            # Webhook
            if "webhook" in self.config:
                self.channels["webhook"] = WebhookNotification(self.config["webhook"])
    
    def start(self) -> bool:
        """Start the notification background thread"""
        with self.lock:
            if self.is_running:
                return True
            
            self.is_running = True
            self.notification_thread = threading.Thread(
                target=self._notification_loop,
                daemon=True,
                name="notification-thread"
            )
            self.notification_thread.start()
            
            return True
    
    def stop(self) -> bool:
        """Stop the notification background thread"""
        with self.lock:
            self.is_running = False
            
            if self.notification_thread:
                self.notification_thread.join(timeout=2.0)
                self.notification_thread = None
            
            return True
    
    def _notification_loop(self) -> None:
        """Background thread for processing notifications"""
        while self.is_running:
            try:
                # Process notifications in queue
                with self.lock:
                    queue = self.notification_queue.copy()
                    self.notification_queue = []
                
                for item in queue:
                    alert, camera, screenshot_path = item
                    self._send_notifications(alert, camera, screenshot_path)
                
                # Sleep to avoid high CPU usage
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in notification thread: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(5.0)  # Sleep longer on error
    
    def notify(
        self,
        alert_id: int,
        channels: Optional[List[str]] = None
    ) -> bool:
        """
        Queue notification for an alert.
        
        Args:
            alert_id: Alert ID
            channels: Optional list of specific channels to notify
            
        Returns:
            True if notification was queued
        """
        try:
            # Get alert from database
            alert = self.db.query(Alert).filter(Alert.id == alert_id).first()
            if not alert:
                logger.warning(f"Alert {alert_id} not found for notification")
                return False
            
            # Get camera from database
            camera = self.db.query(Camera).filter(Camera.id == alert.camera_id).first()
            
            # Convert to dictionaries
            alert_dict = {
                "id": alert.id,
                "camera_id": alert.camera_id,
                "violation_type": alert.violation_type,
                "confidence": alert.confidence,
                "screenshot_path": alert.screenshot_path,
                "created_at": alert.created_at.isoformat() if alert.created_at else None,
                "resolved": alert.resolved
            }
            
            camera_dict = None
            if camera:
                camera_dict = {
                    "id": camera.id,
                    "name": camera.name,
                    "location": camera.location,
                    "url": camera.url
                }
            
            # Queue notification
            with self.lock:
                self.notification_queue.append((
                    alert_dict,
                    camera_dict,
                    alert.screenshot_path
                ))
            
            logger.info(f"Queued notification for alert {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error queueing notification for alert {alert_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _send_notifications(
        self,
        alert: Dict[str, Any],
        camera: Optional[Dict[str, Any]],
        screenshot_path: Optional[str],
        channels: Optional[List[str]] = None
    ) -> None:
        """
        Send notifications to all enabled channels.
        
        Args:
            alert: Alert data
            camera: Camera data
            screenshot_path: Path to screenshot
            channels: Optional list of specific channels to notify
        """
        try:
            # Get list of channels to notify
            channel_list = channels or list(self.channels.keys())
            
            # Send to each channel
            for channel_name in channel_list:
                if channel_name in self.channels:
                    channel = self.channels[channel_name]
                    
                    try:
                        success = channel.send_notification(alert, camera, screenshot_path)
                        if success:
                            logger.info(f"Sent {channel_name} notification for alert {alert.get('id')}")
                        else:
                            logger.warning(f"Failed to send {channel_name} notification for alert {alert.get('id')}")
                    except Exception as e:
                        logger.error(f"Error sending {channel_name} notification: {str(e)}")
                        logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error sending notifications: {str(e)}")
            logger.error(traceback.format_exc())
    
    def update_channel_config(self, channel_name: str, config: Dict[str, Any]) -> bool:
        """
        Update configuration for a notification channel.
        
        Args:
            channel_name: Channel name
            config: Channel configuration
            
        Returns:
            True if update was successful
        """
        with self.lock:
            # Check if channel exists in config
            if channel_name not in self.config:
                logger.warning(f"Channel {channel_name} not found in config")
                return False
            
            # Update config
            self.config[channel_name].update(config)
            
            # Save config
            if not self._save_config():
                return False
            
            # Re-initialize channels
            self._initialize_channels()
            
            logger.info(f"Updated configuration for {channel_name} channel")
            return True
    
    def get_channel_config(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a notification channel.
        
        Args:
            channel_name: Channel name
            
        Returns:
            Channel configuration or None if not found
        """
        with self.lock:
            return self.config.get(channel_name)
    
    def get_channels(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all notification channels and their status.
        
        Returns:
            Dictionary mapping channel names to status dictionaries
        """
        with self.lock:
            result = {}
            
            for name, channel in self.channels.items():
                result[name] = {
                    "enabled": channel.enabled,
                    "type": channel.__class__.__name__,
                    "throttle_seconds": channel.throttle_seconds
                }
            
            return result
    
    def test_channel(self, channel_name: str) -> bool:
        """
        Send a test notification to a channel.
        
        Args:
            channel_name: Channel name
            
        Returns:
            True if test was successful
        """
        with self.lock:
            if channel_name not in self.channels:
                logger.warning(f"Channel {channel_name} not found")
                return False
            
            # Create test alert
            test_alert = {
                "id": 0,
                "camera_id": 0,
                "violation_type": "test_notification",
                "confidence": 1.0,
                "created_at": datetime.now().isoformat()
            }
            
            # Create test camera
            test_camera = {
                "id": 0,
                "name": "Test Camera",
                "location": "Test Location",
                "url": ""
            }
            
            # Send test notification
            channel = self.channels[channel_name]
            success = channel.send_notification(test_alert, test_camera, None)
            
            if success:
                logger.info(f"Sent test notification to {channel_name} channel")
            else:
                logger.warning(f"Failed to send test notification to {channel_name} channel")
            
            return success


# Factory function to create notification service
def get_notification_service(db: Session) -> NotificationService:
    """Create notification service with database session."""
    return NotificationService(db)