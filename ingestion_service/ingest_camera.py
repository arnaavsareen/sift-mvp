#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import cv2
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------
REQUIRED_ENV_VARS = [
    "CAMERA_URL",
    "FRAME_INTERVAL",
    "S3_BUCKET",
    "SQS_QUEUE_URL",
    "AWS_REGION",
]

MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # exponential backoff multiplier

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("ingest_camera")

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def load_and_validate_env():
    """Load .env and validate required environment variables."""
    load_dotenv()
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        sys.exit(1)

    try:
        interval = float(os.getenv("FRAME_INTERVAL"))
        assert interval > 0
    except (ValueError, AssertionError):
        logger.error("FRAME_INTERVAL must be a positive number.")
        sys.exit(1)
    
    return {
        "camera_url": os.getenv("CAMERA_URL"),
        "frame_interval": interval,
        "s3_bucket": os.getenv("S3_BUCKET"),
        "sqs_url": os.getenv("SQS_QUEUE_URL"),
        "aws_region": os.getenv("AWS_REGION"),
    }


def create_boto_clients(region: str):
    """Create boto3 Session and clients for S3 and SQS."""
    session = boto3.Session(region_name=region)
    return session.client("s3"), session.client("sqs")


def open_video_source(url: str) -> cv2.VideoCapture:
    """Attempt to open the video source; retry as file if initial RTSP fails."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.warning(f"Cannot open camera '{url}', trying as file path.")
        cap = cv2.VideoCapture(str(Path(url)))
        if not cap.isOpened():
            logger.error(f"Unable to open video source '{url}'.")
            sys.exit(1)
    return cap


def safe_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def retry_operation(fn, *args, **kwargs):
    """Retry a function up to MAX_RETRIES with exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (BotoCoreError, ClientError) as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                logger.error("Max retries reached; giving up.")
                raise
            time.sleep(BACKOFF_FACTOR ** (attempt - 1))


# ------------------------------------------------------------------------------
# Main Ingestion Loop
# ------------------------------------------------------------------------------
def main():
    # Load config
    cfg = load_and_validate_env()
    logger.info(f"Configuration: {cfg}")

    # Prepare AWS clients
    s3_client, sqs_client = create_boto_clients(cfg["aws_region"])

    # Prepare local frame directory
    frames_dir = Path("frames")
    frames_dir.mkdir(exist_ok=True)

    # Open video capture
    cap = open_video_source(cfg["camera_url"])
    is_file_source = Path(cfg["camera_url"]).is_file()

    logger.info("Starting ingestion pipeline.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_file_source:
                    logger.info("End of video file reached; rewinding.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.error("Failed to read frame from camera; exiting loop.")
                    break

            ts = safe_timestamp()
            local_path = frames_dir / f"{ts}.jpg"
            cv2.imwrite(str(local_path), frame)
            logger.info(f"Captured frame {local_path.name}")

            s3_key = f"frames/{local_path.name}"
            try:
                retry_operation(s3_client.upload_file, str(local_path), cfg["s3_bucket"], s3_key)
                logger.info(f"Uploaded to S3: s3://{cfg['s3_bucket']}/{s3_key}")
            except Exception:
                # already logged in retry_operation
                local_path.unlink(missing_ok=True)
                time.sleep(cfg["frame_interval"])
                continue

            message = json.dumps({"s3_key": s3_key})
            try:
                retry_operation(sqs_client.send_message, QueueUrl=cfg["sqs_url"], MessageBody=message)
                logger.info(f"Enqueued SQS message: {message}")
            except Exception:
                # already logged in retry_operation
                pass

            # Cleanup local file
            try:
                local_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {local_path}: {e}")

            time.sleep(cfg["frame_interval"])

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; shutting down.")
    finally:
        cap.release()
        logger.info("Video capture released. Ingestion service stopped.")


if __name__ == "__main__":
    main()
