"""
FastAPI application and endpoints for the SIFT Model Service.
"""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.database import get_db
from app.models.schemas import (
    DetectionResponse, 
    ProcessImageRequest,
    HealthResponse,
    QueueStatusResponse,
    ProcessorStatusResponse
)
from app.models.detection import Detection
from app.models.model_handler import PPEDetector
from app.services.aws import S3Service, SQSService
from app.services.queue_manager import (
    queue_manager,
    initialize_queue_manager,
    start_processing,
    stop_processing,
    get_processor_status,
    run_processor_once,
    health_check
)

# Setup logging
logger = setup_logging()

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Ensure required directories exist
    os.makedirs("./temp", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Log startup information
    logger.info(f"Starting {settings.PROJECT_NAME}")
    logger.info(f"API version: {settings.API_V1_STR}")
    logger.info(f"Debug mode: {settings.DEBUG_MODE}")
    logger.info(f"SQS Queue: {settings.SQS_QUEUE_URL}")
    logger.info(f"S3 Bucket: {settings.S3_BUCKET_NAME}")
    
    # Initialize queue manager
    await initialize_queue_manager()
    
    # Start processing automatically if configured
    if getattr(settings, "AUTO_START_PROCESSING", False):
        logger.info("Auto-starting SQS processing")
        await start_processing()

# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("API shutting down, stopping processor")
    await stop_processing()

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def api_health_check():
    """
    Health check endpoint for the API service.
    
    This endpoint checks if the API itself is running properly.
    For SQS processor health check, use /api/v1/processor/health.
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now()
    }

# Process image API endpoint
@app.post(f"{settings.API_V1_STR}/process", response_model=DetectionResponse, tags=["Detection"])
async def process_image(
    request: ProcessImageRequest,
    db: Session = Depends(get_db)
):
    """
    Process an image for PPE detection.
    
    This endpoint allows manual processing of an image from S3.
    """
    logger.info(f"Processing image request: {request.image_id} from {request.s3_path}")
    
    # Initialize services
    s3_service = S3Service()
    detector = PPEDetector(
        model_path=settings.MODEL_PATH,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD
    )
    
    # Download image from S3
    temp_image_path = f"./temp/{request.image_id}.jpg"
    s3_key = s3_service.get_s3_path_from_full_url(request.s3_path)
    
    if not s3_service.download_image(s3_key, temp_image_path):
        raise HTTPException(status_code=404, detail=f"Failed to download image from S3: {request.s3_path}")
    
    # Process image
    try:
        # Detect PPE violations
        violations, detections = detector.detect_violations(temp_image_path)
        
        # Generate summary
        detection_summary = detector.get_summary(violations)
        processing_time = 0.0  # We don't have this from the detector currently
        
        # Save annotated image
        annotated_path = f"./temp/{request.image_id}_annotated.jpg"
        detector.save_annotated_image(
            image=temp_image_path,
            output_path=annotated_path,
            detections=detections,
            violations=violations,
            show_all_detections=True
        )
        
        # Upload annotated image to S3
        annotated_s3_key = f"annotated/{request.image_id}.jpg"
        s3_service.s3_client.upload_file(
            annotated_path,
            s3_service.bucket_name,
            annotated_s3_key
        )
        
        # Format detection results for database
        detection_results = {
            "timestamp": datetime.now().isoformat(),
            "image_id": request.image_id,
            "original_path": request.s3_path,
            "annotated_path": f"s3://{s3_service.bucket_name}/{annotated_s3_key}",
            "source_id": request.source_id,
            "source_type": request.source_type,
            "num_detections": len(detections),
            "num_violations": len(violations),
            "ppe_detected": detection_summary["has_violations"],
            "violations": detection_summary["violation_counts"],
            "confidence_threshold": detector.confidence_threshold,
            "model_version": detector.get_model_version(),
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
        
        # Create database record
        detection = Detection(
            image_id=request.image_id,
            image_path=request.s3_path,
            source_id=request.source_id,
            source_type=request.source_type,
            num_detections=len(detections),
            ppe_detected=detection_summary["has_violations"],
            violations_detected=detection_summary["has_violations"],
            detection_results=detection_results,
            confidence_threshold=detector.confidence_threshold,
            model_version=detector.get_model_version(),
            processing_time=processing_time
        )
        
        db.add(detection)
        db.commit()
        db.refresh(detection)
        
        # Prepare response
        response = {
            "id": detection.id,
            "image_id": detection.image_id,
            "timestamp": detection.timestamp,
            "ppe_detected": detection.ppe_detected,
            "violations_detected": detection.violations_detected,
            "num_detections": detection.num_detections,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": d.bbox
                } for d in detections
            ],
            "processing_time": processing_time
        }
        
        # Clean up temporary files
        try:
            os.remove(temp_image_path)
            os.remove(annotated_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary images: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

#
# SQS Processor Endpoints
#

# Processor health check endpoint
@app.get(f"{settings.API_V1_STR}/processor/health", tags=["Processor"])
async def processor_health():
    """
    Check the health of the SQS processor pipeline.
    
    This endpoint performs a comprehensive health check of the entire
    processing pipeline, including SQS queue, model, and database.
    """
    return await health_check()

# Start processor endpoint
@app.post(f"{settings.API_V1_STR}/processor/start", tags=["Processor"])
async def start_processor(
    background_tasks: BackgroundTasks,
    background: bool = Query(True, description="Whether to start the processor in the background")
):
    """
    Start the SQS processor to continuously process messages.
    
    This endpoint starts the SQS processor in the background,
    which will continuously poll the queue for new messages.
    """
    if background:
        return await start_processing(background_tasks)
    else:
        return await start_processing()

# Stop processor endpoint
@app.post(f"{settings.API_V1_STR}/processor/stop", tags=["Processor"])
async def stop_processor():
    """
    Stop the SQS processor.
    
    This endpoint stops the SQS processor, which will stop
    polling the queue for new messages.
    """
    return await stop_processing()

# Get processor status endpoint
@app.get(f"{settings.API_V1_STR}/processor/status", response_model=ProcessorStatusResponse, tags=["Processor"])
async def processor_status():
    """
    Get the status of the SQS processor.
    
    This endpoint returns the current status of the SQS processor,
    including metrics and configuration.
    """
    return await get_processor_status()

# Process a batch of messages endpoint
@app.post(f"{settings.API_V1_STR}/processor/run-once", tags=["Processor"])
async def run_once(
    max_messages: int = Query(10, description="Maximum number of messages to process"),
    wait_time: int = Query(5, description="SQS long-polling wait time in seconds")
):
    """
    Process a batch of messages from the SQS queue once.
    
    This endpoint processes a single batch of messages from the SQS queue,
    without starting the continuous processing loop.
    """
    return await run_processor_once(
        max_messages=max_messages,
        wait_time=wait_time
    )

# Process queue endpoint (legacy, for backward compatibility)
@app.post(f"{settings.API_V1_STR}/process-queue", tags=["Queue"])
async def process_queue(
    background_tasks: BackgroundTasks,
    max_messages: int = Query(10, description="Maximum number of messages to process")
):
    """
    Process messages from the SQS queue once in the background.
    
    This endpoint is provided for backward compatibility.
    For continuous processing, use /processor/start.
    """
    logger.info(f"Processing queue for up to {max_messages} messages (legacy endpoint)")
    
    # Run processor once in the background
    async def _run_once():
        await run_processor_once(max_messages=max_messages)
    
    background_tasks.add_task(_run_once)
    
    return {"status": "processing_started", "max_messages": max_messages}
