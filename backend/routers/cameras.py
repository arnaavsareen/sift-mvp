from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
from fastapi.responses import StreamingResponse, FileResponse
import os
from pathlib import Path
import time
import tempfile
import subprocess
import uuid
import shutil
import httpx

from backend.database import get_db
from backend.models import Camera
from backend.services.detection import get_detection_service
from backend.services.alert import get_alert_service
from backend.services.video import start_processor, stop_processor, get_processor, get_all_processors
from backend.services.config_service import get_config_service

# Pydantic models for request/response
from pydantic import BaseModel
from datetime import datetime

class CameraBase(BaseModel):
    name: str
    url: str
    location: Optional[str] = None
    is_active: bool = True

class CameraCreate(CameraBase):
    pass

class CameraResponse(CameraBase):
    id: int
    created_at: datetime
    
    class Config:
        orm_mode = True

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    location: Optional[str] = None
    is_active: Optional[bool] = None


# Create router
router = APIRouter()
logger = logging.getLogger(__name__)

# Global storage for active HLS streams
_active_hls_streams = {}

# Check if FFmpeg is available
def is_ffmpeg_available():
    """Check if FFmpeg is available on the system."""
    try:
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        return result.returncode == 0
    except Exception:
        return False

_FFMPEG_AVAILABLE = is_ffmpeg_available()
logger.info(f"FFmpeg available: {_FFMPEG_AVAILABLE}")

@router.post("/", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(camera: CameraCreate, db: Session = Depends(get_db)):
    """Create a new camera."""
    db_camera = Camera(**camera.dict())
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    return db_camera

@router.get("/", response_model=List[CameraResponse])
def get_cameras(
    skip: int = 0, 
    limit: int = 100, 
    active_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get all cameras."""
    query = db.query(Camera)
    
    if active_only:
        query = query.filter(Camera.is_active == True)
    
    cameras = query.offset(skip).limit(limit).all()
    return cameras

@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: int, db: Session = Depends(get_db)):
    """Get camera by ID."""
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera

@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(
    camera_id: int, 
    camera_update: CameraUpdate, 
    db: Session = Depends(get_db)
):
    """Update camera."""
    db_camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Update fields
    update_data = camera_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_camera, key, value)
    
    db.commit()
    db.refresh(db_camera)
    return db_camera

@router.delete("/{camera_id}", response_model=dict)
def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    """Delete camera."""
    # First stop processing if running
    processor = get_processor(camera_id)
    if processor:
        stop_processor(camera_id)
    
    # Then delete from database
    db_camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Get config service and delete camera configuration
    config_service = get_config_service(db)
    config_service.delete_camera_config(camera_id)
    
    # Delete the camera from the database
    db.delete(db_camera)
    db.commit()
    
    # Return a success message
    return {"status": "success", "message": f"Camera {camera_id} deleted successfully"}

@router.post("/{camera_id}/start", response_model=dict)
def start_camera(camera_id: int, db: Session = Depends(get_db)):
    """Start processing camera feed."""
    # Get camera from database
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Check if already processing
    if get_processor(camera_id):
        return {"status": "already_running", "camera_id": camera_id}
    
    # Get services
    detection_service = get_detection_service()
    alert_service = get_alert_service(db)
    
    # Start processor
    success = start_processor(
        camera_id=camera.id,
        camera_url=camera.url,
        detection_service=detection_service,
        alert_service=alert_service
    )
    
    if not success:
        raise HTTPException(
            status_code=500, 
            detail="Failed to start camera processing"
        )
    
    return {"status": "started", "camera_id": camera_id}

@router.post("/{camera_id}/stop", response_model=dict)
def stop_camera(camera_id: int, db: Session = Depends(get_db)):
    """Stop processing camera feed."""
    # Check if camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Stop processor
    success = stop_processor(camera_id)
    
    if not success:
        return {"status": "not_running", "camera_id": camera_id}
    
    return {"status": "stopped", "camera_id": camera_id}

@router.get("/{camera_id}/status", response_model=dict)
def get_camera_status(camera_id: int, db: Session = Depends(get_db)):
    """Get camera processing status."""
    # Check if camera exists
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Get processor
    processor = get_processor(camera_id)
    
    return {
        "camera_id": camera_id,
        "is_processing": processor is not None,
        "frame_count": getattr(processor, "frame_count", 0) if processor else 0,
        "last_frame_time": getattr(processor, "last_frame_time", None)
    }

@router.get("/{camera_id}/stream")
async def stream_video(
    camera_id: int, 
    format: str = Query("video", description="Format to stream - 'video' for direct streaming, 'hls' for HLS streaming"),
    db: Session = Depends(get_db)
):
    """Stream camera feed.
    
    Args:
        camera_id: Camera ID
        format: Streaming format - 'video' for direct streaming, 'hls' for HLS streaming
    """
    try:
        # Get camera from database
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        # Get URL from camera
        url = camera.url
        logger.info(f"Streaming camera {camera_id} with URL: {url} in format: {format}")
        
        # Handle file:/// URLs - Direct file streaming
        if url.startswith("file:///"):
            # Extract the file path from the file:/// URL
            file_path = url.replace("file:///", "/")
            
            # Check if file exists
            logger.info(f"Checking for file at path: {file_path}")
            if not os.path.exists(file_path):
                logger.error(f"Video file not found: {file_path}")
                raise HTTPException(status_code=404, detail=f"Video file not found: {file_path}")
            
            # Define a generator function to yield video chunks
            def video_generator():
                try:
                    with open(file_path, mode="rb") as file_like:
                        logger.info(f"Starting to stream file: {file_path}")
                        chunk_size = 1024 * 1024  # 1MB chunks
                        while True:
                            chunk = file_like.read(chunk_size)
                            if not chunk:
                                break
                            yield chunk
                except Exception as e:
                    logger.error(f"Error streaming file {file_path}: {str(e)}")
                    raise
            
            # Get the file's media type based on extension
            file_ext = Path(file_path).suffix.lower()
            media_type = "video/mp4"  # Default to MP4
            
            if file_ext == ".mp4":
                media_type = "video/mp4"
            elif file_ext == ".webm":
                media_type = "video/webm"
            elif file_ext == ".mov":
                media_type = "video/quicktime"
            elif file_ext == ".avi":
                media_type = "video/x-msvideo"
            
            logger.info(f"Streaming {file_path} as {media_type}")
            
            # Return a streaming response
            return StreamingResponse(
                video_generator(),
                media_type=media_type,
                headers={
                    "Accept-Ranges": "bytes",
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
        
        # Handle HTTP/HTTPS URLs - Proxy the content
        elif url.startswith("http://") or url.startswith("https://"):
            logger.info(f"Proxying HTTP(S) video content: {url}")
            
            # Determine content type from URL extension
            media_type = "video/mp4"  # Default to MP4
            path = url.split("?")[0]  # Remove query parameters
            
            if path.endswith(".mp4"):
                media_type = "video/mp4"
            elif path.endswith(".webm"):
                media_type = "video/webm"
            elif path.endswith(".mov"):
                media_type = "video/quicktime"
            elif path.endswith(".avi"):
                media_type = "video/x-msvideo"
            elif path.endswith(".m3u8"):
                media_type = "application/vnd.apple.mpegurl"
            
            # Use httpx to stream the content
            async def stream_http_video():
                try:
                    timeout = httpx.Timeout(15.0, connect=10.0)
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        # First make a HEAD request to check if the resource exists
                        head_response = await client.head(url)
                        head_response.raise_for_status()
                        
                        # Then do a streaming GET request
                        async with client.stream("GET", url) as response:
                            response.raise_for_status()
                            
                            # Stream the content in chunks
                            async for chunk in response.aiter_bytes(chunk_size=1024*1024):
                                yield chunk
                                
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error while streaming video: {e.response.status_code} - {str(e)}")
                    yield f"Error: HTTP {e.response.status_code}".encode()
                except httpx.RequestError as e:
                    logger.error(f"Request error while streaming video: {str(e)}")
                    yield b"Error: Failed to connect to video source"
                except Exception as e:
                    logger.error(f"Error streaming HTTP video: {str(e)}")
                    yield b"Error: Unknown error occurred while streaming video"
            
            logger.info(f"Streaming {url} as {media_type}")
            
            # Return streaming response
            return StreamingResponse(
                stream_http_video(),
                media_type=media_type,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
        
        # Handle RTSP URLs (requires ffmpeg to be installed)
        elif url.startswith("rtsp://"):
            # Check if ffmpeg is available for HLS conversion
            _FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
            
            # If requested format is HLS, try to convert RTSP to HLS
            if format == "hls":
                if not _FFMPEG_AVAILABLE:
                    logger.warning("FFmpeg not available, cannot convert RTSP to HLS")
                    # If HLS is requested but not available, return an error with a message to try direct mode
                    return {
                        "status": "error", 
                        "message": "FFmpeg not available on server. HLS conversion not possible.",
                        "url": url,  # Return original URL as fallback
                        "tryDirectMode": True  # Hint to client to try direct mode
                    }
                    
                # Create a unique session ID for this stream
                session_id = str(uuid.uuid4())
                
                # Create a temporary directory for this stream
                temp_dir = Path(tempfile.gettempdir()) / f"rtsp_stream_{session_id}"
                os.makedirs(temp_dir, exist_ok=True)
                
                # HLS playlist file
                playlist_file = temp_dir / "playlist.m3u8"
                
                try:
                    # Start ffmpeg process to convert RTSP to HLS
                    # This runs in the background and writes HLS segments to the temp directory
                    process = subprocess.Popen([
                        "ffmpeg",
                        "-i", url,
                        "-c:v", "copy",  # Copy video stream without re-encoding
                        "-c:a", "copy",  # Copy audio stream without re-encoding
                        "-f", "hls",
                        "-hls_time", "2",  # 2-second segments
                        "-hls_list_size", "10",  # Keep 10 segments in the playlist
                        "-hls_flags", "delete_segments",  # Delete old segments
                        "-hls_segment_filename", f"{temp_dir}/segment_%03d.ts",
                        "-method", "PUT",
                        "-loglevel", "warning",  # Reduce log verbosity
                        str(playlist_file)
                    ], stderr=subprocess.PIPE)
                    
                    # Check if process started successfully
                    if process.poll() is not None:
                        error_output = process.stderr.read().decode("utf-8")
                        logger.error(f"Failed to start HLS conversion: {error_output}")
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Failed to start HLS conversion process: {error_output}"
                        )
                    
                    # Return the playlist file URL
                    logger.info(f"Started HLS conversion for RTSP stream: {url}")
                    return {"status": "success", "url": f"/api/cameras/hls/{session_id}/playlist.m3u8"}
                    
                except Exception as e:
                    logger.error(f"Error setting up HLS conversion: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # If HLS conversion fails, suggest direct streaming as fallback
                    return {
                        "status": "error", 
                        "message": f"Error setting up HLS conversion: {str(e)}",
                        "tryDirectMode": True
                    }
            
            # For direct streaming of RTSP (when format=video), proxy through ffmpeg
            elif format == "video" and _FFMPEG_AVAILABLE:
                # Stream RTSP directly using FFmpeg to convert to MJPEG
                return StreamingResponse(
                    stream_rtsp_mjpeg(url),
                    media_type="multipart/x-mixed-replace; boundary=frame",
                    headers={
                        "Cache-Control": "no-cache, no-store, must-revalidate",
                        "Connection": "close",
                        "Pragma": "no-cache",
                        "Expires": "0"
                    }
                )
            else:
                # For other formats or if ffmpeg is not available, return the original URL
                # Let the client decide how to handle it
                return {
                    "status": "info",
                    "message": "Direct RTSP streaming requested. RTSP URLs must be handled by the client directly.",
                    "url": url
                }
        else:
            # For other URL types we don't recognize, throw an error
            logger.warning(f"Unsupported URL format for streaming: {url}")
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported URL format for streaming: {url}. Only file://, http(s):// and rtsp:// are supported."
            )
    except Exception as e:
        logger.error(f"Error in stream_video for camera {camera_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Function to convert RTSP stream to MJPEG for direct browser viewing
async def stream_rtsp_mjpeg(rtsp_url):
    """
    Stream RTSP as Motion JPEG for direct browser viewing.
    Uses FFmpeg to decode the RTSP stream and converts each frame to JPEG.
    """
    import asyncio
    import shlex
    
    cmd = f"ffmpeg -i {shlex.quote(rtsp_url)} -f image2pipe -vcodec mjpeg -q:v 5 -"
    
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Read FFmpeg's output and yield frames
    try:
        # Buffer for collecting frame data
        frame_buffer = bytearray()
        jpeg_start = b'\xff\xd8'
        jpeg_end = b'\xff\xd9'
        
        # Process FFmpeg output
        while True:
            # Read chunk from ffmpeg's stdout
            chunk = await process.stdout.read(4096)
            if not chunk:
                break
                
            # Add chunk to our buffer
            frame_buffer.extend(chunk)
            
            # Check if we have a complete JPEG frame
            start_idx = frame_buffer.find(jpeg_start)
            if start_idx >= 0:
                # Look for the end marker after the start marker
                end_idx = frame_buffer.find(jpeg_end, start_idx)
                
                # If we found a complete frame
                if end_idx >= 0:
                    # Extract the frame (including end marker)
                    frame = frame_buffer[start_idx:end_idx+2]
                    
                    # Clear buffer up to the end of the frame we just extracted
                    frame_buffer = frame_buffer[end_idx+2:]
                    
                    # Yield the frame in multipart format
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    
    finally:
        # Ensure we clean up the subprocess
        if process.returncode is None:
            process.kill()
            await process.wait()
            
# Improved HLS streaming endpoint - ensure it's accessible through API
@router.get("/hls/{session_id}/{file_name}")
async def serve_hls_stream(session_id: str, file_name: str):
    """Serve HLS streaming files for converted RTSP streams.
    
    Args:
        session_id: Unique session ID for the HLS stream
        file_name: Name of the HLS file (playlist.m3u8 or segment_xxx.ts)
    """
    # Global dictionary to track active HLS streams
    global _active_hls_streams
    if not hasattr(serve_hls_stream, "_active_hls_streams"):
        serve_hls_stream._active_hls_streams = {}
    
    # Get the path to the HLS file
    temp_dir = Path(tempfile.gettempdir()) / f"rtsp_stream_{session_id}"
    
    # Check if the directory exists
    if not temp_dir.exists():
        logger.error(f"HLS stream directory not found: {temp_dir}")
        raise HTTPException(status_code=404, detail="HLS stream not found")
    
    file_path = temp_dir / file_name
    
    # Check if the requested file exists
    if not file_path.exists():
        logger.error(f"HLS file not found: {file_path}")
        raise HTTPException(status_code=404, detail=f"HLS file not found: {file_name}")
    
    # Update last access time for this stream
    serve_hls_stream._active_hls_streams[session_id] = time.time()
    
    # Clean up old streams
    current_time = time.time()
    for sid, last_access in list(serve_hls_stream._active_hls_streams.items()):
        # If a stream hasn't been accessed in over 5 minutes, clean it up
        if current_time - last_access > 300:  # 5 minutes
            stream_dir = Path(tempfile.gettempdir()) / f"rtsp_stream_{sid}"
            try:
                # Try to delete the directory and its contents
                shutil.rmtree(stream_dir, ignore_errors=True)
                logger.info(f"Cleaned up HLS stream directory for session {sid}")
            except Exception as e:
                logger.error(f"Error cleaning up HLS stream directory: {str(e)}")
            
            # Remove from tracking dictionary
            del serve_hls_stream._active_hls_streams[sid]
    
    # Determine the content type
    content_type = "application/vnd.apple.mpegurl" if file_name.endswith(".m3u8") else "video/mp2t"
    
    # Read and return the file
    return FileResponse(
        file_path, 
        media_type=content_type,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )