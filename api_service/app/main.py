"""
Main application entry point for the SIFT API Service.
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import time
from contextlib import asynccontextmanager

from app.api.api_v1.api import api_router
from app.core.config import settings
from app.core.exceptions import (
    APIError, 
    api_error_handler,
    validation_error_handler,
    integrity_error_handler,
    sqlalchemy_error_handler,
    jwt_error_handler,
    generic_error_handler
)
from app.core.logging import logger, setup_logging
from app.db.database import Base, engine, SessionLocal


# Setup application logging
setup_logging()


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events for the application.
    """
    # Startup: Create database tables if they don't exist
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Startup complete
    logger.info("SIFT API Service started")
    yield
    
    # Shutdown: Clean up resources
    logger.info("SIFT API Service shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url=None,  # Disable default docs URL
    redoc_url=None,  # Disable default redoc URL
    lifespan=lifespan
)


# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handlers
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(Exception, generic_error_handler)


# Performance middleware to log request processing time
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    """
    Middleware to log request processing time.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.debug(f"Request {request.method} {request.url.path} processed in {process_time:.4f}s")
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)


# Custom documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Custom Swagger UI endpoint.
    """
    return get_swagger_ui_html(
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        title=f"{settings.PROJECT_NAME} - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )


# Root endpoint
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Root endpoint for the API service.
    """
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "message": "Welcome to the SIFT API Service",
        "docs_url": "/docs",
        "api_prefix": settings.API_V1_STR
    }


# Add a WebSocket connection endpoint for real-time notifications
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """
    WebSocket endpoint for real-time notifications.
    This is a placeholder for future implementation.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    # Use this for development only
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )
