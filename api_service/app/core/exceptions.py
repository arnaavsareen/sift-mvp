"""
Custom exceptions and exception handlers for the API service.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from jose.exceptions import JWTError

from app.core.logging import logger


class APIError(Exception):
    """Base class for API exceptions."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message


class NotFoundError(APIError):
    """Exception raised when a resource is not found."""
    def __init__(self, message: str = "Resource not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, message=message)


class UnauthorizedError(APIError):
    """Exception raised when a user is not authorized."""
    def __init__(self, message: str = "Not authorized"):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, message=message)


class ForbiddenError(APIError):
    """Exception raised when a user doesn't have permissions."""
    def __init__(self, message: str = "Forbidden"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, message=message)


class BadRequestError(APIError):
    """Exception raised for validation or client-side errors."""
    def __init__(self, message: str = "Bad request"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, message=message)


class ConflictError(APIError):
    """Exception raised when there's a conflict with existing data."""
    def __init__(self, message: str = "Conflict with existing resource"):
        super().__init__(status_code=status.HTTP_409_CONFLICT, message=message)


class RateLimitError(APIError):
    """Exception raised when rate limit is exceeded."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(status_code=status.HTTP_429_TOO_MANY_REQUESTS, message=message)


class ServiceUnavailableError(APIError):
    """Exception raised when a service is unavailable."""
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, message=message)


# Exception handlers

async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handler for custom API exceptions."""
    logger.error(f"API Error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": exc.message}
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handler for validation errors."""
    errors = []
    for error in exc.errors():
        # Extract field and error message
        location = " -> ".join([str(loc) for loc in error.get("loc", [])])
        message = error.get("msg", "Validation error")
        errors.append(f"{location}: {message}")
    
    error_message = "Validation error" if len(errors) == 0 else errors[0]
    logger.warning(f"Validation Error: {', '.join(errors)}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": error_message,
            "errors": errors
        }
    )


async def integrity_error_handler(request: Request, exc: IntegrityError) -> JSONResponse:
    """Handler for database integrity errors."""
    logger.error(f"Database Integrity Error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "success": False,
            "message": "Database conflict error. The resource might already exist."
        }
    )


async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """Handler for SQLAlchemy errors."""
    logger.error(f"Database Error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "Database error occurred"
        }
    )


async def jwt_error_handler(request: Request, exc: JWTError) -> JSONResponse:
    """Handler for JWT errors."""
    logger.error(f"JWT Error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={
            "success": False,
            "message": "Invalid or expired token"
        }
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for unhandled exceptions."""
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "Internal server error"
        }
    )
