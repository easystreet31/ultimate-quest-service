# logging_utils.py
"""
Structured logging utilities for Ultimate Quest Service.
Provides consistent, contextual logging across the app.
"""

import logging
import json
from typing import Any, Dict, Optional
from datetime import datetime
from functools import wraps
import time

import config

# ============================================================================
# Logger Setup
# ============================================================================

def setup_logging():
    """Initialize logging with structured format."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format=config.LOG_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)


# ============================================================================
# Standard Loggers
# ============================================================================

logger_main = get_logger("main")
logger_core = get_logger("app_core")
logger_routing = get_logger("routing_helpers")
logger_http = get_logger("http")
logger_cache = get_logger("cache")


# ============================================================================
# Helper Functions
# ============================================================================

def log_request(logger: logging.Logger, method: str, path: str, **kwargs):
    """Log an incoming HTTP request."""
    logger.info(
        f"{method} {path}",
        extra={
            "method": method,
            "path": path,
            **kwargs
        }
    )


def log_response(logger: logging.Logger, status_code: int, duration_ms: float, **kwargs):
    """Log a completed HTTP response."""
    level = logging.INFO if 200 <= status_code < 300 else logging.WARNING
    logger.log(
        level,
        f"Response {status_code} ({duration_ms:.1f}ms)",
        extra={
            "status_code": status_code,
            "duration_ms": duration_ms,
            **kwargs
        }
    )


def log_error(logger: logging.Logger, error_type: str, message: str, exc_info=False, **kwargs):
    """Log an error with context."""
    logger.error(
        f"{error_type}: {message}",
        exc_info=exc_info,
        extra={
            "error_type": error_type,
            "message": message,
            **kwargs
        }
    )


def log_computation(logger: logging.Logger, operation: str, num_players: int = 0, **kwargs):
    """Log a computation/business logic operation."""
    logger.info(
        f"Computing {operation}",
        extra={
            "operation": operation,
            "num_players": num_players,
            **kwargs
        }
    )


# ============================================================================
# Decorators
# ============================================================================

def timed_operation(operation_name: str, logger: logging.Logger = None):
    """Decorator to log operation timing."""
    if logger is None:
        logger = logger_core
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                logger.info(
                    f"{operation_name} completed",
                    extra={
                        "operation": operation_name,
                        "duration_ms": f"{duration_ms:.1f}",
                        "status": "success"
                    }
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                logger.error(
                    f"{operation_name} failed",
                    exc_info=True,
                    extra={
                        "operation": operation_name,
                        "duration_ms": f"{duration_ms:.1f}",
                        "status": "failed",
                        "error": str(e)
                    }
                )
                raise
        return wrapper
    return decorator


# ============================================================================
# Error Response Helper
# ============================================================================

def create_error_response(
    error_code: str,
    detail: str,
    status_code: int = 400,
    context: Optional[Dict[str, Any]] = None
) -> tuple[Dict[str, Any], int]:
    """
    Create a standardized error response.
    
    Args:
        error_code: Machine-readable error identifier (e.g., 'missing_urls')
        detail: Human-readable error message
        status_code: HTTP status code
        context: Optional additional context
    
    Returns:
        Tuple of (response_dict, status_code)
    """
    return {
        "ok": False,
        "error_code": error_code,
        "detail": detail,
        "timestamp": datetime.utcnow().isoformat(),
        "context": context or {}
    }, status_code


# ============================================================================
# Success Response Helper
# ============================================================================

def create_success_response(data: Dict[str, Any], **extra) -> Dict[str, Any]:
    """Create a standardized success response."""
    return {
        "ok": True,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
        **extra
    }
