"""Custom middleware for the API."""

import time
import logging
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("diffusion_boltzmann.api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and timing."""

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Log request and measure response time."""
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time
        duration_ms = round(duration * 1000, 2)

        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"- {response.status_code} ({duration_ms}ms)"
        )

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for catching and formatting unhandled exceptions."""

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Catch exceptions and return formatted error responses."""
        try:
            return await call_next(request)
        except Exception as exc:
            logger.exception(f"Unhandled exception: {exc}")

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(exc),
                    "path": str(request.url.path),
                },
            )


def setup_middleware(app) -> None:
    """Configure all middleware for the application.

    Args:
        app: FastAPI application instance
    """
    # Order matters - error handling should wrap everything
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
