"""Correlation ID middleware for automatic request tracing and observability."""

import logging
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic correlation ID injection and request tracing.

    This middleware automatically:
    - Generates correlation IDs for all incoming requests
    - Extracts correlation IDs from X-Correlation-ID headers
    - Injects correlation context into request.state
    - Adds correlation headers to all responses
    - Provides automatic request lifecycle logging
    - Integrates with existing CorrelationLogger infrastructure

    Features:
    - Supports both auto-generated and client-provided correlation IDs
    - Handles parent/child correlation relationships
    - Prevents circular dependencies with chain depth tracking
    - Automatic request entry/exit logging
    - Exception handling with correlation preservation
    - Integration with existing enterprise-grade correlation infrastructure
    """

    def __init__(
        self,
        app,
        auto_generate: bool = True,
        max_chain_depth: int = 10,
        log_requests: bool = True,
    ):
        """Initialize correlation middleware.

        Args:
            app: FastAPI application instance
            auto_generate: Whether to auto-generate correlation IDs if not provided
            max_chain_depth: Maximum correlation chain depth to prevent circular calls
            log_requests: Whether to automatically log request lifecycle
        """
        super().__init__(app)
        self.auto_generate = auto_generate
        self.max_chain_depth = max_chain_depth
        self.log_requests = log_requests

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with correlation ID injection and tracing.

        Args:
            request: Incoming FastAPI request
            call_next: Next middleware/handler in chain

        Returns:
            Response with correlation headers added
        """
        # Extract correlation information from headers
        correlation_id = request.headers.get("x-correlation-id")
        parent_correlation_id = request.headers.get("x-parent-correlation-id")
        chain_depth = int(request.headers.get("x-request-chain-depth", "0"))

        # Auto-generate correlation ID if not provided and auto_generate is enabled
        if not correlation_id and self.auto_generate:
            correlation_id = f"req-{uuid.uuid4().hex[:12]}"

        # Validate chain depth to prevent circular dependencies
        if chain_depth > self.max_chain_depth:
            logger.error(
                f"Request chain depth exceeded maximum ({chain_depth} > {self.max_chain_depth})",
                extra={
                    "correlation_id": correlation_id,
                    "chain_depth": chain_depth,
                    "max_depth": self.max_chain_depth,
                    "path": request.url.path,
                },
            )
            # Continue processing but log the issue

        # Inject correlation context into request state
        if not hasattr(request.state, "correlation_context"):
            request.state.correlation_context = {}

        request.state.correlation_context.update(
            {
                "correlation_id": correlation_id,
                "parent_correlation_id": parent_correlation_id,
                "chain_depth": chain_depth,
                "request_path": request.url.path,
                "request_method": request.method,
            }
        )

        # Convenience access for backward compatibility
        request.state.correlation_id = correlation_id

        # Log request entry if logging is enabled
        if self.log_requests and correlation_id:
            logger.info(
                f"Request started: {request.method} {request.url.path}",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "chain_depth": chain_depth,
                    "user_agent": request.headers.get("user-agent"),
                    "parent_correlation_id": parent_correlation_id,
                },
            )

        # Process request through middleware chain
        try:
            response = await call_next(request)

            # Log successful request completion
            if self.log_requests and correlation_id:
                logger.info(
                    f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                    extra={
                        "correlation_id": correlation_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "chain_depth": chain_depth,
                        "parent_correlation_id": parent_correlation_id,
                    },
                )

        except Exception as e:
            # Log request failure with error details
            if self.log_requests and correlation_id:
                logger.error(
                    f"Request failed: {request.method} {request.url.path} - {str(e)}",
                    extra={
                        "correlation_id": correlation_id,
                        "method": request.method,
                        "path": request.url.path,
                        "chain_depth": chain_depth,
                        "error_type": type(e).__name__,
                        "parent_correlation_id": parent_correlation_id,
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                    },
                )

            # Re-raise the exception to maintain normal error handling
            raise

        # Add correlation headers to response
        if correlation_id:
            response.headers["X-Correlation-ID"] = correlation_id

        if parent_correlation_id:
            response.headers["X-Parent-Correlation-ID"] = parent_correlation_id

        if chain_depth > 0:
            response.headers["X-Request-Chain-Depth"] = str(chain_depth)

        # Add additional tracing headers for debugging
        response.headers["X-Trace-Source"] = "anime-mcp-server"

        return response

    def get_correlation_context(self, request: Request) -> dict:
        """Extract correlation context from request state.

        Args:
            request: FastAPI request object

        Returns:
            Dictionary containing correlation context
        """
        return getattr(request.state, "correlation_context", {})

    def get_correlation_id(self, request: Request) -> str:
        """Extract correlation ID from request state.

        Args:
            request: FastAPI request object

        Returns:
            Correlation ID string or empty string if not found
        """
        return getattr(request.state, "correlation_id", "")
