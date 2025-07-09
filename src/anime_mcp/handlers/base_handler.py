"""
Base handler for anime MCP operations.

Following modern MCP architecture patterns for separation of concerns.
"""

import logging
from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context

from ...config import Settings
from ...exceptions import (
    AnimeServerError,
    ClientNotInitializedError,
    DatabaseOperationError,
)
from ...vector.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class BaseAnimeHandler:
    """Base handler providing common functionality for anime operations."""

    def __init__(self, qdrant_client: Optional[QdrantClient], settings: Settings):
        """Initialize base handler.

        Args:
            qdrant_client: Qdrant vector database client
            settings: Application settings
        """
        self.qdrant_client = qdrant_client
        self.settings = settings

    def verify_client(self, operation: str) -> QdrantClient:
        """Verify Qdrant client is available.

        Args:
            operation: Name of operation requiring client

        Returns:
            QdrantClient instance

        Raises:
            ClientNotInitializedError: If client not initialized
        """
        if not self.qdrant_client:
            raise ClientNotInitializedError(operation)
        return self.qdrant_client

    async def handle_error(
        self, error: Exception, operation: str, ctx: Optional[Context] = None
    ) -> None:
        """Handle errors with proper logging and context reporting.

        Args:
            error: Exception that occurred
            operation: Operation name for context
            ctx: Optional MCP context for reporting
        """
        error_msg = str(error)

        if ctx:
            await ctx.error(f"{operation} failed: {error_msg}")

        logger.error(f"{operation} error: {error_msg}")

        # Re-raise as domain-specific error if not already
        if not isinstance(error, AnimeServerError):
            raise DatabaseOperationError(operation, error_msg) from error
        raise

    async def log_operation_start(
        self, operation: str, details: str, ctx: Optional[Context] = None
    ) -> None:
        """Log operation start with context.

        Args:
            operation: Operation name
            details: Operation details
            ctx: Optional MCP context
        """
        if ctx:
            await ctx.info(f"Starting {operation}: {details}")
        logger.info(f"{operation} request: {details}")

    async def log_operation_success(
        self, operation: str, result_count: int, ctx: Optional[Context] = None
    ) -> None:
        """Log successful operation completion.

        Args:
            operation: Operation name
            result_count: Number of results returned
            ctx: Optional MCP context
        """
        message = f"{operation} completed: {result_count} results"
        if ctx:
            await ctx.info(message)
        logger.info(message)

    def validate_limit(self, limit: int, max_limit: int) -> int:
        """Validate and clamp limit parameter.

        Args:
            limit: Requested limit
            max_limit: Maximum allowed limit

        Returns:
            Validated limit value
        """
        return min(max(1, limit), max_limit)

    def check_multi_vector_support(self, operation: str) -> bool:
        """Check if multi-vector support is available.

        Args:
            operation: Operation name for error context

        Returns:
            True if multi-vector is supported

        Raises:
            MultiVectorNotAvailableError: If not available
        """
        from ...exceptions import MultiVectorNotAvailableError

        if not self.qdrant_client or not getattr(
            self.qdrant_client, "_supports_multi_vector", False
        ):
            raise MultiVectorNotAvailableError(operation)
        return True

    def create_success_response(self, data: Any) -> Dict[str, Any]:
        """Create standardized success response.

        Args:
            data: Response data

        Returns:
            Formatted response dictionary
        """
        return {"success": True, "data": data}

    def create_error_response(self, error: AnimeServerError) -> Dict[str, Any]:
        """Create standardized error response.

        Args:
            error: Anime server error

        Returns:
            Formatted error response
        """
        return {
            "success": False,
            "error": {
                "code": error.error_code,
                "operation": getattr(error, "operation", "unknown"),
                "message": str(error),
            },
        }
