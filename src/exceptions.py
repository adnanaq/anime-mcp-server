"""
Custom exception classes for Anime MCP Server

Provides specific exception types for better error handling and debugging
across the anime search and recommendation system.
"""

import asyncio
from typing import Any, Dict, Optional


class AnimeServerError(Exception):
    """Base exception class for all anime server errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize anime server error.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class VectorDatabaseError(AnimeServerError):
    """Base exception for vector database related errors."""

    pass


class QdrantConnectionError(VectorDatabaseError):
    """Raised when Qdrant connection fails or times out."""

    def __init__(self, url: str, message: Optional[str] = None):
        self.url = url
        default_message = f"Failed to connect to Qdrant at {url}"
        super().__init__(
            message or default_message,
            error_code="QDRANT_CONNECTION_FAILED",
            details={"qdrant_url": url},
        )


class QdrantCollectionError(VectorDatabaseError):
    """Raised when Qdrant collection operations fail."""

    def __init__(
        self, collection_name: str, operation: str, message: Optional[str] = None
    ):
        self.collection_name = collection_name
        self.operation = operation
        default_message = f"Collection '{collection_name}' {operation} operation failed"
        super().__init__(
            message or default_message,
            error_code="QDRANT_COLLECTION_ERROR",
            details={"collection_name": collection_name, "operation": operation},
        )


class EmbeddingGenerationError(VectorDatabaseError):
    """Raised when FastEmbed embedding generation fails."""

    def __init__(self, text: str, model: str, message: Optional[str] = None):
        self.text = text
        self.model = model
        default_message = f"Failed to generate embedding for text using model {model}"
        super().__init__(
            message or default_message,
            error_code="EMBEDDING_GENERATION_FAILED",
            details={"model": model, "text_length": len(text)},
        )


class SearchError(VectorDatabaseError):
    """Raised when vector search operations fail."""

    def __init__(self, query: str, message: Optional[str] = None):
        self.query = query
        default_message = f"Search operation failed for query: {query}"
        super().__init__(
            message or default_message,
            error_code="SEARCH_OPERATION_FAILED",
            details={"query": query},
        )


class DataProcessingError(AnimeServerError):
    """Base exception for data processing related errors."""

    pass


class AnimeDataDownloadError(DataProcessingError):
    """Raised when anime database download fails."""

    def __init__(
        self, url: str, status_code: Optional[int] = None, message: Optional[str] = None
    ):
        self.url = url
        self.status_code = status_code
        default_message = f"Failed to download anime data from {url}"
        if status_code:
            default_message += f" (HTTP {status_code})"
        super().__init__(
            message or default_message,
            error_code="ANIME_DATA_DOWNLOAD_FAILED",
            details={"url": url, "status_code": status_code},
        )


class AnimeDataValidationError(DataProcessingError):
    """Raised when anime data validation fails."""

    def __init__(
        self, anime_id: str, field: str, value: Any, message: Optional[str] = None
    ):
        self.anime_id = anime_id
        self.field = field
        self.value = value
        default_message = f"Invalid {field} value for anime {anime_id}: {value}"
        super().__init__(
            message or default_message,
            error_code="ANIME_DATA_VALIDATION_FAILED",
            details={"anime_id": anime_id, "field": field, "value": str(value)},
        )


class PlatformIDExtractionError(DataProcessingError):
    """Raised when platform ID extraction fails."""

    def __init__(self, platform: str, url: str, message: Optional[str] = None):
        self.platform = platform
        self.url = url
        default_message = f"Failed to extract {platform} ID from URL: {url}"
        super().__init__(
            message or default_message,
            error_code="PLATFORM_ID_EXTRACTION_FAILED",
            details={"platform": platform, "url": url},
        )


class ConfigurationError(AnimeServerError):
    """Base exception for configuration related errors."""

    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration validation fails."""

    def __init__(self, setting: str, value: Any, reason: str):
        self.setting = setting
        self.value = value
        self.reason = reason
        message = f"Invalid configuration for {setting}: {reason}"
        super().__init__(
            message,
            error_code="INVALID_CONFIGURATION",
            details={"setting": setting, "value": str(value), "reason": reason},
        )


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, setting: str, message: Optional[str] = None):
        self.setting = setting
        default_message = f"Required configuration setting '{setting}' is missing"
        super().__init__(
            message or default_message,
            error_code="MISSING_CONFIGURATION",
            details={"setting": setting},
        )


class APIError(AnimeServerError):
    """Base exception for API related errors."""

    pass


class InvalidSearchQueryError(APIError):
    """Raised when search query is invalid or malformed."""

    def __init__(self, query: str, reason: str):
        self.query = query
        self.reason = reason
        message = f"Invalid search query '{query}': {reason}"
        super().__init__(
            message,
            error_code="INVALID_SEARCH_QUERY",
            details={"query": query, "reason": reason},
        )


class AnimeNotFoundError(APIError):
    """Raised when requested anime is not found."""

    def __init__(self, identifier: str, identifier_type: str = "anime_id"):
        self.identifier = identifier
        self.identifier_type = identifier_type
        message = f"Anime not found with {identifier_type}: {identifier}"
        super().__init__(
            message,
            error_code="ANIME_NOT_FOUND",
            details={"identifier": identifier, "identifier_type": identifier_type},
        )


class RateLimitExceededError(APIError):
    """Raised when rate limits are exceeded."""

    def __init__(self, limit: int, window: str, message: Optional[str] = None):
        self.limit = limit
        self.window = window
        default_message = f"Rate limit exceeded: {limit} requests per {window}"
        super().__init__(
            message or default_message,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"limit": limit, "window": window},
        )


class MCPError(AnimeServerError):
    """Base exception for MCP (Model Context Protocol) related errors."""

    pass


class MCPToolError(MCPError):
    """Raised when MCP tool execution fails."""

    def __init__(self, tool_name: str, arguments: Dict[str, Any], reason: str):
        self.tool_name = tool_name
        self.arguments = arguments
        self.reason = reason
        message = f"MCP tool '{tool_name}' failed: {reason}"
        super().__init__(
            message,
            error_code="MCP_TOOL_EXECUTION_FAILED",
            details={"tool_name": tool_name, "arguments": arguments, "reason": reason},
        )


class MCPClientError(MCPError):
    """Raised when MCP client communication fails."""

    def __init__(self, client_info: str, message: Optional[str] = None):
        self.client_info = client_info
        default_message = f"MCP client communication failed: {client_info}"
        super().__init__(
            message or default_message,
            error_code="MCP_CLIENT_ERROR",
            details={"client_info": client_info},
        )


class UpdateServiceError(AnimeServerError):
    """Base exception for update service related errors."""

    pass


class UpdateSchedulingError(UpdateServiceError):
    """Raised when update scheduling fails."""

    def __init__(self, reason: str, next_attempt: Optional[str] = None):
        self.reason = reason
        self.next_attempt = next_attempt
        message = f"Update scheduling failed: {reason}"
        super().__init__(
            message,
            error_code="UPDATE_SCHEDULING_FAILED",
            details={"reason": reason, "next_attempt": next_attempt},
        )


class ConcurrentUpdateError(UpdateServiceError):
    """Raised when concurrent update operations conflict."""

    def __init__(self, operation: str, message: Optional[str] = None):
        self.operation = operation
        default_message = f"Concurrent update conflict for operation: {operation}"
        super().__init__(
            message or default_message,
            error_code="CONCURRENT_UPDATE_CONFLICT",
            details={"operation": operation},
        )


# Exception mapping for HTTP status codes
HTTP_EXCEPTION_MAP = {
    400: InvalidSearchQueryError,
    404: AnimeNotFoundError,
    429: RateLimitExceededError,
    500: AnimeServerError,
    503: QdrantConnectionError,
}


def map_to_http_exception(status_code: int, message: str, **kwargs) -> AnimeServerError:
    """Map HTTP status code to appropriate exception type.

    Args:
        status_code: HTTP status code
        message: Error message
        **kwargs: Additional exception-specific arguments

    Returns:
        AnimeServerError: Appropriate exception instance
    """
    exception_class = HTTP_EXCEPTION_MAP.get(status_code, AnimeServerError)

    # Handle specific exception constructors
    if exception_class == InvalidSearchQueryError and "query" in kwargs:
        return exception_class(kwargs["query"], message)
    elif exception_class == AnimeNotFoundError and "identifier" in kwargs:
        return exception_class(
            kwargs["identifier"], kwargs.get("identifier_type", "anime_id")
        )
    elif exception_class == QdrantConnectionError and "url" in kwargs:
        return exception_class(kwargs["url"], message)
    else:
        return exception_class(message)


def handle_exception_safely(func):
    """Decorator for safe exception handling with logging.

    Converts generic exceptions to appropriate AnimeServerError types
    and ensures all exceptions are properly logged.
    """
    import functools
    import logging

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            return await func(*args, **kwargs)
        except AnimeServerError:
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise AnimeServerError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={"function": func.__name__, "original_error": str(e)},
            )

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except AnimeServerError:
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise AnimeServerError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={"function": func.__name__, "original_error": str(e)},
            )

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
