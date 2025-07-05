"""
Domain-specific error types for MCP anime server.

Following modern MCP patterns for structured error handling.
"""


class AnimeServerError(Exception):
    """Base exception for anime MCP server operations."""

    def __init__(self, code: str, operation: str, message: str):
        """Initialize anime server error.

        Args:
            code: Error code for categorization
            operation: Operation that failed
            message: Human-readable error message
        """
        self.code = code
        self.operation = operation
        super().__init__(message)


class ClientNotInitializedError(AnimeServerError):
    """Raised when Qdrant client is not initialized."""

    def __init__(self, operation: str):
        super().__init__(
            code="CLIENT_NOT_INITIALIZED",
            operation=operation,
            message="Qdrant client not initialized. Server may be starting up.",
        )


class SearchValidationError(AnimeServerError):
    """Raised when search parameters are invalid."""

    def __init__(self, operation: str, details: str):
        super().__init__(
            code="SEARCH_VALIDATION_ERROR",
            operation=operation,
            message=f"Invalid search parameters: {details}",
        )


class MultiVectorNotAvailableError(AnimeServerError):
    """Raised when multi-vector features are requested but not enabled."""

    def __init__(self, operation: str):
        super().__init__(
            code="MULTI_VECTOR_NOT_AVAILABLE",
            operation=operation,
            message="Multi-vector image search not enabled. Enable in server configuration.",
        )


class AnimeNotFoundError(AnimeServerError):
    """Raised when requested anime is not found."""

    def __init__(self, operation: str, anime_id: str):
        super().__init__(
            code="ANIME_NOT_FOUND",
            operation=operation,
            message=f"Anime not found: {anime_id}",
        )


class DatabaseOperationError(AnimeServerError):
    """Raised when database operations fail."""

    def __init__(self, operation: str, details: str):
        super().__init__(
            code="DATABASE_OPERATION_ERROR",
            operation=operation,
            message=f"Database operation failed: {details}",
        )