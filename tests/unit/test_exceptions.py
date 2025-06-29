"""Comprehensive unit tests for custom exception classes."""

import asyncio
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from src.exceptions import (
    AnimeDataDownloadError,
    AnimeDataValidationError,
    AnimeNotFoundError,
    AnimeServerError,
    ConcurrentUpdateError,
    ConfigurationError,
    EmbeddingGenerationError,
    InvalidConfigurationError,
    InvalidSearchQueryError,
    MCPClientError,
    MCPError,
    MCPToolError,
    MissingConfigurationError,
    PlatformIDExtractionError,
    QdrantCollectionError,
    QdrantConnectionError,
    RateLimitExceededError,
    SearchError,
    UpdateSchedulingError,
    UpdateServiceError,
    VectorDatabaseError,
    map_to_http_exception,
    handle_exception_safely,
    HTTP_EXCEPTION_MAP,
)


class TestAnimeServerError:
    """Test the base AnimeServerError class."""

    def test_basic_initialization(self):
        """Test basic error initialization."""
        error = AnimeServerError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.details == {}

    def test_initialization_with_error_code(self):
        """Test error initialization with error code."""
        error = AnimeServerError("Test error", error_code="TEST_ERROR")
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {}

    def test_initialization_with_details(self):
        """Test error initialization with details."""
        details = {"field": "test_field", "value": "test_value"}
        error = AnimeServerError("Test error", details=details)
        assert error.message == "Test error"
        assert error.error_code is None
        assert error.details == details

    def test_full_initialization(self):
        """Test error initialization with all parameters."""
        details = {"field": "test_field", "value": "test_value"}
        error = AnimeServerError("Test error", error_code="TEST_ERROR", details=details)
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == details

    def test_to_dict_method(self):
        """Test the to_dict method."""
        details = {"field": "test_field", "value": "test_value"}
        error = AnimeServerError("Test error", error_code="TEST_ERROR", details=details)
        
        result = error.to_dict()
        expected = {
            "error": "AnimeServerError",
            "message": "Test error",
            "error_code": "TEST_ERROR",
            "details": details,
        }
        assert result == expected

    def test_to_dict_minimal(self):
        """Test to_dict with minimal initialization."""
        error = AnimeServerError("Simple error")
        result = error.to_dict()
        expected = {
            "error": "AnimeServerError",
            "message": "Simple error",
            "error_code": None,
            "details": {},
        }
        assert result == expected


class TestVectorDatabaseErrors:
    """Test vector database error classes."""

    def test_vector_database_error_inheritance(self):
        """Test VectorDatabaseError inherits from AnimeServerError."""
        error = VectorDatabaseError("Test vector error")
        assert isinstance(error, AnimeServerError)
        assert error.message == "Test vector error"

    def test_qdrant_connection_error_default_message(self):
        """Test QdrantConnectionError with default message."""
        error = QdrantConnectionError("http://localhost:6333")
        assert "Failed to connect to Qdrant at http://localhost:6333" in error.message
        assert error.error_code == "QDRANT_CONNECTION_FAILED"
        assert error.details["qdrant_url"] == "http://localhost:6333"
        assert error.url == "http://localhost:6333"

    def test_qdrant_connection_error_custom_message(self):
        """Test QdrantConnectionError with custom message."""
        custom_message = "Custom connection error"
        error = QdrantConnectionError("http://localhost:6333", custom_message)
        assert error.message == custom_message
        assert error.error_code == "QDRANT_CONNECTION_FAILED"
        assert error.url == "http://localhost:6333"

    def test_qdrant_collection_error_default_message(self):
        """Test QdrantCollectionError with default message."""
        error = QdrantCollectionError("anime_database", "create")
        assert "Collection 'anime_database' create operation failed" in error.message
        assert error.error_code == "QDRANT_COLLECTION_ERROR"
        assert error.details["collection_name"] == "anime_database"
        assert error.details["operation"] == "create"
        assert error.collection_name == "anime_database"
        assert error.operation == "create"

    def test_qdrant_collection_error_custom_message(self):
        """Test QdrantCollectionError with custom message."""
        custom_message = "Custom collection error"
        error = QdrantCollectionError("anime_database", "delete", custom_message)
        assert error.message == custom_message
        assert error.error_code == "QDRANT_COLLECTION_ERROR"
        assert error.collection_name == "anime_database"
        assert error.operation == "delete"

    def test_embedding_generation_error(self):
        """Test EmbeddingGenerationError."""
        text = "test anime description"
        model = "BAAI/bge-small-en-v1.5"
        
        error = EmbeddingGenerationError(text, model)
        assert f"Failed to generate embedding for text using model {model}" in error.message
        assert error.error_code == "EMBEDDING_GENERATION_FAILED"
        assert error.details["model"] == model
        assert error.details["text_length"] == len(text)
        assert error.text == text
        assert error.model == model

    def test_embedding_generation_error_custom_message(self):
        """Test EmbeddingGenerationError with custom message."""
        text = "test text"
        model = "test-model"
        custom_message = "Custom embedding error"
        
        error = EmbeddingGenerationError(text, model, custom_message)
        assert error.message == custom_message
        assert error.text == text
        assert error.model == model

    def test_search_error(self):
        """Test SearchError."""
        query = "test search query"
        error = SearchError(query)
        assert f"Search operation failed for query: {query}" in error.message
        assert error.error_code == "SEARCH_OPERATION_FAILED"
        assert error.details["query"] == query
        assert error.query == query

    def test_search_error_custom_message(self):
        """Test SearchError with custom message."""
        query = "test query"
        custom_message = "Custom search error"
        error = SearchError(query, custom_message)
        assert error.message == custom_message
        assert error.query == query


class TestDataProcessingErrors:
    """Test data processing error classes."""

    def test_anime_data_download_error_basic(self):
        """Test AnimeDataDownloadError basic functionality."""
        url = "https://example.com/anime.json"
        error = AnimeDataDownloadError(url)
        assert f"Failed to download anime data from {url}" in error.message
        assert error.error_code == "ANIME_DATA_DOWNLOAD_FAILED"
        assert error.details["url"] == url
        assert error.details["status_code"] is None
        assert error.url == url

    def test_anime_data_download_error_with_status_code(self):
        """Test AnimeDataDownloadError with status code."""
        url = "https://example.com/anime.json"
        status_code = 404
        error = AnimeDataDownloadError(url, status_code)
        assert f"Failed to download anime data from {url} (HTTP {status_code})" in error.message
        assert error.status_code == status_code

    def test_anime_data_download_error_custom_message(self):
        """Test AnimeDataDownloadError with custom message."""
        url = "https://example.com/anime.json"
        custom_message = "Custom download error"
        error = AnimeDataDownloadError(url, message=custom_message)
        assert error.message == custom_message

    def test_anime_data_validation_error(self):
        """Test AnimeDataValidationError."""
        anime_id = "anime123"
        field = "episodes"
        value = -1
        error = AnimeDataValidationError(anime_id, field, value)
        assert f"Invalid {field} value for anime {anime_id}: {value}" in error.message
        assert error.error_code == "ANIME_DATA_VALIDATION_FAILED"
        assert error.details["anime_id"] == anime_id
        assert error.details["field"] == field
        assert error.details["value"] == str(value)
        assert error.anime_id == anime_id
        assert error.field == field
        assert error.value == value

    def test_anime_data_validation_error_custom_message(self):
        """Test AnimeDataValidationError with custom message."""
        custom_message = "Custom validation error"
        error = AnimeDataValidationError("test", "field", "value", custom_message)
        assert error.message == custom_message

    def test_platform_id_extraction_error(self):
        """Test PlatformIDExtractionError."""
        platform = "myanimelist"
        url = "https://myanimelist.net/anime/invalid"
        error = PlatformIDExtractionError(platform, url)
        assert f"Failed to extract {platform} ID from URL: {url}" in error.message
        assert error.error_code == "PLATFORM_ID_EXTRACTION_FAILED"
        assert error.details["platform"] == platform
        assert error.details["url"] == url
        assert error.platform == platform
        assert error.url == url

    def test_platform_id_extraction_error_custom_message(self):
        """Test PlatformIDExtractionError with custom message."""
        custom_message = "Custom extraction error"
        error = PlatformIDExtractionError("test", "url", custom_message)
        assert error.message == custom_message


class TestConfigurationErrors:
    """Test configuration error classes."""

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from AnimeServerError."""
        error = ConfigurationError("Test config error")
        assert isinstance(error, AnimeServerError)

    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError."""
        setting = "qdrant_url"
        value = "invalid-url"
        reason = "URL must start with http:// or https://"
        error = InvalidConfigurationError(setting, value, reason)
        assert f"Invalid configuration for {setting}: {reason}" in error.message
        assert error.error_code == "INVALID_CONFIGURATION"
        assert error.details["setting"] == setting
        assert error.details["value"] == str(value)
        assert error.details["reason"] == reason
        assert error.setting == setting
        assert error.value == value
        assert error.reason == reason

    def test_missing_configuration_error_default_message(self):
        """Test MissingConfigurationError with default message."""
        setting = "QDRANT_URL"
        error = MissingConfigurationError(setting)
        assert f"Required configuration setting '{setting}' is missing" in error.message
        assert error.error_code == "MISSING_CONFIGURATION"
        assert error.details["setting"] == setting
        assert error.setting == setting

    def test_missing_configuration_error_custom_message(self):
        """Test MissingConfigurationError with custom message."""
        setting = "TEST_SETTING"
        custom_message = "Custom missing config error"
        error = MissingConfigurationError(setting, custom_message)
        assert error.message == custom_message
        assert error.setting == setting


class TestAPIErrors:
    """Test API error classes."""

    def test_invalid_search_query_error(self):
        """Test InvalidSearchQueryError."""
        query = "invalid query"
        reason = "Query contains invalid characters"
        error = InvalidSearchQueryError(query, reason)
        assert f"Invalid search query '{query}': {reason}" in error.message
        assert error.error_code == "INVALID_SEARCH_QUERY"
        assert error.details["query"] == query
        assert error.details["reason"] == reason
        assert error.query == query
        assert error.reason == reason

    def test_anime_not_found_error_default_type(self):
        """Test AnimeNotFoundError with default identifier type."""
        identifier = "123456"
        error = AnimeNotFoundError(identifier)
        assert f"Anime not found with anime_id: {identifier}" in error.message
        assert error.error_code == "ANIME_NOT_FOUND"
        assert error.details["identifier"] == identifier
        assert error.details["identifier_type"] == "anime_id"
        assert error.identifier == identifier
        assert error.identifier_type == "anime_id"

    def test_anime_not_found_error_custom_type(self):
        """Test AnimeNotFoundError with custom identifier type."""
        identifier = "123456"
        identifier_type = "myanimelist_id"
        error = AnimeNotFoundError(identifier, identifier_type)
        assert f"Anime not found with {identifier_type}: {identifier}" in error.message
        assert error.identifier_type == identifier_type

    def test_rate_limit_exceeded_error_default_message(self):
        """Test RateLimitExceededError with default message."""
        limit = 100
        window = "hour"
        error = RateLimitExceededError(limit, window)
        assert f"Rate limit exceeded: {limit} requests per {window}" in error.message
        assert error.error_code == "RATE_LIMIT_EXCEEDED"
        assert error.details["limit"] == limit
        assert error.details["window"] == window
        assert error.limit == limit
        assert error.window == window

    def test_rate_limit_exceeded_error_custom_message(self):
        """Test RateLimitExceededError with custom message."""
        custom_message = "Custom rate limit error"
        error = RateLimitExceededError(50, "minute", custom_message)
        assert error.message == custom_message


class TestMCPErrors:
    """Test MCP (Model Context Protocol) error classes."""

    def test_mcp_error_inheritance(self):
        """Test MCPError inherits from AnimeServerError."""
        error = MCPError("Test MCP error")
        assert isinstance(error, AnimeServerError)

    def test_mcp_tool_error(self):
        """Test MCPToolError."""
        tool_name = "search_anime"
        arguments = {"query": "test", "limit": 10}
        reason = "Vector database unavailable"
        error = MCPToolError(tool_name, arguments, reason)
        assert f"MCP tool '{tool_name}' failed: {reason}" in error.message
        assert error.error_code == "MCP_TOOL_EXECUTION_FAILED"
        assert error.details["tool_name"] == tool_name
        assert error.details["arguments"] == arguments
        assert error.details["reason"] == reason
        assert error.tool_name == tool_name
        assert error.arguments == arguments
        assert error.reason == reason

    def test_mcp_client_error_default_message(self):
        """Test MCPClientError with default message."""
        client_info = "Client disconnected"
        error = MCPClientError(client_info)
        assert f"MCP client communication failed: {client_info}" in error.message
        assert error.error_code == "MCP_CLIENT_ERROR"
        assert error.details["client_info"] == client_info
        assert error.client_info == client_info

    def test_mcp_client_error_custom_message(self):
        """Test MCPClientError with custom message."""
        client_info = "test info"
        custom_message = "Custom client error"
        error = MCPClientError(client_info, custom_message)
        assert error.message == custom_message


class TestUpdateServiceErrors:
    """Test update service error classes."""

    def test_update_service_error_inheritance(self):
        """Test UpdateServiceError inherits from AnimeServerError."""
        error = UpdateServiceError("Test update error")
        assert isinstance(error, AnimeServerError)

    def test_update_scheduling_error_minimal(self):
        """Test UpdateSchedulingError with minimal parameters."""
        reason = "Scheduler not available"
        error = UpdateSchedulingError(reason)
        assert f"Update scheduling failed: {reason}" in error.message
        assert error.error_code == "UPDATE_SCHEDULING_FAILED"
        assert error.details["reason"] == reason
        assert error.details["next_attempt"] is None
        assert error.reason == reason
        assert error.next_attempt is None

    def test_update_scheduling_error_with_next_attempt(self):
        """Test UpdateSchedulingError with next attempt time."""
        reason = "Scheduler not available"
        next_attempt = "2025-01-22T10:00:00Z"
        error = UpdateSchedulingError(reason, next_attempt)
        assert error.details["next_attempt"] == next_attempt
        assert error.next_attempt == next_attempt

    def test_concurrent_update_error_default_message(self):
        """Test ConcurrentUpdateError with default message."""
        operation = "data_download"
        error = ConcurrentUpdateError(operation)
        assert f"Concurrent update conflict for operation: {operation}" in error.message
        assert error.error_code == "CONCURRENT_UPDATE_CONFLICT"
        assert error.details["operation"] == operation
        assert error.operation == operation

    def test_concurrent_update_error_custom_message(self):
        """Test ConcurrentUpdateError with custom message."""
        operation = "test_operation"
        custom_message = "Custom concurrent error"
        error = ConcurrentUpdateError(operation, custom_message)
        assert error.message == custom_message


class TestHTTPExceptionMapping:
    """Test HTTP exception mapping functionality."""

    def test_http_exception_map_contents(self):
        """Test HTTP_EXCEPTION_MAP contains expected mappings."""
        expected_mappings = {
            400: InvalidSearchQueryError,
            404: AnimeNotFoundError,
            429: RateLimitExceededError,
            500: AnimeServerError,
            503: QdrantConnectionError,
        }
        assert HTTP_EXCEPTION_MAP == expected_mappings

    def test_map_to_http_exception_invalid_search_query(self):
        """Test mapping to InvalidSearchQueryError."""
        query = "invalid query"
        message = "Bad query"
        result = map_to_http_exception(400, message, query=query)
        assert isinstance(result, InvalidSearchQueryError)
        assert result.query == query
        assert result.reason == message

    def test_map_to_http_exception_anime_not_found(self):
        """Test mapping to AnimeNotFoundError."""
        identifier = "123456"
        identifier_type = "myanimelist_id"
        message = "Not found"
        result = map_to_http_exception(
            404, message, identifier=identifier, identifier_type=identifier_type
        )
        assert isinstance(result, AnimeNotFoundError)
        assert result.identifier == identifier
        assert result.identifier_type == identifier_type

    def test_map_to_http_exception_anime_not_found_default_type(self):
        """Test mapping to AnimeNotFoundError with default type."""
        identifier = "123456"
        message = "Not found"
        result = map_to_http_exception(404, message, identifier=identifier)
        assert isinstance(result, AnimeNotFoundError)
        assert result.identifier == identifier
        assert result.identifier_type == "anime_id"

    def test_map_to_http_exception_qdrant_connection(self):
        """Test mapping to QdrantConnectionError."""
        url = "http://localhost:6333"
        message = "Connection failed"
        result = map_to_http_exception(503, message, url=url)
        assert isinstance(result, QdrantConnectionError)
        assert result.url == url

    def test_map_to_http_exception_fallback_to_base(self):
        """Test fallback to AnimeServerError for unknown status codes."""
        message = "Unknown error"
        result = map_to_http_exception(418, message)
        assert isinstance(result, AnimeServerError)
        assert result.message == message

    def test_map_to_http_exception_missing_kwargs(self):
        """Test mapping when required kwargs are missing."""
        # Should use fallback values when specific kwargs missing
        result = map_to_http_exception(400, "Bad request")
        assert isinstance(result, InvalidSearchQueryError)
        assert result.query == "unknown"
        assert result.reason == "Bad request"
        
        result = map_to_http_exception(404, "Not found")
        assert isinstance(result, AnimeNotFoundError)
        assert result.identifier == "unknown"
        assert result.identifier_type == "anime_id"
        
        result = map_to_http_exception(503, "Service unavailable")
        assert isinstance(result, QdrantConnectionError)
        assert result.url == "unknown"
        
        result = map_to_http_exception(429, "Rate limit")
        assert isinstance(result, RateLimitExceededError)
        assert result.limit == 100
        assert result.window == "hour"


class TestExceptionHandlerDecorator:
    """Test the handle_exception_safely decorator."""

    def test_async_function_pass_through_custom_exception(self):
        """Test async function passes through custom exceptions."""
        
        @handle_exception_safely
        async def test_function():
            raise QdrantConnectionError("http://localhost:6333")
        
        with pytest.raises(QdrantConnectionError):
            asyncio.run(test_function())

    def test_async_function_wraps_generic_exception(self):
        """Test async function wraps generic exceptions."""
        
        @handle_exception_safely
        async def test_function():
            raise ValueError("Generic error")
        
        with pytest.raises(AnimeServerError) as exc_info:
            asyncio.run(test_function())
        
        assert "Unexpected error in test_function: Generic error" in str(exc_info.value)
        assert exc_info.value.error_code == "UNEXPECTED_ERROR"
        assert exc_info.value.details["function"] == "test_function"
        assert exc_info.value.details["original_error"] == "Generic error"

    def test_sync_function_pass_through_custom_exception(self):
        """Test sync function passes through custom exceptions."""
        
        @handle_exception_safely
        def test_function():
            raise InvalidSearchQueryError("test query", "invalid")
        
        with pytest.raises(InvalidSearchQueryError):
            test_function()

    def test_sync_function_wraps_generic_exception(self):
        """Test sync function wraps generic exceptions."""
        
        @handle_exception_safely
        def test_function():
            raise RuntimeError("Runtime error")
        
        with pytest.raises(AnimeServerError) as exc_info:
            test_function()
        
        assert "Unexpected error in test_function: Runtime error" in str(exc_info.value)
        assert exc_info.value.error_code == "UNEXPECTED_ERROR"

    def test_async_function_success_case(self):
        """Test async function success case."""
        
        @handle_exception_safely
        async def test_function():
            return "success"
        
        result = asyncio.run(test_function())
        assert result == "success"

    def test_sync_function_success_case(self):
        """Test sync function success case."""
        
        @handle_exception_safely
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"

    @patch('logging.getLogger')
    def test_decorator_logging(self, mock_get_logger):
        """Test that decorator logs errors properly."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @handle_exception_safely
        def test_function():
            raise RuntimeError("Test error")
        
        with pytest.raises(AnimeServerError):
            test_function()
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "Unexpected error in test_function: Test error" in call_args[0][0]
        assert call_args[1]["exc_info"] is True

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        
        @handle_exception_safely
        def test_function():
            """Test function docstring."""
            return "test"
        
        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test function docstring."


class TestExceptionInheritanceHierarchy:
    """Test the exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_anime_server_error(self):
        """Test that all custom exceptions inherit from AnimeServerError."""
        exception_classes = [
            VectorDatabaseError,
            QdrantConnectionError,
            QdrantCollectionError,
            EmbeddingGenerationError,
            SearchError,
            AnimeDataDownloadError,
            AnimeDataValidationError,
            PlatformIDExtractionError,
            ConfigurationError,
            InvalidConfigurationError,
            MissingConfigurationError,
            InvalidSearchQueryError,
            AnimeNotFoundError,
            RateLimitExceededError,
            MCPError,
            MCPToolError,
            MCPClientError,
            UpdateServiceError,
            UpdateSchedulingError,
            ConcurrentUpdateError,
        ]
        
        for exception_class in exception_classes:
            # Create instance and check inheritance
            if exception_class == InvalidConfigurationError:
                instance = exception_class("setting", "value", "reason")
            elif exception_class == AnimeDataValidationError:
                instance = exception_class("anime_id", "field", "value")
            elif exception_class == InvalidSearchQueryError:
                instance = exception_class("query", "reason")
            elif exception_class == AnimeNotFoundError:
                instance = exception_class("identifier")
            elif exception_class == RateLimitExceededError:
                instance = exception_class(100, "hour")
            elif exception_class == QdrantConnectionError:
                instance = exception_class("http://localhost:6333")
            elif exception_class == QdrantCollectionError:
                instance = exception_class("collection", "operation")
            elif exception_class == EmbeddingGenerationError:
                instance = exception_class("text", "model")
            elif exception_class == SearchError:
                instance = exception_class("query")
            elif exception_class == AnimeDataDownloadError:
                instance = exception_class("http://example.com")
            elif exception_class == PlatformIDExtractionError:
                instance = exception_class("platform", "url")
            elif exception_class == MissingConfigurationError:
                instance = exception_class("setting")
            elif exception_class == MCPToolError:
                instance = exception_class("tool", {}, "reason")
            elif exception_class == MCPClientError:
                instance = exception_class("client_info")
            elif exception_class == UpdateSchedulingError:
                instance = exception_class("reason")
            elif exception_class == ConcurrentUpdateError:
                instance = exception_class("operation")
            else:
                instance = exception_class("Test error")
            
            assert isinstance(instance, AnimeServerError)
            assert isinstance(instance, Exception)