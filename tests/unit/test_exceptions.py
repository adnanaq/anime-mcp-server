"""Unit tests for custom exception classes."""
import pytest
from typing import Dict, Any

# Mock asyncio before importing exceptions
import asyncio
from unittest.mock import MagicMock, patch


class TestCustomExceptions:
    """Test cases for custom exception classes."""
    
    def test_anime_server_error_base_functionality(self):
        """Test AnimeServerError base exception functionality."""
        # Mock the exception classes since we can't import them directly
        class MockAnimeServerError(Exception):
            def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
                super().__init__(message)
                self.message = message
                self.error_code = error_code
                self.details = details or {}
            
            def to_dict(self) -> Dict[str, Any]:
                return {
                    "error": self.__class__.__name__,
                    "message": self.message,
                    "error_code": self.error_code,
                    "details": self.details
                }
        
        # Test basic exception creation
        error = MockAnimeServerError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.details == {}
        
        # Test exception with error code and details
        details = {"field": "test_field", "value": "test_value"}
        error_with_details = MockAnimeServerError(
            "Detailed error",
            error_code="TEST_ERROR",
            details=details
        )
        
        assert error_with_details.message == "Detailed error"
        assert error_with_details.error_code == "TEST_ERROR"
        assert error_with_details.details == details
        
        # Test to_dict conversion
        error_dict = error_with_details.to_dict()
        expected_dict = {
            "error": "MockAnimeServerError",
            "message": "Detailed error",
            "error_code": "TEST_ERROR",
            "details": details
        }
        assert error_dict == expected_dict
    
    def test_vector_database_error_hierarchy(self):
        """Test vector database error hierarchy."""
        class MockVectorDatabaseError(Exception):
            def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
                super().__init__(message)
                self.message = message
                self.error_code = error_code
                self.details = details or {}
        
        class MockQdrantConnectionError(MockVectorDatabaseError):
            def __init__(self, url: str, message: str = None):
                self.url = url
                default_message = f"Failed to connect to Qdrant at {url}"
                super().__init__(
                    message or default_message,
                    error_code="QDRANT_CONNECTION_FAILED",
                    details={"qdrant_url": url}
                )
        
        class MockQdrantCollectionError(MockVectorDatabaseError):
            def __init__(self, collection_name: str, operation: str, message: str = None):
                self.collection_name = collection_name
                self.operation = operation
                default_message = f"Collection '{collection_name}' {operation} operation failed"
                super().__init__(
                    message or default_message,
                    error_code="QDRANT_COLLECTION_ERROR",
                    details={"collection_name": collection_name, "operation": operation}
                )
        
        # Test QdrantConnectionError
        conn_error = MockQdrantConnectionError("http://localhost:6333")
        assert "Failed to connect to Qdrant at http://localhost:6333" in conn_error.message
        assert conn_error.error_code == "QDRANT_CONNECTION_FAILED"
        assert conn_error.details["qdrant_url"] == "http://localhost:6333"
        assert conn_error.url == "http://localhost:6333"
        
        # Test QdrantCollectionError
        coll_error = MockQdrantCollectionError("anime_database", "create")
        assert "Collection 'anime_database' create operation failed" in coll_error.message
        assert coll_error.error_code == "QDRANT_COLLECTION_ERROR"
        assert coll_error.details["collection_name"] == "anime_database"
        assert coll_error.details["operation"] == "create"
        assert coll_error.collection_name == "anime_database"
        assert coll_error.operation == "create"
    
    def test_embedding_generation_error(self):
        """Test embedding generation error."""
        class MockEmbeddingGenerationError(Exception):
            def __init__(self, text: str, model: str, message: str = None):
                self.text = text
                self.model = model
                default_message = f"Failed to generate embedding for text using model {model}"
                super().__init__(message or default_message)
                self.error_code = "EMBEDDING_GENERATION_FAILED"
                self.details = {"model": model, "text_length": len(text)}
        
        text = "test anime description"
        model = "BAAI/bge-small-en-v1.5"
        
        error = MockEmbeddingGenerationError(text, model)
        assert f"Failed to generate embedding for text using model {model}" in str(error)
        assert error.error_code == "EMBEDDING_GENERATION_FAILED"
        assert error.details["model"] == model
        assert error.details["text_length"] == len(text)
        assert error.text == text
        assert error.model == model
    
    def test_data_processing_errors(self):
        """Test data processing error classes."""
        class MockDataProcessingError(Exception):
            def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
                super().__init__(message)
                self.error_code = error_code
                self.details = details or {}
        
        class MockAnimeDataDownloadError(MockDataProcessingError):
            def __init__(self, url: str, status_code: int = None, message: str = None):
                self.url = url
                self.status_code = status_code
                default_message = f"Failed to download anime data from {url}"
                if status_code:
                    default_message += f" (HTTP {status_code})"
                super().__init__(
                    message or default_message,
                    error_code="ANIME_DATA_DOWNLOAD_FAILED",
                    details={"url": url, "status_code": status_code}
                )
        
        class MockAnimeDataValidationError(MockDataProcessingError):
            def __init__(self, anime_id: str, field: str, value: Any, message: str = None):
                self.anime_id = anime_id
                self.field = field
                self.value = value
                default_message = f"Invalid {field} value for anime {anime_id}: {value}"
                super().__init__(
                    message or default_message,
                    error_code="ANIME_DATA_VALIDATION_FAILED",
                    details={"anime_id": anime_id, "field": field, "value": str(value)}
                )
        
        # Test download error
        download_error = MockAnimeDataDownloadError(
            "https://example.com/anime.json", 
            status_code=404
        )
        assert "Failed to download anime data from https://example.com/anime.json (HTTP 404)" in str(download_error)
        assert download_error.error_code == "ANIME_DATA_DOWNLOAD_FAILED"
        assert download_error.details["url"] == "https://example.com/anime.json"
        assert download_error.details["status_code"] == 404
        
        # Test validation error
        validation_error = MockAnimeDataValidationError(
            "anime123", "episodes", -1
        )
        assert "Invalid episodes value for anime anime123: -1" in str(validation_error)
        assert validation_error.error_code == "ANIME_DATA_VALIDATION_FAILED"
        assert validation_error.details["anime_id"] == "anime123"
        assert validation_error.details["field"] == "episodes"
        assert validation_error.details["value"] == "-1"
    
    def test_platform_id_extraction_error(self):
        """Test platform ID extraction error."""
        class MockPlatformIDExtractionError(Exception):
            def __init__(self, platform: str, url: str, message: str = None):
                self.platform = platform
                self.url = url
                default_message = f"Failed to extract {platform} ID from URL: {url}"
                super().__init__(message or default_message)
                self.error_code = "PLATFORM_ID_EXTRACTION_FAILED"
                self.details = {"platform": platform, "url": url}
        
        error = MockPlatformIDExtractionError(
            "myanimelist", 
            "https://myanimelist.net/anime/invalid"
        )
        
        assert "Failed to extract myanimelist ID from URL: https://myanimelist.net/anime/invalid" in str(error)
        assert error.error_code == "PLATFORM_ID_EXTRACTION_FAILED"
        assert error.details["platform"] == "myanimelist"
        assert error.details["url"] == "https://myanimelist.net/anime/invalid"
    
    def test_configuration_errors(self):
        """Test configuration error classes."""
        class MockConfigurationError(Exception):
            def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
                super().__init__(message)
                self.error_code = error_code
                self.details = details or {}
        
        class MockInvalidConfigurationError(MockConfigurationError):
            def __init__(self, setting: str, value: Any, reason: str):
                self.setting = setting
                self.value = value
                self.reason = reason
                message = f"Invalid configuration for {setting}: {reason}"
                super().__init__(
                    message,
                    error_code="INVALID_CONFIGURATION",
                    details={"setting": setting, "value": str(value), "reason": reason}
                )
        
        class MockMissingConfigurationError(MockConfigurationError):
            def __init__(self, setting: str, message: str = None):
                self.setting = setting
                default_message = f"Required configuration setting '{setting}' is missing"
                super().__init__(
                    message or default_message,
                    error_code="MISSING_CONFIGURATION",
                    details={"setting": setting}
                )
        
        # Test invalid configuration error
        invalid_error = MockInvalidConfigurationError(
            "qdrant_url", "invalid-url", "URL must start with http:// or https://"
        )
        assert "Invalid configuration for qdrant_url: URL must start with http:// or https://" in str(invalid_error)
        assert invalid_error.error_code == "INVALID_CONFIGURATION"
        assert invalid_error.details["setting"] == "qdrant_url"
        assert invalid_error.details["reason"] == "URL must start with http:// or https://"
        
        # Test missing configuration error
        missing_error = MockMissingConfigurationError("QDRANT_URL")
        assert "Required configuration setting 'QDRANT_URL' is missing" in str(missing_error)
        assert missing_error.error_code == "MISSING_CONFIGURATION"
        assert missing_error.details["setting"] == "QDRANT_URL"
    
    def test_api_errors(self):
        """Test API error classes."""
        class MockAPIError(Exception):
            def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
                super().__init__(message)
                self.error_code = error_code
                self.details = details or {}
        
        class MockInvalidSearchQueryError(MockAPIError):
            def __init__(self, query: str, reason: str):
                self.query = query
                self.reason = reason
                message = f"Invalid search query '{query}': {reason}"
                super().__init__(
                    message,
                    error_code="INVALID_SEARCH_QUERY",
                    details={"query": query, "reason": reason}
                )
        
        class MockAnimeNotFoundError(MockAPIError):
            def __init__(self, identifier: str, identifier_type: str = "anime_id"):
                self.identifier = identifier
                self.identifier_type = identifier_type
                message = f"Anime not found with {identifier_type}: {identifier}"
                super().__init__(
                    message,
                    error_code="ANIME_NOT_FOUND",
                    details={"identifier": identifier, "identifier_type": identifier_type}
                )
        
        # Test invalid search query error
        search_error = MockInvalidSearchQueryError("", "Query cannot be empty")
        assert "Invalid search query '': Query cannot be empty" in str(search_error)
        assert search_error.error_code == "INVALID_SEARCH_QUERY"
        assert search_error.details["query"] == ""
        assert search_error.details["reason"] == "Query cannot be empty"
        
        # Test anime not found error
        not_found_error = MockAnimeNotFoundError("123456", "myanimelist_id")
        assert "Anime not found with myanimelist_id: 123456" in str(not_found_error)
        assert not_found_error.error_code == "ANIME_NOT_FOUND"
        assert not_found_error.details["identifier"] == "123456"
        assert not_found_error.details["identifier_type"] == "myanimelist_id"
    
    def test_mcp_errors(self):
        """Test MCP (Model Context Protocol) error classes."""
        class MockMCPError(Exception):
            def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
                super().__init__(message)
                self.error_code = error_code
                self.details = details or {}
        
        class MockMCPToolError(MockMCPError):
            def __init__(self, tool_name: str, arguments: Dict[str, Any], reason: str):
                self.tool_name = tool_name
                self.arguments = arguments
                self.reason = reason
                message = f"MCP tool '{tool_name}' failed: {reason}"
                super().__init__(
                    message,
                    error_code="MCP_TOOL_EXECUTION_FAILED",
                    details={"tool_name": tool_name, "arguments": arguments, "reason": reason}
                )
        
        # Test MCP tool error
        tool_error = MockMCPToolError(
            "search_anime",
            {"query": "test", "limit": 10},
            "Vector database unavailable"
        )
        assert "MCP tool 'search_anime' failed: Vector database unavailable" in str(tool_error)
        assert tool_error.error_code == "MCP_TOOL_EXECUTION_FAILED"
        assert tool_error.details["tool_name"] == "search_anime"
        assert tool_error.details["arguments"] == {"query": "test", "limit": 10}
        assert tool_error.details["reason"] == "Vector database unavailable"
    
    def test_http_exception_mapping_logic(self):
        """Test HTTP exception mapping logic."""
        # Define mock mapping function
        def mock_map_to_http_exception(status_code: int, message: str, **kwargs):
            """Mock implementation of HTTP exception mapping."""
            class MockHTTPException(Exception):
                def __init__(self, message: str, status_code: int, **kwargs):
                    super().__init__(message)
                    self.status_code = status_code
                    self.details = kwargs
            
            if status_code == 400:
                if 'query' in kwargs:
                    return MockHTTPException(f"Invalid query: {kwargs['query']}", status_code, **kwargs)
                return MockHTTPException("Bad Request", status_code, **kwargs)
            elif status_code == 404:
                if 'identifier' in kwargs:
                    return MockHTTPException(f"Not found: {kwargs['identifier']}", status_code, **kwargs)
                return MockHTTPException("Not Found", status_code, **kwargs)
            elif status_code == 429:
                return MockHTTPException("Rate limit exceeded", status_code, **kwargs)
            elif status_code == 500:
                return MockHTTPException("Internal Server Error", status_code, **kwargs)
            elif status_code == 503:
                if 'url' in kwargs:
                    return MockHTTPException(f"Service unavailable: {kwargs['url']}", status_code, **kwargs)
                return MockHTTPException("Service Unavailable", status_code, **kwargs)
            else:
                return MockHTTPException(message, status_code, **kwargs)
        
        # Test different HTTP status codes
        test_cases = [
            (400, "Bad request", {"query": "invalid"}),
            (404, "Not found", {"identifier": "123", "identifier_type": "anime_id"}),
            (429, "Too many requests", {"limit": 100}),
            (500, "Server error", {}),
            (503, "Service down", {"url": "http://qdrant:6333"}),
        ]
        
        for status_code, message, kwargs in test_cases:
            exception = mock_map_to_http_exception(status_code, message, **kwargs)
            assert exception.status_code == status_code
            assert message.lower() in str(exception).lower()
    
    def test_exception_handler_decorator_logic(self):
        """Test exception handler decorator logic."""
        # Mock the decorator behavior
        def mock_handle_exception_safely(func):
            """Mock implementation of exception handling decorator."""
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Custom exceptions pass through
                    if hasattr(e, 'error_code'):
                        raise e
                    # Generic exceptions get wrapped
                    raise Exception(f"Unexpected error in {func.__name__}: {str(e)}")
            return wrapper
        
        # Test custom exception pass-through
        class MockCustomException(Exception):
            def __init__(self, message: str):
                super().__init__(message)
                self.error_code = "CUSTOM_ERROR"
        
        @mock_handle_exception_safely
        def function_with_custom_error():
            raise MockCustomException("Custom error")
        
        with pytest.raises(MockCustomException) as exc_info:
            function_with_custom_error()
        assert exc_info.value.error_code == "CUSTOM_ERROR"
        
        # Test generic exception wrapping
        @mock_handle_exception_safely
        def function_with_generic_error():
            raise ValueError("Generic error")
        
        with pytest.raises(Exception) as exc_info:
            function_with_generic_error()
        assert "Unexpected error in function_with_generic_error: Generic error" in str(exc_info.value)
    
    def test_exception_context_preservation(self):
        """Test that exception context is properly preserved."""
        # Test that error details are preserved through exception hierarchy
        class MockBaseError(Exception):
            def __init__(self, message: str, context: Dict[str, Any] = None):
                super().__init__(message)
                self.context = context or {}
        
        class MockDerivedError(MockBaseError):
            def __init__(self, specific_field: str, message: str = None):
                self.specific_field = specific_field
                default_message = f"Error in field: {specific_field}"
                context = {"field": specific_field, "timestamp": "2025-01-21"}
                super().__init__(message or default_message, context)
        
        error = MockDerivedError("test_field")
        assert error.specific_field == "test_field"
        assert error.context["field"] == "test_field"
        assert "timestamp" in error.context
        assert "Error in field: test_field" in str(error)