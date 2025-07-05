"""Comprehensive tests for BaseClient functionality."""

import json
import unittest.mock
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from src.exceptions import APIError
from src.integrations.clients.base_client import BaseClient
from src.integrations.error_handling import (
    CircuitBreaker,
    CorrelationLogger,
    ErrorContext,
    ErrorSeverity,
    ExecutionTracer,
    GracefulDegradation,
)


class TestBaseClient:
    """Test BaseClient comprehensive functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        return CircuitBreaker()

    @pytest.fixture
    def cache_manager(self):
        """Create mock cache manager."""
        cache = Mock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        return cache

    @pytest.fixture
    def error_handler(self):
        """Create mock error handler."""
        handler = Mock()
        handler.handle_error = AsyncMock()
        return handler

    @pytest.fixture
    def base_client(self, circuit_breaker, cache_manager, error_handler):
        """Create BaseClient instance for testing."""
        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            return BaseClient(
                service_name="test_service",
                circuit_breaker=circuit_breaker,
                cache_manager=cache_manager,
                error_handler=error_handler,
                timeout=10.0,
            )

    def test_base_client_initialization(
        self, base_client, circuit_breaker, cache_manager, error_handler
    ):
        """Test BaseClient initializes correctly."""
        assert base_client.service_name == "test_service"
        assert base_client.timeout == 10.0
        assert base_client.circuit_breaker is circuit_breaker
        assert base_client.cache_manager is cache_manager
        assert base_client.error_handler is error_handler
        assert base_client.logger.name == "integrations.test_service"

    def test_base_client_default_initialization(self):
        """Test BaseClient with default parameters."""
        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            client = BaseClient("test")
            assert client.service_name == "test"
            assert client.timeout == 30.0
            assert isinstance(client.circuit_breaker, CircuitBreaker)
            assert client.cache_manager is None
            assert client.error_handler is None

    @pytest.mark.asyncio
    async def test_make_request_success_json(self, base_client):
        """Test successful HTTP request with JSON response."""
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(
                    return_value={"data": "success", "id": 123}
                )
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await base_client.make_request(
                    "https://api.example.com/test", method="POST", priority=2
                )

                assert result == {"data": "success", "id": 123}
                mock_rate_manager.acquire.assert_called_once_with(
                    service_name="test_service", priority=2, endpoint=""
                )
                # Check that record_response was called with the service name, status, and 0, plus the response object
                mock_rate_manager.record_response.assert_called_once()
                call_args = mock_rate_manager.record_response.call_args[0]
                assert call_args[0] == "test_service"  # service_name
                assert call_args[1] == 200  # status_code
                assert call_args[2] == 0    # response_time
                # Fourth argument should be the response object

    @pytest.mark.asyncio
    async def test_make_request_success_text(self, base_client):
        """Test successful HTTP request with text response."""
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"content-type": "text/plain"}
                mock_response.text = AsyncMock(return_value="plain text response")
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await base_client.make_request("https://api.example.com/test")

                assert result == "plain text response"

    @pytest.mark.asyncio
    async def test_make_request_rate_limit_exceeded(self, base_client):
        """Test rate limit exceeded scenario."""
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=False)

            with pytest.raises(APIError) as exc_info:
                await base_client.make_request("https://api.example.com/test")

            assert "Rate limit exceeded for test_service" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_make_request_http_error_with_json(self, base_client):
        """Test HTTP error response with JSON error details."""
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 404
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(
                    return_value={"error": "Resource not found", "code": "NOT_FOUND"}
                )
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                with pytest.raises(APIError) as exc_info:
                    await base_client.make_request("https://api.example.com/test")

                assert "test_service API error: 404 - Resource not found" in str(
                    exc_info.value
                )

    @pytest.mark.asyncio
    async def test_make_request_http_error_with_message(self, base_client):
        """Test HTTP error response with message field."""
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 400
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(
                    return_value={"message": "Bad request parameters"}
                )
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                with pytest.raises(APIError) as exc_info:
                    await base_client.make_request("https://api.example.com/test")

                assert "test_service API error: 400 - Bad request parameters" in str(
                    exc_info.value
                )

    @pytest.mark.asyncio
    async def test_make_request_connection_error(self, base_client):
        """Test connection error handling."""
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session_class.side_effect = aiohttp.ClientError(
                    "Connection failed"
                )

                with pytest.raises(APIError) as exc_info:
                    await base_client.make_request("https://api.example.com/test")

                assert "test_service connection error: Connection failed" in str(
                    exc_info.value
                )

    @pytest.mark.asyncio
    async def test_make_request_with_circuit_breaker(self, base_client):
        """Test request with circuit breaker."""

        # Mock circuit breaker call
        async def mock_circuit_call(func):
            return await func()

        base_client.circuit_breaker.call_with_breaker = AsyncMock(
            side_effect=mock_circuit_call
        )

        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(return_value={"success": True})
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await base_client.make_request("https://api.example.com/test")

                assert result == {"success": True}
                base_client.circuit_breaker.call_with_breaker.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_request_without_circuit_breaker(self):
        """Test request without circuit breaker directly calls _request function."""
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            # Manually set circuit_breaker to None after initialization to test the else branch
            client = BaseClient("test")
            client.circuit_breaker = None

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(return_value={"direct": True})
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await client.make_request("https://api.example.com/test")

                assert result == {"direct": True}
                # Verify we manually set circuit breaker to None
                assert client.circuit_breaker is None
            assert client.correlation_logger is None
            assert client.execution_tracer is None

    @pytest.mark.asyncio
    async def test_with_circuit_breaker_method(self, base_client):
        """Test with_circuit_breaker method."""

        async def test_func():
            return {"test": "result"}

        base_client.circuit_breaker.call_with_breaker = AsyncMock(
            return_value={"test": "result"}
        )

        result = await base_client.with_circuit_breaker(test_func)

        assert result == {"test": "result"}
        base_client.circuit_breaker.call_with_breaker.assert_called_once_with(test_func)

    @pytest.mark.asyncio
    async def test_with_circuit_breaker_without_breaker(self):
        """Test with_circuit_breaker method when no circuit breaker."""
        client = BaseClient("test")
        client.circuit_breaker = None  # Manually set to None to test else branch

        async def test_func():
            return {"direct": "call"}

        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            result = await client.with_circuit_breaker(test_func)

        assert result == {"direct": "call"}

    @pytest.mark.asyncio
    async def test_handle_graceful_degradation_success(self, base_client):
        """Test graceful degradation when primary succeeds."""

        async def primary_func():
            return {"primary": "success"}

        async def fallback_func():
            return {"fallback": "success"}

        result = await base_client.handle_graceful_degradation(
            primary_func, fallback_func
        )

        assert result == {"primary": "success"}

    @pytest.mark.asyncio
    async def test_handle_graceful_degradation_cache_fallback(self, base_client):
        """Test graceful degradation with cache fallback."""

        async def primary_func():
            raise APIError("Primary failed")

        # Mock cache returning data
        base_client.cache_manager.get = AsyncMock(return_value={"cached": "data"})

        with patch.object(base_client.logger, "info") as mock_log:
            result = await base_client.handle_graceful_degradation(
                primary_func, cache_key="test_key"
            )

        assert result == {"cached": "data"}
        base_client.cache_manager.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_handle_graceful_degradation_fallback_service(self, base_client):
        """Test graceful degradation with fallback service."""

        async def primary_func():
            raise APIError("Primary failed")

        async def fallback_func():
            return {"fallback": "success"}

        # Mock cache miss
        base_client.cache_manager.get = AsyncMock(return_value=None)

        with patch.object(base_client.logger, "info") as mock_log:
            result = await base_client.handle_graceful_degradation(
                primary_func, fallback_func, cache_key="test_key"
            )

        assert result == {"fallback": "success"}

    @pytest.mark.asyncio
    async def test_handle_graceful_degradation_no_fallback(self, base_client):
        """Test graceful degradation with no fallback available."""

        async def primary_func():
            raise APIError("Primary failed")

        # Mock cache miss
        base_client.cache_manager.get = AsyncMock(return_value=None)

        with patch.object(base_client.logger, "error") as mock_log:
            with pytest.raises(APIError) as exc_info:
                await base_client.handle_graceful_degradation(primary_func)

        assert "Primary failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_response_json(self, base_client):
        """Test _parse_response with JSON content."""
        mock_response = AsyncMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={"parsed": "json"})

        result = await base_client._parse_response(mock_response)

        assert result == {"parsed": "json"}
        mock_response.json.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_response_xml(self, base_client):
        """Test _parse_response with XML content."""
        mock_response = AsyncMock()
        mock_response.headers = {"content-type": "application/xml"}
        mock_response.text = AsyncMock(return_value="<xml>data</xml>")

        result = await base_client._parse_response(mock_response)

        assert result == "<xml>data</xml>"
        mock_response.text.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_response_text(self, base_client):
        """Test _parse_response with plain text content."""
        mock_response = AsyncMock()
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = AsyncMock(return_value="plain text")

        result = await base_client._parse_response(mock_response)

        assert result == "plain text"
        mock_response.text.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_response_fallback_on_json_error(self, base_client):
        """Test _parse_response fallback when JSON parsing fails."""
        mock_response = AsyncMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(
            side_effect=json.JSONDecodeError("Invalid JSON", "", 0)
        )
        mock_response.text = AsyncMock(return_value="fallback text")

        result = await base_client._parse_response(mock_response)

        assert result == "fallback text"
        mock_response.json.assert_called_once()
        mock_response.text.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_response_complete_failure(self, base_client):
        """Test _parse_response when all parsing fails."""
        mock_response = AsyncMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(side_effect=Exception("JSON failed"))
        mock_response.text = AsyncMock(side_effect=Exception("Text failed"))

        result = await base_client._parse_response(mock_response)

        assert result == {"raw_response": "Unable to parse response"}

    @pytest.mark.asyncio
    async def test_make_request_custom_timeout(self, base_client):
        """Test make_request with custom timeout."""
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(return_value={"success": True})
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await base_client.make_request(
                    "https://api.example.com/test",
                    timeout=aiohttp.ClientTimeout(total=5),
                )

                assert result == {"success": True}
                # Verify timeout was passed through
                call_args = mock_session.request.call_args
                assert "timeout" in call_args[1]

    @pytest.mark.asyncio
    async def test_handle_graceful_degradation_cache_error_fallback_service(
        self, base_client
    ):
        """Test graceful degradation when cache fails but fallback service works."""

        async def primary_func():
            raise APIError("Primary failed")

        async def fallback_func():
            return {"fallback": "service"}

        # Mock cache raising exception
        base_client.cache_manager.get = AsyncMock(side_effect=Exception("Cache error"))

        result = await base_client.handle_graceful_degradation(
            primary_func, fallback_func, cache_key="test_key"
        )

        assert result == {"fallback": "service"}

    @pytest.mark.asyncio
    async def test_handle_graceful_degradation_fallback_service_fails(
        self, base_client
    ):
        """Test graceful degradation when fallback service also fails."""

        async def primary_func():
            raise APIError("Primary failed")

        async def fallback_func():
            raise Exception("Fallback also failed")

        # Mock cache miss
        base_client.cache_manager.get = AsyncMock(return_value=None)

        with pytest.raises(APIError) as exc_info:
            await base_client.handle_graceful_degradation(
                primary_func, fallback_func, cache_key="test_key"
            )

        assert "Primary failed" in str(exc_info.value)


class TestBaseClientEnhancedErrorHandling:
    """Test BaseClient integration with enhanced error handling infrastructure."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        return CircuitBreaker()

    @pytest.fixture
    def cache_manager(self):
        """Create mock cache manager."""
        cache = Mock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        return cache

    @pytest.fixture
    def correlation_logger(self):
        """Create correlation logger for testing."""
        return CorrelationLogger()

    @pytest.fixture
    def execution_tracer(self):
        """Create execution tracer for testing."""
        return ExecutionTracer()

    @pytest.fixture
    def enhanced_client(
        self, circuit_breaker, cache_manager, correlation_logger, execution_tracer
    ):
        """Create BaseClient instance with enhanced error handling."""
        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            return BaseClient(
                service_name="enhanced_service",
                circuit_breaker=circuit_breaker,
                cache_manager=cache_manager,
                correlation_logger=correlation_logger,
                execution_tracer=execution_tracer,
                timeout=15.0,
            )

    def test_enhanced_client_initialization(
        self, enhanced_client, correlation_logger, execution_tracer
    ):
        """Test BaseClient initializes with enhanced error handling components."""
        assert enhanced_client.service_name == "enhanced_service"
        assert enhanced_client.correlation_logger is correlation_logger
        assert enhanced_client.execution_tracer is execution_tracer
        assert enhanced_client.timeout == 15.0

    @pytest.mark.asyncio
    async def test_make_request_with_correlation_logging(self, enhanced_client):
        """Test make_request uses correlation logging."""
        correlation_id = "test-corr-123"

        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(return_value={"data": "success"})
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                enhanced_client.correlation_logger.log_with_correlation = AsyncMock()

                result = await enhanced_client.make_request_with_correlation(
                    "https://api.example.com/test",
                    correlation_id=correlation_id,
                    method="GET",
                )

                assert result == {"data": "success"}
                # Verify correlation logging was called
                enhanced_client.correlation_logger.log_with_correlation.assert_called()
                call_args = (
                    enhanced_client.correlation_logger.log_with_correlation.call_args_list
                )
                assert any(correlation_id in str(call) for call in call_args)

    @pytest.mark.asyncio
    async def test_make_request_with_execution_tracing(self, enhanced_client):
        """Test make_request uses execution tracing."""
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(return_value={"traced": "data"})
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                enhanced_client.execution_tracer.start_trace = AsyncMock(
                    return_value="trace-123"
                )
                enhanced_client.execution_tracer.add_trace_step = AsyncMock()
                enhanced_client.execution_tracer.end_trace = AsyncMock()

                result = await enhanced_client.make_request_with_tracing(
                    "https://api.example.com/test", operation="api_request"
                )

                assert result == {"traced": "data"}
                # Verify tracing was used
                enhanced_client.execution_tracer.start_trace.assert_called_once_with(
                    operation="api_request",
                    context={
                        "url": "https://api.example.com/test",
                        "service": "enhanced_service",
                        "method": "GET",
                    },
                )
                enhanced_client.execution_tracer.end_trace.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_error_with_context(self, enhanced_client):
        """Test error handling creates proper ErrorContext."""
        error = APIError("Test API error")

        with patch.object(
            enhanced_client.correlation_logger, "log_with_correlation"
        ) as mock_log:
            error_context = await enhanced_client.create_error_context(
                error,
                correlation_id="err-123",
                user_message="Service temporarily unavailable",
                trace_data={"endpoint": "/test", "status": 500},
            )

            assert isinstance(error_context, ErrorContext)
            assert error_context.correlation_id == "err-123"
            assert error_context.user_message == "Service temporarily unavailable"
            assert error_context.debug_info == "APIError: Test API error"
            assert error_context.trace_data["endpoint"] == "/test"
            assert error_context.severity == ErrorSeverity.ERROR

            # Verify error was logged with correlation
            mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhanced_graceful_degradation(self, enhanced_client):
        """Test graceful degradation uses enhanced GracefulDegradation."""
        correlation_id = "deg-456"

        async def failing_primary():
            raise APIError("Primary service down")

        with patch.object(
            GracefulDegradation, "execute_degradation_cascade"
        ) as mock_cascade:
            mock_cascade.return_value = {
                "strategy": "secondary_cache",
                "data": {"cached": "result"},
                "quality_score": 0.8,
            }

            result = await enhanced_client.handle_enhanced_graceful_degradation(
                failing_primary,
                correlation_id=correlation_id,
                context={"operation": "get_data", "service": "enhanced_service"},
            )

            assert result["data"] == {"cached": "result"}
            assert result["strategy"] == "secondary_cache"
            assert result["quality_score"] == 0.8

            # Verify cascade was called with proper context
            mock_cascade.assert_called_once()
            call_args = mock_cascade.call_args[0][0]
            assert call_args["correlation_id"] == correlation_id
            assert call_args["operation"] == "get_data"
            assert call_args["service"] == "enhanced_service"

    @pytest.mark.asyncio
    async def test_make_request_error_with_enhanced_handling(self, enhanced_client):
        """Test make_request error creates enhanced ErrorContext."""
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 500
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(
                    return_value={"error": "Internal server error"}
                )
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                enhanced_client.correlation_logger.log_with_correlation = AsyncMock()
                enhanced_client.execution_tracer.start_trace = AsyncMock(
                    return_value="trace-err"
                )
                enhanced_client.execution_tracer.end_trace = AsyncMock()

                with pytest.raises(APIError) as exc_info:
                    await enhanced_client.make_request_with_enhanced_error_handling(
                        "https://api.example.com/error", correlation_id="err-789"
                    )

                # Verify error context was created (correlation logger called)
                enhanced_client.correlation_logger.log_with_correlation.assert_called()
                # Verify trace was completed with error
                enhanced_client.execution_tracer.end_trace.assert_called_with(
                    trace_id="trace-err", status="error", error=unittest.mock.ANY
                )

    @pytest.mark.asyncio
    async def test_propagate_correlation_id_through_request_chain(
        self, enhanced_client
    ):
        """Test correlation ID propagation through request chains."""
        parent_correlation_id = "parent-123"

        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(return_value={"chain": "success"})
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                enhanced_client.correlation_logger.log_with_correlation = AsyncMock()

                # Simulate request chain with parent correlation ID
                result = await enhanced_client.make_request_with_correlation_chain(
                    "https://api.example.com/chain",
                    parent_correlation_id=parent_correlation_id,
                )

                assert result == {"chain": "success"}

                # Verify parent correlation ID was used in logging
                log_calls = (
                    enhanced_client.correlation_logger.log_with_correlation.call_args_list
                )
                assert any(
                    call[1].get("parent_correlation_id") == parent_correlation_id
                    for call in log_calls
                )

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration_with_enhanced_error_context(
        self, enhanced_client
    ):
        """Test circuit breaker integration creates enhanced error context."""
        # Mock circuit breaker to trigger failure
        enhanced_client.correlation_logger.log_with_correlation = AsyncMock()

        # Mock the rate limiter to allow request but have the HTTP request fail
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager, patch(
            "src.integrations.clients.base_client.aiohttp.ClientSession"
        ) as mock_session_class:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            # Mock session to raise a connection error
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.request = Mock(side_effect=aiohttp.ClientError("Connection failed"))

            with pytest.raises(APIError) as exc_info:
                await enhanced_client.make_request_with_enhanced_error_handling(
                    "https://api.example.com/breaker", correlation_id="breaker-123"
                )

            # Verify error was logged with correlation
            enhanced_client.correlation_logger.log_with_correlation.assert_called()
            log_calls = (
                enhanced_client.correlation_logger.log_with_correlation.call_args_list
            )
            assert any("breaker-123" in str(call) for call in log_calls)

    @pytest.mark.asyncio
    async def test_parse_response_methods_coverage(self, enhanced_client):
        """Test _parse_response method edge cases for coverage."""
        # Test various response types for complete coverage
        mock_response = AsyncMock()

        # Test JSON response
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={"test": "json"})
        result = await enhanced_client._parse_response(mock_response)
        assert result == {"test": "json"}

        # Test XML response
        mock_response.headers = {"content-type": "application/xml"}
        mock_response.text = AsyncMock(return_value="<xml>test</xml>")
        result = await enhanced_client._parse_response(mock_response)
        assert result == "<xml>test</xml>"

        # Test plain text response
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = AsyncMock(return_value="plain text")
        result = await enhanced_client._parse_response(mock_response)
        assert result == "plain text"

    @pytest.mark.asyncio
    async def test_graceful_degradation_cache_exception_coverage(self, enhanced_client):
        """Test graceful degradation when cache manager throws exception."""

        async def failing_primary():
            raise APIError("Primary failed")

        async def working_fallback():
            return {"fallback": "data"}

        # Mock cache manager to throw exception
        enhanced_client.cache_manager.get = AsyncMock(
            side_effect=Exception("Cache error")
        )

        result = await enhanced_client.handle_enhanced_graceful_degradation(
            primary_func=failing_primary,
            correlation_id="cache-err-123",
            context={"operation": "test_cache_error"},
            fallback_func=working_fallback,
            cache_key="test_cache_key",
        )

        # Should use GracefulDegradation.execute_degradation_cascade
        assert "source" in result  # GracefulDegradation returns source field
        assert result["degraded"] is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_correlation_logging_coverage(self, enhanced_client):
        """Test circuit breaker error logging with correlation ID."""
        # Set a correlation ID attribute to trigger the circuit breaker logging path
        enhanced_client._current_correlation_id = "circuit-123"
        enhanced_client.correlation_logger.log_with_correlation = AsyncMock()

        # Mock circuit breaker to fail
        enhanced_client.circuit_breaker.call_with_breaker = AsyncMock(
            side_effect=Exception("Circuit breaker open")
        )

        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)

            with pytest.raises(Exception, match="Circuit breaker open"):
                await enhanced_client.make_request("https://api.example.com/circuit")

            # Verify circuit breaker error was logged with correlation
            enhanced_client.correlation_logger.log_with_correlation.assert_called()

    def test_base_client_without_enhanced_components(self):
        """Test BaseClient initialization without enhanced error handling components."""
        with patch("src.integrations.clients.base_client.rate_limit_manager"):
            client = BaseClient(
                service_name="basic_service",
                correlation_logger=None,
                execution_tracer=None,
            )

            assert client.service_name == "basic_service"
            assert client.correlation_logger is None
            assert client.execution_tracer is None
            assert client.timeout == 30.0  # Default timeout

    @pytest.mark.asyncio
    async def test_make_request_methods_without_enhanced_components(
        self, enhanced_client
    ):
        """Test enhanced methods gracefully handle missing components."""
        # Remove enhanced components
        enhanced_client.correlation_logger = None
        enhanced_client.execution_tracer = None

        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(return_value={"no_enhanced": "success"})
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                # Should work without enhanced components
                result = await enhanced_client.make_request_with_correlation(
                    "https://api.example.com/basic", correlation_id="basic-123"
                )

                assert result == {"no_enhanced": "success"}

                # Should work without tracing
                result = await enhanced_client.make_request_with_tracing(
                    "https://api.example.com/basic", operation="basic_operation"
                )

                assert result == {"no_enhanced": "success"}

    @pytest.mark.asyncio
    async def test_make_request_with_correlation_exception_coverage(
        self, enhanced_client
    ):
        """Test make_request_with_correlation exception path for coverage."""
        enhanced_client.correlation_logger.log_with_correlation = AsyncMock()

        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(
                return_value=False
            )  # Force rate limit failure

            with pytest.raises(APIError, match="Rate limit exceeded"):
                await enhanced_client.make_request_with_correlation(
                    "https://api.example.com/fail", correlation_id="fail-123"
                )

            # Verify error was logged
            enhanced_client.correlation_logger.log_with_correlation.assert_called()
            calls = (
                enhanced_client.correlation_logger.log_with_correlation.call_args_list
            )
            error_calls = [call for call in calls if call[1].get("level") == "error"]
            assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_make_request_with_tracing_exception_coverage(self, enhanced_client):
        """Test make_request_with_tracing exception path for coverage."""
        enhanced_client.execution_tracer.start_trace = AsyncMock(
            return_value="trace-fail"
        )
        enhanced_client.execution_tracer.end_trace = AsyncMock()

        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(
                return_value=False
            )  # Force rate limit failure

            with pytest.raises(APIError, match="Rate limit exceeded"):
                await enhanced_client.make_request_with_tracing(
                    "https://api.example.com/trace-fail", operation="fail_operation"
                )

            # Verify trace was ended with error
            enhanced_client.execution_tracer.end_trace.assert_called_with(
                trace_id="trace-fail", status="error", error=unittest.mock.ANY
            )

    @pytest.mark.asyncio
    async def test_enhanced_graceful_degradation_primary_success_coverage(
        self, enhanced_client
    ):
        """Test enhanced graceful degradation primary success path for coverage."""

        async def successful_primary():
            return {"primary": "success"}

        result = await enhanced_client.handle_enhanced_graceful_degradation(
            primary_func=successful_primary,
            correlation_id="success-123",
            context={"operation": "test_primary_success"},
        )

        # Should return primary success path
        assert result == {
            "data": {"primary": "success"},
            "strategy": "primary",
            "quality_score": 1.0,
        }

    @pytest.mark.asyncio
    async def test_make_request_with_enhanced_error_handling_success_coverage(
        self, enhanced_client
    ):
        """Test make_request_with_enhanced_error_handling success path for coverage."""
        enhanced_client.execution_tracer.start_trace = AsyncMock(
            return_value="success-trace"
        )
        enhanced_client.execution_tracer.end_trace = AsyncMock()
        enhanced_client.correlation_logger.log_with_correlation = AsyncMock()

        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            with patch("aiohttp.ClientSession") as mock_session_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {"content-type": "application/json"}
                mock_response.json = AsyncMock(return_value={"enhanced": "success"})
                mock_session.request = Mock(return_value=mock_response)
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_session_class.return_value.__aenter__ = AsyncMock(
                    return_value=mock_session
                )
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

                result = (
                    await enhanced_client.make_request_with_enhanced_error_handling(
                        "https://api.example.com/enhanced-success",
                        correlation_id="enhanced-success-123",
                    )
                )

                assert result == {"enhanced": "success"}

                # Verify successful trace completion
                enhanced_client.execution_tracer.end_trace.assert_called_with(
                    trace_id="success-trace",
                    status="success",
                    result={
                        "response_size": unittest.mock.ANY,
                        "response_type": "dict",
                    },
                )

    @pytest.mark.asyncio
    async def test_make_request_with_correlation_chain_exception_coverage(
        self, enhanced_client
    ):
        """Test make_request_with_correlation_chain exception path for coverage."""
        enhanced_client.correlation_logger.log_with_correlation = AsyncMock()

        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager:
            mock_rate_manager.acquire = AsyncMock(
                return_value=False
            )  # Force rate limit failure

            with pytest.raises(APIError, match="Rate limit exceeded"):
                await enhanced_client.make_request_with_correlation_chain(
                    "https://api.example.com/chain-fail",
                    parent_correlation_id="parent-123",
                )

            # Verify error was logged in chain context
            enhanced_client.correlation_logger.log_with_correlation.assert_called()
            calls = (
                enhanced_client.correlation_logger.log_with_correlation.call_args_list
            )
            error_calls = [call for call in calls if call[1].get("level") == "error"]
            assert len(error_calls) > 0
            # Verify parent correlation ID was used
            parent_calls = [
                call
                for call in calls
                if call[1].get("parent_correlation_id") == "parent-123"
            ]
            assert len(parent_calls) > 0

    @pytest.mark.asyncio
    async def test_correlation_chain_header_propagation(self, enhanced_client):
        """Test correlation chain adds proper HTTP headers."""
        parent_id = "parent-123"
        mock_response = {"data": "test"}
        enhanced_client.correlation_logger.log_with_correlation = AsyncMock()
        
        with patch.object(
            enhanced_client, "make_request", return_value=mock_response
        ) as mock_request:
            await enhanced_client.make_request_with_correlation_chain(
                url="https://api.test.com/endpoint",
                parent_correlation_id=parent_id,
                method="GET",
                endpoint="/test"
            )
            
            # Check that headers were added to the request
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            headers = call_args.kwargs.get("headers", {})
            
            # Should have correlation headers
            assert "X-Correlation-ID" in headers
            assert "X-Parent-Correlation-ID" in headers
            assert headers["X-Parent-Correlation-ID"] == parent_id
            assert "X-Request-Chain-Depth" in headers
            assert headers["X-Request-Chain-Depth"] == "1"  # First level


class TestBaseClientMissingCoverage:
    """Test class to achieve 100% coverage for missing lines."""

    @pytest.fixture
    def base_client(self):
        """Create a basic BaseClient for testing."""
        return BaseClient("test_service")

    @pytest.mark.asyncio
    async def test_handle_rate_limit_response_429_coverage(self, base_client):
        """Test 429 rate limit response handling (lines 179-181)."""
        with patch(
            "src.integrations.clients.base_client.rate_limit_manager"
        ) as mock_rate_manager, patch(
            "src.integrations.clients.base_client.aiohttp.ClientSession"
        ) as mock_session_class:
            mock_rate_manager.acquire = AsyncMock(return_value=True)
            mock_rate_manager.record_response = Mock()

            # Mock session and response
            mock_session = Mock()
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.headers = {"content-type": "application/json", "Retry-After": "60"}
            mock_response.json = AsyncMock(return_value={"error": "Rate limit exceeded"})
            
            mock_session.request = Mock(return_value=mock_response)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock the handle_rate_limit_response and _request_with_retry methods
            base_client.handle_rate_limit_response = AsyncMock()
            base_client._request_with_retry = AsyncMock(return_value={"success": True})

            result = await base_client.make_request("https://api.example.com/test")
            
            # Verify 429 handling was called
            base_client.handle_rate_limit_response.assert_called_once_with(mock_response)
            base_client._request_with_retry.assert_called_once()
            assert result == {"success": True}

    @pytest.mark.asyncio  
    async def test_handle_rate_limit_response_default_implementation(self, base_client):
        """Test default handle_rate_limit_response implementation (line 654)."""
        mock_response = Mock()
        mock_response.headers = {"Retry-After": "30"}
        
        with patch.object(base_client, '_generic_rate_limit_backoff', new_callable=AsyncMock) as mock_backoff:
            await base_client.handle_rate_limit_response(mock_response)
            mock_backoff.assert_called_once_with(mock_response)

    @pytest.mark.asyncio
    async def test_calculate_backoff_delay_implementation(self, base_client):
        """Test calculate_backoff_delay implementation (lines 681-684)."""
        mock_response = Mock()
        
        with patch('random.uniform', return_value=0.2):
            # Test different attempt numbers
            delay_0 = await base_client.calculate_backoff_delay(mock_response, attempt=0)
            delay_1 = await base_client.calculate_backoff_delay(mock_response, attempt=1)
            delay_2 = await base_client.calculate_backoff_delay(mock_response, attempt=2)
            
            # Verify exponential backoff: 1s, 2s, 4s base + jitter
            assert delay_0 == 1.2  # 1 + (0.2 * 1)
            assert delay_1 == 2.4  # 2 + (0.2 * 2) 
            assert delay_2 == 4.8  # 4 + (0.2 * 4)

    @pytest.mark.asyncio
    async def test_generic_rate_limit_backoff_with_retry_after(self, base_client):
        """Test _generic_rate_limit_backoff with Retry-After header (lines 688-705)."""
        mock_response = Mock()
        mock_response.headers = {"Retry-After": "45"}
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await base_client._generic_rate_limit_backoff(mock_response)
            mock_sleep.assert_called_once_with(45)

    @pytest.mark.asyncio
    async def test_generic_rate_limit_backoff_invalid_retry_after(self, base_client):
        """Test _generic_rate_limit_backoff with invalid Retry-After (lines 695-697)."""
        mock_response = Mock()
        mock_response.headers = {"Retry-After": "invalid"}
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await base_client._generic_rate_limit_backoff(mock_response)
            mock_sleep.assert_called_once_with(60)  # Default fallback

    @pytest.mark.asyncio
    async def test_generic_rate_limit_backoff_no_header(self, base_client):
        """Test _generic_rate_limit_backoff without Retry-After header (lines 698-705)."""
        mock_response = Mock()
        mock_response.headers = {}
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await base_client._generic_rate_limit_backoff(mock_response)
            mock_sleep.assert_called_once_with(60)  # Default backoff

    @pytest.mark.asyncio
    async def test_request_with_retry_success_after_retries(self, base_client):
        """Test _request_with_retry success after initial failures (lines 720-743)."""
        call_count = 0
        
        async def mock_request_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError("429 Rate limit")
            return {"success": True}
        
        with patch('asyncio.sleep', new_callable=AsyncMock), \
             patch.object(base_client, 'calculate_backoff_delay', return_value=1.0):
            
            result = await base_client._request_with_retry(mock_request_func, max_retries=3)
            assert result == {"success": True}
            assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_request_with_retry_exhausted_retries(self, base_client):
        """Test _request_with_retry when all retries exhausted (lines 727-729)."""
        async def mock_request_func():
            raise APIError("429 Rate limit")
        
        with patch('asyncio.sleep', new_callable=AsyncMock), \
             patch.object(base_client, 'calculate_backoff_delay', return_value=1.0):
            
            with pytest.raises(APIError) as exc_info:
                await base_client._request_with_retry(mock_request_func, max_retries=2)
            
            assert "429 Rate limit" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_with_retry_non_rate_limit_error(self, base_client):
        """Test _request_with_retry with non-rate-limit error (lines 742-743)."""
        async def mock_request_func():
            raise APIError("Server error")
        
        with pytest.raises(APIError) as exc_info:
            await base_client._request_with_retry(mock_request_func, max_retries=2)
        
        assert "Server error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_make_graphql_request_with_variables(self, base_client):
        """Test make_graphql_request with variables (lines 773-815)."""
        query = "query GetAnime($id: Int!) { anime(id: $id) { title } }"
        variables = {"id": 123}
        
        with patch.object(base_client, 'make_request_with_enhanced_error_handling', 
                         new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": {"anime": {"title": "Test Anime"}}}
            
            result = await base_client.make_graphql_request(
                url="https://api.example.com/graphql",
                query=query,
                variables=variables,
                correlation_id="test-123"
            )
            
            assert result == {"data": {"anime": {"title": "Test Anime"}}}
            
            # Verify request was made with correct payload
            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args.kwargs
            assert call_kwargs["json"]["query"] == query
            assert call_kwargs["json"]["variables"] == variables
            assert call_kwargs["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_make_graphql_request_without_correlation_id(self, base_client):
        """Test make_graphql_request auto-generates correlation ID (lines 798-807)."""
        query = "{ anime { title } }"
        
        with patch.object(base_client, 'make_request_with_enhanced_error_handling', 
                         new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": {"anime": {"title": "Test"}}}
            
            result = await base_client.make_graphql_request(
                url="https://api.example.com/graphql",
                query=query
            )
            
            assert result == {"data": {"anime": {"title": "Test"}}}
            
            # Verify correlation ID was auto-generated
            mock_request.assert_called_once()
            call_kwargs = mock_request.call_args.kwargs
            assert call_kwargs["correlation_id"].startswith("graphql-")

    @pytest.mark.asyncio
    async def test_make_graphql_request_with_errors(self, base_client):
        """Test make_graphql_request with GraphQL errors (lines 810-813)."""
        query = "{ invalid_field }"
        
        with patch.object(base_client, 'make_request_with_enhanced_error_handling', 
                         new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "errors": [{"message": "Field 'invalid_field' doesn't exist"}]
            }
            
            with pytest.raises(APIError) as exc_info:
                await base_client.make_graphql_request(
                    url="https://api.example.com/graphql",
                    query=query,
                    correlation_id="error-test"
                )
            
            assert "GraphQL error: Field 'invalid_field' doesn't exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_monitor_rate_limits_default_implementation(self, base_client):
        """Test default monitor_rate_limits implementation (no-op)."""
        mock_response = Mock()
        # Should not raise any exceptions - it's a no-op
        await base_client.monitor_rate_limits(mock_response)
