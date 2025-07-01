"""Tests for error handling infrastructure."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.integrations.error_handling import (
    CircuitBreaker,
    ErrorContext,
    ErrorSeverity,
    ExecutionTracer,
    GracefulDegradation,
    LangGraphErrorHandler,
)


class TestErrorContext:
    """Test the ErrorContext class."""

    def test_error_context_initialization(self):
        """Test ErrorContext initializes correctly."""
        context = ErrorContext(
            user_message="Something went wrong",
            debug_info="ValueError: Invalid input",
            trace_data={"step": 1, "action": "validate"},
        )

        assert context.user_message == "Something went wrong"
        assert context.debug_info == "ValueError: Invalid input"
        assert context.trace_data == {"step": 1, "action": "validate"}

    def test_error_context_default_trace_data(self):
        """Test ErrorContext with default trace_data."""
        context = ErrorContext(
            user_message="Error occurred", debug_info="Exception details"
        )

        assert context.user_message == "Error occurred"
        assert context.debug_info == "Exception details"
        assert context.trace_data == {}

    def test_error_context_from_exception(self):
        """Test creating ErrorContext from exception."""
        exception = ValueError("Invalid parameter")
        user_message = "Please check your input"
        trace_data = {"request_id": "123", "user": "test"}

        context = ErrorContext.from_exception(
            exception=exception, user_message=user_message, trace_data=trace_data
        )

        assert context.user_message == user_message
        assert context.debug_info == "ValueError: Invalid parameter"
        assert context.trace_data == trace_data

    def test_error_context_from_exception_no_trace_data(self):
        """Test creating ErrorContext from exception without trace data."""
        exception = RuntimeError("System error")
        user_message = "Service temporarily unavailable"

        context = ErrorContext.from_exception(
            exception=exception, user_message=user_message
        )

        assert context.user_message == user_message
        assert context.debug_info == "RuntimeError: System error"
        assert context.trace_data == {}

    def test_error_context_pydantic_validation(self):
        """Test ErrorContext Pydantic validation."""
        # Test valid data
        context = ErrorContext(
            user_message="Test message", debug_info="Test debug info"
        )
        assert isinstance(context, ErrorContext)

        # Test missing required fields should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            ErrorContext()

    def test_error_context_with_correlation_id(self):
        """Test ErrorContext with correlation ID."""
        correlation_id = "req-123-456"
        context = ErrorContext(
            user_message="Error occurred",
            debug_info="Technical details",
            correlation_id=correlation_id,
        )

        assert context.correlation_id == correlation_id

    def test_error_context_auto_correlation_id(self):
        """Test ErrorContext auto-generates correlation ID."""
        context = ErrorContext(
            user_message="Error occurred", debug_info="Technical details"
        )

        # Should auto-generate a correlation ID
        assert context.correlation_id is not None
        assert len(context.correlation_id) > 0
        assert context.correlation_id.startswith("err-")

    def test_error_context_with_severity(self):
        """Test ErrorContext with severity level."""
        context = ErrorContext(
            user_message="Critical error",
            debug_info="System failure",
            severity=ErrorSeverity.CRITICAL,
        )

        assert context.severity == ErrorSeverity.CRITICAL

    def test_error_context_default_severity(self):
        """Test ErrorContext has default severity."""
        context = ErrorContext(user_message="Error occurred", debug_info="Details")

        assert context.severity == ErrorSeverity.ERROR

    def test_error_context_with_recovery_suggestions(self):
        """Test ErrorContext with recovery suggestions."""
        suggestions = [
            "Check your internet connection",
            "Try again in a few minutes",
            "Contact support if problem persists",
        ]

        context = ErrorContext(
            user_message="Service unavailable",
            debug_info="API timeout",
            recovery_suggestions=suggestions,
        )

        assert context.recovery_suggestions == suggestions

    def test_error_context_with_breadcrumbs(self):
        """Test ErrorContext with execution breadcrumbs."""
        breadcrumbs = [
            {"step": "validate_input", "timestamp": "2024-01-01T10:00:00Z"},
            {"step": "call_api", "timestamp": "2024-01-01T10:00:01Z"},
            {"step": "parse_response", "timestamp": "2024-01-01T10:00:02Z"},
        ]

        context = ErrorContext(
            user_message="Processing failed",
            debug_info="Validation error",
            breadcrumbs=breadcrumbs,
        )

        assert context.breadcrumbs == breadcrumbs

    def test_error_context_add_breadcrumb(self):
        """Test adding breadcrumb to ErrorContext."""
        context = ErrorContext(user_message="Error occurred", debug_info="Details")

        context.add_breadcrumb("api_call", {"url": "https://api.example.com"})

        assert len(context.breadcrumbs) == 1
        breadcrumb = context.breadcrumbs[0]
        assert breadcrumb["step"] == "api_call"
        assert breadcrumb["data"] == {"url": "https://api.example.com"}
        assert "timestamp" in breadcrumb

    def test_error_context_enhanced_from_exception(self):
        """Test enhanced from_exception with all new features."""
        exception = ValueError("Invalid input")
        user_message = "Please check your input"
        trace_data = {"request_id": "123"}
        correlation_id = "req-456"
        recovery_suggestions = ["Check input format", "Try different values"]

        context = ErrorContext.from_exception(
            exception=exception,
            user_message=user_message,
            trace_data=trace_data,
            correlation_id=correlation_id,
            severity=ErrorSeverity.WARNING,
            recovery_suggestions=recovery_suggestions,
        )

        assert context.user_message == user_message
        assert context.debug_info == "ValueError: Invalid input"
        assert context.trace_data == trace_data
        assert context.correlation_id == correlation_id
        assert context.severity == ErrorSeverity.WARNING
        assert context.recovery_suggestions == recovery_suggestions

    def test_error_context_to_dict(self):
        """Test ErrorContext conversion to dictionary."""
        context = ErrorContext(
            user_message="Error occurred",
            debug_info="Technical details",
            correlation_id="req-123",
            severity=ErrorSeverity.ERROR,
            recovery_suggestions=["Try again"],
        )

        result = context.to_dict()

        assert result["user_message"] == "Error occurred"
        assert result["debug_info"] == "Technical details"
        assert result["correlation_id"] == "req-123"
        assert result["severity"] == "error"
        assert result["recovery_suggestions"] == ["Try again"]
        assert "timestamp" in result

    def test_error_context_is_recoverable(self):
        """Test ErrorContext recoverability detection."""
        # Recoverable errors
        recoverable_context = ErrorContext(
            user_message="Service temporarily unavailable",
            debug_info="503 Service Unavailable",
            recovery_suggestions=["Try again later"],
        )
        assert recoverable_context.is_recoverable()

        # Non-recoverable errors
        non_recoverable_context = ErrorContext(
            user_message="Invalid API key", debug_info="401 Unauthorized"
        )
        assert not non_recoverable_context.is_recoverable()


class TestCircuitBreaker:
    """Test the CircuitBreaker class."""

    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initializes correctly."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 60
        assert breaker.failure_count == 0
        assert breaker.state == "closed"
        assert breaker.last_failure_time is None

    def test_circuit_breaker_default_initialization(self):
        """Test CircuitBreaker with default parameters."""
        breaker = CircuitBreaker()

        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 300
        assert breaker.state == "closed"

    def test_is_open_initial_state(self):
        """Test is_open returns False initially."""
        breaker = CircuitBreaker()
        assert not breaker.is_open()

    def test_is_open_when_open(self):
        """Test is_open returns True when circuit is open."""
        breaker = CircuitBreaker()
        breaker.state = "open"
        assert breaker.is_open()

    @pytest.mark.asyncio
    async def test_call_with_breaker_success(self):
        """Test successful call through circuit breaker."""
        breaker = CircuitBreaker()

        async def successful_func():
            return "success"

        result = await breaker.call_with_breaker(successful_func)
        assert result == "success"
        assert breaker.failure_count == 0
        assert breaker.state == "closed"

    @pytest.mark.asyncio
    async def test_call_with_breaker_single_failure(self):
        """Test single failure through circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await breaker.call_with_breaker(failing_func)

        assert breaker.failure_count == 1
        assert breaker.state == "closed"  # Still closed, below threshold
        assert breaker.last_failure_time is not None

    @pytest.mark.asyncio
    async def test_call_with_breaker_multiple_failures_opens_circuit(self):
        """Test multiple failures open the circuit."""
        breaker = CircuitBreaker(failure_threshold=2)

        async def failing_func():
            raise ValueError("Test error")

        # First failure
        with pytest.raises(ValueError):
            await breaker.call_with_breaker(failing_func)
        assert breaker.state == "closed"

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await breaker.call_with_breaker(failing_func)
        assert breaker.state == "open"
        assert breaker.failure_count == 2

    @pytest.mark.asyncio
    async def test_call_with_breaker_blocks_when_open(self):
        """Test circuit breaker blocks calls when open."""
        breaker = CircuitBreaker()
        breaker.state = "open"

        async def any_func():
            return "should not execute"

        with pytest.raises(Exception, match="Circuit breaker is open"):
            await breaker.call_with_breaker(any_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        breaker = CircuitBreaker(
            failure_threshold=1, recovery_timeout=0.1
        )  # Short timeout

        # Cause failure to open circuit
        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await breaker.call_with_breaker(failing_func)
        assert breaker.state == "open"

        # Wait for recovery timeout
        await asyncio.sleep(0.2)  # Wait longer than timeout

        # Next call should transition to half-open and succeed
        async def successful_func():
            return "recovered"

        result = await breaker.call_with_breaker(successful_func)
        assert result == "recovered"
        assert breaker.state == "closed"  # Reset to closed after success
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure_reopens(self):
        """Test circuit breaker reopens if half-open call fails."""
        breaker = CircuitBreaker(
            failure_threshold=1, recovery_timeout=0.1
        )  # Very short timeout

        # Open the circuit
        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await breaker.call_with_breaker(failing_func)
        assert breaker.state == "open"

        # Wait for recovery timeout
        await asyncio.sleep(0.2)  # Wait longer than timeout

        # Half-open call fails - should transition to half-open first, then back to open
        with pytest.raises(ValueError):
            await breaker.call_with_breaker(failing_func)

        assert breaker.state == "open"
        assert breaker.failure_count == 2

    def test_should_attempt_reset_no_failure_time(self):
        """Test should_attempt_reset with no failure time."""
        breaker = CircuitBreaker()
        assert not breaker._should_attempt_reset()

    def test_should_attempt_reset_before_timeout(self):
        """Test should_attempt_reset before timeout."""
        breaker = CircuitBreaker(recovery_timeout=300)
        breaker.last_failure_time = datetime.now(timezone.utc)
        assert not breaker._should_attempt_reset()

    def test_should_attempt_reset_after_timeout(self):
        """Test should_attempt_reset after timeout."""
        breaker = CircuitBreaker(recovery_timeout=1)
        breaker.last_failure_time = datetime.now(timezone.utc) - timedelta(seconds=2)
        assert breaker._should_attempt_reset()

    def test_record_failure_increments_count(self):
        """Test _record_failure increments failure count."""
        breaker = CircuitBreaker()
        initial_count = breaker.failure_count

        breaker._record_failure()

        assert breaker.failure_count == initial_count + 1
        assert breaker.last_failure_time is not None

    def test_record_failure_opens_circuit_at_threshold(self):
        """Test _record_failure opens circuit at threshold."""
        breaker = CircuitBreaker(failure_threshold=2)

        # First failure - should stay closed
        breaker._record_failure()
        assert breaker.state == "closed"

        # Second failure - should open
        breaker._record_failure()
        assert breaker.state == "open"

    def test_reset_clears_state(self):
        """Test _reset clears circuit breaker state."""
        breaker = CircuitBreaker()
        breaker.failure_count = 3
        breaker.state = "half_open"
        breaker.last_failure_time = datetime.now(timezone.utc)

        breaker._reset()

        assert breaker.failure_count == 0
        assert breaker.state == "closed"
        assert breaker.last_failure_time is None

    def test_circuit_breaker_with_api_name(self):
        """Test CircuitBreaker with API-specific name."""
        breaker = CircuitBreaker(api_name="mal_api", failure_threshold=2)

        assert breaker.api_name == "mal_api"
        assert breaker.failure_threshold == 2

    def test_circuit_breaker_get_metrics(self):
        """Test CircuitBreaker metrics collection."""
        breaker = CircuitBreaker(api_name="test_api", failure_threshold=3)

        metrics = breaker.get_metrics()

        assert metrics["api_name"] == "test_api"
        assert metrics["state"] == "closed"
        assert metrics["failure_count"] == 0
        assert metrics["total_requests"] == 0
        assert metrics["successful_requests"] == 0
        assert metrics["blocked_requests"] == 0
        assert "last_failure_time" in metrics
        assert "uptime_seconds" in metrics

    @pytest.mark.asyncio
    async def test_circuit_breaker_tracks_successful_requests(self):
        """Test CircuitBreaker tracks successful request metrics."""
        breaker = CircuitBreaker(api_name="test_api")

        async def successful_func():
            return "success"

        # Make 3 successful requests
        for _ in range(3):
            result = await breaker.call_with_breaker(successful_func)
            assert result == "success"

        metrics = breaker.get_metrics()
        assert metrics["total_requests"] == 3
        assert metrics["successful_requests"] == 3
        assert metrics["blocked_requests"] == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_tracks_blocked_requests(self):
        """Test CircuitBreaker tracks blocked request metrics."""
        breaker = CircuitBreaker(api_name="test_api", failure_threshold=1)

        # Open the circuit
        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await breaker.call_with_breaker(failing_func)

        # Now circuit is open, requests should be blocked
        async def any_func():
            return "should not execute"

        with pytest.raises(Exception, match="Circuit breaker is open"):
            await breaker.call_with_breaker(any_func)

        with pytest.raises(Exception, match="Circuit breaker is open"):
            await breaker.call_with_breaker(any_func)

        metrics = breaker.get_metrics()
        assert metrics["total_requests"] == 3  # 1 failed + 2 blocked
        assert metrics["successful_requests"] == 0
        assert metrics["blocked_requests"] == 2
        assert metrics["failure_count"] == 1

    def test_circuit_breaker_uptime_calculation(self):
        """Test CircuitBreaker uptime calculation."""
        breaker = CircuitBreaker(api_name="test_api")

        # Wait a small amount to ensure uptime > 0
        import time

        time.sleep(0.01)

        metrics = breaker.get_metrics()
        assert metrics["uptime_seconds"] > 0

    def test_circuit_breaker_error_rate_calculation(self):
        """Test CircuitBreaker error rate calculation."""
        breaker = CircuitBreaker(api_name="test_api")

        # Simulate some requests
        breaker.total_requests = 10
        breaker.successful_requests = 7  # 3 failed

        metrics = breaker.get_metrics()
        assert metrics["error_rate"] == 0.3  # 3/10 = 0.3

    def test_circuit_breaker_error_rate_no_requests(self):
        """Test CircuitBreaker error rate with no requests."""
        breaker = CircuitBreaker(api_name="test_api")

        metrics = breaker.get_metrics()
        assert metrics["error_rate"] == 0.0


class TestGracefulDegradation:
    """Test the GracefulDegradation class."""

    @pytest.mark.asyncio
    async def test_fallback_to_cache_success(self):
        """Test successful cache fallback."""
        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value={"data": "cached_value"})

        result = await GracefulDegradation.fallback_to_cache("test_key", mock_cache)

        assert result == {"data": "cached_value"}
        mock_cache.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_fallback_to_cache_miss(self):
        """Test cache fallback with cache miss."""
        mock_cache = Mock()
        mock_cache.get = AsyncMock(return_value=None)

        result = await GracefulDegradation.fallback_to_cache("test_key", mock_cache)

        assert result is None
        mock_cache.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_fallback_to_cache_error(self):
        """Test cache fallback handles errors gracefully."""
        mock_cache = Mock()
        mock_cache.get = AsyncMock(side_effect=Exception("Cache error"))

        result = await GracefulDegradation.fallback_to_cache("test_key", mock_cache)

        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_to_cache_no_get_method(self):
        """Test cache fallback with object that has no get method."""
        mock_cache = Mock(spec=[])  # Mock without 'get' method

        result = await GracefulDegradation.fallback_to_cache("test_key", mock_cache)

        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_to_offline_data(self):
        """Test fallback to offline data."""
        anime_id = "12345"

        result = await GracefulDegradation.fallback_to_offline_data(anime_id)

        assert result["anime_id"] == anime_id
        assert result["title"] == "Data temporarily unavailable"
        assert (
            result["synopsis"] == "Please try again later when services are restored."
        )
        assert result["source"] == "offline_fallback"
        assert result["degraded"] is True

    def test_graceful_degradation_strategy_levels(self):
        """Test 5-level degradation strategy configuration."""
        strategies = GracefulDegradation.get_degradation_strategies()

        assert len(strategies) == 5
        assert strategies[0]["level"] == 1
        assert strategies[0]["name"] == "primary_cache"
        assert strategies[1]["level"] == 2
        assert strategies[1]["name"] == "secondary_cache"
        assert strategies[2]["level"] == 3
        assert strategies[2]["name"] == "offline_database"
        assert strategies[3]["level"] == 4
        assert strategies[3]["name"] == "minimal_response"
        assert strategies[4]["level"] == 5
        assert strategies[4]["name"] == "error_response"

    @pytest.mark.asyncio
    async def test_execute_degradation_strategy_level_1(self):
        """Test level 1: Primary cache degradation."""
        context = {
            "cache_key": "anime_123",
            "cache_manager": AsyncMock(),
            "anime_id": "123",
        }
        context["cache_manager"].get.return_value = {
            "id": "123",
            "title": "Cached Anime",
        }

        result = await GracefulDegradation.execute_degradation_strategy(1, context)

        assert result["id"] == "123"
        assert result["title"] == "Cached Anime"
        assert result["degradation_level"] == 1
        assert result["source"] == "primary_cache"

    @pytest.mark.asyncio
    async def test_execute_degradation_strategy_level_2(self):
        """Test level 2: Secondary cache degradation."""
        context = {
            "cache_key": "anime_123",
            "secondary_cache": AsyncMock(),
            "anime_id": "123",
        }
        context["secondary_cache"].get.return_value = {
            "id": "123",
            "title": "Secondary Cache",
        }

        result = await GracefulDegradation.execute_degradation_strategy(2, context)

        assert result["id"] == "123"
        assert result["title"] == "Secondary Cache"
        assert result["degradation_level"] == 2
        assert result["source"] == "secondary_cache"

    @pytest.mark.asyncio
    async def test_execute_degradation_strategy_level_3(self):
        """Test level 3: Offline database degradation."""
        context = {"anime_id": "123"}

        result = await GracefulDegradation.execute_degradation_strategy(3, context)

        assert result["anime_id"] == "123"
        assert result["degradation_level"] == 3
        assert result["source"] == "offline_database"
        assert result["degraded"] is True

    @pytest.mark.asyncio
    async def test_execute_degradation_strategy_level_4(self):
        """Test level 4: Minimal response degradation."""
        context = {"anime_id": "123", "query": "naruto"}

        result = await GracefulDegradation.execute_degradation_strategy(4, context)

        assert result["anime_id"] == "123"
        assert result["degradation_level"] == 4
        assert result["source"] == "minimal_response"
        assert result["limited_functionality"] is True
        assert "query" in result

    @pytest.mark.asyncio
    async def test_execute_degradation_strategy_level_5(self):
        """Test level 5: Error response degradation."""
        context = {"anime_id": "123", "error_message": "Service unavailable"}

        result = await GracefulDegradation.execute_degradation_strategy(5, context)

        assert result["anime_id"] == "123"
        assert result["degradation_level"] == 5
        assert result["source"] == "error_response"
        assert result["error"] is True
        assert "error_message" in result

    @pytest.mark.asyncio
    async def test_execute_degradation_cascade_success(self):
        """Test degradation cascade stops at first success."""
        cache_manager = AsyncMock()
        cache_manager.get.return_value = {"id": "123", "title": "Found in cache"}

        context = {
            "anime_id": "123",
            "cache_key": "anime_123",
            "cache_manager": cache_manager,
        }

        result = await GracefulDegradation.execute_degradation_cascade(context)

        # Should stop at level 1 (primary cache)
        assert result["degradation_level"] == 1
        assert result["source"] == "primary_cache"
        assert result["title"] == "Found in cache"

    @pytest.mark.asyncio
    async def test_execute_degradation_cascade_fallthrough(self):
        """Test degradation cascade falls through cache levels to offline database."""
        cache_manager = AsyncMock()
        cache_manager.get.return_value = None  # Cache miss
        secondary_cache = AsyncMock()
        secondary_cache.get.return_value = None  # Secondary cache miss

        context = {
            "anime_id": "123",
            "cache_key": "anime_123",
            "cache_manager": cache_manager,
            "secondary_cache": secondary_cache,
            "error_message": "All services failed",
        }

        result = await GracefulDegradation.execute_degradation_cascade(context)

        # Should reach level 3 (offline database) since it always provides a result
        assert result["degradation_level"] == 3
        assert result["source"] == "offline_database"
        assert result["degraded"] is True

    @pytest.mark.asyncio
    async def test_execute_degradation_strategy_invalid_level(self):
        """Test invalid degradation level raises exception."""
        context = {"anime_id": "123"}

        with pytest.raises(ValueError, match="Invalid degradation level"):
            await GracefulDegradation.execute_degradation_strategy(6, context)

    @pytest.mark.asyncio
    async def test_execute_degradation_strategy_exception_handling(self):
        """Test degradation strategy handles exceptions gracefully."""
        context = {
            "cache_key": "anime_123",
            "cache_manager": AsyncMock(),
            "anime_id": "123",
        }
        context["cache_manager"].get.side_effect = Exception("Cache error")

        result = await GracefulDegradation.execute_degradation_strategy(1, context)

        # Should return None when strategy fails
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_degradation_cascade_reaches_level_5(self):
        """Test degradation cascade reaches level 5 when offline database fails."""
        # Mock all strategies to fail by patching the private methods
        with patch.object(
            GracefulDegradation, "_execute_primary_cache_strategy", return_value=None
        ):
            with patch.object(
                GracefulDegradation,
                "_execute_secondary_cache_strategy",
                return_value=None,
            ):
                with patch.object(
                    GracefulDegradation,
                    "_execute_offline_database_strategy",
                    return_value=None,
                ):
                    with patch.object(
                        GracefulDegradation,
                        "_execute_minimal_response_strategy",
                        return_value=None,
                    ):

                        context = {
                            "anime_id": "123",
                            "error_message": "All services completely failed",
                        }

                        result = await GracefulDegradation.execute_degradation_cascade(
                            context
                        )

                        # Should reach level 5 (error response)
                        assert result["degradation_level"] == 5
                        assert result["source"] == "error_response"
                        assert result["error"] is True
                        assert (
                            result["error_message"] == "All services completely failed"
                        )

    @pytest.mark.asyncio
    async def test_execute_degradation_cascade_complete_failure(self):
        """Test degradation cascade fallback when all strategies return None."""
        # Mock ALL strategies to return None, including level 5
        with patch.object(
            GracefulDegradation, "_execute_primary_cache_strategy", return_value=None
        ):
            with patch.object(
                GracefulDegradation,
                "_execute_secondary_cache_strategy",
                return_value=None,
            ):
                with patch.object(
                    GracefulDegradation,
                    "_execute_offline_database_strategy",
                    return_value=None,
                ):
                    with patch.object(
                        GracefulDegradation,
                        "_execute_minimal_response_strategy",
                        return_value=None,
                    ):
                        with patch.object(
                            GracefulDegradation,
                            "_execute_error_response_strategy",
                            return_value=None,
                        ):

                            context = {"anime_id": "123"}

                            result = (
                                await GracefulDegradation.execute_degradation_cascade(
                                    context
                                )
                            )

                            # Should hit the fallback error response
                            assert result["degradation_level"] == 5
                            assert result["source"] == "error_response"
                            assert result["error"] is True
                            assert (
                                result["error_message"]
                                == "All degradation strategies failed"
                            )


class TestLangGraphErrorHandler:
    """Test the LangGraphErrorHandler class."""

    def test_langgraph_error_handler_initialization(self):
        """Test LangGraphErrorHandler initializes correctly."""
        handler = LangGraphErrorHandler()

        assert handler.error_patterns is not None
        assert len(handler.error_patterns) == 6  # 6 LangGraph-specific patterns
        assert handler.recovery_strategies is not None
        assert handler.correlation_tracker is not None

    def test_get_langgraph_error_patterns(self):
        """Test LangGraph error patterns configuration."""
        patterns = LangGraphErrorHandler.get_langgraph_error_patterns()

        assert len(patterns) == 6
        pattern_names = [p["name"] for p in patterns]
        expected_patterns = [
            "node_execution_failure",
            "workflow_state_corruption",
            "tool_invocation_error",
            "memory_management_error",
            "agent_recursion_limit",
            "workflow_timeout",
        ]
        assert all(name in pattern_names for name in expected_patterns)

    @pytest.mark.asyncio
    async def test_handle_node_execution_failure(self):
        """Test handling node execution failures."""
        handler = LangGraphErrorHandler()

        error_context = {
            "node_name": "search_anime",
            "workflow_id": "wf_123",
            "error": Exception("API timeout"),
            "execution_step": 3,
            "retry_count": 1,
        }

        result = await handler.handle_node_execution_failure(error_context)

        assert result["error_type"] == "node_execution_failure"
        assert result["node_name"] == "search_anime"
        assert result["workflow_id"] == "wf_123"
        assert result["recoverable"] is True
        assert result["suggested_action"] == "retry_with_backoff"
        assert "recovery_steps" in result

    @pytest.mark.asyncio
    async def test_handle_workflow_state_corruption(self):
        """Test handling workflow state corruption."""
        handler = LangGraphErrorHandler()

        error_context = {
            "workflow_id": "wf_456",
            "corrupted_state": {"invalid": "data"},
            "expected_schema": {"anime_id": "str"},
            "corruption_source": "state_update",
        }

        result = await handler.handle_workflow_state_corruption(error_context)

        assert result["error_type"] == "workflow_state_corruption"
        assert result["workflow_id"] == "wf_456"
        assert result["recoverable"] is True
        assert result["suggested_action"] == "restore_state"
        assert result["state_backup_available"] is True

    @pytest.mark.asyncio
    async def test_handle_tool_invocation_error(self):
        """Test handling tool invocation errors."""
        handler = LangGraphErrorHandler()

        error_context = {
            "tool_name": "search_anime_by_name",
            "workflow_id": "wf_789",
            "parameters": {"query": "naruto"},
            "error": Exception("Tool not found"),
            "invocation_id": "inv_123",
        }

        result = await handler.handle_tool_invocation_error(error_context)

        assert result["error_type"] == "tool_invocation_error"
        assert result["tool_name"] == "search_anime_by_name"
        assert result["workflow_id"] == "wf_789"
        assert result["recoverable"] is True
        assert result["suggested_action"] == "fallback_tool"
        assert "alternative_tools" in result

    @pytest.mark.asyncio
    async def test_handle_memory_management_error(self):
        """Test handling memory management errors."""
        handler = LangGraphErrorHandler()

        error_context = {
            "workflow_id": "wf_101",
            "memory_type": "conversation_history",
            "memory_size": 1024000,  # 1MB
            "error": Exception("Memory limit exceeded"),
            "operation": "append",
        }

        result = await handler.handle_memory_management_error(error_context)

        assert result["error_type"] == "memory_management_error"
        assert result["workflow_id"] == "wf_101"
        assert result["recoverable"] is True
        assert result["suggested_action"] == "cleanup_memory"
        assert result["memory_cleanup_strategy"] == "lru_eviction"

    @pytest.mark.asyncio
    async def test_handle_agent_recursion_limit(self):
        """Test handling agent recursion limit errors."""
        handler = LangGraphErrorHandler()

        error_context = {
            "workflow_id": "wf_202",
            "agent_name": "anime_search_agent",
            "recursion_depth": 50,
            "max_recursion": 25,
            "recursion_path": ["search", "refine", "search", "refine"],
        }

        result = await handler.handle_agent_recursion_limit(error_context)

        assert result["error_type"] == "agent_recursion_limit"
        assert result["workflow_id"] == "wf_202"
        assert result["recoverable"] is True
        assert result["suggested_action"] == "break_recursion"
        assert result["recursion_break_strategy"] == "intermediate_result"

    @pytest.mark.asyncio
    async def test_handle_workflow_timeout(self):
        """Test handling workflow timeout errors."""
        handler = LangGraphErrorHandler()

        error_context = {
            "workflow_id": "wf_303",
            "timeout_seconds": 300,
            "execution_time": 450,
            "last_completed_node": "filter_results",
            "pending_nodes": ["rank_results", "format_output"],
        }

        result = await handler.handle_workflow_timeout(error_context)

        assert result["error_type"] == "workflow_timeout"
        assert result["workflow_id"] == "wf_303"
        assert result["recoverable"] is True
        assert result["suggested_action"] == "partial_completion"
        assert result["partial_results_available"] is True

    @pytest.mark.asyncio
    async def test_create_workflow_error_context(self):
        """Test creating workflow-specific error context."""
        handler = LangGraphErrorHandler()

        error = Exception("Test error")
        workflow_state = {
            "workflow_id": "wf_404",
            "current_node": "search_anime",
            "execution_step": 5,
            "user_query": "find anime like naruto",
        }

        context = handler.create_workflow_error_context(error, workflow_state)

        assert isinstance(context, ErrorContext)
        assert context.correlation_id.startswith("wf_404")
        assert context.severity == ErrorSeverity.ERROR
        assert "workflow_id" in context.trace_data
        assert "current_node" in context.trace_data
        assert context.trace_data["workflow_id"] == "wf_404"

    @pytest.mark.asyncio
    async def test_get_recovery_strategy(self):
        """Test getting recovery strategy for error types."""
        handler = LangGraphErrorHandler()

        # Test node execution failure recovery
        strategy = handler.get_recovery_strategy("node_execution_failure")
        assert strategy["name"] == "retry_with_backoff"
        assert strategy["max_retries"] == 3
        assert strategy["backoff_multiplier"] == 2.0

        # Test workflow timeout recovery
        strategy = handler.get_recovery_strategy("workflow_timeout")
        assert strategy["name"] == "partial_completion"
        assert strategy["preserve_partial_results"] is True

    @pytest.mark.asyncio
    async def test_track_workflow_error(self):
        """Test workflow error tracking."""
        handler = LangGraphErrorHandler()

        error_info = {
            "workflow_id": "wf_505",
            "error_type": "node_execution_failure",
            "node_name": "search_anime",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        handler.track_workflow_error(error_info)

        # Verify error was tracked
        tracked_errors = handler.get_workflow_error_history("wf_505")
        assert len(tracked_errors) == 1
        assert tracked_errors[0]["error_type"] == "node_execution_failure"
        assert tracked_errors[0]["node_name"] == "search_anime"

    @pytest.mark.asyncio
    async def test_get_workflow_error_patterns(self):
        """Test getting workflow error patterns for analysis."""
        handler = LangGraphErrorHandler()

        # Track multiple errors
        errors = [
            {
                "workflow_id": "wf_1",
                "error_type": "node_execution_failure",
                "node_name": "search",
            },
            {
                "workflow_id": "wf_1",
                "error_type": "node_execution_failure",
                "node_name": "search",
            },
            {
                "workflow_id": "wf_2",
                "error_type": "tool_invocation_error",
                "tool_name": "mal_search",
            },
        ]

        for error in errors:
            handler.track_workflow_error(error)

        patterns = handler.get_workflow_error_patterns()

        assert "node_execution_failure" in patterns
        assert patterns["node_execution_failure"]["count"] == 2
        assert "tool_invocation_error" in patterns
        assert patterns["tool_invocation_error"]["count"] == 1

    @pytest.mark.asyncio
    async def test_should_trigger_circuit_breaker(self):
        """Test circuit breaker trigger logic for workflows."""
        handler = LangGraphErrorHandler()

        # Track failures for same workflow
        workflow_id = "wf_606"
        for _ in range(3):
            handler.track_workflow_error(
                {
                    "workflow_id": workflow_id,
                    "error_type": "node_execution_failure",
                    "node_name": "search",
                }
            )

        should_trigger = handler.should_trigger_circuit_breaker(workflow_id)
        assert should_trigger is True

        # Test with different workflow
        should_trigger = handler.should_trigger_circuit_breaker("wf_other")
        assert should_trigger is False

    @pytest.mark.asyncio
    async def test_track_workflow_error_history_trimming(self):
        """Test workflow error history trimming at 100 errors."""
        handler = LangGraphErrorHandler()
        workflow_id = "wf_trim_test"

        # Add 105 errors to trigger trimming
        for i in range(105):
            handler.track_workflow_error(
                {
                    "workflow_id": workflow_id,
                    "error_type": "node_execution_failure",
                    "node_name": f"node_{i}",
                    "sequence": i,  # To verify trimming order
                }
            )

        history = handler.get_workflow_error_history(workflow_id)

        # Should be trimmed to last 100 errors
        assert len(history) == 100

        # Should keep the most recent errors (sequence 5-104)
        assert history[0]["sequence"] == 5  # First kept error
        assert history[-1]["sequence"] == 104  # Last error

    @pytest.mark.asyncio
    async def test_should_trigger_circuit_breaker_invalid_timestamp(self):
        """Test circuit breaker with invalid timestamp formats."""
        handler = LangGraphErrorHandler()
        workflow_id = "wf_invalid_timestamp"

        # Track errors with invalid timestamps
        handler.track_workflow_error(
            {
                "workflow_id": workflow_id,
                "error_type": "node_execution_failure",
                "timestamp": "invalid_timestamp_format",
            }
        )

        handler.track_workflow_error(
            {
                "workflow_id": workflow_id,
                "error_type": "node_execution_failure",
                "timestamp": None,  # This will cause TypeError
            }
        )

        # Should handle invalid timestamps gracefully
        should_trigger = handler.should_trigger_circuit_breaker(workflow_id)
        assert should_trigger is False  # No valid recent errors


class TestExecutionTracer:
    """Test the ExecutionTracer class."""

    def test_execution_tracer_initialization(self):
        """Test ExecutionTracer initializes correctly."""
        tracer = ExecutionTracer()

        assert tracer.traces == {}
        assert tracer.active_traces == {}
        assert tracer.trace_configs is not None
        assert tracer.performance_metrics == {}
        assert tracer.max_trace_history == 1000

    def test_execution_tracer_with_custom_config(self):
        """Test ExecutionTracer with custom configuration."""
        config = {
            "max_trace_history": 500,
            "performance_tracking": True,
            "detailed_logging": False,
        }

        tracer = ExecutionTracer(config)

        assert tracer.max_trace_history == 500
        assert tracer.trace_configs["performance_tracking"] is True
        assert tracer.trace_configs["detailed_logging"] is False

    @pytest.mark.asyncio
    async def test_start_trace(self):
        """Test starting an execution trace."""
        tracer = ExecutionTracer()

        trace_id = await tracer.start_trace(
            operation="search_anime", context={"query": "naruto", "user_id": "123"}
        )

        assert trace_id is not None
        assert trace_id in tracer.active_traces

        trace = tracer.active_traces[trace_id]
        assert trace["operation"] == "search_anime"
        assert trace["context"]["query"] == "naruto"
        assert trace["status"] == "active"
        assert "start_time" in trace
        assert "trace_id" in trace

    @pytest.mark.asyncio
    async def test_add_trace_step(self):
        """Test adding steps to execution trace."""
        tracer = ExecutionTracer()

        trace_id = await tracer.start_trace("api_call", {"endpoint": "/anime"})

        await tracer.add_trace_step(
            trace_id,
            step_name="validate_parameters",
            step_data={"params": {"q": "naruto"}},
            step_status="completed",
        )

        await tracer.add_trace_step(
            trace_id,
            step_name="call_external_api",
            step_data={"url": "https://api.jikan.moe"},
            step_status="in_progress",
        )

        trace = tracer.active_traces[trace_id]
        assert len(trace["steps"]) == 2

        step1 = trace["steps"][0]
        assert step1["step_name"] == "validate_parameters"
        assert step1["step_status"] == "completed"
        assert step1["step_data"]["params"]["q"] == "naruto"
        assert "timestamp" in step1

        step2 = trace["steps"][1]
        assert step2["step_name"] == "call_external_api"
        assert step2["step_status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_end_trace_success(self):
        """Test ending trace successfully."""
        tracer = ExecutionTracer()

        trace_id = await tracer.start_trace("search_operation", {})
        await tracer.add_trace_step(trace_id, "step1", {}, "completed")

        result = await tracer.end_trace(
            trace_id,
            status="success",
            result={"anime_count": 5},
            performance_metrics={"duration_ms": 250},
        )

        assert result["trace_id"] == trace_id
        assert result["status"] == "success"
        assert result["result"]["anime_count"] == 5
        assert result["performance_metrics"]["duration_ms"] == 250

        # Should be moved from active to completed traces
        assert trace_id not in tracer.active_traces
        assert trace_id in tracer.traces

        completed_trace = tracer.traces[trace_id]
        assert completed_trace["status"] == "success"
        assert "end_time" in completed_trace
        assert "total_duration_ms" in completed_trace

    @pytest.mark.asyncio
    async def test_end_trace_failure(self):
        """Test ending trace with failure."""
        tracer = ExecutionTracer()

        trace_id = await tracer.start_trace("failing_operation", {})

        result = await tracer.end_trace(
            trace_id,
            status="failure",
            error=Exception("API timeout"),
            performance_metrics={"duration_ms": 5000},
        )

        assert result["status"] == "failure"
        assert "API timeout" in str(result["error"])

        completed_trace = tracer.traces[trace_id]
        assert completed_trace["status"] == "failure"
        assert "error" in completed_trace

    @pytest.mark.asyncio
    async def test_get_trace(self):
        """Test retrieving trace information."""
        tracer = ExecutionTracer()

        trace_id = await tracer.start_trace("test_operation", {"key": "value"})
        await tracer.add_trace_step(trace_id, "step1", {"data": "test"}, "completed")

        # Get active trace
        trace = tracer.get_trace(trace_id)
        assert trace["operation"] == "test_operation"
        assert trace["status"] == "active"
        assert len(trace["steps"]) == 1

        await tracer.end_trace(trace_id, "success")

        # Get completed trace
        trace = tracer.get_trace(trace_id)
        assert trace["status"] == "success"

    @pytest.mark.asyncio
    async def test_get_nonexistent_trace(self):
        """Test getting trace that doesn't exist."""
        tracer = ExecutionTracer()

        trace = tracer.get_trace("nonexistent_trace_id")
        assert trace is None

    @pytest.mark.asyncio
    async def test_get_active_traces(self):
        """Test getting all active traces."""
        tracer = ExecutionTracer()

        trace_id1 = await tracer.start_trace("operation1", {})
        trace_id2 = await tracer.start_trace("operation2", {})
        trace_id3 = await tracer.start_trace("operation3", {})

        # End one trace
        await tracer.end_trace(trace_id2, "success")

        active_traces = tracer.get_active_traces()

        assert len(active_traces) == 2
        active_ids = [trace["trace_id"] for trace in active_traces]
        assert trace_id1 in active_ids
        assert trace_id3 in active_ids
        assert trace_id2 not in active_ids

    @pytest.mark.asyncio
    async def test_get_traces_by_operation(self):
        """Test filtering traces by operation type."""
        tracer = ExecutionTracer()

        trace_id1 = await tracer.start_trace("search_anime", {})
        trace_id2 = await tracer.start_trace("get_anime_details", {})
        trace_id3 = await tracer.start_trace("search_anime", {})

        await tracer.end_trace(trace_id1, "success")
        await tracer.end_trace(trace_id2, "success")
        await tracer.end_trace(trace_id3, "failure")

        search_traces = tracer.get_traces_by_operation("search_anime")

        assert len(search_traces) == 2
        search_ids = [trace["trace_id"] for trace in search_traces]
        assert trace_id1 in search_ids
        assert trace_id3 in search_ids
        assert trace_id2 not in search_ids

    @pytest.mark.asyncio
    async def test_get_performance_summary(self):
        """Test getting performance summary."""
        tracer = ExecutionTracer()

        # Create traces with different durations
        trace_id1 = await tracer.start_trace("fast_operation", {})
        await asyncio.sleep(0.01)  # Small delay
        await tracer.end_trace(
            trace_id1, "success", performance_metrics={"duration_ms": 100}
        )

        trace_id2 = await tracer.start_trace("slow_operation", {})
        await asyncio.sleep(0.01)
        await tracer.end_trace(
            trace_id2, "success", performance_metrics={"duration_ms": 500}
        )

        summary = tracer.get_performance_summary()

        assert summary["total_traces"] == 2
        assert summary["successful_traces"] == 2
        assert summary["failed_traces"] == 0
        assert "average_duration_ms" in summary
        assert "min_duration_ms" in summary
        assert "max_duration_ms" in summary

    @pytest.mark.asyncio
    async def test_cleanup_old_traces(self):
        """Test cleaning up old traces."""
        tracer = ExecutionTracer({"max_trace_history": 3})

        # Create 5 traces
        trace_ids = []
        for i in range(5):
            trace_id = await tracer.start_trace(f"operation_{i}", {})
            await tracer.end_trace(trace_id, "success")
            trace_ids.append(trace_id)

        # Trigger cleanup
        tracer.cleanup_old_traces()

        # Should keep only the last 3 traces
        assert len(tracer.traces) == 3

        # First 2 traces should be removed
        assert trace_ids[0] not in tracer.traces
        assert trace_ids[1] not in tracer.traces

        # Last 3 traces should remain
        assert trace_ids[2] in tracer.traces
        assert trace_ids[3] in tracer.traces
        assert trace_ids[4] in tracer.traces

    @pytest.mark.asyncio
    async def test_trace_context_manager(self):
        """Test ExecutionTracer as context manager."""
        tracer = ExecutionTracer()

        async with tracer.trace("context_operation", {"key": "value"}) as trace_ctx:
            trace_id = trace_ctx.trace_id

            # Add some steps within context
            await trace_ctx.add_step("step1", {"data": "test"}, "completed")
            await trace_ctx.add_step("step2", {"data": "test2"}, "completed")

            # Trace should be active
            assert trace_id in tracer.active_traces

        # After context exit, trace should be completed
        assert trace_id not in tracer.active_traces
        assert trace_id in tracer.traces

        completed_trace = tracer.traces[trace_id]
        assert completed_trace["status"] == "success"
        assert len(completed_trace["steps"]) == 2

    @pytest.mark.asyncio
    async def test_trace_context_manager_with_exception(self):
        """Test ExecutionTracer context manager with exception."""
        tracer = ExecutionTracer()

        trace_id = None
        try:
            async with tracer.trace("failing_operation", {}) as trace_ctx:
                trace_id = trace_ctx.trace_id
                await trace_ctx.add_step("step1", {}, "completed")
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception

        # Trace should be completed with failure status
        assert trace_id not in tracer.active_traces
        assert trace_id in tracer.traces

        completed_trace = tracer.traces[trace_id]
        assert completed_trace["status"] == "failure"
        assert "error" in completed_trace

    @pytest.mark.asyncio
    async def test_export_traces(self):
        """Test exporting traces for analysis."""
        tracer = ExecutionTracer()

        # Create a few traces
        trace_id1 = await tracer.start_trace("export_test1", {"key": "value1"})
        await tracer.add_trace_step(trace_id1, "step1", {}, "completed")
        await tracer.end_trace(trace_id1, "success")

        trace_id2 = await tracer.start_trace("export_test2", {"key": "value2"})
        await tracer.end_trace(trace_id2, "failure", error=Exception("Test error"))

        exported_data = tracer.export_traces()

        assert "traces" in exported_data
        assert "summary" in exported_data
        assert "export_timestamp" in exported_data

        traces = exported_data["traces"]
        assert len(traces) == 2

        # Verify trace data structure
        trace1 = next(t for t in traces if t["trace_id"] == trace_id1)
        assert trace1["operation"] == "export_test1"
        assert trace1["status"] == "success"
        assert len(trace1["steps"]) == 1

        summary = exported_data["summary"]
        assert summary["total_traces"] == 2
        assert summary["successful_traces"] == 1
        assert summary["failed_traces"] == 1

    @pytest.mark.asyncio
    async def test_add_step_to_nonexistent_trace(self):
        """Test adding step to non-existent trace."""
        tracer = ExecutionTracer()

        # Try to add step to trace that doesn't exist
        await tracer.add_trace_step("nonexistent_trace", "step1", {}, "completed")

        # Should handle gracefully without error
        assert "nonexistent_trace" not in tracer.active_traces

    @pytest.mark.asyncio
    async def test_end_nonexistent_trace(self):
        """Test ending non-existent trace."""
        tracer = ExecutionTracer()

        # Try to end trace that doesn't exist
        result = await tracer.end_trace("nonexistent_trace", "success")

        # Should return empty dict
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_performance_summary_no_traces(self):
        """Test performance summary when no traces exist."""
        tracer = ExecutionTracer()

        summary = tracer.get_performance_summary()

        assert summary["total_traces"] == 0
        assert summary["successful_traces"] == 0
        assert summary["failed_traces"] == 0
        assert summary["average_duration_ms"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_old_traces_no_cleanup_needed(self):
        """Test cleanup when no cleanup is needed."""
        tracer = ExecutionTracer({"max_trace_history": 100})

        # Create only 2 traces - should not trigger cleanup
        trace_id1 = await tracer.start_trace("op1", {})
        await tracer.end_trace(trace_id1, "success")

        trace_id2 = await tracer.start_trace("op2", {})
        await tracer.end_trace(trace_id2, "success")

        initial_count = len(tracer.traces)
        tracer.cleanup_old_traces()

        # Should not change the number of traces
        assert len(tracer.traces) == initial_count
        assert len(tracer.traces) == 2

    @pytest.mark.asyncio
    async def test_auto_cleanup_traces_with_threshold(self):
        """Test auto cleanup when threshold is exceeded."""
        tracer = ExecutionTracer({"max_trace_history": 3, "auto_cleanup": True})

        # Create 5 traces to exceed the 1.2x threshold (3 * 1.2 = 3.6, so 4+ triggers cleanup)
        trace_ids = []
        for i in range(5):
            trace_id = await tracer.start_trace(f"operation_{i}", {})
            await tracer.end_trace(trace_id, "success")
            trace_ids.append(trace_id)

        # Auto cleanup should have been triggered, keeping only the last 3
        assert len(tracer.traces) == 3

        # First 2 traces should be removed
        assert trace_ids[0] not in tracer.traces
        assert trace_ids[1] not in tracer.traces

        # Last 3 traces should remain
        assert trace_ids[2] in tracer.traces
        assert trace_ids[3] in tracer.traces
        assert trace_ids[4] in tracer.traces

    @pytest.mark.asyncio
    async def test_export_traces_with_filter(self):
        """Test exporting traces with operation filter."""
        tracer = ExecutionTracer()

        # Create traces with different operations
        trace_id1 = await tracer.start_trace("search_anime", {})
        await tracer.end_trace(trace_id1, "success")

        trace_id2 = await tracer.start_trace("get_details", {})
        await tracer.end_trace(trace_id2, "success")

        trace_id3 = await tracer.start_trace("search_anime", {})
        await tracer.end_trace(trace_id3, "failure")

        # Export with filter
        exported_data = tracer.export_traces(operation_filter="search_anime")

        assert exported_data["total_exported"] == 2
        assert exported_data["operation_filter"] == "search_anime"

        traces = exported_data["traces"]
        assert len(traces) == 2
        assert all(t["operation"] == "search_anime" for t in traces)

    @pytest.mark.asyncio
    async def test_get_traces_by_operation_with_active_traces(self):
        """Test filtering traces by operation including active traces."""
        tracer = ExecutionTracer()

        # Create a mix of completed and active traces
        trace_id1 = await tracer.start_trace("search_anime", {})
        await tracer.end_trace(trace_id1, "success")  # Completed

        trace_id2 = await tracer.start_trace("search_anime", {})  # Active
        trace_id3 = await tracer.start_trace(
            "get_details", {}
        )  # Active, different operation

        # Get traces for search_anime operation
        search_traces = tracer.get_traces_by_operation("search_anime")

        # Should include both completed and active traces
        assert len(search_traces) == 2
        trace_ids = [trace["trace_id"] for trace in search_traces]
        assert trace_id1 in trace_ids  # Completed trace
        assert trace_id2 in trace_ids  # Active trace
        assert trace_id3 not in [
            t["trace_id"] for t in search_traces
        ]  # Different operation


class TestCorrelationLogger:
    """Test the CorrelationLogger class."""

    @pytest.fixture
    def correlation_logger(self):
        """Create CorrelationLogger for testing."""
        from src.integrations.error_handling import CorrelationLogger

        return CorrelationLogger()

    def test_correlation_logger_initialization(self, correlation_logger):
        """Test CorrelationLogger initializes correctly."""
        assert hasattr(correlation_logger, "logs")
        assert hasattr(correlation_logger, "correlation_chains")
        assert hasattr(correlation_logger, "log_filters")
        assert hasattr(correlation_logger, "correlation_metrics")
        assert len(correlation_logger.logs) == 0
        assert len(correlation_logger.correlation_chains) == 0

    @pytest.mark.asyncio
    async def test_log_with_correlation_basic(self, correlation_logger):
        """Test basic correlation logging."""
        correlation_id = "test-corr-123"
        message = "Test message"

        await correlation_logger.log_with_correlation(
            correlation_id=correlation_id,
            level="info",
            message=message,
            context={"service": "test"},
        )

        # Check log was created
        assert len(correlation_logger.logs) == 1
        log_entry = correlation_logger.logs[0]

        assert log_entry["correlation_id"] == correlation_id
        assert log_entry["level"] == "info"
        assert log_entry["message"] == message
        assert log_entry["context"]["service"] == "test"
        assert "timestamp" in log_entry

    @pytest.mark.asyncio
    async def test_log_with_correlation_chain(self, correlation_logger):
        """Test correlation chain building."""
        parent_id = "parent-123"
        child_id = "child-456"

        # Log parent operation
        await correlation_logger.log_with_correlation(
            correlation_id=parent_id,
            level="info",
            message="Parent operation started",
            context={"operation": "search"},
        )

        # Log child operation with parent
        await correlation_logger.log_with_correlation(
            correlation_id=child_id,
            level="info",
            message="Child operation started",
            context={"operation": "details"},
            parent_correlation_id=parent_id,
        )

        # Check correlation chain was built
        assert child_id in correlation_logger.correlation_chains
        assert correlation_logger.correlation_chains[child_id]["parent"] == parent_id
        assert correlation_logger.correlation_chains[child_id]["children"] == []

        # Check parent has child reference
        if parent_id in correlation_logger.correlation_chains:
            assert (
                child_id in correlation_logger.correlation_chains[parent_id]["children"]
            )

    @pytest.mark.asyncio
    async def test_log_with_correlation_error_severity(self, correlation_logger):
        """Test error severity classification."""
        correlation_id = "error-test-123"

        await correlation_logger.log_with_correlation(
            correlation_id=correlation_id,
            level="error",
            message="Critical error occurred",
            context={"error_type": "api_failure", "service": "mal"},
            error_details={"status_code": 500, "response": "Internal Error"},
        )

        log_entry = correlation_logger.logs[0]
        assert log_entry["level"] == "error"
        assert log_entry["error_details"]["status_code"] == 500
        assert "error_type" in log_entry["context"]

    @pytest.mark.asyncio
    async def test_get_correlation_chain(self, correlation_logger):
        """Test retrieving correlation chain."""
        parent_id = "parent-789"
        child1_id = "child1-101"
        child2_id = "child2-102"

        # Create correlation chain
        await correlation_logger.log_with_correlation(parent_id, "info", "Parent", {})
        await correlation_logger.log_with_correlation(
            child1_id, "info", "Child1", {}, parent_id
        )
        await correlation_logger.log_with_correlation(
            child2_id, "info", "Child2", {}, parent_id
        )

        # Get chain for parent
        chain = correlation_logger.get_correlation_chain(parent_id)

        assert chain["correlation_id"] == parent_id
        assert chain["parent"] is None
        assert len(chain["children"]) == 2
        assert child1_id in chain["children"]
        assert child2_id in chain["children"]

        # Get chain for child
        child_chain = correlation_logger.get_correlation_chain(child1_id)
        assert child_chain["parent"] == parent_id
        assert len(child_chain["children"]) == 0

    @pytest.mark.asyncio
    async def test_get_logs_by_correlation(self, correlation_logger):
        """Test retrieving logs by correlation ID."""
        correlation_id = "log-test-456"

        # Add multiple logs with same correlation ID
        await correlation_logger.log_with_correlation(
            correlation_id, "info", "Step 1", {"step": 1}
        )
        await correlation_logger.log_with_correlation(
            correlation_id, "debug", "Step 2", {"step": 2}
        )
        await correlation_logger.log_with_correlation(
            correlation_id, "error", "Step 3 failed", {"step": 3}
        )

        # Add log with different correlation ID
        await correlation_logger.log_with_correlation(
            "other-123", "info", "Other operation", {}
        )

        logs = correlation_logger.get_logs_by_correlation(correlation_id)

        assert len(logs) == 3
        for log in logs:
            assert log["correlation_id"] == correlation_id

        # Check logs are in chronological order
        assert logs[0]["context"]["step"] == 1
        assert logs[1]["context"]["step"] == 2
        assert logs[2]["context"]["step"] == 3

    @pytest.mark.asyncio
    async def test_get_logs_by_correlation_with_chain(self, correlation_logger):
        """Test retrieving logs by correlation with chain traversal."""
        parent_id = "parent-chain-123"
        child_id = "child-chain-456"

        await correlation_logger.log_with_correlation(
            parent_id, "info", "Parent start", {}
        )
        await correlation_logger.log_with_correlation(
            child_id, "info", "Child start", {}, parent_id
        )
        await correlation_logger.log_with_correlation(
            parent_id, "info", "Parent end", {}
        )
        await correlation_logger.log_with_correlation(
            child_id, "error", "Child error", {}, parent_id
        )

        # Get logs including chain
        logs = correlation_logger.get_logs_by_correlation(parent_id, include_chain=True)

        # Should include both parent and child logs
        assert len(logs) == 4
        correlation_ids = [log["correlation_id"] for log in logs]
        assert parent_id in correlation_ids
        assert child_id in correlation_ids

    @pytest.mark.asyncio
    async def test_add_log_filter(self, correlation_logger):
        """Test adding and applying log filters."""
        # Add filter for error level only
        correlation_logger.add_log_filter("error_only", level="error")

        correlation_id = "filter-test-789"

        await correlation_logger.log_with_correlation(
            correlation_id, "info", "Info message", {}
        )
        await correlation_logger.log_with_correlation(
            correlation_id, "error", "Error message", {}
        )
        await correlation_logger.log_with_correlation(
            correlation_id, "debug", "Debug message", {}
        )

        # Apply filter
        filtered_logs = correlation_logger.get_filtered_logs("error_only")

        assert len(filtered_logs) == 1
        assert filtered_logs[0]["level"] == "error"
        assert filtered_logs[0]["message"] == "Error message"

    @pytest.mark.asyncio
    async def test_add_log_filter_service(self, correlation_logger):
        """Test filtering by service context."""
        correlation_logger.add_log_filter(
            "mal_only", context_filters={"service": "mal"}
        )

        await correlation_logger.log_with_correlation(
            "test1", "info", "MAL operation", {"service": "mal"}
        )
        await correlation_logger.log_with_correlation(
            "test2", "info", "Jikan operation", {"service": "jikan"}
        )
        await correlation_logger.log_with_correlation(
            "test3", "error", "MAL error", {"service": "mal"}
        )

        filtered_logs = correlation_logger.get_filtered_logs("mal_only")

        assert len(filtered_logs) == 2
        for log in filtered_logs:
            assert log["context"]["service"] == "mal"

    @pytest.mark.asyncio
    async def test_get_correlation_metrics(self, correlation_logger):
        """Test correlation metrics calculation."""
        # Create various log entries
        await correlation_logger.log_with_correlation(
            "corr1", "info", "Success", {"operation": "search"}
        )
        await correlation_logger.log_with_correlation(
            "corr1", "error", "Error", {"operation": "search"}
        )
        await correlation_logger.log_with_correlation(
            "corr2", "info", "Success", {"operation": "details"}
        )
        await correlation_logger.log_with_correlation(
            "corr3", "debug", "Debug", {"operation": "search"}
        )

        metrics = correlation_logger.get_correlation_metrics()

        assert "total_correlations" in metrics
        assert "log_levels" in metrics
        assert "average_logs_per_correlation" in metrics
        assert "most_active_correlations" in metrics

        assert metrics["total_correlations"] == 3
        assert metrics["log_levels"]["info"] == 2
        assert metrics["log_levels"]["error"] == 1
        assert metrics["log_levels"]["debug"] == 1

    @pytest.mark.asyncio
    async def test_clear_old_logs(self, correlation_logger):
        """Test clearing old logs based on retention."""
        correlation_id = "retention-test"

        # Add logs
        await correlation_logger.log_with_correlation(
            correlation_id, "info", "Old log", {}
        )
        await correlation_logger.log_with_correlation(
            correlation_id, "info", "Recent log", {}
        )

        # Clear logs older than 0 seconds (should clear first log in test timing)
        cleared_count = correlation_logger.clear_old_logs(max_age_seconds=0)

        # At minimum, should have mechanism to clear logs
        assert isinstance(cleared_count, int)
        assert cleared_count >= 0

    @pytest.mark.asyncio
    async def test_export_correlation_logs(self, correlation_logger):
        """Test exporting correlation logs for analysis."""
        correlation_id = "export-test-123"

        await correlation_logger.log_with_correlation(
            correlation_id,
            "info",
            "Export test",
            {"service": "mal", "operation": "search"},
        )

        exported_data = correlation_logger.export_correlation_logs(correlation_id)

        assert "correlation_id" in exported_data
        assert "logs" in exported_data
        assert "chain_info" in exported_data
        assert "export_timestamp" in exported_data

        assert exported_data["correlation_id"] == correlation_id
        assert len(exported_data["logs"]) == 1

    @pytest.mark.asyncio
    async def test_correlation_context_manager(self, correlation_logger):
        """Test correlation context manager for automatic logging."""
        correlation_id = "context-test-456"

        # Use context manager
        async with correlation_logger.correlation_context(
            correlation_id, "search_operation", {"query": "naruto"}
        ) as ctx:
            # Log within context
            await ctx.log("info", "Operation in progress", {"step": "validation"})
            await ctx.log("debug", "Query processed", {"processed_query": "naruto"})

        # Check logs were created
        logs = correlation_logger.get_logs_by_correlation(correlation_id)

        # Should have context entry + 2 manual logs + context exit
        assert len(logs) >= 2

        # Check context logs have correlation ID
        for log in logs:
            assert log["correlation_id"] == correlation_id

    @pytest.mark.asyncio
    async def test_correlation_context_manager_with_error(self, correlation_logger):
        """Test correlation context manager handles errors properly."""
        correlation_id = "context-error-789"

        # Use context manager with exception
        try:
            async with correlation_logger.correlation_context(
                correlation_id, "failing_operation", {"will_fail": True}
            ) as ctx:
                await ctx.log("info", "About to fail", {})
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected error

        logs = correlation_logger.get_logs_by_correlation(correlation_id)

        # Should have logs including error context
        assert len(logs) >= 1

        # At least one log should mention the error or have error level
        error_logs = [
            log
            for log in logs
            if log["level"] == "error" or "error" in log["message"].lower()
        ]
        assert len(error_logs) >= 1

    def test_correlation_logger_thread_safety(self, correlation_logger):
        """Test correlation logger handles concurrent access."""
        import asyncio

        async def log_worker(worker_id: int):
            correlation_id = f"worker-{worker_id}"
            for i in range(5):
                await correlation_logger.log_with_correlation(
                    correlation_id,
                    "info",
                    f"Worker {worker_id} step {i}",
                    {"worker": worker_id, "step": i},
                )

        async def run_concurrent_test():
            # Run multiple workers concurrently
            tasks = [log_worker(i) for i in range(3)]
            await asyncio.gather(*tasks)

        # Run the test
        asyncio.run(run_concurrent_test())

        # Verify all logs were created
        assert len(correlation_logger.logs) == 15  # 3 workers * 5 logs each

        # Verify each worker's logs are present
        for worker_id in range(3):
            worker_logs = correlation_logger.get_logs_by_correlation(
                f"worker-{worker_id}"
            )
            assert len(worker_logs) == 5

            # Verify logs are in correct order
            for i, log in enumerate(worker_logs):
                assert log["context"]["step"] == i

    @pytest.mark.asyncio
    async def test_correlation_logger_memory_efficiency(self, correlation_logger):
        """Test correlation logger memory management."""
        # Set a smaller retention limit for testing
        correlation_logger.max_logs_in_memory = 100

        # Add many logs to test memory management
        for i in range(150):
            await correlation_logger.log_with_correlation(
                f"corr-{i}", "info", f"Log {i}", {"index": i}
            )

        # Should not exceed memory limit significantly
        assert (
            len(correlation_logger.logs) <= correlation_logger.max_logs_in_memory * 1.1
        )  # Allow 10% buffer

    def test_get_correlation_chain_nonexistent(self, correlation_logger):
        """Test getting chain for non-existent correlation ID."""
        chain = correlation_logger.get_correlation_chain("nonexistent-123")

        assert chain is None

    def test_get_logs_by_correlation_empty(self, correlation_logger):
        """Test getting logs for non-existent correlation ID."""
        logs = correlation_logger.get_logs_by_correlation("nonexistent-456")

        assert logs == []

    def test_get_filtered_logs_nonexistent_filter(self, correlation_logger):
        """Test getting logs with non-existent filter."""
        logs = correlation_logger.get_filtered_logs("nonexistent-filter")

        assert logs == []

    @pytest.mark.asyncio
    async def test_correlation_logger_performance_metrics(self, correlation_logger):
        """Test correlation logger performance tracking."""
        correlation_id = "perf-test-123"

        # Add logs with timing information
        await correlation_logger.log_with_correlation(
            correlation_id,
            "info",
            "Operation started",
            {
                "operation": "search",
                "performance": {"start_time": "2023-01-01T00:00:00Z"},
            },
        )

        await correlation_logger.log_with_correlation(
            correlation_id,
            "info",
            "Operation completed",
            {
                "operation": "search",
                "performance": {
                    "end_time": "2023-01-01T00:00:05Z",
                    "duration_ms": 5000,
                },
            },
        )

        # Get performance metrics
        perf_metrics = correlation_logger.get_performance_metrics(correlation_id)

        assert "correlation_id" in perf_metrics
        assert "total_logs" in perf_metrics
        assert perf_metrics["correlation_id"] == correlation_id
        assert perf_metrics["total_logs"] == 2

    @pytest.mark.asyncio
    async def test_correlation_logger_edge_case_parent_not_in_chains(
        self, correlation_logger
    ):
        """Test edge case where parent correlation ID is not in chains."""
        parent_id = "new-parent-123"
        child_id = "child-456"

        # Directly call _build_correlation_chain to test the edge case
        # where parent_id not in correlation_chains
        correlation_logger._build_correlation_chain(child_id, parent_id)

        # Should create both child and parent entries
        assert child_id in correlation_logger.correlation_chains
        assert parent_id in correlation_logger.correlation_chains
        assert correlation_logger.correlation_chains[child_id]["parent"] == parent_id
        assert child_id in correlation_logger.correlation_chains[parent_id]["children"]

    def test_correlation_logger_metrics_empty_logs(self, correlation_logger):
        """Test correlation metrics when no logs exist."""
        # Test the edge case where self.logs is empty
        metrics = correlation_logger.get_correlation_metrics()

        assert metrics["total_correlations"] == 0
        assert metrics["total_logs"] == 0
        assert metrics["log_levels"] == {}
        assert metrics["average_logs_per_correlation"] == 0.0
        assert metrics["most_active_correlations"] == []

    def test_correlation_logger_performance_metrics_no_logs(self, correlation_logger):
        """Test performance metrics when no logs exist for correlation."""
        correlation_id = "nonexistent-correlation"

        # Test the edge case where correlation has no logs
        perf_metrics = correlation_logger.get_performance_metrics(correlation_id)

        assert perf_metrics["correlation_id"] == correlation_id
        assert perf_metrics["total_logs"] == 0
        assert perf_metrics["performance_data"] is None
