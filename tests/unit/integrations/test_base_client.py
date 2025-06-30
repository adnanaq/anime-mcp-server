"""
Comprehensive test suite for BaseClient class.

Tests all functionality including:
- Error handling infrastructure
- Circuit breaker integration
- Rate limiting
- Collaborative caching
- Request deduplication
- LangGraph error handling
"""

import asyncio
import pytest
import aiohttp
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Import the classes we're testing (these will be implemented)
from src.integrations.clients.base_client import BaseClient
from src.integrations.error_handling import ErrorContext, CircuitBreaker, GracefulDegradation
from src.integrations.cache_manager import CollaborativeCacheSystem, RequestDeduplication
from src.integrations.execution_tracing import LangGraphErrorHandler


class TestErrorContext:
    """Test the three-layer error context preservation."""
    
    def test_error_context_creation(self):
        """Test creating an ErrorContext with all layers."""
        context = ErrorContext(
            user_message="Unable to fetch anime data. Please try again.",
            debug_info="AniList API returned 429: Rate limit exceeded",
            trace_data={"execution_id": "abc123", "tool": "search_anime", "timestamp": "2025-01-01T12:00:00Z"}
        )
        
        assert context.user_message == "Unable to fetch anime data. Please try again."
        assert context.debug_info == "AniList API returned 429: Rate limit exceeded"
        assert context.trace_data["execution_id"] == "abc123"
    
    def test_error_context_from_exception(self):
        """Test creating ErrorContext from an exception."""
        try:
            raise ValueError("Invalid anime ID format")
        except ValueError as e:
            context = ErrorContext.from_exception(
                e,
                user_message="Invalid input provided",
                trace_data={"tool": "get_anime_by_id", "input": "invalid_id"}
            )
            
            assert context.user_message == "Invalid input provided"
            assert "ValueError: Invalid anime ID format" in context.debug_info
            assert context.trace_data["tool"] == "get_anime_by_id"


class TestCircuitBreaker:
    """Test circuit breaker functionality for API failure prevention."""
    
    @pytest.fixture
    def circuit_breaker(self):
        return CircuitBreaker(failure_threshold=3, recovery_timeout=300)
    
    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initializes with correct parameters."""
        assert circuit_breaker.failure_threshold == 3
        assert circuit_breaker.recovery_timeout == 300
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success_path(self, circuit_breaker):
        """Test circuit breaker allows calls when closed."""
        mock_func = AsyncMock(return_value={"data": "success"})
        
        result = await circuit_breaker.call_with_breaker(mock_func)
        
        assert result == {"data": "success"}
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_counting(self, circuit_breaker):
        """Test circuit breaker counts failures correctly."""
        mock_func = AsyncMock(side_effect=aiohttp.ClientError("Connection failed"))
        
        # First 2 failures should not open circuit
        for i in range(2):
            with pytest.raises(aiohttp.ClientError):
                await circuit_breaker.call_with_breaker(mock_func)
        
        assert circuit_breaker.failure_count == 2
        assert circuit_breaker.state == "closed"
        
        # Third failure should open circuit
        with pytest.raises(aiohttp.ClientError):
            await circuit_breaker.call_with_breaker(mock_func)
        
        assert circuit_breaker.failure_count == 3
        assert circuit_breaker.state == "open"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state_blocks_calls(self, circuit_breaker):
        """Test circuit breaker blocks calls when open."""
        mock_func = AsyncMock(side_effect=aiohttp.ClientError("Connection failed"))
        
        # Trigger circuit breaker to open
        for i in range(3):
            with pytest.raises(aiohttp.ClientError):
                await circuit_breaker.call_with_breaker(mock_func)
        
        # Now circuit should be open and block calls
        mock_func.reset_mock()
        mock_func.side_effect = None
        mock_func.return_value = {"data": "success"}
        
        with pytest.raises(Exception) as exc_info:
            await circuit_breaker.call_with_breaker(mock_func)
        
        assert "Circuit breaker is open" in str(exc_info.value)
        mock_func.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, circuit_breaker):
        """Test circuit breaker recovery through half-open state."""
        mock_func = AsyncMock(side_effect=aiohttp.ClientError("Connection failed"))
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(aiohttp.ClientError):
                await circuit_breaker.call_with_breaker(mock_func)
        
        # Fast-forward time to trigger half-open state
        circuit_breaker.last_failure_time = datetime.utcnow() - timedelta(seconds=301)
        circuit_breaker.state = "half_open"
        
        # Success should close the circuit
        mock_func.side_effect = None
        mock_func.return_value = {"data": "recovered"}
        
        result = await circuit_breaker.call_with_breaker(mock_func)
        
        assert result == {"data": "recovered"}
        assert circuit_breaker.state == "closed"
        assert circuit_breaker.failure_count == 0


class TestCollaborativeCacheSystem:
    """Test collaborative community caching functionality."""
    
    @pytest.fixture
    def cache_system(self):
        return CollaborativeCacheSystem()
    
    @pytest.mark.asyncio
    async def test_get_enhanced_data_with_personal_quota(self, cache_system):
        """Test getting enhanced data when user has personal quota."""
        cache_system.has_personal_quota = AsyncMock(return_value=True)
        cache_system.fetch_with_user_quota = AsyncMock(return_value={"synopsis": "Enhanced data"})
        cache_system.share_with_community = AsyncMock()
        
        result = await cache_system.get_enhanced_data("anime123", "user456", "anilist")
        
        assert result == {"synopsis": "Enhanced data"}
        cache_system.has_personal_quota.assert_called_once_with("user456", "anilist")
        cache_system.fetch_with_user_quota.assert_called_once_with("anime123", "anilist", "user456")
        cache_system.share_with_community.assert_called_once_with("anime123", "anilist", {"synopsis": "Enhanced data"})
    
    @pytest.mark.asyncio
    async def test_get_enhanced_data_from_community_cache(self, cache_system):
        """Test getting enhanced data from community cache when no personal quota."""
        cache_system.has_personal_quota = AsyncMock(return_value=False)
        cache_system.community_cache = AsyncMock()
        cache_system.community_cache.get = AsyncMock(return_value={
            "data": {"synopsis": "Community cached data"},
            "fetched_at": datetime.utcnow(),
            "source": "anilist"
        })
        cache_system.track_cache_usage = AsyncMock()
        
        result = await cache_system.get_enhanced_data("anime123", "user456", "anilist")
        
        assert result == {"synopsis": "Community cached data"}
        cache_system.track_cache_usage.assert_called_once_with("user456", "anime123", "community_hit")
    
    @pytest.mark.asyncio
    async def test_get_enhanced_data_with_quota_donor(self, cache_system):
        """Test getting enhanced data using quota donor when no cache available."""
        cache_system.has_personal_quota = AsyncMock(return_value=False)
        cache_system.community_cache = AsyncMock()
        cache_system.community_cache.get = AsyncMock(return_value=None)
        cache_system.find_quota_donor = AsyncMock(return_value="donor_user789")
        cache_system.fetch_with_user_quota = AsyncMock(return_value={"synopsis": "Donor fetched data"})
        cache_system.share_with_community = AsyncMock()
        cache_system.credit_donor = AsyncMock()
        
        result = await cache_system.get_enhanced_data("anime123", "user456", "anilist")
        
        assert result == {"synopsis": "Donor fetched data"}
        cache_system.find_quota_donor.assert_called_once_with("anilist")
        cache_system.fetch_with_user_quota.assert_called_once_with("anime123", "anilist", "donor_user789")
        cache_system.credit_donor.assert_called_once_with("donor_user789", "anime123")
    
    @pytest.mark.asyncio
    async def test_get_enhanced_data_fallback_to_degraded(self, cache_system):
        """Test fallback to degraded data when all options exhausted."""
        cache_system.has_personal_quota = AsyncMock(return_value=False)
        cache_system.community_cache = AsyncMock()
        cache_system.community_cache.get = AsyncMock(return_value=None)
        cache_system.find_quota_donor = AsyncMock(return_value=None)
        cache_system.get_degraded_data = AsyncMock(return_value={"title": "Basic offline data"})
        
        result = await cache_system.get_enhanced_data("anime123", "user456", "anilist")
        
        assert result == {"title": "Basic offline data"}
        cache_system.get_degraded_data.assert_called_once_with("anime123")


class TestRequestDeduplication:
    """Test request deduplication functionality."""
    
    @pytest.fixture
    def deduplicator(self):
        return RequestDeduplication()
    
    @pytest.mark.asyncio
    async def test_deduplicate_request_first_call(self, deduplicator):
        """Test first call to deduplicate_request executes function."""
        mock_fetch = AsyncMock(return_value={"data": "fetched"})
        
        result = await deduplicator.deduplicate_request("key1", mock_fetch)
        
        assert result == {"data": "fetched"}
        mock_fetch.assert_called_once()
        assert "key1" not in deduplicator.active_requests  # Should be cleaned up
    
    @pytest.mark.asyncio
    async def test_deduplicate_request_concurrent_calls(self, deduplicator):
        """Test concurrent calls with same key share result."""
        call_count = 0
        
        async def mock_fetch():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate async work
            return {"data": f"fetched_{call_count}"}
        
        # Start multiple concurrent requests with same key
        tasks = [
            deduplicator.deduplicate_request("key1", mock_fetch),
            deduplicator.deduplicate_request("key1", mock_fetch),
            deduplicator.deduplicate_request("key1", mock_fetch)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should return the same result
        assert all(result == {"data": "fetched_1"} for result in results)
        # Function should only be called once
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_deduplicate_request_different_keys(self, deduplicator):
        """Test different keys execute separately."""
        mock_fetch1 = AsyncMock(return_value={"data": "fetch1"})
        mock_fetch2 = AsyncMock(return_value={"data": "fetch2"})
        
        task1 = deduplicator.deduplicate_request("key1", mock_fetch1)
        task2 = deduplicator.deduplicate_request("key2", mock_fetch2)
        
        result1, result2 = await asyncio.gather(task1, task2)
        
        assert result1 == {"data": "fetch1"}
        assert result2 == {"data": "fetch2"}
        mock_fetch1.assert_called_once()
        mock_fetch2.assert_called_once()


class TestLangGraphErrorHandler:
    """Test LangGraph-specific error handling."""
    
    @pytest.fixture
    def error_handler(self):
        return LangGraphErrorHandler()
    
    @pytest.mark.asyncio
    async def test_detect_tool_selection_loop(self, error_handler):
        """Test detection of tool selection loops."""
        # No loop - different tools
        history1 = ["search_anime", "get_anime_details", "find_similar"]
        assert not await error_handler.detect_tool_selection_loop(history1)
        
        # Loop detected - same tool called 3+ times
        history2 = ["search_anime", "search_anime", "search_anime"]
        assert await error_handler.detect_tool_selection_loop(history2)
        
        # Edge case - 2 consecutive calls should not trigger
        history3 = ["search_anime", "search_anime", "get_anime_details"]
        assert not await error_handler.detect_tool_selection_loop(history3)
    
    @pytest.mark.asyncio
    async def test_handle_parameter_extraction_failure(self, error_handler):
        """Test parameter extraction failure handling."""
        result = await error_handler.handle_parameter_extraction_failure("complex query", attempt=1)
        
        assert "simplified" in result
        assert result["attempt"] == 1
        assert result["fallback_strategy"] is True
    
    @pytest.mark.asyncio
    async def test_detect_state_corruption(self, error_handler):
        """Test state corruption detection."""
        # Valid state
        valid_state = {
            "messages": [{"role": "user", "content": "test"}],
            "current_tool": "search_anime",
            "results": []
        }
        assert not await error_handler.detect_state_corruption(valid_state)
        
        # Corrupted state - missing required fields
        corrupted_state = {
            "messages": None,
            "current_tool": "",
            "results": "invalid_type"
        }
        assert await error_handler.detect_state_corruption(corrupted_state)
    
    @pytest.mark.asyncio
    async def test_handle_graph_timeout(self, error_handler):
        """Test graph timeout handling."""
        partial_results = {
            "partial_data": [{"title": "Anime 1"}],
            "completed_steps": ["search", "filter"]
        }
        
        result = await error_handler.handle_graph_timeout("exec123", partial_results)
        
        assert result["timeout"] is True
        assert result["execution_id"] == "exec123"
        assert result["partial_results"] == partial_results
        assert "timeout_message" in result
    
    @pytest.mark.asyncio
    async def test_detect_memory_explosion(self, error_handler):
        """Test memory explosion detection."""
        # Normal state size
        assert not await error_handler.detect_memory_explosion(1000, 10000)
        
        # Memory explosion
        assert await error_handler.detect_memory_explosion(15000, 10000)
    
    @pytest.mark.asyncio
    async def test_prune_conversation_history(self, error_handler):
        """Test conversation history pruning."""
        large_state = {
            "messages": [f"message_{i}" for i in range(20)],
            "other_data": "preserved"
        }
        
        result = await error_handler.prune_conversation_history(large_state, keep_last_n=5)
        
        assert len(result["messages"]) == 5
        assert result["messages"] == [f"message_{i}" for i in range(15, 20)]
        assert result["other_data"] == "preserved"


class TestBaseClient:
    """Test the BaseClient class that integrates all error handling components."""
    
    @pytest.fixture
    def mock_dependencies(self):
        return {
            "circuit_breaker": Mock(spec=CircuitBreaker),
            "rate_limiter": Mock(),
            "cache_manager": Mock(spec=CollaborativeCacheSystem),
            "error_handler": Mock(spec=ErrorContext)
        }
    
    @pytest.fixture
    def base_client(self, mock_dependencies):
        return BaseClient(**mock_dependencies)
    
    def test_base_client_initialization(self, base_client, mock_dependencies):
        """Test BaseClient initializes with all dependencies."""
        assert base_client.circuit_breaker == mock_dependencies["circuit_breaker"]
        assert base_client.rate_limiter == mock_dependencies["rate_limiter"]
        assert base_client.cache_manager == mock_dependencies["cache_manager"]
        assert base_client.error_handler == mock_dependencies["error_handler"]
    
    @pytest.mark.asyncio
    async def test_make_request_success_path(self, base_client):
        """Test successful request through BaseClient."""
        base_client.circuit_breaker.call_with_breaker = AsyncMock(return_value={"data": "success"})
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"data": "success"})
            mock_session.get = AsyncMock(return_value=mock_response)
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await base_client.make_request("https://api.example.com/anime/123")
            
            assert result == {"data": "success"}
    
    @pytest.mark.asyncio
    async def test_make_request_with_rate_limiting(self, base_client):
        """Test request with rate limiting."""
        # Mock rate limiter
        rate_limiter_mock = AsyncMock()
        base_client.rate_limiter = rate_limiter_mock
        
        # Mock circuit breaker to actually call the function
        async def mock_circuit_call(func):
            return await func()
        base_client.circuit_breaker.call_with_breaker = AsyncMock(side_effect=mock_circuit_call)
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"data": "success"})
            mock_response.raise_for_status = Mock()
            mock_session.request = Mock(return_value=mock_response)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await base_client.make_request("https://api.example.com/anime/123")
            
            # Rate limiter should be called
            rate_limiter_mock.__aenter__.assert_called_once()
            assert result == {"data": "success"}
    
    @pytest.mark.asyncio
    async def test_make_request_with_circuit_breaker_open(self, base_client):
        """Test request when circuit breaker is open."""
        base_client.circuit_breaker.call_with_breaker = AsyncMock(
            side_effect=Exception("Circuit breaker is open")
        )
        
        with pytest.raises(Exception) as exc_info:
            await base_client.make_request("https://api.example.com/anime/123")
        
        assert "Circuit breaker is open" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_make_request_handles_rate_limit_response(self, base_client):
        """Test handling of 429 rate limit responses."""
        # Mock rate limiter
        base_client.rate_limiter = AsyncMock()
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.headers = {"Retry-After": "60"}
            mock_response.raise_for_status = Mock(side_effect=Exception("Rate limited: 429"))
            mock_session.request = Mock(return_value=mock_response)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock circuit breaker to actually call the function
            async def mock_circuit_call(func):
                return await func()
            base_client.circuit_breaker.call_with_breaker = AsyncMock(side_effect=mock_circuit_call)
            
            with pytest.raises(Exception, match="Rate limited: 429"):
                await base_client.make_request("https://api.example.com/anime/123")
    
    @pytest.mark.asyncio
    async def test_with_circuit_breaker_decorator(self, base_client):
        """Test the circuit breaker decorator functionality."""
        mock_func = AsyncMock(return_value="success")
        base_client.circuit_breaker.call_with_breaker = AsyncMock(return_value="success")
        
        result = await base_client.with_circuit_breaker(mock_func)
        
        assert result == "success"
        base_client.circuit_breaker.call_with_breaker.assert_called_once_with(mock_func)
    
    @pytest.mark.asyncio
    async def test_error_context_integration(self, base_client):
        """Test error context creation during failures."""
        # Mock rate limiter
        base_client.rate_limiter = AsyncMock()
        
        base_client.error_handler.from_exception = Mock(return_value=ErrorContext(
            user_message="Service temporarily unavailable",
            debug_info="Connection timeout after 30s",
            trace_data={"url": "https://api.example.com", "timeout": 30}
        ))
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_session.request = Mock(side_effect=asyncio.TimeoutError("Request timeout"))
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock circuit breaker to let exception through
            async def mock_circuit_call(func):
                return await func()
            base_client.circuit_breaker.call_with_breaker = AsyncMock(side_effect=mock_circuit_call)
            
            with pytest.raises(asyncio.TimeoutError):
                await base_client.make_request("https://api.example.com/anime/123")


class TestIntegrationScenarios:
    """Test complete integration scenarios combining all components."""
    
    @pytest.mark.asyncio
    async def test_complete_error_recovery_flow(self):
        """Test complete error recovery flow with all components."""
        # Setup all components
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=300)
        cache_system = CollaborativeCacheSystem()
        deduplicator = RequestDeduplication()
        error_handler = LangGraphErrorHandler()
        
        # Mock external dependencies
        cache_system.has_personal_quota = AsyncMock(return_value=False)
        cache_system.community_cache = AsyncMock()
        cache_system.community_cache.get = AsyncMock(return_value={
            "data": {"title": "Cached Anime", "synopsis": "From cache"},
            "fetched_at": datetime.utcnow()
        })
        cache_system.track_cache_usage = AsyncMock()
        
        # Test degraded service with cache fallback
        result = await cache_system.get_enhanced_data("anime123", "user456", "anilist")
        
        assert result["title"] == "Cached Anime"
        assert result["synopsis"] == "From cache"
        cache_system.track_cache_usage.assert_called_once_with("user456", "anime123", "community_hit")
    
    @pytest.mark.asyncio
    async def test_langgraph_error_handling_integration(self):
        """Test LangGraph error handling with circuit breaker integration."""
        error_handler = LangGraphErrorHandler()
        circuit_breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
        
        # Simulate tool selection loop
        execution_history = ["search_anime", "search_anime", "search_anime"]
        loop_detected = await error_handler.detect_tool_selection_loop(execution_history)
        
        assert loop_detected
        
        # Test forced tool change
        new_tool = await error_handler.force_tool_change("search_anime", ["get_anime_details", "find_similar"])
        
        assert new_tool in ["get_anime_details", "find_similar"]
        assert new_tool != "search_anime"
    
    @pytest.mark.asyncio
    async def test_request_deduplication_with_cache_integration(self):
        """Test request deduplication working with collaborative cache."""
        deduplicator = RequestDeduplication()
        cache_system = CollaborativeCacheSystem()
        
        # Mock cache to return data for one call
        cache_system.get_enhanced_data = AsyncMock(return_value={"title": "Shared Result"})
        
        # Multiple concurrent requests should be deduplicated
        async def fetch_func():
            return await cache_system.get_enhanced_data("anime123", "user456", "anilist")
        
        tasks = [
            deduplicator.deduplicate_request("anime123:anilist", fetch_func),
            deduplicator.deduplicate_request("anime123:anilist", fetch_func),
            deduplicator.deduplicate_request("anime123:anilist", fetch_func)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should return same result
        assert all(result == {"title": "Shared Result"} for result in results)
        # Cache should only be called once due to deduplication
        assert cache_system.get_enhanced_data.call_count == 1


@pytest.mark.asyncio
async def test_full_base_client_integration():
    """Test complete BaseClient integration with all error handling components."""
    # Create real instances (not mocks) for integration test
    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=300)
    cache_system = CollaborativeCacheSystem()
    error_handler = LangGraphErrorHandler()
    
    # Mock the external dependencies that aren't implemented yet
    mock_rate_limiter = AsyncMock()
    mock_rate_limiter.__aenter__ = AsyncMock(return_value=None)
    mock_rate_limiter.__aexit__ = AsyncMock(return_value=None)
    
    base_client = BaseClient(
        circuit_breaker=circuit_breaker,
        rate_limiter=mock_rate_limiter,
        cache_manager=cache_system,
        error_handler=error_handler
    )
    
    # Test that all components are properly integrated
    assert base_client.circuit_breaker == circuit_breaker
    assert base_client.cache_manager == cache_system
    assert base_client.error_handler == error_handler
    
    # Test error handling integration
    assert hasattr(base_client, 'make_request')
    assert hasattr(base_client, 'with_circuit_breaker')
    assert hasattr(base_client, 'handle_rate_limit')


# Additional tests for BaseScraper coverage
class TestBaseScraperCoverage:
    """Test cases for BaseScraper coverage (added to base_client tests to avoid new files)."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        return {
            "circuit_breaker": Mock(),
            "rate_limiter": AsyncMock(),
            "cache_manager": AsyncMock(),
            "error_handler": AsyncMock()
        }
    
    @pytest.fixture
    def base_scraper(self, mock_dependencies):
        """Create BaseScraper instance for testing."""
        from src.integrations.scrapers.base_scraper import BaseScraper
        mock_dependencies["circuit_breaker"].is_open = Mock(return_value=False)
        return BaseScraper(**mock_dependencies)
    
    @pytest.mark.asyncio
    async def test_make_request_http_error_coverage(self, base_scraper):
        """Test HTTP error handling coverage (line 67)."""
        url = "https://example.com/test"
        
        with patch('cloudscraper.create_scraper') as mock_create:
            mock_scraper = Mock()
            mock_response = Mock()
            mock_response.status_code = 404
            mock_scraper.get.return_value = mock_response
            mock_create.return_value = mock_scraper
            
            with pytest.raises(Exception, match="HTTP 404 error"):
                await base_scraper._make_request(url)
    
    @pytest.mark.asyncio
    async def test_make_request_exception_with_error_handler(self, base_scraper):
        """Test exception handling with error handler (lines 82-87)."""
        url = "https://example.com/test"
        
        with patch('cloudscraper.create_scraper') as mock_create:
            mock_scraper = Mock()
            mock_scraper.get.side_effect = Exception("Network error")
            mock_create.return_value = mock_scraper
            
            with pytest.raises(Exception, match="Scraping error for"):
                await base_scraper._make_request(url)
            
            # Verify error handler was called
            base_scraper.error_handler.handle_error.assert_called_once()
    
    def test_extract_json_ld_no_script_string(self, base_scraper):
        """Test JSON-LD extraction with empty script (lines 109-110)."""
        from bs4 import BeautifulSoup
        html = '<script type="application/ld+json"></script>'
        soup = BeautifulSoup(html, 'html.parser')
        
        result = base_scraper._extract_json_ld(soup)
        assert result is None
    
    def test_extract_json_ld_invalid_json(self, base_scraper):
        """Test JSON-LD extraction with invalid JSON (lines 127-128)."""
        from bs4 import BeautifulSoup
        html = '<script type="application/ld+json">invalid json{</script>'
        soup = BeautifulSoup(html, 'html.parser')
        
        result = base_scraper._extract_json_ld(soup)
        assert result is None
    
    def test_extract_json_ld_list_with_matching_type(self, base_scraper):
        """Test JSON-LD extraction with list containing matching type (lines 120-126)."""
        from bs4 import BeautifulSoup
        import json
        json_data = [
            {"@type": "Organization", "name": "Test"},
            {"@type": "TVSeries", "name": "Anime", "description": "Test anime"}
        ]
        html = f'<script type="application/ld+json">{json.dumps(json_data)}</script>'
        soup = BeautifulSoup(html, 'html.parser')
        
        result = base_scraper._extract_json_ld(soup)
        assert result is not None
        assert result["@type"] == "TVSeries"
        assert result["name"] == "Anime"
    
    def test_extract_meta_tags_coverage(self, base_scraper):
        """Test meta tags extraction edge cases (lines 148-157)."""
        from bs4 import BeautifulSoup
        html = '''
        <html>
        <head>
            <meta name="description" content="Test description">
            <meta name="keywords">
            <meta name="author" content="">
            <meta name="title" content="Test title">
        </head>
        </html>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        
        result = base_scraper._extract_meta_tags(soup)
        
        assert result["description"] == "Test description"
        assert result["title"] == "Test title"
        assert "keywords" not in result  # No content
        assert "author" not in result  # Empty content
    
    def test_clean_text_empty_input(self, base_scraper):
        """Test text cleaning with empty input (line 162)."""
        result = base_scraper._clean_text("")
        assert result == ""
        
        result = base_scraper._clean_text(None)
        assert result == ""
    
    def test_extract_base_data_no_json_ld(self, base_scraper):
        """Test base data extraction without JSON-LD (line 199)."""
        from bs4 import BeautifulSoup
        html = '''
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Test description">
        </head>
        </html>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        url = "https://example.com/test"
        
        result = base_scraper._extract_base_data(soup, url)
        
        assert result["json_ld"] is None
        assert result["page_title"] == "Test Page"
        assert result["meta_description"] == "Test description"
    
    @pytest.mark.asyncio
    async def test_close_method_coverage(self, base_scraper):
        """Test close method coverage (lines 210-212)."""
        # Set up a scraper instance
        base_scraper.scraper = Mock()
        
        await base_scraper.close()
        
        assert base_scraper.scraper is None
    
    def test_extract_json_ld_dict_matching_type_coverage(self, base_scraper):
        """Test JSON-LD extraction with dict having matching type (lines 117-119)."""
        from bs4 import BeautifulSoup
        import json
        json_data = {"@type": "Movie", "name": "Anime Movie", "description": "Test movie"}
        html = f'<script type="application/ld+json">{json.dumps(json_data)}</script>'
        soup = BeautifulSoup(html, 'html.parser')
        
        result = base_scraper._extract_json_ld(soup)
        assert result is not None
        assert result["@type"] == "Movie"
        assert result["name"] == "Anime Movie"
    
    def test_extract_base_data_with_json_ld_coverage(self, base_scraper):
        """Test base data extraction with JSON-LD present (line 199)."""
        from bs4 import BeautifulSoup
        import json
        json_data = {"@type": "TVSeries", "name": "Test Series"}
        html = f'''
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Test description">
            <script type="application/ld+json">{json.dumps(json_data)}</script>
        </head>
        </html>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        url = "https://example.com/test"
        
        result = base_scraper._extract_base_data(soup, url)
        
        assert result["json_ld"] is not None
        assert result["json_ld"]["@type"] == "TVSeries"
        assert result["page_title"] == "Test Page"
        assert result["meta_description"] == "Test description"