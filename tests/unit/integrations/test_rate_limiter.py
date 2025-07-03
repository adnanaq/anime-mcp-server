"""Comprehensive tests for rate limiting system with 100% coverage."""

import asyncio
import time
from unittest.mock import patch

import pytest

from src.integrations.rate_limiter import (
    RateLimitConfig,
    RateLimitedRequest,
    RateLimitManager,
    RateLimitStrategy,
    ServiceRateLimiter,
    TokenBucket,
    rate_limit_manager,
)


class TestTokenBucket:
    """Test token bucket implementation."""

    @pytest.fixture
    def bucket(self):
        """Create token bucket for testing."""
        return TokenBucket(capacity=10, refill_rate=5.0)

    @pytest.mark.asyncio
    async def test_token_bucket_initialization(self, bucket):
        """Test token bucket initializes correctly."""
        assert bucket.capacity == 10
        assert bucket.refill_rate == 5.0
        assert bucket.tokens == 10.0
        assert bucket.last_refill > 0

    @pytest.mark.asyncio
    async def test_consume_tokens_success(self, bucket):
        """Test successful token consumption."""
        result = await bucket.consume(5)
        assert result is True
        assert bucket.tokens == 5.0

    @pytest.mark.asyncio
    async def test_consume_tokens_insufficient(self, bucket):
        """Test token consumption with insufficient tokens."""
        # Mock time to prevent refill
        with patch("time.time") as mock_time:
            # Set a consistent time
            fixed_time = bucket.last_refill
            mock_time.return_value = fixed_time

            # Consume all tokens first
            await bucket.consume(10)

            # Try to consume more (still at same time, no refill)
            result = await bucket.consume(1)
            assert result is False
            assert bucket.tokens == 0.0

    @pytest.mark.asyncio
    async def test_token_refill_over_time(self, bucket):
        """Test tokens refill over time."""
        # Consume all tokens
        await bucket.consume(10)
        assert bucket.tokens == 0.0

        # Mock time to simulate passage
        with patch("time.time") as mock_time:
            # Simulate 1 second passing (5 tokens should refill)
            mock_time.side_effect = [bucket.last_refill + 1.0, bucket.last_refill + 1.0]

            # Try to consume 3 tokens (should succeed)
            result = await bucket.consume(3)
            assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_tokens_success(self, bucket):
        """Test waiting for tokens successfully."""
        # Consume most tokens
        await bucket.consume(9)

        # Should be able to wait for 1 token
        result = await bucket.wait_for_tokens(1, timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_tokens_timeout(self, bucket):
        """Test waiting for tokens times out."""
        # Consume all tokens
        await bucket.consume(10)

        # Should timeout waiting for 10 tokens
        result = await bucket.wait_for_tokens(10, timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_available_tokens(self, bucket):
        """Test available tokens calculation."""
        initial_tokens = bucket.available_tokens()
        assert initial_tokens == 10

        # Consume some tokens
        await bucket.consume(3)
        remaining_tokens = bucket.available_tokens()
        assert remaining_tokens == 7

    @pytest.mark.asyncio
    async def test_capacity_limit(self, bucket):
        """Test tokens don't exceed capacity."""
        # Wait longer than needed to refill
        with patch("time.time") as mock_time:
            mock_time.side_effect = [
                bucket.last_refill + 10.0,
                bucket.last_refill + 10.0,
            ]

            # Try to consume 1 token
            result = await bucket.consume(1)
            assert result is True

            # Should have capacity - 1 tokens
            assert bucket.tokens == 9.0


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_rate_limit_config_defaults(self):
        """Test rate limit config with defaults."""
        config = RateLimitConfig(requests_per_second=2.0)

        assert config.requests_per_second == 2.0
        assert config.requests_per_minute is None
        assert config.burst_size == 10
        assert config.strategy == RateLimitStrategy.TOKEN_BUCKET
        assert config.max_queue_size == 1000

    def test_rate_limit_config_custom(self):
        """Test rate limit config with custom values."""
        config = RateLimitConfig(
            requests_per_second=5.0,
            requests_per_minute=120,
            burst_size=20,
            strategy=RateLimitStrategy.ADAPTIVE,
            max_queue_size=500,
        )

        assert config.requests_per_second == 5.0
        assert config.requests_per_minute == 120
        assert config.burst_size == 20
        assert config.strategy == RateLimitStrategy.ADAPTIVE
        assert config.max_queue_size == 500


class TestRateLimitedRequest:
    """Test rate limited request data structure."""

    def test_rate_limited_request_initialization(self):
        """Test rate limited request initializes correctly."""
        future = asyncio.Future()
        request = RateLimitedRequest(future=future, priority=1)

        assert request.future == future
        assert request.priority == 1
        assert request.timestamp > 0
        assert request.service == ""
        assert request.endpoint == ""
        assert request.retry_count == 0

    def test_rate_limited_request_with_values(self):
        """Test rate limited request with custom values."""
        future = asyncio.Future()
        request = RateLimitedRequest(
            future=future, priority=0, service="test_service", endpoint="/api/test"
        )

        assert request.future == future
        assert request.priority == 0
        assert request.service == "test_service"
        assert request.endpoint == "/api/test"


class TestServiceRateLimiter:
    """Test service-specific rate limiter."""

    @pytest.fixture
    def config(self):
        """Create rate limit config for testing."""
        return RateLimitConfig(
            requests_per_second=2.0,
            requests_per_minute=60,
            burst_size=5,
            max_queue_size=10,
        )

    @pytest.fixture
    def limiter(self, config):
        """Create service rate limiter for testing."""
        return ServiceRateLimiter("test_service", config)

    def test_service_rate_limiter_initialization(self, limiter, config):
        """Test service rate limiter initializes correctly."""
        assert limiter.service_name == "test_service"
        assert limiter.config == config
        assert limiter.primary_bucket is not None
        assert limiter.minute_bucket is not None
        assert limiter.hour_bucket is None
        assert limiter.day_bucket is None
        assert len(limiter.request_queues) == config.priority_levels
        assert limiter.queue_size == 0

    @pytest.mark.asyncio
    async def test_acquire_immediate_success(self, limiter):
        """Test immediate acquisition success."""
        result = await limiter.acquire(priority=1, timeout=1.0)
        assert result is True
        assert limiter.total_requests == 1
        assert limiter.successful_requests == 1

    @pytest.mark.asyncio
    async def test_acquire_queue_full(self, limiter):
        """Test acquisition when queue is full."""
        # First consume all tokens to force queueing
        await limiter.primary_bucket.consume(limiter.primary_bucket.capacity)
        await limiter.minute_bucket.consume(limiter.minute_bucket.capacity)

        # Then manually set queue as full
        limiter.queue_size = limiter.config.max_queue_size

        result = await limiter.acquire(priority=1, timeout=0.1)
        assert result is False
        assert limiter.rate_limited_requests == 1

    @pytest.mark.asyncio
    async def test_try_immediate_acquire_success(self, limiter):
        """Test immediate acquisition from buckets."""
        result = await limiter._try_immediate_acquire()
        assert result is True

    @pytest.mark.asyncio
    async def test_try_immediate_acquire_failure(self, limiter):
        """Test immediate acquisition failure."""
        # Consume all tokens from primary bucket
        await limiter.primary_bucket.consume(limiter.primary_bucket.capacity)

        result = await limiter._try_immediate_acquire()
        assert result is False

    @pytest.mark.asyncio
    async def test_try_immediate_acquire_minute_bucket_failure(self, limiter):
        """Test immediate acquisition failure from minute bucket."""
        # Consume all tokens from minute bucket
        await limiter.minute_bucket.consume(limiter.minute_bucket.capacity)

        result = await limiter._try_immediate_acquire()
        assert result is False

    @pytest.mark.asyncio
    async def test_get_next_request_empty_queues(self, limiter):
        """Test getting next request from empty queues."""
        request = await limiter._get_next_request()
        assert request is None

    @pytest.mark.asyncio
    async def test_get_next_request_priority_order(self, limiter):
        """Test getting requests in priority order."""
        # Add requests to different priority queues
        future1 = asyncio.Future()
        future2 = asyncio.Future()
        future3 = asyncio.Future()

        request1 = RateLimitedRequest(future=future1, priority=2)  # Low priority
        request2 = RateLimitedRequest(future=future2, priority=0)  # High priority
        request3 = RateLimitedRequest(future=future3, priority=1)  # Medium priority

        limiter.request_queues[2].append(request1)
        limiter.request_queues[0].append(request2)
        limiter.request_queues[1].append(request3)
        limiter.queue_size = 3

        # Should get highest priority first
        next_request = await limiter._get_next_request()
        assert next_request == request2
        assert limiter.queue_size == 2

    @pytest.mark.asyncio
    async def test_wait_for_all_buckets_success(self, limiter):
        """Test waiting for all buckets successfully."""
        result = await limiter._wait_for_all_buckets()
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_all_buckets_timeout(self, limiter):
        """Test waiting for all buckets times out."""
        # Consume all tokens first
        await limiter.primary_bucket.consume(limiter.primary_bucket.capacity)
        await limiter.minute_bucket.consume(limiter.minute_bucket.capacity)

        # Mock the bucket's wait_for_tokens to simulate timeout
        with patch.object(limiter.primary_bucket, "wait_for_tokens") as mock_wait:
            mock_wait.return_value = False  # Simulate timeout
            
            result = await limiter._wait_for_all_buckets()
            assert result is False

    def test_record_response_success(self, limiter):
        """Test recording successful response."""
        limiter.record_response(200, 0.5)

        assert len(limiter.recent_responses) == 1
        response = limiter.recent_responses[0]
        assert response["status_code"] == 200
        assert response["response_time"] == 0.5

    def test_record_response_429(self, limiter):
        """Test recording 429 response."""
        original_backoff = limiter.current_backoff
        original_rate = limiter.primary_bucket.refill_rate

        limiter.record_response(429, 1.0)

        assert limiter.current_backoff > original_backoff
        assert limiter.last_429_time > 0
        # For adaptive strategy, rate should decrease
        if limiter.config.strategy == RateLimitStrategy.ADAPTIVE:
            assert limiter.primary_bucket.refill_rate < original_rate

    def test_handle_429_response(self, limiter):
        """Test handling 429 response."""
        limiter._handle_429_response()

        assert limiter.current_backoff > 0
        assert limiter.last_429_time > 0

    def test_handle_success_response(self, limiter):
        """Test handling successful response."""
        # Set some backoff first
        limiter.current_backoff = 10.0

        limiter._handle_success_response()

        assert limiter.current_backoff < 10.0

    def test_handle_success_response_adaptive_increase(self, limiter):
        """Test adaptive rate increase on success."""
        # Set adaptive strategy
        limiter.config.strategy = RateLimitStrategy.ADAPTIVE
        limiter.last_429_time = time.time() - 400  # More than 5 minutes ago
        original_rate = limiter.primary_bucket.refill_rate

        limiter._handle_success_response()

        assert limiter.primary_bucket.refill_rate >= original_rate

    def test_get_stats(self, limiter):
        """Test getting rate limiter statistics."""
        limiter.total_requests = 100
        limiter.successful_requests = 95
        limiter.rate_limited_requests = 3
        limiter.queue_timeouts = 2

        stats = limiter.get_stats()

        assert stats["service"] == "test_service"
        assert stats["total_requests"] == 100
        assert stats["successful_requests"] == 95
        assert stats["rate_limited_requests"] == 3
        assert stats["queue_timeouts"] == 2
        assert "current_queue_size" in stats
        assert "available_tokens" in stats


class TestRateLimitManager:
    """Test rate limit manager."""

    @pytest.fixture
    def manager(self):
        """Create rate limit manager for testing."""
        return RateLimitManager()

    def test_rate_limit_manager_initialization(self, manager):
        """Test rate limit manager initializes correctly."""
        assert len(manager.service_limiters) == 0
        assert len(manager._default_configs) > 0
        assert "mal" in manager._default_configs
        assert "anilist" in manager._default_configs

    def test_get_limiter_new_service(self, manager):
        """Test getting limiter for new service."""
        limiter = manager.get_limiter("mal")

        assert limiter.service_name == "mal"
        assert "mal" in manager.service_limiters
        assert manager.service_limiters["mal"] == limiter

    def test_get_limiter_existing_service(self, manager):
        """Test getting limiter for existing service."""
        limiter1 = manager.get_limiter("mal")
        limiter2 = manager.get_limiter("mal")

        assert limiter1 == limiter2

    def test_get_limiter_unknown_service(self, manager):
        """Test getting limiter for unknown service."""
        limiter = manager.get_limiter("unknown_service")

        assert limiter.service_name == "unknown_service"
        assert limiter.config.requests_per_second == 1.0  # Default fallback

    @pytest.mark.asyncio
    async def test_acquire_success(self, manager):
        """Test acquiring rate limit permission successfully."""
        result = await manager.acquire("mal", priority=1, timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_with_endpoint(self, manager):
        """Test acquiring with endpoint specification."""
        result = await manager.acquire("mal", endpoint="/api/anime", timeout=1.0)
        assert result is True

    def test_record_response_existing_service(self, manager):
        """Test recording response for existing service."""
        # Create limiter first
        manager.get_limiter("mal")

        # Record response
        manager.record_response("mal", 200, 0.5)

        limiter = manager.service_limiters["mal"]
        assert len(limiter.recent_responses) == 1

    def test_record_response_nonexistent_service(self, manager):
        """Test recording response for non-existent service."""
        # Should not raise exception
        manager.record_response("nonexistent", 200, 0.5)

    def test_get_all_stats_empty(self, manager):
        """Test getting stats with no services."""
        stats = manager.get_all_stats()
        assert stats == {}

    def test_get_all_stats_with_services(self, manager):
        """Test getting stats with services."""
        manager.get_limiter("mal")
        manager.get_limiter("anilist")

        stats = manager.get_all_stats()

        assert "mal" in stats
        assert "anilist" in stats
        assert stats["mal"]["service"] == "mal"
        assert stats["anilist"]["service"] == "anilist"

    def test_update_config(self, manager):
        """Test updating service configuration."""
        new_config = RateLimitConfig(requests_per_second=10.0)

        manager.update_config("test_service", new_config)

        assert manager._default_configs["test_service"] == new_config

    def test_update_config_existing_limiter(self, manager):
        """Test updating config for existing limiter."""
        # Create limiter first
        manager.get_limiter("mal")
        original_limiter = manager.service_limiters["mal"]

        # Update config
        new_config = RateLimitConfig(requests_per_second=10.0)
        manager.update_config("mal", new_config)

        # Should remove existing limiter
        assert "mal" not in manager.service_limiters

        # New limiter should use new config
        new_limiter = manager.get_limiter("mal")
        assert new_limiter != original_limiter


class TestGlobalRateLimitManager:
    """Test global rate limit manager instance."""

    def test_global_instance_exists(self):
        """Test global rate limit manager instance exists."""
        assert rate_limit_manager is not None
        assert isinstance(rate_limit_manager, RateLimitManager)

    def test_global_instance_default_configs(self):
        """Test global instance has default configurations."""
        configs = rate_limit_manager._default_configs

        # Test specific service configs
        assert "mal" in configs
        assert configs["mal"].requests_per_second == 2.0
        assert configs["mal"].requests_per_minute == 60

        assert "anilist" in configs
        assert configs["anilist"].requests_per_second == 1.5
        assert configs["anilist"].requests_per_minute == 90

        assert "kitsu" in configs
        assert configs["kitsu"].requests_per_second == 8.0

        # Test scraper configs are more conservative
        assert "animeplanet" in configs
        assert configs["animeplanet"].requests_per_second == 0.5
        assert configs["animeplanet"].backoff_factor == 3.0


class TestRateLimitStrategies:
    """Test different rate limiting strategies."""

    def test_token_bucket_strategy(self):
        """Test token bucket strategy configuration."""
        config = RateLimitConfig(
            requests_per_second=2.0, strategy=RateLimitStrategy.TOKEN_BUCKET
        )

        assert config.strategy == RateLimitStrategy.TOKEN_BUCKET

    def test_adaptive_strategy(self):
        """Test adaptive strategy configuration."""
        config = RateLimitConfig(
            requests_per_second=2.0, strategy=RateLimitStrategy.ADAPTIVE
        )

        assert config.strategy == RateLimitStrategy.ADAPTIVE
        assert config.adaptive_decrease_factor == 0.8
        assert config.adaptive_increase_factor == 1.1

    def test_sliding_window_strategy(self):
        """Test sliding window strategy configuration."""
        config = RateLimitConfig(
            requests_per_second=2.0, strategy=RateLimitStrategy.SLIDING_WINDOW
        )

        assert config.strategy == RateLimitStrategy.SLIDING_WINDOW

    def test_fixed_window_strategy(self):
        """Test fixed window strategy configuration."""
        config = RateLimitConfig(
            requests_per_second=2.0, strategy=RateLimitStrategy.FIXED_WINDOW
        )

        assert config.strategy == RateLimitStrategy.FIXED_WINDOW


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_acquire_timeout_removes_from_queue(self):
        """Test that timeout removes request from queue."""
        config = RateLimitConfig(requests_per_second=0.1)  # Very slow
        limiter = ServiceRateLimiter("test", config)

        # Consume tokens so we need to queue
        await limiter.primary_bucket.consume(limiter.primary_bucket.capacity)

        # This should timeout and be removed from queue
        result = await limiter.acquire(timeout=0.1)
        assert result is False
        assert limiter.queue_timeouts == 1

    @pytest.mark.asyncio
    async def test_process_queue_exception_handling(self):
        """Test process queue handles exceptions properly."""
        config = RateLimitConfig(requests_per_second=1.0)
        limiter = ServiceRateLimiter("test", config)

        # Mock _wait_for_all_buckets to raise exception
        async def mock_wait():
            raise Exception("Test exception")

        limiter._wait_for_all_buckets = mock_wait

        # Add a request to queue
        future = asyncio.Future()
        request = RateLimitedRequest(future=future)
        limiter.request_queues[1].append(request)
        limiter.queue_size = 1

        # Process queue should handle exception
        await limiter._process_queue()

        # Future should have exception set
        assert future.done()
        with pytest.raises(Exception, match="Test exception"):
            future.result()

    def test_token_bucket_with_zero_capacity(self):
        """Test token bucket with zero capacity."""
        bucket = TokenBucket(capacity=0, refill_rate=1.0)
        assert bucket.capacity == 0
        assert bucket.tokens == 0.0

    def test_rate_limit_config_minimal(self):
        """Test rate limit config with minimal parameters."""
        config = RateLimitConfig(requests_per_second=1.0)

        assert config.requests_per_second == 1.0
        assert config.requests_per_minute is None
        assert config.burst_size == 10
        assert config.strategy == RateLimitStrategy.TOKEN_BUCKET
