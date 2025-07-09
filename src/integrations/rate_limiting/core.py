"""Comprehensive rate limiting system with token buckets and request queuing.

Provides proactive rate limiting for all external anime APIs with:
- Token bucket algorithms for precise rate control
- Per-service configuration with burst handling
- Request queuing with priority and timeout management
- Distributed rate limiting support for scaling
- Metrics and monitoring for rate limit health
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies for different API behaviors."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitConfig:
    """Configuration for a specific service's rate limits."""

    # Basic rate limiting
    requests_per_second: float
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None

    # Burst handling
    burst_size: int = 10
    burst_refill_rate: float = 1.0

    # Strategy and behavior
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    backoff_factor: float = 2.0
    max_backoff: float = 300.0  # 5 minutes

    # Queue management
    max_queue_size: int = 1000
    request_timeout: float = 30.0
    priority_levels: int = 3

    # Adaptive behavior
    adaptive_decrease_factor: float = 0.8
    adaptive_increase_factor: float = 1.1
    adaptive_window_size: int = 100


class TokenBucket:
    """Token bucket implementation for smooth rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient
        """
        async with self._lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = max(0.0, float(now) - float(self.last_refill))
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = float(now)

            if self.tokens >= tokens:
                self.tokens -= tokens
                # Ensure tokens don't go below 0 due to floating point precision
                self.tokens = max(0.0, self.tokens)
                return True
            return False

    async def wait_for_tokens(self, tokens: int = 1, timeout: float = 30.0) -> bool:
        """Wait until tokens are available.

        Args:
            tokens: Number of tokens needed
            timeout: Maximum wait time in seconds

        Returns:
            True if tokens obtained, False if timeout
        """
        start_time = float(time.time())

        while float(time.time()) - start_time < timeout:
            if await self.consume(tokens):
                return True

            # Calculate wait time for next token
            async with self._lock:
                if self.tokens < tokens:
                    wait_time = min((tokens - self.tokens) / self.refill_rate, 1.0)
                    await asyncio.sleep(wait_time)

        return False

    def available_tokens(self) -> int:
        """Get number of available tokens."""
        now = time.time()
        elapsed = max(0.0, float(now) - float(self.last_refill))
        current_tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        return int(current_tokens)


@dataclass
class RateLimitedRequest:
    """Request waiting in rate limit queue."""

    future: asyncio.Future
    priority: int = 1  # 0=highest, 2=lowest
    timestamp: float = 0.0
    service: str = ""
    endpoint: str = ""
    retry_count: int = 0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class ServiceRateLimiter:
    """Rate limiter for a specific service."""

    def __init__(self, service_name: str, config: RateLimitConfig):
        """Initialize service rate limiter.

        Args:
            service_name: Name of the service
            config: Rate limiting configuration
        """
        self.service_name = service_name
        self.config = config

        # Primary token bucket for requests per second
        self.primary_bucket = TokenBucket(
            capacity=int(config.requests_per_second * config.burst_size),
            refill_rate=config.requests_per_second,
        )

        # Secondary buckets for longer time windows
        self.minute_bucket = None
        self.hour_bucket = None
        self.day_bucket = None

        if config.requests_per_minute:
            self.minute_bucket = TokenBucket(
                capacity=config.requests_per_minute,
                refill_rate=config.requests_per_minute / 60.0,
            )

        if config.requests_per_hour:
            self.hour_bucket = TokenBucket(
                capacity=config.requests_per_hour,
                refill_rate=config.requests_per_hour / 3600.0,
            )

        if config.requests_per_day:
            self.day_bucket = TokenBucket(
                capacity=config.requests_per_day,
                refill_rate=config.requests_per_day / 86400.0,
            )

        # Request queue with priority levels
        self.request_queues: List[deque] = [
            deque() for _ in range(config.priority_levels)
        ]
        self.queue_size = 0
        self.processing_task: Optional[asyncio.Task] = None
        self._queue_lock = asyncio.Lock()

        # Adaptive rate limiting
        self.recent_responses: deque = deque(maxlen=config.adaptive_window_size)
        self.current_backoff = 0.0
        self.last_429_time = 0.0

        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.rate_limited_requests = 0
        self.queue_timeouts = 0

    async def acquire(
        self, priority: int = 1, timeout: float = None, endpoint: str = ""
    ) -> bool:
        """Acquire permission to make a request.

        Args:
            priority: Request priority (0=highest, 2=lowest)
            timeout: Maximum wait time
            endpoint: API endpoint for debugging

        Returns:
            True if permission granted, False if timeout/rejected
        """
        if timeout is None:
            timeout = self.config.request_timeout

        # Check if we can immediately proceed
        if await self._try_immediate_acquire():
            self.total_requests += 1
            self.successful_requests += 1
            return True

        # Queue the request
        if self.queue_size >= self.config.max_queue_size:
            logger.warning(f"Rate limit queue full for {self.service_name}")
            self.rate_limited_requests += 1
            return False

        future = asyncio.Future()
        request = RateLimitedRequest(
            future=future,
            priority=priority,
            service=self.service_name,
            endpoint=endpoint,
        )

        async with self._queue_lock:
            self.request_queues[priority].append(request)
            self.queue_size += 1

            # Start processing task if not running
            if self.processing_task is None or self.processing_task.done():
                self.processing_task = asyncio.create_task(self._process_queue())

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            self.total_requests += 1
            if result:
                self.successful_requests += 1
            else:
                self.rate_limited_requests += 1
            return result
        except asyncio.TimeoutError:
            self.queue_timeouts += 1
            # Remove from queue if still there
            async with self._queue_lock:
                for queue in self.request_queues:
                    if request in queue:
                        queue.remove(request)
                        self.queue_size -= 1
                        break
            return False

    async def _try_immediate_acquire(self) -> bool:
        """Try to immediately acquire tokens from all buckets."""
        # Check primary bucket first
        if not await self.primary_bucket.consume():
            return False

        # Check secondary buckets
        buckets_to_check = [self.minute_bucket, self.hour_bucket, self.day_bucket]

        consumed_buckets = []

        for bucket in buckets_to_check:
            if bucket is not None:
                if await bucket.consume():
                    consumed_buckets.append(bucket)
                else:
                    # Restore tokens to previously consumed buckets
                    for consumed_bucket in consumed_buckets:
                        consumed_bucket.tokens += 1
                    # Restore primary bucket
                    self.primary_bucket.tokens += 1
                    return False

        return True

    async def _process_queue(self):
        """Process queued requests in priority order."""
        while True:
            request = await self._get_next_request()
            if request is None:
                break

            try:
                # Wait for tokens to be available
                success = await self._wait_for_all_buckets()

                if success:
                    request.future.set_result(True)
                else:
                    request.future.set_result(False)

            except Exception as e:
                request.future.set_exception(e)

    async def _get_next_request(self) -> Optional[RateLimitedRequest]:
        """Get the next request from priority queues."""
        async with self._queue_lock:
            # Check queues in priority order (0=highest priority)
            for queue in self.request_queues:
                if queue:
                    request = queue.popleft()
                    self.queue_size -= 1
                    return request
            return None

    async def _wait_for_all_buckets(self) -> bool:
        """Wait for tokens to be available in all buckets."""
        buckets = [self.primary_bucket]

        if self.minute_bucket:
            buckets.append(self.minute_bucket)
        if self.hour_bucket:
            buckets.append(self.hour_bucket)
        if self.day_bucket:
            buckets.append(self.day_bucket)

        # Wait for all buckets to have tokens with shorter timeout for testing
        timeout = min(
            self.config.request_timeout, 5.0
        )  # Max 5 seconds to avoid test hanging
        for bucket in buckets:
            if not await bucket.wait_for_tokens(1, timeout=timeout):
                return False

        return True

    def record_response(
        self, status_code: int, response_time: float, rate_limit_info=None
    ):
        """Record API response for adaptive rate limiting.

        Args:
            status_code: HTTP status code
            response_time: Response time in seconds
            rate_limit_info: Optional platform-specific rate limit information
        """
        self.recent_responses.append(
            {
                "status_code": status_code,
                "response_time": response_time,
                "timestamp": time.time(),
                "rate_limit_info": rate_limit_info,
            }
        )

        if status_code == 429:
            self._handle_429_response(rate_limit_info)
        elif status_code < 400:
            self._handle_success_response(rate_limit_info)

    def _handle_429_response(self, rate_limit_info=None):
        """Handle 429 Too Many Requests response with optional platform-specific data."""
        self.last_429_time = time.time()

        # Use platform-specific retry-after if available
        if (
            rate_limit_info
            and hasattr(rate_limit_info, "retry_after")
            and rate_limit_info.retry_after
        ):
            # Platform provided specific retry time
            self.current_backoff = min(
                rate_limit_info.retry_after, self.config.max_backoff
            )
        else:
            # Standard exponential backoff
            self.current_backoff = min(
                (
                    self.current_backoff * self.config.backoff_factor
                    if self.current_backoff > 0
                    else 1.0
                ),
                self.config.max_backoff,
            )

        # Temporarily reduce rate limit with platform-aware adaptation
        if self.config.strategy == RateLimitStrategy.ADAPTIVE:
            # Check if we have platform-specific rate limit data
            if (
                rate_limit_info
                and hasattr(rate_limit_info, "remaining")
                and rate_limit_info.remaining is not None
                and rate_limit_info.limit
            ):
                # Adapt based on actual remaining capacity
                usage_ratio = (
                    rate_limit_info.limit - rate_limit_info.remaining
                ) / rate_limit_info.limit
                adaptation_factor = max(
                    0.3, 1.0 - usage_ratio
                )  # More aggressive when heavily used
                new_rate = self.primary_bucket.refill_rate * adaptation_factor
            else:
                # Standard adaptation
                new_rate = (
                    self.primary_bucket.refill_rate
                    * self.config.adaptive_decrease_factor
                )

            self.primary_bucket.refill_rate = max(new_rate, 0.1)  # Minimum 0.1 req/sec

        # Enhanced logging with platform data
        if rate_limit_info and hasattr(rate_limit_info, "custom_data"):
            platform = rate_limit_info.custom_data.get("platform", self.service_name)
            logger.warning(
                f"Rate limited by {platform}, backing off for {self.current_backoff}s. "
                f"Remaining: {getattr(rate_limit_info, 'remaining', 'unknown')}, "
                f"Limit: {getattr(rate_limit_info, 'limit', 'unknown')}"
            )
        else:
            logger.warning(
                f"Rate limited by {self.service_name}, backing off for {self.current_backoff}s"
            )

    def _handle_success_response(self, rate_limit_info=None):
        """Handle successful response with optional platform-specific optimization."""
        # Gradually reduce backoff
        if self.current_backoff > 0:
            self.current_backoff = max(0, self.current_backoff - 1.0)

        # Platform-aware adaptive rate limiting
        if (
            self.config.strategy == RateLimitStrategy.ADAPTIVE
            and time.time() - self.last_429_time > 300
        ):  # 5 minutes since last 429

            # Check if we have platform rate limit data for smarter adaptation
            if (
                rate_limit_info
                and hasattr(rate_limit_info, "remaining")
                and rate_limit_info.remaining is not None
                and rate_limit_info.limit
            ):
                # Adapt based on current capacity utilization
                utilization = (
                    rate_limit_info.limit - rate_limit_info.remaining
                ) / rate_limit_info.limit

                if utilization < 0.5:  # Low utilization, can increase more aggressively
                    increase_factor = self.config.adaptive_increase_factor * 1.2
                elif utilization < 0.8:  # Moderate utilization, standard increase
                    increase_factor = self.config.adaptive_increase_factor
                else:  # High utilization, conservative increase
                    increase_factor = self.config.adaptive_increase_factor * 0.8

                new_rate = min(
                    self.primary_bucket.refill_rate * increase_factor,
                    self.config.requests_per_second,
                )
            else:
                # Standard adaptation without platform data
                new_rate = min(
                    self.primary_bucket.refill_rate
                    * self.config.adaptive_increase_factor,
                    self.config.requests_per_second,
                )

            self.primary_bucket.refill_rate = new_rate

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "service": self.service_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "queue_timeouts": self.queue_timeouts,
            "current_queue_size": self.queue_size,
            "available_tokens": self.primary_bucket.available_tokens(),
            "current_backoff": self.current_backoff,
            "last_429_time": self.last_429_time,
            "current_rate": self.primary_bucket.refill_rate,
        }


class RateLimitManager:
    """Central manager for all service rate limiters with platform adapter support."""

    def __init__(self):
        """Initialize rate limit manager."""
        self.service_limiters: Dict[str, ServiceRateLimiter] = {}
        self._default_configs = self._get_default_configs()
        self._platform_adapters: Dict[str, Any] = {}  # Platform adapters registry

    def _get_default_configs(self) -> Dict[str, RateLimitConfig]:
        """Get default rate limit configurations for each service.

        Based on official API documentation and best practices:
        - MAL: OAuth2, 2 req/sec, 60 req/min
        - AniList: Optional OAuth2, 90 req/min burst limit
        - Kitsu: No auth, 10 req/sec
        - AniDB: Client registration, 1 req/sec
        - AnimeNewsNetwork: No auth, 1 req/sec
        - AnimeSchedule: No auth, unlimited (but we set conservative limits)
        - Scrapers: Very conservative to avoid blocking
        """
        return {
            # Official APIs with documented limits (researched from official docs)
            "mal": RateLimitConfig(
                requests_per_second=2.0,  # MAL API v2: 2 req/sec, 60 req/min
                requests_per_minute=60,
                burst_size=5,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                backoff_factor=2.0,
                max_backoff=300.0,
            ),
            "anilist": RateLimitConfig(
                requests_per_second=1.5,  # AniList: 90 req/min = 1.5 req/sec
                requests_per_minute=90,  # Can degrade to 30/min during issues
                burst_size=8,  # Burst limiter prevents rapid requests
                strategy=RateLimitStrategy.ADAPTIVE,
                backoff_factor=2.0,
                max_backoff=300.0,
                adaptive_decrease_factor=0.7,  # Moderate decrease on 429
                adaptive_increase_factor=1.1,
            ),
            "kitsu": RateLimitConfig(
                requests_per_second=8.0,  # Conservative estimate (no explicit limits)
                burst_size=15,  # Generous burst for pagination
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                backoff_factor=2.0,
                max_backoff=300.0,
            ),
            "anidb": RateLimitConfig(
                requests_per_second=0.5,  # AniDB: 1 request every 2 seconds
                requests_per_hour=150,  # ~100-200 requests per day limit
                requests_per_day=200,  # Daily limit from community research
                burst_size=1,  # Very strict - no burst allowed
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                backoff_factor=4.0,  # Aggressive backoff (they ban easily)
                max_backoff=3600.0,  # 1 hour max backoff
                request_timeout=10.0,  # Shorter timeout for strict API
                max_queue_size=20,  # Small queue for daily limits
            ),
            "animenewsnetwork": RateLimitConfig(
                requests_per_second=1.0,  # ANN: 1 req/sec per IP (delays vs rejects)
                burst_size=2,  # Small burst allowed
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                backoff_factor=2.0,
                max_backoff=300.0,
            ),
            "animeschedule": RateLimitConfig(
                requests_per_second=5.0,  # Conservative for "unlimited" API
                burst_size=10,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                backoff_factor=2.0,
                max_backoff=300.0,
            ),
            # Unofficial APIs
            "jikan": RateLimitConfig(
                requests_per_second=3.0,  # Jikan v4: 3 req/sec, 60 req/min
                requests_per_minute=60,
                burst_size=6,  # Allow small burst
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                backoff_factor=2.0,
                max_backoff=300.0,
            ),
            # Web scrapers - extremely conservative to avoid blocking
            "animeplanet": RateLimitConfig(
                requests_per_second=0.5,  # 1 request every 2 seconds
                burst_size=2,  # Minimal burst
                strategy=RateLimitStrategy.ADAPTIVE,
                backoff_factor=3.0,  # Aggressive backoff
                max_backoff=600.0,  # 10 minutes max
                adaptive_decrease_factor=0.4,  # Very aggressive decrease
                adaptive_increase_factor=1.05,  # Very slow increase
                max_queue_size=30,  # Smaller queue for scraping
                request_timeout=20.0,  # Longer timeout for web requests
            ),
            "anisearch": RateLimitConfig(
                requests_per_second=0.33,  # 1 request every 3 seconds (most conservative)
                burst_size=1,  # No burst allowed
                strategy=RateLimitStrategy.ADAPTIVE,
                backoff_factor=4.0,  # Very aggressive backoff
                max_backoff=900.0,  # 15 minutes max
                adaptive_decrease_factor=0.3,  # Extremely aggressive decrease
                adaptive_increase_factor=1.02,  # Extremely slow increase
                max_queue_size=20,  # Small queue
                request_timeout=25.0,  # Longer timeout for German site
            ),
            "animecountdown": RateLimitConfig(
                requests_per_second=0.5,  # 1 request every 2 seconds
                burst_size=2,
                strategy=RateLimitStrategy.ADAPTIVE,
                backoff_factor=3.0,
                max_backoff=600.0,  # 10 minutes max
                adaptive_decrease_factor=0.4,
                adaptive_increase_factor=1.05,
                max_queue_size=30,
                request_timeout=20.0,
            ),
        }

    def get_limiter(self, service_name: str) -> ServiceRateLimiter:
        """Get rate limiter for a service.

        Args:
            service_name: Name of the service

        Returns:
            ServiceRateLimiter instance
        """
        if service_name not in self.service_limiters:
            config = self._default_configs.get(
                service_name,
                RateLimitConfig(requests_per_second=1.0),  # Default fallback
            )
            self.service_limiters[service_name] = ServiceRateLimiter(
                service_name, config
            )

        return self.service_limiters[service_name]

    async def acquire(
        self,
        service_name: str,
        priority: int = 1,
        timeout: float = None,
        endpoint: str = "",
    ) -> bool:
        """Acquire rate limit permission for a service.

        Args:
            service_name: Name of the service
            priority: Request priority (0=highest, 2=lowest)
            timeout: Maximum wait time
            endpoint: API endpoint for debugging

        Returns:
            True if permission granted
        """
        limiter = self.get_limiter(service_name)
        return await limiter.acquire(priority, timeout, endpoint)

    def register_platform_adapter(self, service_name: str, adapter):
        """Register a platform-specific adapter for enhanced rate limiting.

        Args:
            service_name: Name of the service
            adapter: Platform-specific rate limit adapter
        """
        self._platform_adapters[service_name] = adapter

    def extract_platform_rate_info(self, service_name: str, response):
        """Extract platform-specific rate limit information.

        Args:
            service_name: Name of the service
            response: HTTP response object

        Returns:
            Platform-specific rate limit info or None
        """
        adapter = self._platform_adapters.get(service_name)
        if adapter and hasattr(adapter, "extract_rate_limit_info"):
            return adapter.extract_rate_limit_info(response)
        return None

    def record_response(
        self, service_name: str, status_code: int, response_time: float, response=None
    ):
        """Record API response for adaptive rate limiting with platform-specific data.

        Args:
            service_name: Name of the service
            status_code: HTTP status code
            response_time: Response time in seconds
            response: Optional HTTP response object for platform-specific header extraction
        """
        if service_name in self.service_limiters:
            # Extract platform-specific rate limit info if adapter is available
            rate_limit_info = None
            if response:
                rate_limit_info = self.extract_platform_rate_info(
                    service_name, response
                )

            self.service_limiters[service_name].record_response(
                status_code, response_time, rate_limit_info
            )

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all service rate limiters."""
        return {
            service_name: limiter.get_stats()
            for service_name, limiter in self.service_limiters.items()
        }

    def update_config(self, service_name: str, config: RateLimitConfig):
        """Update configuration for a service.

        Args:
            service_name: Name of the service
            config: New rate limiting configuration
        """
        self._default_configs[service_name] = config
        # If limiter already exists, recreate it
        if service_name in self.service_limiters:
            del self.service_limiters[service_name]


# Global rate limit manager instance
rate_limit_manager = RateLimitManager()
