"""Platform-specific rate limiting strategies following Strategy + Adapter patterns.

This module implements the Strategy pattern for platform-specific rate limiting behavior
and the Adapter pattern for extracting platform-specific rate limit information.

Based on industry best practices from GitHub, Twitter, and other major APIs.
"""

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp


@dataclass
class RateLimitInfo:
    """Standardized rate limit information extracted from platform-specific responses.

    This serves as the common interface for the Adapter pattern.
    """

    remaining: Optional[int] = None
    limit: Optional[int] = None
    reset_time: Optional[int] = None
    retry_after: Optional[int] = None
    degraded: bool = False
    custom_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_data is None:
            self.custom_data = {}


class RateLimitStrategy(ABC):
    """Strategy pattern interface for platform-specific rate limiting behavior.

    Each platform can implement its own strategy for handling rate limits,
    backoff calculations, and response processing.
    """

    @abstractmethod
    async def handle_rate_limit_response(self, rate_info: RateLimitInfo) -> None:
        """Handle a rate limit response with platform-specific logic.

        Args:
            rate_info: Standardized rate limit information
        """

    @abstractmethod
    async def calculate_backoff_delay(
        self, rate_info: RateLimitInfo, attempt: int = 0
    ) -> float:
        """Calculate platform-specific backoff delay.

        Args:
            rate_info: Rate limit information
            attempt: Current retry attempt number

        Returns:
            Delay in seconds
        """

    @abstractmethod
    def should_proactive_throttle(self, rate_info: RateLimitInfo) -> bool:
        """Determine if proactive throttling should be applied.

        Args:
            rate_info: Current rate limit state

        Returns:
            True if requests should be proactively slowed
        """


class PlatformRateLimitAdapter(ABC):
    """Adapter pattern interface for platform-specific header extraction.

    Each platform implements this to extract rate limit information from
    their specific response headers and formats.
    """

    @abstractmethod
    def extract_rate_limit_info(
        self, response: aiohttp.ClientResponse
    ) -> RateLimitInfo:
        """Extract rate limit information from platform-specific response.

        Args:
            response: HTTP response from the platform API

        Returns:
            Standardized rate limit information
        """

    @abstractmethod
    def get_strategy(self) -> RateLimitStrategy:
        """Get the rate limiting strategy for this platform.

        Returns:
            Platform-specific rate limiting strategy
        """


class AniListRateLimitStrategy(RateLimitStrategy):
    """AniList-specific rate limiting strategy implementation.

    Based on AniList's documented behavior:
    - 90 requests/minute standard, 30 requests/minute degraded
    - Burst limiter prevents rapid consecutive requests
    - X-RateLimit-* headers provide state information
    """

    def __init__(self, service_name: str = "anilist"):
        self.service_name = service_name

    async def handle_rate_limit_response(self, rate_info: RateLimitInfo) -> None:
        """Handle AniList 429 response with intelligent backoff."""
        # Calculate backoff based on AniList-specific behavior
        if rate_info.retry_after:
            base_delay = rate_info.retry_after
        else:
            # AniList-specific: start with 60 seconds for unknown 429s
            base_delay = 60

        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.3) * base_delay
        total_delay = base_delay + jitter

        # Log with AniList-specific context
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"AniList rate limit hit. Backing off for {total_delay:.1f}s. "
            f"Remaining: {rate_info.remaining}, Reset: {rate_info.reset_time}"
        )

        await asyncio.sleep(total_delay)

    async def calculate_backoff_delay(
        self, rate_info: RateLimitInfo, attempt: int = 0
    ) -> float:
        """Calculate AniList-specific backoff with enhanced jitter."""
        # AniList-specific: conservative backoff due to strict limits
        base_delay = min(60, 2**attempt)  # Cap at 60 seconds

        # Enhanced jitter for AniList
        jitter = random.uniform(0.2, 0.5) * base_delay

        return base_delay + jitter

    def should_proactive_throttle(self, rate_info: RateLimitInfo) -> bool:
        """AniList-specific proactive throttling logic."""
        if not rate_info.remaining or not rate_info.limit:
            return False

        # Calculate usage percentage
        usage_percent = (
            (rate_info.limit - rate_info.remaining) / rate_info.limit
        ) * 100

        # AniList-specific thresholds for proactive throttling
        return usage_percent >= 85  # Throttle when 85%+ used


class AniListRateLimitAdapter(PlatformRateLimitAdapter):
    """AniList-specific adapter for extracting rate limit information.

    Extracts AniList's specific headers:
    - X-RateLimit-Remaining: Requests left in current window
    - X-RateLimit-Limit: Total requests allowed
    - X-RateLimit-Reset: When the limit resets
    - Retry-After: Backoff time on 429 responses
    """

    def __init__(self):
        self._strategy = AniListRateLimitStrategy()

    def extract_rate_limit_info(
        self, response: aiohttp.ClientResponse
    ) -> RateLimitInfo:
        """Extract AniList rate limit headers."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        limit = response.headers.get("X-RateLimit-Limit")
        reset_time = response.headers.get("X-RateLimit-Reset")
        retry_after = response.headers.get("Retry-After")

        # Convert string values to integers where possible
        try:
            remaining = int(remaining) if remaining else None
        except (ValueError, TypeError):
            remaining = None

        try:
            limit = int(limit) if limit else None
        except (ValueError, TypeError):
            limit = None

        try:
            reset_time = int(reset_time) if reset_time else None
        except (ValueError, TypeError):
            reset_time = None

        try:
            retry_after = int(retry_after) if retry_after else None
        except (ValueError, TypeError):
            retry_after = None

        # Detect degraded mode (AniList sometimes reduces limits)
        degraded = False
        if limit and limit < 90:  # Standard AniList limit is 90/min
            degraded = True

        return RateLimitInfo(
            remaining=remaining,
            limit=limit,
            reset_time=reset_time,
            retry_after=retry_after,
            degraded=degraded,
            custom_data={
                "platform": "anilist",
                "headers": dict(response.headers),
                "status_code": response.status,
            },
        )

    def get_strategy(self) -> RateLimitStrategy:
        """Get AniList rate limiting strategy."""
        return self._strategy


class GenericRateLimitStrategy(RateLimitStrategy):
    """Generic rate limiting strategy for platforms without specific behavior.

    Provides sensible defaults for platforms that don't have documented
    rate limiting behavior or specific header formats.
    """

    def __init__(self, service_name: str):
        self.service_name = service_name

    async def handle_rate_limit_response(self, rate_info: RateLimitInfo) -> None:
        """Generic 429 handling with standard exponential backoff."""
        if rate_info.retry_after:
            delay = rate_info.retry_after
        else:
            delay = 60  # Default backoff

        # Add minimal jitter
        jitter = random.uniform(0.1, 0.2) * delay
        total_delay = delay + jitter

        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"{self.service_name} rate limit hit. Backing off for {total_delay:.1f}s."
        )

        await asyncio.sleep(total_delay)

    async def calculate_backoff_delay(
        self, rate_info: RateLimitInfo, attempt: int = 0
    ) -> float:
        """Generic exponential backoff calculation."""
        base_delay = 2**attempt  # Standard exponential backoff
        jitter = random.uniform(0.1, 0.3) * base_delay
        return base_delay + jitter

    def should_proactive_throttle(self, rate_info: RateLimitInfo) -> bool:
        """Conservative proactive throttling."""
        if not rate_info.remaining or not rate_info.limit:
            return False

        usage_percent = (
            (rate_info.limit - rate_info.remaining) / rate_info.limit
        ) * 100
        return usage_percent >= 90  # Conservative threshold


class MALRateLimitStrategy(RateLimitStrategy):
    """MAL API-specific rate limiting strategy implementation.

    Based on MAL API v2 documentation:
    - 2 requests/second, 60 requests/minute documented limits
    - OAuth2 authentication with possible token refresh needed
    - DoS protection (403) triggers aggressive backoff
    - Generally permissive in practice but follows documented limits
    """

    def __init__(self, service_name: str = "mal"):
        self.service_name = service_name

    async def handle_rate_limit_response(self, rate_info: RateLimitInfo) -> None:
        """Handle MAL 429/403 response with appropriate backoff."""
        # MAL-specific: 403 indicates DoS protection, needs longer backoff
        if rate_info.custom_data and rate_info.custom_data.get("status_code") == 403:
            base_delay = 300  # 5 minutes for DoS protection
        elif rate_info.retry_after:
            base_delay = rate_info.retry_after
        else:
            # MAL default: 2 req/sec means 30s should be safe
            base_delay = 30

        # Add jitter for MAL
        jitter = random.uniform(0.1, 0.25) * base_delay
        total_delay = base_delay + jitter

        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"MAL rate limit hit. Backing off for {total_delay:.1f}s. "
            f"Status: {rate_info.custom_data.get('status_code', 'unknown')}"
        )

        await asyncio.sleep(total_delay)

    async def calculate_backoff_delay(
        self, rate_info: RateLimitInfo, attempt: int = 0
    ) -> float:
        """Calculate MAL-specific backoff delay."""
        # MAL-specific: moderate backoff since API is generally permissive
        base_delay = min(30, 1.5**attempt)  # Gentle exponential, cap at 30s

        # MAL jitter
        jitter = random.uniform(0.15, 0.35) * base_delay

        return base_delay + jitter

    def should_proactive_throttle(self, rate_info: RateLimitInfo) -> bool:
        """MAL-specific proactive throttling logic."""
        # MAL appears very permissive, so conservative throttling
        if not rate_info.remaining or not rate_info.limit:
            return False

        usage_percent = (
            (rate_info.limit - rate_info.remaining) / rate_info.limit
        ) * 100

        # MAL-specific: only throttle at very high usage
        return usage_percent >= 95  # Very conservative since MAL is permissive


class MALRateLimitAdapter(PlatformRateLimitAdapter):
    """MAL API-specific adapter for extracting rate limit information.

    Based on testing, MAL API doesn't provide standard rate limit headers.
    This adapter focuses on status code analysis and fallback strategies.
    """

    def __init__(self):
        self._strategy = MALRateLimitStrategy()

    def extract_rate_limit_info(
        self, response: aiohttp.ClientResponse
    ) -> RateLimitInfo:
        """Extract MAL rate limit info (primarily status code based)."""
        # MAL doesn't provide standard rate limit headers
        # Focus on status codes and response analysis

        retry_after = response.headers.get("Retry-After")

        # Convert retry_after if present
        try:
            retry_after = int(retry_after) if retry_after else None
        except (ValueError, TypeError):
            retry_after = None

        # Check for potential rate limiting indicators
        degraded = False
        if response.status in [403, 429]:  # 403 = DoS protection, 429 = rate limit
            degraded = True

        return RateLimitInfo(
            remaining=None,  # MAL doesn't provide this
            limit=None,  # MAL doesn't provide this
            reset_time=None,  # MAL doesn't provide this
            retry_after=retry_after,
            degraded=degraded,
            custom_data={
                "platform": "mal",
                "headers_available": False,
                "status_code": response.status,
                "dos_protection": response.status == 403,
                "headers": dict(response.headers),
            },
        )

    def get_strategy(self) -> RateLimitStrategy:
        """Get MAL rate limiting strategy."""
        return self._strategy


class JikanRateLimitStrategy(RateLimitStrategy):
    """Jikan API-specific rate limiting strategy implementation.

    Based on Jikan API documentation and testing:
    - 3 requests/second, 60 requests/minute documented limits
    - No rate limit headers provided
    - Simple 429 responses with JSON error messages
    - Conservative backoff needed due to lack of header information
    """

    def __init__(self, service_name: str = "jikan"):
        self.service_name = service_name

    async def handle_rate_limit_response(self, rate_info: RateLimitInfo) -> None:
        """Handle Jikan 429 response with conservative backoff."""
        # Jikan provides no headers, so use conservative backoff
        if rate_info.retry_after:
            base_delay = rate_info.retry_after
        else:
            # Jikan default: 60/min limit means 60s should reset the window
            base_delay = 60

        # Add jitter for Jikan
        jitter = random.uniform(0.1, 0.2) * base_delay
        total_delay = base_delay + jitter

        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"Jikan rate limit hit. Backing off for {total_delay:.1f}s. "
            f"No headers available for precise timing."
        )

        await asyncio.sleep(total_delay)

    async def calculate_backoff_delay(
        self, rate_info: RateLimitInfo, attempt: int = 0
    ) -> float:
        """Calculate Jikan-specific backoff delay."""
        # Jikan-specific: conservative backoff due to no header info
        base_delay = min(60, 2**attempt)  # Exponential up to 60s

        # Jikan jitter
        jitter = random.uniform(0.2, 0.4) * base_delay

        return base_delay + jitter

    def should_proactive_throttle(self, rate_info: RateLimitInfo) -> bool:
        """Jikan-specific proactive throttling logic."""
        # Jikan provides no headers, so no proactive throttling possible
        return False


class JikanRateLimitAdapter(PlatformRateLimitAdapter):
    """Jikan API-specific adapter for extracting rate limit information.

    Based on testing, Jikan API provides no rate limit headers.
    This adapter handles the JSON error response structure.
    """

    def __init__(self):
        self._strategy = JikanRateLimitStrategy()

    def extract_rate_limit_info(
        self, response: aiohttp.ClientResponse
    ) -> RateLimitInfo:
        """Extract Jikan rate limit info (no headers available)."""
        # Jikan provides no rate limit headers
        retry_after = response.headers.get("Retry-After")

        # Convert retry_after if present (unlikely but check anyway)
        try:
            retry_after = int(retry_after) if retry_after else None
        except (ValueError, TypeError):
            retry_after = None

        # Jikan only indicates rate limiting via 429 status
        degraded = response.status == 429

        return RateLimitInfo(
            remaining=None,  # Jikan doesn't provide this
            limit=None,  # Jikan doesn't provide this
            reset_time=None,  # Jikan doesn't provide this
            retry_after=retry_after,
            degraded=degraded,
            custom_data={
                "platform": "jikan",
                "headers_available": False,
                "status_code": response.status,
                "response_format": "json_error",
                "headers": dict(response.headers),
            },
        )

    def get_strategy(self) -> RateLimitStrategy:
        """Get Jikan rate limiting strategy."""
        return self._strategy


class GenericRateLimitAdapter(PlatformRateLimitAdapter):
    """Generic adapter for platforms with standard rate limiting headers.

    Attempts to extract common headers like:
    - X-RateLimit-Remaining, X-Rate-Limit-Remaining
    - X-RateLimit-Limit, X-Rate-Limit-Limit
    - Retry-After
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._strategy = GenericRateLimitStrategy(service_name)

    def extract_rate_limit_info(
        self, response: aiohttp.ClientResponse
    ) -> RateLimitInfo:
        """Extract rate limit info from common headers."""
        # Try various common header formats
        remaining = (
            response.headers.get("X-RateLimit-Remaining")
            or response.headers.get("X-Rate-Limit-Remaining")
            or response.headers.get("RateLimit-Remaining")
        )

        limit = (
            response.headers.get("X-RateLimit-Limit")
            or response.headers.get("X-Rate-Limit-Limit")
            or response.headers.get("RateLimit-Limit")
        )

        reset_time = (
            response.headers.get("X-RateLimit-Reset")
            or response.headers.get("X-Rate-Limit-Reset")
            or response.headers.get("RateLimit-Reset")
        )

        retry_after = response.headers.get("Retry-After")

        # Convert to integers
        try:
            remaining = int(remaining) if remaining else None
        except (ValueError, TypeError):
            remaining = None

        try:
            limit = int(limit) if limit else None
        except (ValueError, TypeError):
            limit = None

        try:
            reset_time = int(reset_time) if reset_time else None
        except (ValueError, TypeError):
            reset_time = None

        try:
            retry_after = int(retry_after) if retry_after else None
        except (ValueError, TypeError):
            retry_after = None

        return RateLimitInfo(
            remaining=remaining,
            limit=limit,
            reset_time=reset_time,
            retry_after=retry_after,
            custom_data={
                "platform": self.service_name,
                "headers": dict(response.headers),
                "status_code": response.status,
            },
        )

    def get_strategy(self) -> RateLimitStrategy:
        """Get generic rate limiting strategy."""
        return self._strategy
