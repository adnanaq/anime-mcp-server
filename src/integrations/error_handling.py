"""Error handling infrastructure for anime MCP server.

Provides three-layer error context preservation, circuit breaker patterns,
and graceful degradation strategies for robust API integrations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import aiohttp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ErrorContext(BaseModel):
    """Three-layer error context preservation.
    
    Provides structured error information at different levels:
    - user_message: Friendly, actionable message for end users
    - debug_info: Technical context for developers
    - trace_data: Complete execution path for debugging
    """
    
    user_message: str = Field(
        description="User-friendly error message that is actionable"
    )
    debug_info: str = Field(
        description="Technical error information for developers"
    )
    trace_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete execution context and trace information"
    )
    
    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        user_message: str,
        trace_data: Optional[Dict[str, Any]] = None
    ) -> "ErrorContext":
        """Create ErrorContext from an exception.
        
        Args:
            exception: The original exception
            user_message: User-friendly message
            trace_data: Additional trace information
            
        Returns:
            ErrorContext with populated debug info
        """
        debug_info = f"{type(exception).__name__}: {str(exception)}"
        
        return cls(
            user_message=user_message,
            debug_info=debug_info,
            trace_data=trace_data or {}
        )


class CircuitBreaker:
    """Circuit breaker pattern for API failure prevention.
    
    Prevents cascading failures by monitoring API call success rates
    and temporarily blocking requests when failure threshold is exceeded.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "closed"  # closed, open, half_open
        self.last_failure_time = None
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "open"
        
    async def call_with_breaker(self, func):
        """Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        # Check if we should transition to half-open
        if self.state == "open" and self._should_attempt_reset():
            self.state = "half_open"
            logger.info("Circuit breaker transitioning to half-open")
            
        # Block calls if circuit is open
        if self.state == "open":
            raise Exception("Circuit breaker is open - blocking request")
            
        try:
            result = await func()
            
            # Success - reset failure count and close circuit
            if self.state == "half_open":
                self._reset()
                logger.info("Circuit breaker reset - returning to closed state")
                
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if not self.last_failure_time:
            return False
            
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
        
    def _record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )
            
    def _reset(self):
        """Reset circuit breaker to healthy state."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None


class GracefulDegradation:
    """Graceful degradation strategies for service failures.
    
    Provides fallback mechanisms when primary services are unavailable.
    """
    
    @staticmethod
    async def fallback_to_cache(cache_key: str, cache_manager) -> Optional[Dict[str, Any]]:
        """Attempt to retrieve cached data as fallback.
        
        Args:
            cache_key: Key to look up in cache
            cache_manager: Cache management instance
            
        Returns:
            Cached data if available, None otherwise
        """
        try:
            if hasattr(cache_manager, 'get'):
                cached_data = await cache_manager.get(cache_key)
                if cached_data:
                    logger.info(f"Using cached fallback for {cache_key}")
                    return cached_data
        except Exception as e:
            logger.warning(f"Cache fallback failed: {e}")
            
        return None
        
    @staticmethod
    async def fallback_to_offline_data(anime_id: str) -> Dict[str, Any]:
        """Fallback to offline anime database.
        
        Args:
            anime_id: ID of anime to retrieve
            
        Returns:
            Basic anime data from offline sources
        """
        # This would integrate with the existing offline database
        return {
            "anime_id": anime_id,
            "title": "Data temporarily unavailable",
            "synopsis": "Please try again later when services are restored.",
            "source": "offline_fallback",
            "degraded": True
        }
