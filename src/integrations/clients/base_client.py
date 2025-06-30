"""Base client for API integrations with error handling foundation."""

import asyncio
import aiohttp
from typing import Any, Dict, Optional, Callable


class BaseClient:
    """Base client that integrates all error handling components."""
    
    def __init__(
        self,
        circuit_breaker=None,
        rate_limiter=None,
        cache_manager=None,
        error_handler=None
    ):
        """Initialize BaseClient with error handling dependencies.
        
        Args:
            circuit_breaker: Circuit breaker instance
            rate_limiter: Rate limiter instance
            cache_manager: Cache manager instance
            error_handler: Error handler instance
        """
        self.circuit_breaker = circuit_breaker
        self.rate_limiter = rate_limiter
        self.cache_manager = cache_manager
        self.error_handler = error_handler
        
    async def make_request(self, url: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling.
        
        Args:
            url: URL to request
            method: HTTP method
            **kwargs: Additional request parameters
            
        Returns:
            Response data
        """
        async def _request():
            # Apply rate limiting if available
            if self.rate_limiter:
                # Use rate limiter as context manager
                async with self.rate_limiter:
                    # Make actual HTTP request
                    async with aiohttp.ClientSession() as session:
                        async with session.request(method, url, **kwargs) as response:
                            if response.status == 429:  # Rate limited
                                raise Exception(f"Rate limited: {response.status}")
                            response.raise_for_status()
                            return await response.json()
            else:
                # Make actual HTTP request without rate limiting
                async with aiohttp.ClientSession() as session:
                    async with session.request(method, url, **kwargs) as response:
                        if response.status == 429:  # Rate limited
                            raise Exception(f"Rate limited: {response.status}")
                        response.raise_for_status()
                        return await response.json()
        
        # Use circuit breaker if available
        if self.circuit_breaker:
            return await self.circuit_breaker.call_with_breaker(_request)
        else:
            return await _request()
            
    async def with_circuit_breaker(self, func: Callable) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            
        Returns:
            Function result
        """
        if self.circuit_breaker:
            return await self.circuit_breaker.call_with_breaker(func)
        else:
            return await func()
            
    async def handle_rate_limit(self, response) -> None:
        """Handle rate limit response.
        
        Args:
            response: HTTP response with rate limit info
        """
        # Basic rate limit handling - can be extended
        if hasattr(response, 'headers') and 'Retry-After' in response.headers:
            retry_after = response.headers.get('Retry-After')
            # Could implement actual retry logic here
            pass
