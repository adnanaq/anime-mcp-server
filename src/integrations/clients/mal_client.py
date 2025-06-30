"""MAL/Jikan REST client implementation."""

import aiohttp
import asyncio
from typing import Any, Dict, List, Optional
from .base_client import BaseClient


class MALClient(BaseClient):
    """MAL/Jikan REST API client with dual API strategy."""
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        circuit_breaker=None,
        rate_limiter=None,
        cache_manager=None,
        error_handler=None
    ):
        """Initialize MAL client.
        
        Args:
            client_id: MAL OAuth2 client ID
            client_secret: MAL OAuth2 client secret
            circuit_breaker: Circuit breaker instance
            rate_limiter: Rate limiter instance
            cache_manager: Cache manager instance
            error_handler: Error handler instance
        """
        super().__init__(circuit_breaker, rate_limiter, cache_manager, error_handler)
        self.mal_base_url = "https://api.myanimelist.net/v2"
        self.jikan_base_url = "https://api.jikan.moe/v4"
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.refresh_token = None
    
    async def _make_mal_request(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make request to official MAL API."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Add X-MAL-CLIENT-ID header if client_id is available
        if self.client_id:
            headers["X-MAL-CLIENT-ID"] = self.client_id
        
        # Add Authorization header if access_token is available
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        elif not self.client_id:
            raise Exception("MAL API requires either client_id or access_token")
        
        url = f"{self.mal_base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After", "60")
                    raise Exception(f"MAL rate limit exceeded. Retry after {retry_after} seconds")
                
                if response.status == 401:
                    raise Exception("MAL API unauthorized - invalid or expired token")
                
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"MAL API error: {error_data.get('error', 'Unknown error')}")
                
                return await response.json()
    
    async def _make_jikan_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Make request to Jikan API."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        url = f"{self.jikan_base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After", "60") 
                    raise Exception(f"Jikan rate limit exceeded. Retry after {retry_after} seconds")
                
                if response.status == 404:
                    error_data = await response.json()
                    raise Exception(f"Jikan API error: {error_data.get('message', 'Resource does not exist')}")
                
                if response.status != 200:
                    try:
                        error_data = await response.json()
                        raise Exception(f"Jikan API error: {error_data.get('message', 'Unknown error')}")
                    except:
                        raise Exception(f"Jikan API HTTP {response.status} error")
                
                return await response.json()
    
    async def _make_jikan_request_with_retry(self, endpoint: str, max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """Make Jikan request with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return await self._make_jikan_request(endpoint, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                if "500" in str(e) or "502" in str(e) or "503" in str(e):
                    # Exponential backoff for server errors
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    # Don't retry for other errors
                    raise e
    
    async def get_anime_by_id(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get anime by ID using dual API strategy."""
        # Check cache first
        cache_key = f"mal_anime_{anime_id}"
        if self.cache_manager:
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    return cached_result
            except:
                # Cache miss or error, continue to API call
                pass
        
        # Check circuit breaker before making API call
        if self.circuit_breaker and self.circuit_breaker.is_open():
            raise Exception("Circuit breaker is open")
        
        # Dual API strategy: Try official MAL API first if auth available, otherwise use Jikan
        if self.client_id:
            try:
                endpoint = f"/anime/{anime_id}"
                result = await self._make_mal_request(endpoint)
                return result
            except Exception:
                # MAL API failed, fallback to Jikan
                pass
        
        # Fallback to Jikan API
        try:
            endpoint = f"/anime/{anime_id}"
            response = await self._make_jikan_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception:
            pass
        
        return None
    
    async def search_anime(
        self,
        query: Optional[str] = None,
        genres: Optional[List[int]] = None,
        status: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search anime using Jikan API."""
        # Jikan search endpoint
        endpoint = "/anime"
        params = {}
        
        if query:
            params["q"] = query
        if genres:
            params["genres"] = ",".join(map(str, genres))
        if status:
            params["status"] = status
        if limit:
            params["limit"] = limit
            
        try:
            response = await self._make_jikan_request(endpoint, params=params)
            if response and "data" in response:
                return response["data"]
        except Exception:
            pass
        
        return []
    
    async def get_seasonal_anime(self, year: int, season: str) -> List[Dict[str, Any]]:
        """Get seasonal anime using Jikan API."""
        endpoint = f"/seasons/{year}/{season}"
        
        try:
            response = await self._make_jikan_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception:
            pass
        
        return []
    
    async def get_anime_statistics(self, anime_id: int) -> Dict[str, Any]:
        """Get anime statistics using Jikan API."""
        endpoint = f"/anime/{anime_id}/statistics"
        
        try:
            response = await self._make_jikan_request(endpoint)
            if response and "data" in response:
                return response["data"]
        except Exception:
            pass
        
        return {}
    
    async def refresh_access_token(self) -> None:
        """Refresh OAuth2 access token for MAL API."""
        if not self.client_id or not self.client_secret or not self.refresh_token:
            raise Exception("OAuth2 credentials required for token refresh")
        
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post("https://myanimelist.net/v1/oauth2/token", data=data) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"Token refresh failed: {error_data.get('error', 'Unknown error')}")
                
                token_data = await response.json()
                self.access_token = token_data["access_token"]
                self.refresh_token = token_data["refresh_token"]