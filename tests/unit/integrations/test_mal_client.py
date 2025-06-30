"""Tests for MAL/Jikan REST client."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional
import json

from src.integrations.clients.mal_client import MALClient
from src.integrations.error_handling import ErrorContext, CircuitBreaker
from src.integrations.cache_manager import CollaborativeCacheSystem


class TestMALClient:
    """Test the MAL/Jikan REST client."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for MAL client."""
        cache_manager = Mock(spec=CollaborativeCacheSystem)
        cache_manager.get = AsyncMock(return_value=None)
        
        circuit_breaker = Mock(spec=CircuitBreaker)
        circuit_breaker.is_open = Mock(return_value=False)
        
        return {
            "circuit_breaker": circuit_breaker,
            "rate_limiter": Mock(),
            "cache_manager": cache_manager,
            "error_handler": Mock(spec=ErrorContext)
        }
    
    @pytest.fixture
    def mal_client(self, mock_dependencies):
        """Create MAL client with mocked dependencies."""
        return MALClient(**mock_dependencies)
    
    @pytest.fixture
    def mal_client_with_auth(self, mock_dependencies):
        """Create MAL client with OAuth2 credentials."""
        return MALClient(
            client_id="195a1bf1e8043ed0507576d020e9e17d",
            client_secret="1b6bf289306de009ebb73533811500c88d75c527800cf5048f535c6966af53b0",
            **mock_dependencies
        )
    
    @pytest.fixture
    def sample_mal_anime_response(self):
        """Sample MAL API anime response."""
        return {
            "id": 21,
            "title": "One Piece",
            "main_picture": {
                "medium": "https://cdn.myanimelist.net/images/anime/6/73245.jpg",
                "large": "https://cdn.myanimelist.net/images/anime/6/73245l.jpg"
            },
            "alternative_titles": {
                "synonyms": ["OP"],
                "en": "One Piece",
                "ja": "ワンピース"
            },
            "start_date": "1999-10-20",
            "end_date": None,
            "synopsis": "Gold Roger was known as the Pirate King...",
            "mean": 9.0,
            "rank": 5,
            "popularity": 11,
            "num_list_users": 1300000,
            "num_scoring_users": 850000,
            "nsfw": "white",
            "media_type": "tv",
            "status": "currently_airing",
            "genres": [
                {"id": 1, "name": "Action"},
                {"id": 2, "name": "Adventure"}
            ],
            "num_episodes": 0,
            "start_season": {
                "year": 1999,
                "season": "fall"
            },
            "studios": [
                {"id": 18, "name": "Toei Animation"}
            ]
        }
    
    @pytest.fixture
    def sample_jikan_anime_response(self):
        """Sample Jikan API anime response."""
        return {
            "data": {
                "mal_id": 21,
                "url": "https://myanimelist.net/anime/21/One_Piece",
                "images": {
                    "jpg": {
                        "image_url": "https://cdn.myanimelist.net/images/anime/6/73245.jpg",
                        "small_image_url": "https://cdn.myanimelist.net/images/anime/6/73245t.jpg",
                        "large_image_url": "https://cdn.myanimelist.net/images/anime/6/73245l.jpg"
                    }
                },
                "title": "One Piece",
                "title_english": "One Piece",
                "title_japanese": "ワンピース",
                "type": "TV",
                "source": "Manga",
                "episodes": None,
                "status": "Currently Airing",
                "airing": True,
                "aired": {
                    "from": "1999-10-20T00:00:00+00:00",
                    "to": None
                },
                "duration": "24 min",
                "rating": "PG-13 - Teens 13 or older",
                "score": 9.0,
                "scored_by": 850000,
                "rank": 5,
                "popularity": 11,
                "synopsis": "Gold Roger was known as the Pirate King...",
                "genres": [
                    {"mal_id": 1, "type": "anime", "name": "Action", "url": "https://myanimelist.net/anime/genre/1/Action"},
                    {"mal_id": 2, "type": "anime", "name": "Adventure", "url": "https://myanimelist.net/anime/genre/2/Adventure"}
                ],
                "studios": [
                    {"mal_id": 18, "type": "anime", "name": "Toei Animation", "url": "https://myanimelist.net/anime/producer/18/Toei_Animation"}
                ]
            }
        }
    
    @pytest.fixture
    def sample_jikan_search_response(self):
        """Sample Jikan search response."""
        return {
            "data": [
                {
                    "mal_id": 21,
                    "title": "One Piece",
                    "score": 9.0,
                    "episodes": None,
                    "status": "Currently Airing"
                },
                {
                    "mal_id": 11757,
                    "title": "Sword Art Online",
                    "score": 7.2,
                    "episodes": 25,
                    "status": "Finished Airing"
                }
            ],
            "pagination": {
                "last_visible_page": 100,
                "has_next_page": True,
                "current_page": 1,
                "items": {
                    "count": 25,
                    "total": 2500,
                    "per_page": 25
                }
            }
        }

    def test_client_initialization_without_auth(self, mock_dependencies):
        """Test MAL client initialization without authentication."""
        client = MALClient(**mock_dependencies)
        
        assert client.circuit_breaker == mock_dependencies["circuit_breaker"]
        assert client.rate_limiter == mock_dependencies["rate_limiter"]
        assert client.cache_manager == mock_dependencies["cache_manager"]
        assert client.error_handler == mock_dependencies["error_handler"]
        assert client.mal_base_url == "https://api.myanimelist.net/v2"
        assert client.jikan_base_url == "https://api.jikan.moe/v4"
        assert client.client_id is None
        assert client.client_secret is None
        assert client.access_token is None

    def test_client_initialization_with_auth(self, mock_dependencies):
        """Test MAL client initialization with OAuth2 credentials."""
        client_id = "195a1bf1e8043ed0507576d020e9e17d"
        client_secret = "1b6bf289306de009ebb73533811500c88d75c527800cf5048f535c6966af53b0"
        
        client = MALClient(
            client_id=client_id,
            client_secret=client_secret,
            **mock_dependencies
        )
        
        assert client.client_id == client_id
        assert client.client_secret == client_secret
        assert client.access_token is None

    @pytest.mark.asyncio
    async def test_get_anime_by_id_official_mal_success(self, mal_client_with_auth, sample_mal_anime_response):
        """Test successful anime retrieval using official MAL API."""
        anime_id = 21
        
        with patch.object(mal_client_with_auth, '_make_mal_request', new_callable=AsyncMock) as mock_mal_request:
            mock_mal_request.return_value = sample_mal_anime_response
            
            result = await mal_client_with_auth.get_anime_by_id(anime_id)
            
            assert result is not None
            assert result["id"] == 21
            assert result["title"] == "One Piece"
            assert result["mean"] == 9.0
            assert len(result["genres"]) == 2
            
            # Verify MAL API was called with correct parameters
            mock_mal_request.assert_called_once()
            call_args = mock_mal_request.call_args
            assert f"/anime/{anime_id}" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_anime_by_id_jikan_fallback(self, mal_client, sample_jikan_anime_response):
        """Test anime retrieval falling back to Jikan API."""
        anime_id = 21
        
        with patch.object(mal_client, '_make_jikan_request', new_callable=AsyncMock) as mock_jikan_request:
            mock_jikan_request.return_value = sample_jikan_anime_response
            
            result = await mal_client.get_anime_by_id(anime_id)
            
            assert result is not None
            assert result["mal_id"] == 21
            assert result["title"] == "One Piece"
            assert result["score"] == 9.0
            
            # Verify Jikan API was called
            mock_jikan_request.assert_called_once()
            call_args = mock_jikan_request.call_args
            assert f"/anime/{anime_id}" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_anime_by_id_mal_failure_jikan_fallback(self, mal_client_with_auth, sample_jikan_anime_response):
        """Test MAL API failure with successful Jikan fallback."""
        anime_id = 21
        
        with patch.object(mal_client_with_auth, '_make_mal_request', new_callable=AsyncMock) as mock_mal_request:
            with patch.object(mal_client_with_auth, '_make_jikan_request', new_callable=AsyncMock) as mock_jikan_request:
                # MAL API fails
                mock_mal_request.side_effect = Exception("MAL API rate limit exceeded")
                # Jikan API succeeds
                mock_jikan_request.return_value = sample_jikan_anime_response
                
                result = await mal_client_with_auth.get_anime_by_id(anime_id)
                
                assert result is not None
                assert result["mal_id"] == 21
                
                # Verify both APIs were called
                mock_mal_request.assert_called_once()
                mock_jikan_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_anime_jikan(self, mal_client, sample_jikan_search_response):
        """Test anime search using Jikan API."""
        query = "One Piece"
        
        with patch.object(mal_client, '_make_jikan_request', new_callable=AsyncMock) as mock_jikan_request:
            mock_jikan_request.return_value = sample_jikan_search_response
            
            results = await mal_client.search_anime(query=query, limit=10)
            
            assert len(results) == 2
            assert results[0]["mal_id"] == 21
            assert results[0]["title"] == "One Piece"
            assert results[1]["mal_id"] == 11757
            
            # Verify search parameters
            call_args = mock_jikan_request.call_args
            assert "/anime" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_search_anime_with_filters(self, mal_client, sample_jikan_search_response):
        """Test anime search with genre and status filters."""
        query = "action"
        
        with patch.object(mal_client, '_make_jikan_request', new_callable=AsyncMock) as mock_jikan_request:
            mock_jikan_request.return_value = sample_jikan_search_response
            
            results = await mal_client.search_anime(
                query=query,
                genres=[1, 2],  # Action, Adventure
                status="airing",
                limit=5
            )
            
            assert isinstance(results, list)
            
            # Verify filter parameters were included
            call_args = mock_jikan_request.call_args
            assert "/anime" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_seasonal_anime(self, mal_client):
        """Test retrieving seasonal anime."""
        year = 2023
        season = "fall"
        seasonal_response = {
            "data": [
                {
                    "mal_id": 12345,
                    "title": "Fall 2023 Anime",
                    "score": 8.5,
                    "episodes": 12,
                    "status": "Currently Airing"
                }
            ]
        }
        
        with patch.object(mal_client, '_make_jikan_request', new_callable=AsyncMock) as mock_jikan_request:
            mock_jikan_request.return_value = seasonal_response
            
            results = await mal_client.get_seasonal_anime(year, season)
            
            assert len(results) == 1
            assert results[0]["mal_id"] == 12345
            assert results[0]["title"] == "Fall 2023 Anime"
            
            # Verify seasonal endpoint was called
            call_args = mock_jikan_request.call_args
            assert f"/seasons/{year}/{season}" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_anime_statistics(self, mal_client):
        """Test retrieving anime statistics."""
        anime_id = 21
        stats_response = {
            "data": {
                "watching": 500000,
                "completed": 300000,
                "on_hold": 50000,
                "dropped": 25000,
                "plan_to_watch": 200000,
                "total": 1075000,
                "scores": {
                    "1": {"votes": 1000, "percentage": 0.1},
                    "10": {"votes": 150000, "percentage": 15.0}
                }
            }
        }
        
        with patch.object(mal_client, '_make_jikan_request', new_callable=AsyncMock) as mock_jikan_request:
            mock_jikan_request.return_value = stats_response
            
            stats = await mal_client.get_anime_statistics(anime_id)
            
            assert stats["watching"] == 500000
            assert stats["completed"] == 300000
            assert stats["total"] == 1075000
            
            # Verify statistics endpoint was called
            call_args = mock_jikan_request.call_args
            assert f"/anime/{anime_id}/statistics" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_mal_request_with_auth(self, mal_client_with_auth):
        """Test MAL API request with authentication."""
        endpoint = "/anime/21"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.json = AsyncMock(return_value={"id": 21, "title": "One Piece"})
            mock_response.status = 200
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock access token
            mal_client_with_auth.access_token = "test_access_token"
            
            result = await mal_client_with_auth._make_mal_request(endpoint)
            
            # Verify authorization header was included
            call_args = mock_get.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test_access_token"
            assert result["id"] == 21

    @pytest.mark.asyncio
    async def test_mal_request_without_auth_fails(self, mal_client):
        """Test MAL API request without authentication fails."""
        endpoint = "/anime/21"
        
        with pytest.raises(Exception) as exc_info:
            await mal_client._make_mal_request(endpoint)
        
        assert "client_id or access_token" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_jikan_request_success(self, mal_client):
        """Test successful Jikan API request."""
        endpoint = "/anime/21"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.json = AsyncMock(return_value={"data": {"mal_id": 21, "title": "One Piece"}})
            mock_response.status = 200
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await mal_client._make_jikan_request(endpoint)
            
            # Verify no authentication header needed for Jikan
            call_args = mock_get.call_args
            headers = call_args[1].get("headers", {})
            assert "Authorization" not in headers
            assert result["data"]["mal_id"] == 21

    @pytest.mark.asyncio
    async def test_rate_limit_handling_mal(self, mal_client_with_auth):
        """Test MAL API rate limit handling."""
        endpoint = "/anime/21"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 429
            mock_response.headers = {"Retry-After": "60"}
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            mal_client_with_auth.access_token = "test_token"
            
            with pytest.raises(Exception) as exc_info:
                await mal_client_with_auth._make_mal_request(endpoint)
            
            assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rate_limit_handling_jikan(self, mal_client):
        """Test Jikan API rate limit handling."""
        endpoint = "/anime/21"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 429
            mock_response.headers = {"Retry-After": "60"}
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with pytest.raises(Exception) as exc_info:
                await mal_client._make_jikan_request(endpoint)
            
            assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, mal_client):
        """Test circuit breaker integration."""
        # Mock circuit breaker to simulate open state
        mal_client.circuit_breaker.is_open = Mock(return_value=True)
        
        with pytest.raises(Exception) as exc_info:
            await mal_client.get_anime_by_id(21)
        
        assert "circuit breaker" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cache_integration(self, mal_client, sample_jikan_anime_response):
        """Test cache integration."""
        anime_id = 21
        cache_key = f"mal_anime_{anime_id}"
        
        # Mock cache hit
        mal_client.cache_manager.get = AsyncMock(return_value=sample_jikan_anime_response["data"])
        
        result = await mal_client.get_anime_by_id(anime_id)
        
        assert result["mal_id"] == 21
        mal_client.cache_manager.get.assert_called_once_with(cache_key)

    @pytest.mark.asyncio
    async def test_oauth2_token_refresh(self, mal_client_with_auth):
        """Test OAuth2 token refresh functionality."""
        refresh_token = "test_refresh_token"
        mal_client_with_auth.refresh_token = refresh_token
        
        token_response = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "token_type": "Bearer",
            "expires_in": 3600
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.json = AsyncMock(return_value=token_response)
            mock_response.status = 200
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)
            
            await mal_client_with_auth.refresh_access_token()
            
            assert mal_client_with_auth.access_token == "new_access_token"
            assert mal_client_with_auth.refresh_token == "new_refresh_token"

    @pytest.mark.asyncio
    async def test_mal_error_handling(self, mal_client_with_auth):
        """Test MAL API specific error handling."""
        endpoint = "/anime/21"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 401
            mock_response.json = AsyncMock(return_value={"error": "unauthorized"})
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            mal_client_with_auth.access_token = "invalid_token"
            
            with pytest.raises(Exception) as exc_info:
                await mal_client_with_auth._make_mal_request(endpoint)
            
            assert "unauthorized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_jikan_error_handling(self, mal_client):
        """Test Jikan API specific error handling."""
        endpoint = "/anime/999999"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 404
            mock_response.json = AsyncMock(return_value={
                "status": 404,
                "type": "BadResponseException",
                "message": "Resource does not exist",
                "error": None
            })
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with pytest.raises(Exception) as exc_info:
                await mal_client._make_jikan_request(endpoint)
            
            assert "resource does not exist" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_exponential_backoff_retry(self, mal_client):
        """Test exponential backoff retry logic."""
        endpoint = "/anime/21"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # First call fails with 500
            mock_response_fail = Mock()
            mock_response_fail.status = 500
            mock_response_fail.json = AsyncMock(return_value={"error": "internal server error"})
            
            # Second call succeeds
            mock_response_success = Mock()
            mock_response_success.status = 200
            mock_response_success.json = AsyncMock(return_value={"data": {"mal_id": 21}})
            
            mock_get.return_value.__aenter__ = AsyncMock(side_effect=[mock_response_fail, mock_response_success])
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with patch('asyncio.sleep') as mock_sleep:
                result = await mal_client._make_jikan_request_with_retry(endpoint, max_retries=2)
                
                assert result["data"]["mal_id"] == 21
                assert mock_get.call_count == 2
                mock_sleep.assert_called_once()  # Exponential backoff sleep