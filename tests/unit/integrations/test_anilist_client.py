"""Tests for AniList GraphQL client."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.integrations.cache_manager import CollaborativeCacheSystem
from src.integrations.clients.anilist_client import AniListClient
from src.integrations.error_handling import CircuitBreaker, ErrorContext


class TestAniListClient:
    """Test the AniList GraphQL client."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for AniList client."""
        cache_manager = Mock(spec=CollaborativeCacheSystem)
        cache_manager.get = AsyncMock(return_value=None)

        circuit_breaker = Mock(spec=CircuitBreaker)
        circuit_breaker.is_open = Mock(return_value=False)

        return {
            "circuit_breaker": circuit_breaker,
            "rate_limiter": Mock(),
            "cache_manager": cache_manager,
            "error_handler": Mock(spec=ErrorContext),
        }

    @pytest.fixture
    def anilist_client(self, mock_dependencies):
        """Create AniList client with mocked dependencies."""
        return AniListClient(**mock_dependencies)

    @pytest.fixture
    def sample_anime_response(self):
        """Sample AniList API response."""
        return {
            "data": {
                "Media": {
                    "id": 21,
                    "title": {
                        "romaji": "One Piece",
                        "english": "One Piece",
                        "native": "ワンピース",
                    },
                    "description": "Gold Roger was known as the Pirate King...",
                    "episodes": 1000,
                    "duration": 24,
                    "status": "RELEASING",
                    "seasonYear": 1999,
                    "genres": ["Adventure", "Comedy", "Drama"],
                    "averageScore": 90,
                    "popularity": 275000,
                    "coverImage": {
                        "large": "https://s4.anilist.co/file/anilistcdn/media/anime/cover/large/bx21-YCDoj1EkAxFn.jpg"
                    },
                    "studios": {"nodes": [{"name": "Toei Animation", "isMain": True}]},
                    "characters": {
                        "edges": [
                            {
                                "node": {"name": {"full": "Monkey D. Luffy"}},
                                "role": "MAIN",
                            }
                        ]
                    },
                }
            }
        }

    @pytest.fixture
    def sample_search_response(self):
        """Sample AniList search response."""
        return {
            "data": {
                "Page": {
                    "media": [
                        {
                            "id": 21,
                            "title": {"romaji": "One Piece"},
                            "averageScore": 90,
                        },
                        {
                            "id": 11757,
                            "title": {"romaji": "Sword Art Online"},
                            "averageScore": 72,
                        },
                    ]
                }
            }
        }

    def test_client_initialization(self, mock_dependencies):
        """Test AniList client initialization."""
        client = AniListClient(**mock_dependencies)

        assert client.circuit_breaker == mock_dependencies["circuit_breaker"]
        assert client.rate_limiter == mock_dependencies["rate_limiter"]
        assert client.cache_manager == mock_dependencies["cache_manager"]
        assert client.error_handler == mock_dependencies["error_handler"]
        assert client.base_url == "https://graphql.anilist.co"
        assert client.auth_token is None

    def test_client_initialization_with_auth(self, mock_dependencies):
        """Test AniList client initialization with auth token."""
        auth_token = "test_token_123"
        client = AniListClient(auth_token=auth_token, **mock_dependencies)

        assert client.auth_token == auth_token

    @pytest.mark.asyncio
    async def test_get_anime_by_id_success(self, anilist_client, sample_anime_response):
        """Test successful anime retrieval by ID."""
        anime_id = 21

        with patch.object(
            anilist_client, "_make_graphql_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = sample_anime_response

            result = await anilist_client.get_anime_by_id(anime_id)

            assert result is not None
            assert result["id"] == 21
            assert result["title"]["romaji"] == "One Piece"
            assert result["genres"] == ["Adventure", "Comedy", "Drama"]

            # Verify GraphQL query was called with correct parameters
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "query" in call_args[1]
            assert "variables" in call_args[1]
            assert call_args[1]["variables"]["id"] == anime_id

    @pytest.mark.asyncio
    async def test_get_anime_by_id_not_found(self, anilist_client):
        """Test anime retrieval with non-existent ID."""
        anime_id = 999999

        response = {"data": {"Media": None}}

        with patch.object(
            anilist_client, "_make_graphql_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = response

            result = await anilist_client.get_anime_by_id(anime_id)

            assert result is None

    @pytest.mark.asyncio
    async def test_search_anime_by_title(self, anilist_client, sample_search_response):
        """Test anime search by title."""
        query = "One Piece"

        with patch.object(
            anilist_client, "_make_graphql_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = sample_search_response

            results = await anilist_client.search_anime(query=query, limit=10)

            assert len(results) == 2
            assert results[0]["id"] == 21
            assert results[0]["title"]["romaji"] == "One Piece"
            assert results[1]["id"] == 11757

            # Verify search parameters
            call_args = mock_request.call_args
            variables = call_args[1]["variables"]
            assert variables["search"] == query
            assert variables["perPage"] == 10

    @pytest.mark.asyncio
    async def test_search_anime_with_filters(
        self, anilist_client, sample_search_response
    ):
        """Test anime search with genre and year filters."""
        query = "adventure"
        genres = ["Adventure", "Fantasy"]
        year = 2020

        with patch.object(
            anilist_client, "_make_graphql_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = sample_search_response

            results = await anilist_client.search_anime(
                query=query, genres=genres, year=year, limit=5
            )

            assert isinstance(results, list)

            # Verify filter parameters
            call_args = mock_request.call_args
            variables = call_args[1]["variables"]
            assert variables["search"] == query
            assert variables["genre_in"] == genres
            assert variables["seasonYear"] == year
            assert variables["perPage"] == 5

    @pytest.mark.asyncio
    async def test_get_anime_characters(self, anilist_client):
        """Test retrieving anime characters."""
        anime_id = 21
        characters_response = {
            "data": {
                "Media": {
                    "characters": {
                        "edges": [
                            {
                                "node": {
                                    "id": 40,
                                    "name": {"full": "Monkey D. Luffy"},
                                    "image": {"large": "https://example.com/luffy.jpg"},
                                },
                                "role": "MAIN",
                            },
                            {
                                "node": {
                                    "id": 41,
                                    "name": {"full": "Roronoa Zoro"},
                                    "image": {"large": "https://example.com/zoro.jpg"},
                                },
                                "role": "MAIN",
                            },
                        ]
                    }
                }
            }
        }

        with patch.object(
            anilist_client, "_make_graphql_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = characters_response

            characters = await anilist_client.get_anime_characters(anime_id)

            assert len(characters) == 2
            assert characters[0]["name"]["full"] == "Monkey D. Luffy"
            assert characters[0]["role"] == "MAIN"
            assert characters[1]["name"]["full"] == "Roronoa Zoro"

    @pytest.mark.asyncio
    async def test_get_anime_staff(self, anilist_client):
        """Test retrieving anime staff."""
        anime_id = 21
        staff_response = {
            "data": {
                "Media": {
                    "staff": {
                        "edges": [
                            {
                                "node": {"id": 95269, "name": {"full": "Eiichiro Oda"}},
                                "role": "Original Creator",
                            }
                        ]
                    }
                }
            }
        }

        with patch.object(
            anilist_client, "_make_graphql_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = staff_response

            staff = await anilist_client.get_anime_staff(anime_id)

            assert len(staff) == 1
            assert staff[0]["name"]["full"] == "Eiichiro Oda"
            assert staff[0]["role"] == "Original Creator"

    @pytest.mark.asyncio
    async def test_graphql_request_with_auth(self, mock_dependencies):
        """Test GraphQL request with authentication."""
        auth_token = "test_token_123"
        client = AniListClient(auth_token=auth_token, **mock_dependencies)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = Mock()
            mock_response.json = AsyncMock(return_value={"data": {"test": "success"}})
            mock_response.status = 200
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            query = "query { test }"
            variables = {"id": 1}

            result = await client._make_graphql_request(
                query=query, variables=variables
            )

            # Verify authorization header was included
            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == f"Bearer {auth_token}"
            assert result["data"]["test"] == "success"

    @pytest.mark.asyncio
    async def test_graphql_request_without_auth(self, anilist_client):
        """Test GraphQL request without authentication."""
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = Mock()
            mock_response.json = AsyncMock(return_value={"data": {"test": "success"}})
            mock_response.status = 200
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            query = "query { test }"

            result = await anilist_client._make_graphql_request(query=query)

            # Verify no authorization header
            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            assert "Authorization" not in headers
            assert result["data"]["test"] == "success"

    @pytest.mark.asyncio
    async def test_graphql_error_handling(self, anilist_client):
        """Test GraphQL error response handling."""
        error_response = {
            "errors": [
                {"message": "Invalid query", "locations": [{"line": 1, "column": 1}]}
            ]
        }

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = Mock()
            mock_response.json = AsyncMock(return_value=error_response)
            mock_response.status = 200
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            query = "invalid query"

            with pytest.raises(Exception) as exc_info:
                await anilist_client._make_graphql_request(query=query)

            assert "Invalid query" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, anilist_client):
        """Test rate limit handling."""
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = Mock()
            mock_response.status = 429
            mock_response.headers = {"Retry-After": "60"}
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)

            query = "query { test }"

            with pytest.raises(Exception) as exc_info:
                await anilist_client._make_graphql_request(query=query)

            assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, anilist_client):
        """Test circuit breaker integration."""
        # Mock circuit breaker to simulate open state
        anilist_client.circuit_breaker.is_open = Mock(return_value=True)

        with pytest.raises(Exception) as exc_info:
            await anilist_client.get_anime_by_id(21)

        assert "circuit breaker" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cache_integration(self, anilist_client, sample_anime_response):
        """Test cache integration."""
        anime_id = 21
        cache_key = f"anilist_anime_{anime_id}"

        # Mock cache hit
        anilist_client.cache_manager.get = AsyncMock(
            return_value=sample_anime_response["data"]["Media"]
        )

        result = await anilist_client.get_anime_by_id(anime_id)

        assert result["id"] == 21
        anilist_client.cache_manager.get.assert_called_once_with(cache_key)
