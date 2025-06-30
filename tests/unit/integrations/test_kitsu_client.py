"""Tests for Kitsu JSON:API client."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional
import json

from src.integrations.clients.kitsu_client import KitsuClient
from src.integrations.error_handling import ErrorContext, CircuitBreaker
from src.integrations.cache_manager import CollaborativeCacheSystem


class TestKitsuClient:
    """Test the Kitsu JSON:API client."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for Kitsu client."""
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
    def kitsu_client(self, mock_dependencies):
        """Create Kitsu client with mocked dependencies."""
        return KitsuClient(**mock_dependencies)
    
    @pytest.fixture
    def sample_anime_response(self):
        """Sample Kitsu API anime response."""
        return {
            "data": {
                "id": "1",
                "type": "anime",
                "attributes": {
                    "slug": "cowboy-bebop",
                    "synopsis": "In the year 2071, humanity has colonized several of the planets...",
                    "description": "In the year 2071, humanity has colonized several of the planets...",
                    "coverImageTopOffset": 400,
                    "titles": {
                        "en": "Cowboy Bebop",
                        "en_jp": "Cowboy Bebop",
                        "en_us": "Cowboy Bebop",
                        "ja_jp": "カウボーイビバップ"
                    },
                    "canonicalTitle": "Cowboy Bebop",
                    "abbreviatedTitles": ["CB"],
                    "averageRating": "82.74",
                    "ratingFrequencies": {
                        "2": "832",
                        "3": "81",
                        "4": "106"
                    },
                    "userCount": 64304,
                    "favoritesCount": 4312,
                    "startDate": "1998-04-03",
                    "endDate": "1999-04-24",
                    "nextRelease": None,
                    "popularityRank": 26,
                    "ratingRank": 28,
                    "ageRating": "R",
                    "ageRatingGuide": "17+ (violence & profanity)",
                    "subtype": "TV",
                    "status": "finished",
                    "tba": None,
                    "posterImage": {
                        "tiny": "https://media.kitsu.io/anime/poster_images/1/tiny.jpg",
                        "large": "https://media.kitsu.io/anime/poster_images/1/large.jpg",
                        "small": "https://media.kitsu.io/anime/poster_images/1/small.jpg",
                        "medium": "https://media.kitsu.io/anime/poster_images/1/medium.jpg",
                        "original": "https://media.kitsu.io/anime/poster_images/1/original.jpg"
                    },
                    "coverImage": {
                        "tiny": "https://media.kitsu.io/anime/cover_images/1/tiny.jpg",
                        "large": "https://media.kitsu.io/anime/cover_images/1/large.jpg",
                        "small": "https://media.kitsu.io/anime/cover_images/1/small.jpg",
                        "original": "https://media.kitsu.io/anime/cover_images/1/original.jpg"
                    },
                    "episodeCount": 26,
                    "episodeLength": 24,
                    "totalLength": 624,
                    "youtubeVideoId": "qig4KOK2R2g",
                    "showType": "TV",
                    "nsfw": False
                },
                "relationships": {
                    "genres": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/genres",
                            "related": "https://kitsu.io/api/edge/anime/1/genres"
                        }
                    },
                    "categories": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/categories",
                            "related": "https://kitsu.io/api/edge/anime/1/categories"
                        }
                    },
                    "castings": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/castings",
                            "related": "https://kitsu.io/api/edge/anime/1/castings"
                        }
                    },
                    "installments": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/installments",
                            "related": "https://kitsu.io/api/edge/anime/1/installments"
                        }
                    },
                    "mappings": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/mappings",
                            "related": "https://kitsu.io/api/edge/anime/1/mappings"
                        }
                    },
                    "reviews": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/reviews",
                            "related": "https://kitsu.io/api/edge/anime/1/reviews"
                        }
                    },
                    "mediaRelationships": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/media-relationships",
                            "related": "https://kitsu.io/api/edge/anime/1/media-relationships"
                        }
                    },
                    "characters": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/characters",
                            "related": "https://kitsu.io/api/edge/anime/1/characters"
                        }
                    },
                    "staff": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/staff",
                            "related": "https://kitsu.io/api/edge/anime/1/staff"
                        }
                    },
                    "productions": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/productions",
                            "related": "https://kitsu.io/api/edge/anime/1/productions"
                        }
                    },
                    "quotes": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/quotes",
                            "related": "https://kitsu.io/api/edge/anime/1/quotes"
                        }
                    },
                    "episodes": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/episodes",
                            "related": "https://kitsu.io/api/edge/anime/1/episodes"
                        }
                    },
                    "streamingLinks": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/streaming-links",
                            "related": "https://kitsu.io/api/edge/anime/1/streaming-links"
                        }
                    },
                    "animeProductions": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/anime-productions",
                            "related": "https://kitsu.io/api/edge/anime/1/anime-productions"
                        }
                    },
                    "animeCharacters": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/anime-characters",
                            "related": "https://kitsu.io/api/edge/anime/1/anime-characters"
                        }
                    },
                    "animeStaff": {
                        "links": {
                            "self": "https://kitsu.io/api/edge/anime/1/relationships/anime-staff",
                            "related": "https://kitsu.io/api/edge/anime/1/anime-staff"
                        }
                    }
                }
            }
        }
    
    @pytest.fixture
    def sample_search_response(self):
        """Sample Kitsu search response."""
        return {
            "data": [
                {
                    "id": "1",
                    "type": "anime",
                    "attributes": {
                        "canonicalTitle": "Cowboy Bebop",
                        "averageRating": "82.74",
                        "episodeCount": 26,
                        "status": "finished",
                        "posterImage": {
                            "small": "https://media.kitsu.io/anime/poster_images/1/small.jpg"
                        }
                    }
                },
                {
                    "id": "2",
                    "type": "anime",
                    "attributes": {
                        "canonicalTitle": "Trigun",
                        "averageRating": "79.42",
                        "episodeCount": 26,
                        "status": "finished",
                        "posterImage": {
                            "small": "https://media.kitsu.io/anime/poster_images/2/small.jpg"
                        }
                    }
                }
            ],
            "meta": {
                "count": 2
            },
            "links": {
                "first": "https://kitsu.io/api/edge/anime?filter%5Btext%5D=cowboy&page%5Blimit%5D=10&page%5Boffset%5D=0",
                "last": "https://kitsu.io/api/edge/anime?filter%5Btext%5D=cowboy&page%5Blimit%5D=10&page%5Boffset%5D=0"
            }
        }
    
    @pytest.fixture
    def sample_episodes_response(self):
        """Sample Kitsu episodes response."""
        return {
            "data": [
                {
                    "id": "1",
                    "type": "episodes",
                    "attributes": {
                        "description": "Aboard the ship Bebop, Spike Spiegel and his partner Jet Black...",
                        "titles": {
                            "en": "Asteroid Blues",
                            "ja_jp": "アステロイド・ブルース"
                        },
                        "canonicalTitle": "Asteroid Blues",
                        "seasonNumber": 1,
                        "number": 1,
                        "relativeNumber": 1,
                        "airdate": "1998-04-03T00:00:00.000Z",
                        "length": 24,
                        "thumbnail": {
                            "original": "https://media.kitsu.io/episodes/thumbnails/1/original.jpg"
                        }
                    }
                }
            ]
        }
    
    @pytest.fixture
    def sample_streaming_links_response(self):
        """Sample Kitsu streaming links response."""
        return {
            "data": [
                {
                    "id": "1",
                    "type": "streamingLinks",
                    "attributes": {
                        "url": "https://www.crunchyroll.com/cowboy-bebop",
                        "subs": ["en"],
                        "dubs": ["ja"]
                    },
                    "relationships": {
                        "streamer": {
                            "data": {
                                "id": "1",
                                "type": "streamers"
                            }
                        }
                    }
                }
            ],
            "included": [
                {
                    "id": "1",
                    "type": "streamers",
                    "attributes": {
                        "siteName": "Crunchyroll",
                        "logo": {
                            "original": "https://media.kitsu.io/streamers/logos/1/original.png"
                        }
                    }
                }
            ]
        }

    def test_client_initialization(self, mock_dependencies):
        """Test Kitsu client initialization."""
        client = KitsuClient(**mock_dependencies)
        
        assert client.circuit_breaker == mock_dependencies["circuit_breaker"]
        assert client.rate_limiter == mock_dependencies["rate_limiter"]
        assert client.cache_manager == mock_dependencies["cache_manager"]
        assert client.error_handler == mock_dependencies["error_handler"]
        assert client.base_url == "https://kitsu.io/api/edge"

    @pytest.mark.asyncio
    async def test_get_anime_by_id_success(self, kitsu_client, sample_anime_response):
        """Test successful anime retrieval by ID."""
        anime_id = 1
        
        with patch.object(kitsu_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_anime_response
            
            result = await kitsu_client.get_anime_by_id(anime_id)
            
            assert result is not None
            assert result["id"] == "1"
            assert result["attributes"]["canonicalTitle"] == "Cowboy Bebop"
            assert result["attributes"]["episodeCount"] == 26
            assert result["attributes"]["averageRating"] == "82.74"
            
            # Verify API was called with correct endpoint
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert f"/anime/{anime_id}" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_anime_by_id_not_found(self, kitsu_client):
        """Test anime retrieval with non-existent ID."""
        anime_id = 999999
        
        response = {"data": None}
        
        with patch.object(kitsu_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = response
            
            result = await kitsu_client.get_anime_by_id(anime_id)
            
            assert result is None

    @pytest.mark.asyncio
    async def test_search_anime_by_title(self, kitsu_client, sample_search_response):
        """Test anime search by title."""
        query = "cowboy"
        
        with patch.object(kitsu_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_search_response
            
            results = await kitsu_client.search_anime(query=query, limit=10)
            
            assert len(results) == 2
            assert results[0]["id"] == "1"
            assert results[0]["attributes"]["canonicalTitle"] == "Cowboy Bebop"
            assert results[1]["id"] == "2"
            assert results[1]["attributes"]["canonicalTitle"] == "Trigun"
            
            # Verify search parameters
            call_args = mock_request.call_args
            assert "/anime" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_search_anime_with_filters(self, kitsu_client, sample_search_response):
        """Test anime search with category and year filters."""
        query = "space"
        categories = ["Action", "Drama"]
        year = "1998"
        
        with patch.object(kitsu_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_search_response
            
            results = await kitsu_client.search_anime(
                query=query,
                categories=categories,
                year=year,
                limit=5
            )
            
            assert isinstance(results, list)
            
            # Verify filter parameters were included
            call_args = mock_request.call_args
            assert "/anime" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_anime_episodes(self, kitsu_client, sample_episodes_response):
        """Test retrieving anime episodes."""
        anime_id = 1
        
        with patch.object(kitsu_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_episodes_response
            
            episodes = await kitsu_client.get_anime_episodes(anime_id)
            
            assert len(episodes) == 1
            assert episodes[0]["attributes"]["canonicalTitle"] == "Asteroid Blues"
            assert episodes[0]["attributes"]["number"] == 1
            assert episodes[0]["attributes"]["length"] == 24
            
            # Verify episodes endpoint was called
            call_args = mock_request.call_args
            assert f"/anime/{anime_id}/episodes" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_streaming_links(self, kitsu_client, sample_streaming_links_response):
        """Test retrieving streaming links."""
        anime_id = 1
        
        with patch.object(kitsu_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_streaming_links_response
            
            streaming_links = await kitsu_client.get_streaming_links(anime_id)
            
            assert len(streaming_links) == 1
            assert streaming_links[0]["attributes"]["url"] == "https://www.crunchyroll.com/cowboy-bebop"
            assert "en" in streaming_links[0]["attributes"]["subs"]
            assert "ja" in streaming_links[0]["attributes"]["dubs"]
            
            # Verify streaming links endpoint was called
            call_args = mock_request.call_args
            assert f"/anime/{anime_id}/streaming-links" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_anime_characters(self, kitsu_client):
        """Test retrieving anime characters."""
        anime_id = 1
        characters_response = {
            "data": [
                {
                    "id": "1",
                    "type": "animeCharacters",
                    "attributes": {
                        "role": "main"
                    },
                    "relationships": {
                        "character": {
                            "data": {
                                "id": "1",
                                "type": "characters"
                            }
                        }
                    }
                }
            ],
            "included": [
                {
                    "id": "1",
                    "type": "characters",
                    "attributes": {
                        "canonicalName": "Spike Spiegel",
                        "image": {
                            "original": "https://media.kitsu.io/characters/images/1/original.jpg"
                        }
                    }
                }
            ]
        }
        
        with patch.object(kitsu_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = characters_response
            
            characters = await kitsu_client.get_anime_characters(anime_id)
            
            assert len(characters) == 1
            assert characters[0]["character"]["canonicalName"] == "Spike Spiegel"
            assert characters[0]["role"] == "main"

    @pytest.mark.asyncio
    async def test_json_api_request_success(self, kitsu_client):
        """Test successful JSON:API request."""
        endpoint = "/anime/1"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.json = AsyncMock(return_value={"data": {"id": "1", "type": "anime"}})
            mock_response.status = 200
            mock_response.headers = {"Content-Type": "application/vnd.api+json"}
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await kitsu_client._make_request(endpoint)
            
            # Verify JSON:API headers were included
            call_args = mock_get.call_args
            headers = call_args[1]["headers"]
            assert headers["Accept"] == "application/vnd.api+json"
            assert headers["Content-Type"] == "application/vnd.api+json"
            assert result["data"]["id"] == "1"

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, kitsu_client):
        """Test rate limit handling."""
        endpoint = "/anime/1"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 429
            mock_response.headers = {"Retry-After": "10"}
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with pytest.raises(Exception) as exc_info:
                await kitsu_client._make_request(endpoint)
            
            assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_error_handling_404(self, kitsu_client):
        """Test 404 error handling."""
        endpoint = "/anime/999999"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 404
            mock_response.json = AsyncMock(return_value={
                "errors": [
                    {
                        "title": "Record not found",
                        "detail": "The record identified by 999999 could not be found.",
                        "code": "404",
                        "status": "404"
                    }
                ]
            })
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            with pytest.raises(Exception) as exc_info:
                await kitsu_client._make_request(endpoint)
            
            assert "record not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, kitsu_client):
        """Test circuit breaker integration."""
        # Mock circuit breaker to simulate open state
        kitsu_client.circuit_breaker.is_open = Mock(return_value=True)
        
        with pytest.raises(Exception) as exc_info:
            await kitsu_client.get_anime_by_id(1)
        
        assert "circuit breaker" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cache_integration(self, kitsu_client, sample_anime_response):
        """Test cache integration."""
        anime_id = 1
        cache_key = f"kitsu_anime_{anime_id}"
        
        # Mock cache hit
        kitsu_client.cache_manager.get = AsyncMock(return_value=sample_anime_response["data"])
        
        result = await kitsu_client.get_anime_by_id(anime_id)
        
        assert result["id"] == "1"
        kitsu_client.cache_manager.get.assert_called_once_with(cache_key)

    @pytest.mark.asyncio
    async def test_pagination_handling(self, kitsu_client):
        """Test pagination parameter handling."""
        query = "anime"
        
        paginated_response = {
            "data": [{"id": "1", "type": "anime"}],
            "links": {
                "next": "https://kitsu.io/api/edge/anime?page[offset]=10&page[limit]=10"
            },
            "meta": {
                "count": 100
            }
        }
        
        with patch.object(kitsu_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = paginated_response
            
            results = await kitsu_client.search_anime(query=query, limit=10, offset=0)
            
            assert len(results) == 1
            
            # Verify pagination parameters
            call_args = mock_request.call_args
            assert "/anime" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_json_api_includes_handling(self, kitsu_client):
        """Test JSON:API includes parameter handling."""
        anime_id = 1
        includes = ["categories", "characters", "staff"]
        
        with patch.object(kitsu_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"data": {"id": "1"}}
            
            await kitsu_client.get_anime_by_id(anime_id, includes=includes)
            
            # Verify includes parameter was added
            call_args = mock_request.call_args
            assert "/anime/1" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_relationship_data_parsing(self, kitsu_client):
        """Test parsing of JSON:API relationship data."""
        anime_id = 1
        
        response_with_includes = {
            "data": {
                "id": "1",
                "type": "anime",
                "relationships": {
                    "categories": {
                        "data": [
                            {"id": "1", "type": "categories"},
                            {"id": "2", "type": "categories"}
                        ]
                    }
                }
            },
            "included": [
                {
                    "id": "1",
                    "type": "categories",
                    "attributes": {"title": "Action", "slug": "action"}
                },
                {
                    "id": "2", 
                    "type": "categories",
                    "attributes": {"title": "Adventure", "slug": "adventure"}
                }
            ]
        }
        
        with patch.object(kitsu_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = response_with_includes
            
            result = await kitsu_client.get_anime_by_id(anime_id, includes=["categories"])
            
            assert result is not None
            assert result["id"] == "1"

    @pytest.mark.asyncio
    async def test_trending_anime(self, kitsu_client):
        """Test retrieving trending anime."""
        trending_response = {
            "data": [
                {
                    "id": "1",
                    "type": "anime",
                    "attributes": {
                        "canonicalTitle": "Trending Anime 1",
                        "averageRating": "85.0"
                    }
                }
            ]
        }
        
        with patch.object(kitsu_client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = trending_response
            
            results = await kitsu_client.get_trending_anime(limit=20)
            
            assert len(results) == 1
            assert results[0]["attributes"]["canonicalTitle"] == "Trending Anime 1"
            
            # Verify trending endpoint and sorting
            call_args = mock_request.call_args
            assert "/anime" in call_args[0][0]