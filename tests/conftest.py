"""Test configuration and fixtures."""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest

from src.models.anime import AnimeEntry
from src.services.data_service import AnimeDataService


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_anime_data() -> Dict[str, Any]:
    """Sample anime data for testing."""
    return {
        "sources": [
            "https://myanimelist.net/anime/1",
            "https://anilist.co/anime/1",
            "https://kitsu.app/anime/1",
            "https://anidb.net/anime/1",
        ],
        "title": "Test Anime",
        "type": "TV",
        "episodes": 12,
        "status": "FINISHED",
        "animeSeason": {"season": "SPRING", "year": 2023},
        "picture": "https://example.com/image.jpg",
        "thumbnail": "https://example.com/thumb.jpg",
        "synonyms": ["Test Show", "テストアニメ"],
        "tags": ["Action", "Adventure", "Sci-Fi"],
        "studios": ["Test Studio"],
        "producers": ["Test Producer"],
        "synopsis": "A test anime about testing.",
    }


@pytest.fixture
def complex_anime_data() -> Dict[str, Any]:
    """Complex anime data with multiple platform sources."""
    return {
        "sources": [
            "https://myanimelist.net/anime/12345",
            "https://anilist.co/anime/67890",
            "https://kitsu.app/anime/111",
            "https://anidb.net/anime/222",
            "https://anisearch.com/anime/333",
            "https://simkl.com/anime/444",
            "https://livechart.me/anime/555",
            "https://animenewsnetwork.com/encyclopedia/anime.php?id=666",
            "https://anime-planet.com/anime/test-anime-slug",
            "https://notify.moe/anime/ABC123DEF",
            "https://animecountdown.com/777",
        ],
        "title": "Complex Test Anime",
        "type": "Movie",
        "episodes": 1,
        "status": "FINISHED",
        "animeSeason": {"season": "SUMMER", "year": 2024},
        "picture": "https://example.com/complex.jpg",
        "synonyms": ["Complex Show", "複雑なアニメ"],
        "tags": ["Drama", "Thriller", "Mystery"],
        "studios": ["Complex Studio", "Another Studio"],
        "producers": ["Complex Producer"],
        "synopsis": "A complex anime for testing cross-platform ID extraction.",
    }


@pytest.fixture
def anime_entry(sample_anime_data) -> AnimeEntry:
    """Create AnimeEntry instance from sample data."""
    return AnimeEntry(**sample_anime_data)


@pytest.fixture
def data_service() -> AnimeDataService:
    """Create AnimeDataService instance for testing."""
    return AnimeDataService()


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock_client = AsyncMock()
    mock_client.health_check.return_value = True
    mock_client.create_index.return_value = True
    mock_client.add_documents.return_value = {"status": "success"}
    mock_client.search.return_value = []
    mock_client.get_stats.return_value = {
        "total_anime": 100,
        "indexed_anime": 100,
        "last_updated": "2024-01-01T00:00:00Z",
        "index_health": "green",
        "average_quality_score": 0.85,
    }
    return mock_client


@pytest.fixture
def expected_platform_ids() -> Dict[str, Any]:
    """Expected platform IDs from complex anime data."""
    return {
        "myanimelist_id": 12345,
        "anilist_id": 67890,
        "kitsu_id": 111,
        "anidb_id": 222,
        "anisearch_id": 333,
        "simkl_id": 444,
        "livechart_id": 555,
        "animenewsnetwork_id": 666,
        "animeplanet_id": "test-anime-slug",
        "notify_id": "ABC123DEF",
        "animecountdown_id": 777,
    }


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text.return_value = '{"data": []}'
    mock_session.get.return_value.__aenter__.return_value = mock_response
    return mock_session
