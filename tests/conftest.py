"""Test configuration and fixtures."""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch
import os
import sys

import pytest

# Mock external dependencies at import time to prevent hangs during test discovery
sys.modules['qdrant_client'] = Mock()
sys.modules['qdrant_client.http'] = Mock()
sys.modules['qdrant_client.models'] = Mock()
sys.modules['fastembed'] = Mock()
sys.modules['fastembed.TextEmbedding'] = Mock()

# Set test environment variables
os.environ.setdefault('QDRANT_URL', 'http://localhost:6333')
os.environ.setdefault('DEBUG', 'True')
os.environ.setdefault('OPENAI_API_KEY', 'test-key')
os.environ.setdefault('QDRANT_COLLECTION_NAME', 'test_collection')

# Global patches to prevent external connections
@pytest.fixture(scope="session", autouse=True)
def mock_external_services():
    """Mock all external services to prevent network calls during testing."""
    with patch('aiohttp.ClientSession') as mock_session:
        # Mock aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={'data': []})
        mock_response.text = AsyncMock(return_value='{"data": []}')
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
        
        yield

from src.models.anime import AnimeEntry


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
def data_service():
    """Create mocked AnimeDataService instance for testing."""
    with patch('src.services.data_service.AnimeDataService') as mock_service:
        # Mock the service with essential methods
        service_instance = Mock()
        service_instance.download_anime_data = AsyncMock(return_value=True)
        service_instance.process_anime_data = AsyncMock(return_value={'processed': 100})
        service_instance.get_stats = AsyncMock(return_value={
            'total_anime': 100,
            'last_updated': '2024-01-01T00:00:00Z'
        })
        mock_service.return_value = service_instance
        yield service_instance


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
