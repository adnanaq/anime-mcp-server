"""Unit tests for anime models."""

from datetime import datetime
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from src.models.anime import (
    AnimeEntry,
    DatabaseStats,
    SearchRequest,
    SearchResponse,
    SearchResult,
)


class TestAnimeEntry:
    """Test cases for AnimeEntry model."""

    def test_anime_entry_valid_data(self, sample_anime_data: Dict[str, Any]):
        """Test AnimeEntry creation with valid data."""
        anime = AnimeEntry(**sample_anime_data)

        assert anime.title == "Test Anime"
        assert anime.type == "TV"
        assert anime.episodes == 12
        assert anime.status == "FINISHED"
        assert len(anime.sources) == 4
        assert len(anime.tags) == 3
        assert len(anime.studios) == 1

    def test_anime_entry_minimal_data(self):
        """Test AnimeEntry with minimal required data."""
        minimal_data = {
            "sources": ["https://myanimelist.net/anime/1"],
            "title": "Minimal Anime",
            "type": "TV",
            "episodes": 0,
            "status": "FINISHED",
        }

        anime = AnimeEntry(**minimal_data)
        assert anime.title == "Minimal Anime"
        assert anime.episodes == 0
        assert anime.tags == []
        assert anime.studios == []
        assert anime.synopsis is None

    def test_anime_entry_duration_validation(self):
        """Test duration field validation."""
        # Test with integer duration
        data_int = {
            "sources": ["https://myanimelist.net/anime/1"],
            "title": "Test",
            "type": "TV",
            "episodes": 1,
            "status": "FINISHED",
            "duration": 1440,
        }
        anime = AnimeEntry(**data_int)
        assert anime.duration == 1440

        # Test with dict duration
        data_dict = {
            "sources": ["https://myanimelist.net/anime/1"],
            "title": "Test",
            "type": "TV",
            "episodes": 1,
            "status": "FINISHED",
            "duration": {"value": 1440, "unit": "seconds"},
        }
        anime = AnimeEntry(**data_dict)
        assert anime.duration == 1440

        # Test with None
        data_none = {
            "sources": ["https://myanimelist.net/anime/1"],
            "title": "Test",
            "type": "TV",
            "episodes": 1,
            "status": "FINISHED",
            "duration": None,
        }
        anime = AnimeEntry(**data_none)
        assert anime.duration is None

    def test_anime_entry_missing_required_fields(self):
        """Test AnimeEntry validation with missing required fields."""
        with pytest.raises(ValidationError):
            AnimeEntry(title="Test")  # Missing sources, type, status

    def test_anime_entry_invalid_field_types(self):
        """Test AnimeEntry validation with invalid field types."""
        invalid_data = {
            "sources": "not-a-list",  # Should be list
            "title": "Test",
            "type": "TV",
            "episodes": "not-an-int",  # Should be int
            "status": "FINISHED",
        }

        with pytest.raises(ValidationError):
            AnimeEntry(**invalid_data)


class TestSearchModels:
    """Test cases for search-related models."""

    def test_search_request_valid(self):
        """Test SearchRequest with valid data."""
        request = SearchRequest(query="test anime", limit=10)

        assert request.query == "test anime"
        assert request.limit == 10
        assert request.filters is None

    def test_search_request_with_filters(self):
        """Test SearchRequest with filters."""
        filters = {"type": "TV", "year": 2023}
        request = SearchRequest(query="test", limit=20, filters=filters)

        assert request.query == "test"
        assert request.limit == 20
        assert request.filters == filters

    def test_search_request_limit_validation(self):
        """Test SearchRequest limit validation."""
        # Valid limits
        SearchRequest(query="test", limit=1)
        SearchRequest(query="test", limit=100)

        # Invalid limits
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=0)

        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=101)

    def test_search_result_complete(self):
        """Test SearchResult with complete platform data."""
        result = SearchResult(
            anime_id="test123",
            title="Test Anime",
            synopsis="A test anime",
            type="TV",
            episodes=12,
            tags=["Action", "Adventure"],
            studios=["Test Studio"],
            picture="https://example.com/image.jpg",
            score=0.95,
            year=2023,
            season="spring",
            # Platform IDs
            myanimelist_id=12345,
            anilist_id=67890,
            kitsu_id=111,
            anidb_id=222,
            anisearch_id=333,
            simkl_id=444,
            livechart_id=555,
            animenewsnetwork_id=666,
            animeplanet_id="test-slug",
            notify_id="ABC123",
            animecountdown_id=777,
        )

        assert result.anime_id == "test123"
        assert result.title == "Test Anime"
        assert result.myanimelist_id == 12345
        assert result.animeplanet_id == "test-slug"
        assert result.notify_id == "ABC123"

    def test_search_result_minimal(self):
        """Test SearchResult with minimal required data."""
        result = SearchResult(
            anime_id="test123",
            title="Test Anime",
            type="TV",
            episodes=12,
            tags=[],
            studios=[],
            score=0.5,
        )

        assert result.anime_id == "test123"
        assert result.synopsis is None
        assert result.myanimelist_id is None
        assert result.year is None

    def test_search_result_platform_id_types(self):
        """Test SearchResult platform ID type validation."""
        # Numeric IDs should accept integers
        result = SearchResult(
            anime_id="test123",
            title="Test",
            type="TV",
            episodes=1,
            tags=[],
            studios=[],
            score=0.5,
            myanimelist_id=12345,
            anilist_id=67890,
        )

        assert isinstance(result.myanimelist_id, int)
        assert isinstance(result.anilist_id, int)

        # String IDs should accept strings
        result = SearchResult(
            anime_id="test123",
            title="Test",
            type="TV",
            episodes=1,
            tags=[],
            studios=[],
            score=0.5,
            animeplanet_id="test-slug",
            notify_id="ABC123",
        )

        assert isinstance(result.animeplanet_id, str)
        assert isinstance(result.notify_id, str)

    def test_search_response_complete(self):
        """Test SearchResponse with results."""
        results = [
            SearchResult(
                anime_id="test1",
                title="Test 1",
                type="TV",
                episodes=12,
                tags=["Action"],
                studios=["Studio A"],
                score=0.9,
            ),
            SearchResult(
                anime_id="test2",
                title="Test 2",
                type="Movie",
                episodes=1,
                tags=["Drama"],
                studios=["Studio B"],
                score=0.8,
            ),
        ]

        response = SearchResponse(
            query="test anime",
            results=results,
            total_results=2,
            processing_time_ms=150.5,
        )

        assert response.query == "test anime"
        assert len(response.results) == 2
        assert response.total_results == 2
        assert response.processing_time_ms == 150.5

    def test_search_response_empty(self):
        """Test SearchResponse with empty results."""
        response = SearchResponse(
            query="nonexistent anime",
            results=[],
            total_results=0,
            processing_time_ms=50.0,
        )

        assert response.query == "nonexistent anime"
        assert len(response.results) == 0
        assert response.total_results == 0


class TestDatabaseStats:
    """Test cases for DatabaseStats model."""

    def test_database_stats_valid(self):
        """Test DatabaseStats with valid data."""
        stats = DatabaseStats(
            total_anime=38894,
            indexed_anime=38894,
            last_updated=datetime(2024, 1, 1, 12, 0, 0),
            index_health="green",
            average_quality_score=0.85,
        )

        assert stats.total_anime == 38894
        assert stats.indexed_anime == 38894
        assert stats.index_health == "green"
        assert stats.average_quality_score == 0.85

    def test_database_stats_partial_index(self):
        """Test DatabaseStats with partial indexing."""
        stats = DatabaseStats(
            total_anime=38894,
            indexed_anime=17931,
            last_updated=datetime.now(),
            index_health="yellow",
            average_quality_score=0.75,
        )

        assert stats.total_anime > stats.indexed_anime
        assert stats.index_health == "yellow"

    def test_database_stats_validation(self):
        """Test DatabaseStats field validation."""
        # Valid data
        DatabaseStats(
            total_anime=1000,
            indexed_anime=1000,
            last_updated=datetime.now(),
            index_health="green",
            average_quality_score=0.5,
        )

        # Invalid data types
        with pytest.raises(ValidationError):
            DatabaseStats(
                total_anime="not-an-int",
                indexed_anime=1000,
                last_updated=datetime.now(),
                index_health="green",
                average_quality_score=0.5,
            )
