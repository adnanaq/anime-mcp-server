"""Tests for AniSearch query parameter mapper.

Following TDD approach - these tests validate the AniSearch mapper's 
query parameter conversion functionality.
"""

import pytest
from typing import Dict, Any

from src.integrations.mappers.anisearch_mapper import AniSearchMapper
from src.models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class TestAniSearchMapper:
    """Test AniSearchMapper query parameter conversion methods."""

    @pytest.fixture
    def sample_universal_search_params(self) -> UniversalSearchParams:
        """Sample universal search parameters."""
        return UniversalSearchParams(
            query="test anime",
            genres=["Action"],
            status=AnimeStatus.FINISHED,
            type_format=AnimeFormat.TV,
            year=2023,
            min_score=7.0,
            limit=20,
            sort_by="score",
            sort_order="desc"
        )

    def test_to_anisearch_search_params_basic(self, sample_universal_search_params):
        """Test basic universal to AniSearch search parameter conversion."""
        anisearch_params = AniSearchMapper.to_anisearch_search_params(sample_universal_search_params)
        
        # Test basic parameters
        assert anisearch_params["query"] == "test anime"
        assert anisearch_params["limit"] == 20
        assert anisearch_params["type"] == "video.tv_show"  # AniSearch format
        
        # Test score conversion
        assert "min_rating" in anisearch_params

    def test_to_anisearch_search_params_status_mapping(self):
        """Test status mapping from universal to AniSearch."""
        status_test_cases = [
            (AnimeStatus.FINISHED, "COMPLETED"),
            (AnimeStatus.RELEASING, "ONGOING"),
            (AnimeStatus.NOT_YET_RELEASED, "UPCOMING"),
        ]
        
        for universal_status, expected_anisearch in status_test_cases:
            params = UniversalSearchParams(status=universal_status)
            anisearch_params = AniSearchMapper.to_anisearch_search_params(params)
            assert anisearch_params["status"] == expected_anisearch

    def test_to_anisearch_search_params_format_mapping(self):
        """Test format mapping from universal to AniSearch."""
        format_test_cases = [
            (AnimeFormat.TV, "video.tv_show"),
            (AnimeFormat.MOVIE, "video.movie"),
        ]
        
        for universal_format, expected_anisearch in format_test_cases:
            params = UniversalSearchParams(type_format=universal_format)
            anisearch_params = AniSearchMapper.to_anisearch_search_params(params)
            assert anisearch_params["type"] == expected_anisearch

    def test_to_anisearch_search_params_genre_handling(self):
        """Test genre parameter conversion."""
        params = UniversalSearchParams(genres=["Action", "Adventure"])
        anisearch_params = AniSearchMapper.to_anisearch_search_params(params)
        
        assert "genres" in anisearch_params
        assert "Action" in anisearch_params["genres"]
        assert "Adventure" in anisearch_params["genres"]

    def test_to_anisearch_search_params_minimal(self):
        """Test conversion with minimal parameters."""
        params = UniversalSearchParams()
        anisearch_params = AniSearchMapper.to_anisearch_search_params(params)
        
        # Should only contain default values
        assert anisearch_params["limit"] == 20
        assert len(anisearch_params) >= 1

    def test_to_anisearch_search_params_with_offset(self):
        """Test conversion with offset parameter to cover missing line."""
        params = UniversalSearchParams(query="test", offset=50)
        anisearch_params = AniSearchMapper.to_anisearch_search_params(params)
        
        # Should include offset parameter (line 82)
        assert "offset" in anisearch_params
        assert anisearch_params["offset"] == 50

    def test_bidirectional_consistency(self, sample_universal_search_params):
        """Test that converting to AniSearch params preserves critical information."""
        anisearch_params = AniSearchMapper.to_anisearch_search_params(sample_universal_search_params)
        
        # Key parameters should be preserved
        assert "query" in anisearch_params
        assert anisearch_params["query"] == "test anime"