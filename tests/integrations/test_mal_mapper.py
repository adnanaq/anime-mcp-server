"""Tests for MAL API v2 query parameter mapper.

Following TDD approach - these tests validate the MAL mapper's 
query parameter conversion functionality.
"""

import pytest
from typing import Dict, Any

from src.integrations.mappers.mal_mapper import MALMapper
from src.models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class TestMALMapper:
    """Test MALMapper query parameter conversion methods."""

    @pytest.fixture
    def sample_universal_search_params(self) -> UniversalSearchParams:
        """Sample universal search parameters."""
        return UniversalSearchParams(
            query="test anime",
            genres=["Action"],
            genres_exclude=["Horror"],
            status=AnimeStatus.FINISHED,
            type_format=AnimeFormat.TV,
            year=2023,
            season=AnimeSeason.SPRING,
            min_score=7.0,
            max_score=9.0,
            min_episodes=10,
            max_episodes=30,
            include_adult=False,
            limit=20,
            sort_by="score",
            sort_order="desc"
        )

    def test_to_mal_search_params_basic(self, sample_universal_search_params):
        """Test basic universal to MAL search parameter conversion."""
        mal_params = MALMapper.to_mal_search_params(sample_universal_search_params)
        
        # Test basic parameters (only q, limit, offset supported)
        assert mal_params["q"] == "test anime"
        assert mal_params["limit"] == 20
        
        # MAL API v2 does not support filtering by status, format, score, etc.
        # These are response fields only
        assert "status" not in mal_params
        assert "media_type" not in mal_params
        assert "start_date" not in mal_params
        assert "min_score" not in mal_params
        assert "max_score" not in mal_params
        assert "nsfw" not in mal_params

    def test_to_mal_search_params_response_fields(self):
        """Test response field requests generate proper fields parameter."""
        params = UniversalSearchParams(
            query="test",
            title_field=True,
            status_field=True,
            score_field=True,
            mal_nsfw_field=True
        )
        mal_params = MALMapper.to_mal_search_params(params)
        
        # Should generate fields parameter
        assert "fields" in mal_params
        assert "title" in mal_params["fields"]
        assert "status" in mal_params["fields"]
        assert "mean" in mal_params["fields"]  # score_field maps to mean
        assert "nsfw" in mal_params["fields"]

    def test_to_mal_search_params_mal_specific_fields(self):
        """Test MAL-specific response fields generate proper fields parameter."""
        params = UniversalSearchParams(
            query="test",
            mal_alternative_titles_field=True,
            mal_num_list_users_field=True,
            mal_average_episode_duration_field=True,
            mal_start_season_field=True
        )
        mal_params = MALMapper.to_mal_search_params(params)
        
        # Should generate fields parameter with MAL-specific fields
        assert "fields" in mal_params
        assert "alternative_titles" in mal_params["fields"]
        assert "num_list_users" in mal_params["fields"]
        assert "average_episode_duration" in mal_params["fields"]
        assert "start_season" in mal_params["fields"]

    def test_to_mal_search_params_minimal(self):
        """Test conversion with minimal parameters."""
        params = UniversalSearchParams()
        mal_params = MALMapper.to_mal_search_params(params)
        
        # Should only contain default limit
        assert mal_params["limit"] == 20
        assert len(mal_params) == 1  # Only limit field

    def test_to_mal_search_params_unsupported_ignored(self):
        """Test that unsupported filtering parameters are ignored."""
        params = UniversalSearchParams(
            query="test",
            status=AnimeStatus.FINISHED,     # Not supported as query param
            type_format=AnimeFormat.TV,      # Not supported as query param
            min_score=7.0,                   # Not supported as query param
            characters=["Character"],        # Not supported in MAL search
            themes=["Theme"],                # Not directly supported
            staff=["Director"],              # Not supported in basic search
        )
        mal_params = MALMapper.to_mal_search_params(params)
        
        # Should include supported params
        assert mal_params["q"] == "test"
        
        # Should not include any filtering params (MAL doesn't support them)
        assert "status" not in mal_params
        assert "media_type" not in mal_params
        assert "min_score" not in mal_params
        assert "characters" not in mal_params
        assert "themes" not in mal_params  
        assert "staff" not in mal_params

    def test_to_mal_search_params_no_fields_no_filters(self):
        """Test that MAL doesn't generate filtering parameters."""
        params = UniversalSearchParams(
            query="test",
            include_adult=False,
            status=AnimeStatus.FINISHED,
            min_score=8.0
        )
        mal_params = MALMapper.to_mal_search_params(params)
        
        # Should only have basic query params, no filtering
        assert mal_params["q"] == "test"
        assert "nsfw" not in mal_params       # No adult content filtering
        assert "status" not in mal_params     # No status filtering 
        assert "min_score" not in mal_params  # No score filtering

    def test_basic_query_params_only(self, sample_universal_search_params):
        """Test that only basic query parameters are generated."""
        mal_params = MALMapper.to_mal_search_params(sample_universal_search_params)
        
        # Only basic query parameters should be preserved
        assert "q" in mal_params  # query preserved
        assert "limit" in mal_params  # limit preserved
        
        # Filtering parameters should NOT be preserved (MAL doesn't support them)
        assert "status" not in mal_params
        assert "media_type" not in mal_params
        assert "min_score" not in mal_params
        assert "max_score" not in mal_params

    def test_to_mal_search_params_fields_comma_separated(self):
        """Test that multiple fields are properly comma-separated."""
        params = UniversalSearchParams(
            query="test",
            title_field=True,
            status_field=True,
            score_field=True,
            episodes_field=True,
            genres_field=True
        )
        mal_params = MALMapper.to_mal_search_params(params)
        
        # Should generate comma-separated fields
        assert "fields" in mal_params
        fields = mal_params["fields"]
        assert "title" in fields
        assert "status" in fields
        assert "mean" in fields
        assert "num_episodes" in fields
        assert "genres" in fields
        # Should be comma-separated
        assert "," in fields

    def test_to_mal_search_params_offset_handling(self):
        """Test offset parameter handling."""
        params = UniversalSearchParams(query="test", offset=10)
        mal_params = MALMapper.to_mal_search_params(params)
        
        assert mal_params["q"] == "test"
        assert mal_params["offset"] == 10
        assert mal_params["limit"] == 20  # default

    def test_to_mal_search_params_no_filtering_comprehensive(self):
        """Test comprehensive filtering parameters are all ignored."""
        params = UniversalSearchParams(
            query="comprehensive test",
            status=AnimeStatus.RELEASING,
            type_format=AnimeFormat.MOVIE,
            min_score=8.0,
            max_score=10.0,
            min_episodes=1,
            max_episodes=3,
            min_duration=90,
            include_adult=False,
            limit=50,
            sort_by="popularity",
            sort_order="asc"
        )
        
        mal_params = MALMapper.to_mal_search_params(params)
        
        # Only basic query parameters should be included
        assert mal_params["q"] == "comprehensive test"
        assert mal_params["limit"] == 50
        
        # All filtering parameters should be ignored
        assert "status" not in mal_params
        assert "media_type" not in mal_params
        assert "min_score" not in mal_params
        assert "max_score" not in mal_params
        assert "min_episodes" not in mal_params
        assert "max_episodes" not in mal_params
        assert "average_episode_duration" not in mal_params
        assert "nsfw" not in mal_params
        assert "sort" not in mal_params
        assert "order" not in mal_params