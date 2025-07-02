"""Tests for AniDB query parameter mapper.

Following TDD approach - these tests validate the AniDB mapper's 
query parameter conversion functionality based on actual API testing.

AniDB API limitations (verified by testing):
- Only supports `aid` parameter for ID-based lookup
- No search/filter parameters supported
- Requires anime-titles.xml for title-to-ID mapping
"""

import pytest
from typing import Dict, Any

from src.integrations.mappers.anidb_mapper import AniDBMapper
from src.models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class TestAniDBMapper:
    """Test AniDBMapper query parameter conversion methods."""

    @pytest.fixture
    def sample_universal_search_params(self) -> UniversalSearchParams:
        """Sample universal search parameters."""
        return UniversalSearchParams(
            query="123",  # Numeric ID for AniDB
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

    def test_to_anidb_search_params_numeric_query(self, sample_universal_search_params):
        """Test numeric query conversion to aid parameter."""
        anidb_params = AniDBMapper.to_anidb_search_params(sample_universal_search_params)
        
        # Numeric query should be converted to aid parameter
        assert anidb_params["aid"] == 123
        assert len(anidb_params) == 1  # Only aid parameter supported

    def test_to_anidb_search_params_text_query_ignored(self):
        """Test that text queries are ignored since AniDB doesn't support text search."""
        params = UniversalSearchParams(query="naruto")
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        # Text query should be ignored (not convertible to aid)
        assert len(anidb_params) == 0  # No parameters returned
        assert "aid" not in anidb_params

    def test_to_anidb_search_params_no_query(self):
        """Test conversion with no query parameter."""
        params = UniversalSearchParams()
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        # Should return empty params when no query provided
        assert len(anidb_params) == 0

    def test_to_anidb_search_params_all_other_params_ignored(self):
        """Test that all non-query parameters are ignored due to API limitations."""
        params = UniversalSearchParams(
            query="456",  # Only this should work
            status=AnimeStatus.FINISHED,
            type_format=AnimeFormat.TV,
            min_score=7.0,
            max_score=9.0,
            min_episodes=10,
            max_episodes=30,
            genres=["Action"],
            year=2023,
            limit=50,
            offset=10,
            sort_by="score",
            sort_order="desc",
            include_adult=False
        )
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        # Only aid should be returned, all other parameters ignored
        assert anidb_params == {"aid": 456}
        assert "status" not in anidb_params
        assert "type" not in anidb_params
        assert "min_rating" not in anidb_params
        assert "limit" not in anidb_params
        assert "tags" not in anidb_params

    def test_to_anidb_search_params_invalid_numeric_query(self):
        """Test handling of invalid numeric queries."""
        invalid_queries = ["", "0", "-1", "abc123", "12.5"]
        
        for invalid_query in invalid_queries:
            params = UniversalSearchParams(query=invalid_query)
            anidb_params = AniDBMapper.to_anidb_search_params(params)
            
            # Invalid numeric queries should be ignored
            assert len(anidb_params) == 0

    def test_to_anidb_search_params_large_numeric_query(self):
        """Test handling of large numeric IDs."""
        params = UniversalSearchParams(query="999999")
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        # Large valid numeric ID should work
        assert anidb_params["aid"] == 999999

    def test_to_anidb_search_params_edge_cases(self):
        """Test edge cases for query parameter conversion."""
        # Test whitespace
        params_whitespace = UniversalSearchParams(query="  123  ")
        anidb_params_whitespace = AniDBMapper.to_anidb_search_params(params_whitespace)
        assert anidb_params_whitespace["aid"] == 123
        
        # Test leading zeros
        params_zeros = UniversalSearchParams(query="0001")
        anidb_params_zeros = AniDBMapper.to_anidb_search_params(params_zeros)
        assert anidb_params_zeros["aid"] == 1

    def test_to_anidb_search_params_api_limitations_documented(self):
        """Test that the mapper correctly reflects documented API limitations."""
        # This test documents the actual AniDB API behavior
        params = UniversalSearchParams(
            query="123",
            limit=10,  # Not supported
            offset=5,  # Not supported  
            type_format=AnimeFormat.TV,  # Not supported
            status=AnimeStatus.FINISHED  # Not supported
        )
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        # Verify only aid parameter is supported
        expected_params = {"aid": 123}
        assert anidb_params == expected_params
        
        # Verify unsupported parameters are excluded
        unsupported_keys = ["limit", "offset", "type", "status", "min_rating", "tags"]
        for key in unsupported_keys:
            assert key not in anidb_params