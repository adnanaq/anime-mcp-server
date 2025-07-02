"""Tests for Jikan (unofficial MAL) query parameter mapper.

Following TDD approach - these tests validate the Jikan mapper's 
query parameter conversion functionality.
"""

import pytest
from typing import Dict, Any

from src.integrations.mappers.jikan_mapper import JikanMapper
from src.models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class TestJikanMapper:
    """Test JikanMapper query parameter conversion methods."""

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

    def test_to_jikan_search_params_basic(self, sample_universal_search_params):
        """Test basic universal to Jikan search parameter conversion."""
        jikan_params = JikanMapper.to_jikan_search_params(sample_universal_search_params)
        
        # Test basic parameters
        assert jikan_params["q"] == "test anime"
        assert jikan_params["limit"] == 20
        assert jikan_params["status"] == "complete"  # Jikan-specific status
        assert jikan_params["type"] == "tv"
        
        # Test score conversion (Jikan uses same 0-10 scale)
        assert jikan_params["min_score"] == 7.0
        assert jikan_params["max_score"] == 9.0
        
        # Test SFW filter
        assert jikan_params["sfw"] == "true"

    def test_to_jikan_search_params_status_mapping(self):
        """Test status mapping from universal to Jikan."""
        status_test_cases = [
            (AnimeStatus.FINISHED, "complete"),
            (AnimeStatus.RELEASING, "airing"),
            (AnimeStatus.NOT_YET_RELEASED, "upcoming"),
        ]
        
        for universal_status, expected_jikan in status_test_cases:
            params = UniversalSearchParams(status=universal_status)
            jikan_params = JikanMapper.to_jikan_search_params(params)
            assert jikan_params["status"] == expected_jikan

    def test_to_jikan_search_params_format_mapping(self):
        """Test format mapping from universal to Jikan."""
        format_test_cases = [
            (AnimeFormat.TV, "tv"),
            (AnimeFormat.MOVIE, "movie"),
            (AnimeFormat.OVA, "ova"),
            (AnimeFormat.ONA, "ona"),
            (AnimeFormat.SPECIAL, "special"),
            (AnimeFormat.MUSIC, "music"),
        ]
        
        for universal_format, expected_jikan in format_test_cases:
            params = UniversalSearchParams(type_format=universal_format)
            jikan_params = JikanMapper.to_jikan_search_params(params)
            assert jikan_params["type"] == expected_jikan

    def test_to_jikan_search_params_genre_handling(self):
        """Test genre parameter conversion."""
        params = UniversalSearchParams(
            genres=["Action", "Adventure"],
            genres_exclude=["Horror", "Ecchi"]
        )
        jikan_params = JikanMapper.to_jikan_search_params(params)
        
        assert jikan_params["genres"] == "Action,Adventure"
        assert jikan_params["genres_exclude"] == "Horror,Ecchi"

    def test_to_jikan_search_params_sort_mapping(self):
        """Test sort parameter mapping."""
        sort_test_cases = [
            ("score", "desc", "score", "desc"),
            ("popularity", "asc", "popularity", "asc"),
            ("title", "desc", "title", "desc"),
            ("year", "asc", "start_date", "asc"),
            ("episodes", "desc", "episodes", "desc"),
        ]
        
        for sort_by, sort_order, expected_order_by, expected_sort in sort_test_cases:
            params = UniversalSearchParams(sort_by=sort_by, sort_order=sort_order)
            jikan_params = JikanMapper.to_jikan_search_params(params)
            assert jikan_params["order_by"] == expected_order_by
            assert jikan_params["sort"] == expected_sort

    def test_to_jikan_search_params_minimal(self):
        """Test conversion with minimal parameters."""
        params = UniversalSearchParams()
        jikan_params = JikanMapper.to_jikan_search_params(params)
        
        # Should only contain default values
        assert jikan_params["limit"] == 20
        assert jikan_params["sfw"] == "true"  # Default: no adult content
        assert len(jikan_params) == 2  # Only these two fields

    def test_to_jikan_search_params_unsupported_ignored(self):
        """Test that unsupported parameters are gracefully ignored."""
        params = UniversalSearchParams(
            query="test",
            characters=["Character"],  # Not supported in Jikan search
            themes=["Theme"],          # Not directly supported
            staff=["Director"],        # Not supported in basic search
        )
        jikan_params = JikanMapper.to_jikan_search_params(params)
        
        # Should include supported params
        assert jikan_params["q"] == "test"
        
        # Should not include unsupported params
        assert "characters" not in jikan_params
        assert "themes" not in jikan_params  
        assert "staff" not in jikan_params

    def test_bidirectional_consistency(self, sample_universal_search_params):
        """Test that converting to Jikan params preserves critical information."""
        jikan_params = JikanMapper.to_jikan_search_params(sample_universal_search_params)
        
        # Key parameters should be preserved in some form
        assert "q" in jikan_params  # query preserved
        assert "status" in jikan_params  # status preserved
        assert "type" in jikan_params  # format preserved
        assert "min_score" in jikan_params  # min_score preserved
        
        # Parameters should be in Jikan format
        assert jikan_params["status"] == "complete"  # Jikan status format
        assert jikan_params["type"] == "tv"  # Jikan format
        assert isinstance(jikan_params["min_score"], float)  # Jikan score scale

    def test_to_jikan_search_params_end_date(self):
        """Test end_date mapping to Jikan API."""
        params = UniversalSearchParams(end_date="2023-12-31")
        jikan_params = JikanMapper.to_jikan_search_params(params)
        
        assert jikan_params["end_date"] == "2023-12-31"

    def test_to_jikan_search_params_jikan_specific_parameters(self):
        """Test Jikan-specific parameter handling."""
        params = UniversalSearchParams(
            jikan_anime_type="movie",
            jikan_sfw=True,
            jikan_genres_exclude=[1, 9, 12],  # Action, Ecchi, Hentai IDs
            jikan_order_by="score",
            jikan_sort="desc",
            jikan_letter="A",
            jikan_unapproved=False
        )
        jikan_params = JikanMapper.to_jikan_search_params(params)
        
        assert jikan_params["type"] == "movie"
        assert jikan_params["sfw"] == "true"
        assert jikan_params["genres_exclude"] == "1,9,12"
        assert jikan_params["order_by"] == "score"
        assert jikan_params["sort"] == "desc"
        assert jikan_params["letter"] == "A"
        assert jikan_params["unapproved"] == False

    def test_to_jikan_search_params_letter_conflict_prevention(self):
        """Test that letter parameter is excluded when query is present."""
        params = UniversalSearchParams(
            query="naruto",
            jikan_letter="N"
        )
        jikan_params = JikanMapper.to_jikan_search_params(params)
        
        assert jikan_params["q"] == "naruto"
        assert "letter" not in jikan_params  # Should be excluded due to conflict

    def test_to_jikan_search_params_platform_specific_override(self):
        """Test that platform-specific parameters can be passed via jikan_specific dict."""
        universal_params = UniversalSearchParams(query="test")
        jikan_specific = {
            "anime_type": "ona",
            "sfw": False,
            "order_by": "popularity",
            "page": 2
        }
        
        jikan_params = JikanMapper.to_jikan_search_params(universal_params, jikan_specific)
        
        assert jikan_params["q"] == "test"
        assert jikan_params["type"] == "ona"
        assert jikan_params["sfw"] == "true"  # Note: universal include_adult=False overrides jikan_specific sfw=False
        assert jikan_params["order_by"] == "popularity"
        assert jikan_params["page"] == 2

    def test_to_jikan_search_params_comprehensive_mapping(self):
        """Test comprehensive parameter mapping including new features."""
        params = UniversalSearchParams(
            query="comprehensive test",
            status=AnimeStatus.RELEASING,
            type_format=AnimeFormat.OVA,
            min_score=7.5,
            max_score=9.5,
            min_episodes=5,
            max_episodes=15,
            genres=["Action", "Adventure"],
            genres_exclude=["Horror"],
            end_date="2024-06-30",
            include_adult=False,
            limit=25,
            sort_by="popularity",
            sort_order="desc"
        )
        
        jikan_params = JikanMapper.to_jikan_search_params(params)
        
        # Verify all mappings
        assert jikan_params["q"] == "comprehensive test"
        assert jikan_params["status"] == "airing"
        assert jikan_params["type"] == "ova"
        assert jikan_params["min_score"] == 7.5
        assert jikan_params["max_score"] == 9.5
        assert jikan_params["episodes_greater"] == 5
        assert jikan_params["episodes_lesser"] == 15
        assert jikan_params["genres"] == "Action,Adventure"
        assert jikan_params["genres_exclude"] == "Horror"
        assert jikan_params["end_date"] == "2024-06-30"
        assert jikan_params["sfw"] == "true"
        assert jikan_params["limit"] == 25
        assert jikan_params["order_by"] == "popularity"
        assert jikan_params["sort"] == "desc"