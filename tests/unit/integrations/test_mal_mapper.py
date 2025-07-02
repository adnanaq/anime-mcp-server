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
        
        # Test basic parameters
        assert mal_params["q"] == "test anime"
        assert mal_params["limit"] == 20
        assert mal_params["status"] == "finished_airing"  # MAL-specific status
        assert mal_params["media_type"] == "tv"
        
        # Test season/year handling
        assert mal_params.get("start_date") is not None  # Should derive from year/season
        
        # Test score conversion (MAL uses same 0-10 scale)
        assert mal_params["min_score"] == 7.0
        assert mal_params["max_score"] == 9.0
        
        # Test NSFW filter
        assert mal_params["nsfw"] == "white"  # No adult content

    def test_to_mal_search_params_status_mapping(self):
        """Test status mapping from universal to MAL."""
        status_test_cases = [
            (AnimeStatus.FINISHED, "finished_airing"),
            (AnimeStatus.RELEASING, "currently_airing"),
            (AnimeStatus.NOT_YET_RELEASED, "not_yet_aired"),
            (AnimeStatus.HIATUS, "on_hiatus"),
        ]
        
        for universal_status, expected_mal in status_test_cases:
            params = UniversalSearchParams(status=universal_status)
            mal_params = MALMapper.to_mal_search_params(params)
            assert mal_params["status"] == expected_mal

    def test_to_mal_search_params_format_mapping(self):
        """Test format mapping from universal to MAL."""
        format_test_cases = [
            (AnimeFormat.TV, "tv"),
            (AnimeFormat.MOVIE, "movie"),
            (AnimeFormat.OVA, "ova"),
            (AnimeFormat.ONA, "ona"),
            (AnimeFormat.SPECIAL, "special"),
            (AnimeFormat.MUSIC, "music"),
        ]
        
        for universal_format, expected_mal in format_test_cases:
            params = UniversalSearchParams(type_format=universal_format)
            mal_params = MALMapper.to_mal_search_params(params)
            assert mal_params["media_type"] == expected_mal

    def test_to_mal_search_params_season_conversion(self):
        """Test season and year conversion to start_date."""
        params = UniversalSearchParams(
            year=2023,
            season=AnimeSeason.SPRING
        )
        mal_params = MALMapper.to_mal_search_params(params)
        
        # Should convert spring 2023 to start_date
        assert "start_date" in mal_params
        assert "2023" in mal_params["start_date"]

    def test_to_mal_search_params_minimal(self):
        """Test conversion with minimal parameters."""
        params = UniversalSearchParams()
        mal_params = MALMapper.to_mal_search_params(params)
        
        # Should only contain default values
        assert mal_params["limit"] == 20
        assert mal_params["nsfw"] == "white"  # Default: no adult content
        assert len(mal_params) == 2  # Only these two fields

    def test_to_mal_search_params_unsupported_ignored(self):
        """Test that unsupported parameters are gracefully ignored."""
        params = UniversalSearchParams(
            query="test",
            characters=["Character"],  # Not supported in MAL search
            themes=["Theme"],          # Not directly supported
            staff=["Director"],        # Not supported in basic search
        )
        mal_params = MALMapper.to_mal_search_params(params)
        
        # Should include supported params
        assert mal_params["q"] == "test"
        
        # Should not include unsupported params
        assert "characters" not in mal_params
        assert "themes" not in mal_params  
        assert "staff" not in mal_params

    def test_to_mal_search_params_adult_content_handling(self):
        """Test adult content filtering."""
        # Test exclude adult content (default)
        params_sfw = UniversalSearchParams(include_adult=False)
        mal_params_sfw = MALMapper.to_mal_search_params(params_sfw)
        assert mal_params_sfw["nsfw"] == "white"
        
        # Test include adult content
        params_nsfw = UniversalSearchParams(include_adult=True)
        mal_params_nsfw = MALMapper.to_mal_search_params(params_nsfw)
        # MAL API doesn't have include_adult=true, so should omit nsfw filter
        assert "nsfw" not in mal_params_nsfw

    def test_bidirectional_consistency(self, sample_universal_search_params):
        """Test that converting to MAL params preserves critical information."""
        mal_params = MALMapper.to_mal_search_params(sample_universal_search_params)
        
        # Key parameters should be preserved in some form
        assert "q" in mal_params  # query preserved
        assert "status" in mal_params  # status preserved
        assert "media_type" in mal_params  # format preserved
        assert "min_score" in mal_params  # min_score preserved
        
        # Parameters should be in MAL format
        assert mal_params["status"] == "finished_airing"  # MAL status format
        assert mal_params["media_type"] == "tv"  # MAL format
        assert isinstance(mal_params["min_score"], float)  # MAL score scale

    def test_to_mal_search_params_end_date(self):
        """Test end_date mapping to MAL API."""
        params = UniversalSearchParams(end_date="2023-12-31")
        mal_params = MALMapper.to_mal_search_params(params)
        
        assert mal_params["end_date"] == "2023-12-31"

    def test_to_mal_search_params_min_duration_conversion(self):
        """Test min_duration conversion from minutes to seconds for MAL."""
        # Test 20 minutes -> 1200 seconds
        params = UniversalSearchParams(min_duration=20)
        mal_params = MALMapper.to_mal_search_params(params)
        
        assert mal_params["average_episode_duration"] == 1200
        
        # Test 1 minute -> 60 seconds (minimum valid duration)
        params_one = UniversalSearchParams(min_duration=1)
        mal_params_one = MALMapper.to_mal_search_params(params_one)
        
        assert mal_params_one["average_episode_duration"] == 60

    def test_to_mal_search_params_studios(self):
        """Test studios mapping to comma-separated string for MAL."""
        params = UniversalSearchParams(studios=["Studio Ghibli", "Madhouse", "Bones"])
        mal_params = MALMapper.to_mal_search_params(params)
        
        assert mal_params["studios"] == "Studio Ghibli,Madhouse,Bones"
        
        # Test single studio
        params_single = UniversalSearchParams(studios=["Toei Animation"])
        mal_params_single = MALMapper.to_mal_search_params(params_single)
        
        assert mal_params_single["studios"] == "Toei Animation"

    def test_to_mal_search_params_mal_specific_parameters(self):
        """Test MAL-specific parameter handling."""
        params = UniversalSearchParams(
            mal_broadcast_day="monday",
            mal_rating="pg_13",
            mal_nsfw="gray",
            mal_popularity=100
        )
        mal_params = MALMapper.to_mal_search_params(params)
        
        assert mal_params["broadcast_day"] == "monday"
        assert mal_params["rating"] == "pg_13"
        assert mal_params["nsfw"] == "gray"
        assert mal_params["popularity"] == 100

    def test_to_mal_search_params_platform_specific_override(self):
        """Test that platform-specific parameters can be passed via mal_specific dict."""
        universal_params = UniversalSearchParams(query="test")
        mal_specific = {
            "broadcast_day": "friday",
            "rating": "r",
            "num_list_users": 1000
        }
        
        mal_params = MALMapper.to_mal_search_params(universal_params, mal_specific)
        
        assert mal_params["q"] == "test"
        assert mal_params["broadcast_day"] == "friday"
        assert mal_params["rating"] == "r"
        assert mal_params["num_list_users"] == 1000

    def test_to_mal_search_params_comprehensive_mapping(self):
        """Test comprehensive parameter mapping including new features."""
        params = UniversalSearchParams(
            query="comprehensive test",
            status=AnimeStatus.RELEASING,
            type_format=AnimeFormat.MOVIE,
            min_score=8.0,
            max_score=10.0,
            min_episodes=1,
            max_episodes=3,
            end_date="2024-06-30",
            min_duration=90,  # 90 minutes
            studios=["Studio Ghibli", "Pixar"],
            include_adult=False,
            limit=50,
            sort_by="popularity",
            sort_order="asc"
        )
        
        mal_params = MALMapper.to_mal_search_params(params)
        
        # Verify all mappings
        assert mal_params["q"] == "comprehensive test"
        assert mal_params["status"] == "currently_airing"
        assert mal_params["media_type"] == "movie"
        assert mal_params["min_score"] == 8.0
        assert mal_params["max_score"] == 10.0
        assert mal_params["min_episodes"] == 1
        assert mal_params["max_episodes"] == 3
        assert mal_params["end_date"] == "2024-06-30"
        assert mal_params["average_episode_duration"] == 5400  # 90 * 60 seconds
        assert mal_params["studios"] == "Studio Ghibli,Pixar"
        assert mal_params["nsfw"] == "white"
        assert mal_params["limit"] == 50
        assert mal_params["sort"] == "popularity"
        assert mal_params["order"] == "asc"