"""Tests for Kitsu JSON:API query parameter mapper.

Following TDD approach - these tests validate the Kitsu mapper's 
query parameter conversion functionality.
"""

import pytest
from typing import Dict, Any

from src.integrations.mappers.kitsu_mapper import KitsuMapper
from src.models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class TestKitsuMapper:
    """Test KitsuMapper query parameter conversion methods."""

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
            min_duration=20,
            max_duration=30,
            include_adult=False,
            limit=20,
            sort_by="score",
            sort_order="desc"
        )

    def test_to_kitsu_search_params_basic(self, sample_universal_search_params):
        """Test basic universal to Kitsu search parameter conversion."""
        kitsu_params = KitsuMapper.to_kitsu_search_params(sample_universal_search_params)
        
        # Test basic parameters
        assert kitsu_params["filter[text]"] == "test anime"
        assert kitsu_params["page[limit]"] == 20
        assert kitsu_params["filter[status]"] == "finished"  # Kitsu-specific status
        assert kitsu_params["filter[subtype]"] == "TV"
        
        # Test score conversion (0-10 to 1-5 stars)
        assert "filter[averageRating]" in kitsu_params
        # 7.0 -> 4.0, 9.0 -> 5.0 (converted to 1-5 star scale)
        
        # Test NSFW filter
        assert kitsu_params["filter[nsfw]"] == "false"

    def test_to_kitsu_search_params_status_mapping(self):
        """Test status mapping from universal to Kitsu."""
        status_test_cases = [
            (AnimeStatus.FINISHED, "finished"),
            (AnimeStatus.RELEASING, "current"),
            (AnimeStatus.NOT_YET_RELEASED, "upcoming"),
        ]
        
        for universal_status, expected_kitsu in status_test_cases:
            params = UniversalSearchParams(status=universal_status)
            kitsu_params = KitsuMapper.to_kitsu_search_params(params)
            assert kitsu_params["filter[status]"] == expected_kitsu

    def test_to_kitsu_search_params_format_mapping(self):
        """Test format mapping from universal to Kitsu."""
        format_test_cases = [
            (AnimeFormat.TV, "TV"),
            (AnimeFormat.MOVIE, "movie"),
            (AnimeFormat.OVA, "OVA"),
            (AnimeFormat.ONA, "ONA"),
            (AnimeFormat.SPECIAL, "special"),
            (AnimeFormat.MUSIC, "music"),
        ]
        
        for universal_format, expected_kitsu in format_test_cases:
            params = UniversalSearchParams(type_format=universal_format)
            kitsu_params = KitsuMapper.to_kitsu_search_params(params)
            assert kitsu_params["filter[subtype]"] == expected_kitsu

    def test_to_kitsu_search_params_score_conversion(self):
        """Test score conversion from 0-10 to 1-5 star scale."""
        params = UniversalSearchParams(
            min_score=6.0,  # Should convert to ~3.5 stars
            max_score=8.0   # Should convert to ~4.5 stars
        )
        kitsu_params = KitsuMapper.to_kitsu_search_params(params)
        
        # Kitsu uses range format like "3.5..4.5"
        assert "filter[averageRating]" in kitsu_params
        rating_filter = kitsu_params["filter[averageRating]"]
        assert ".." in rating_filter

    def test_to_kitsu_search_params_episode_range(self):
        """Test episode count range filtering."""
        params = UniversalSearchParams(
            min_episodes=12,
            max_episodes=24
        )
        kitsu_params = KitsuMapper.to_kitsu_search_params(params)
        
        assert kitsu_params["filter[episodeCount]"] == "12..24"

    def test_to_kitsu_search_params_duration_range(self):
        """Test episode duration range filtering."""
        params = UniversalSearchParams(
            min_duration=20,
            max_duration=25
        )
        kitsu_params = KitsuMapper.to_kitsu_search_params(params)
        
        assert kitsu_params["filter[episodeLength]"] == "20..25"

    def test_to_kitsu_search_params_year_filter(self):
        """Test year filtering conversion to date range."""
        params = UniversalSearchParams(year=2023)
        kitsu_params = KitsuMapper.to_kitsu_search_params(params)
        
        # Should convert to full year range
        assert kitsu_params["filter[startDate]"] == "2023-01-01..2023-12-31"

    def test_to_kitsu_search_params_category_filtering(self):
        """Test category (genre) filtering."""
        params = UniversalSearchParams(genres=["Action", "Adventure"])
        kitsu_params = KitsuMapper.to_kitsu_search_params(params)
        
        assert kitsu_params["filter[categories]"] == "Action,Adventure"

    def test_to_kitsu_search_params_pagination(self):
        """Test pagination parameter conversion."""
        params = UniversalSearchParams(limit=25, offset=50)
        kitsu_params = KitsuMapper.to_kitsu_search_params(params)
        
        assert kitsu_params["page[limit]"] == 25
        assert kitsu_params["page[number]"] == 3  # (50 / 25) + 1

    def test_to_kitsu_search_params_sort_mapping(self):
        """Test sort parameter mapping."""
        sort_test_cases = [
            ("score", "desc", "-averageRating"),
            ("score", "asc", "averageRating"),
            ("popularity", "desc", "-popularityRank"),
            ("title", "asc", "canonicalTitle"),
            ("year", "desc", "-startDate"),
            ("episodes", "asc", "episodeCount"),
            ("duration", "desc", "-episodeLength"),
            ("rank", "asc", "ratingRank"),
        ]
        
        for sort_by, sort_order, expected_sort in sort_test_cases:
            params = UniversalSearchParams(sort_by=sort_by, sort_order=sort_order)
            kitsu_params = KitsuMapper.to_kitsu_search_params(params)
            assert kitsu_params["sort"] == expected_sort

    def test_to_kitsu_search_params_minimal(self):
        """Test conversion with minimal parameters."""
        params = UniversalSearchParams()
        kitsu_params = KitsuMapper.to_kitsu_search_params(params)
        
        # Should only contain default values
        assert kitsu_params["page[limit]"] == 20
        assert kitsu_params["filter[nsfw]"] == "false"  # Default: no adult content
        assert len(kitsu_params) == 2  # Only these two fields

    def test_to_kitsu_search_params_unsupported_ignored(self):
        """Test that unsupported parameters are gracefully ignored."""
        params = UniversalSearchParams(
            query="test",
            status=AnimeStatus.CANCELLED,  # Not supported in Kitsu
            type_format=AnimeFormat.TV_SHORT,  # Not supported in Kitsu
            characters=["Character"],  # Not supported in basic search
        )
        kitsu_params = KitsuMapper.to_kitsu_search_params(params)
        
        # Should include supported params
        assert kitsu_params["filter[text]"] == "test"
        
        # Should not include unsupported params
        assert "filter[status]" not in kitsu_params  # CANCELLED not supported
        assert "filter[subtype]" not in kitsu_params  # TV_SHORT not supported
        assert "characters" not in kitsu_params

    def test_to_kitsu_search_params_adult_content_handling(self):
        """Test adult content filtering."""
        # Test exclude adult content (default)
        params_sfw = UniversalSearchParams(include_adult=False)
        kitsu_params_sfw = KitsuMapper.to_kitsu_search_params(params_sfw)
        assert kitsu_params_sfw["filter[nsfw]"] == "false"
        
        # Test include adult content
        params_nsfw = UniversalSearchParams(include_adult=True)
        kitsu_params_nsfw = KitsuMapper.to_kitsu_search_params(params_nsfw)
        # Should not include nsfw filter when allowing adult content
        assert "filter[nsfw]" not in kitsu_params_nsfw

    def test_to_kitsu_search_params_includes_handling(self):
        """Test include parameter for related data."""
        params = UniversalSearchParams(
            genres=["Action"],
            studios=["Studio Name"]
        )
        kitsu_params = KitsuMapper.to_kitsu_search_params(params)
        
        # Should include related data for comprehensive responses
        assert "include" in kitsu_params
        includes = kitsu_params["include"].split(",")
        assert "categories" in includes  # For genres
        assert "animeProductions.producer" in includes  # For studios

    def test_bidirectional_consistency(self, sample_universal_search_params):
        """Test that converting to Kitsu params preserves critical information."""
        kitsu_params = KitsuMapper.to_kitsu_search_params(sample_universal_search_params)
        
        # Key parameters should be preserved in some form
        assert "filter[text]" in kitsu_params  # query preserved
        assert "filter[status]" in kitsu_params  # status preserved
        assert "filter[subtype]" in kitsu_params  # format preserved
        assert "filter[averageRating]" in kitsu_params  # score preserved
        
        # Parameters should be in Kitsu format
        assert kitsu_params["filter[status]"] == "finished"  # Kitsu status format
        assert kitsu_params["filter[subtype]"] == "TV"  # Kitsu format
        assert ".." in kitsu_params["filter[averageRating]"]  # Kitsu range format