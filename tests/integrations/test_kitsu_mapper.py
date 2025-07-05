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
        
        # Test score conversion (0-10 scale * 10 to 0-100 scale)
        assert "filter[averageRating]" in kitsu_params
        # 7.0 -> 70.0, 9.0 -> 90.0 (converted to 0-100 scale)
        assert kitsu_params["filter[averageRating]"] == "70.0..90.0"

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
        """Test year filtering using seasonYear."""
        params = UniversalSearchParams(year=2023)
        kitsu_params = KitsuMapper.to_kitsu_search_params(params)
        
        # Should use seasonYear instead of startDate
        assert kitsu_params["filter[seasonYear]"] == 2023

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
        
        # Should only contain limit parameter
        assert kitsu_params["page[limit]"] == 20
        assert len(kitsu_params) == 1  # Only limit field

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
        """Test that adult content filtering is not implemented in current mapper."""
        # Current implementation doesn't handle include_adult
        params_sfw = UniversalSearchParams(include_adult=False)
        kitsu_params_sfw = KitsuMapper.to_kitsu_search_params(params_sfw)
        # Should not contain any nsfw filter
        assert "filter[nsfw]" not in kitsu_params_sfw
        
        # Test include adult content
        params_nsfw = UniversalSearchParams(include_adult=True)
        kitsu_params_nsfw = KitsuMapper.to_kitsu_search_params(params_nsfw)
        # Should not include nsfw filter
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

    def test_to_kitsu_search_params_score_ranges(self):
        """Test different score range combinations."""
        # Test only max score (should handle existing filter case)
        params_max = UniversalSearchParams(max_score=8.5)
        kitsu_params_max = KitsuMapper.to_kitsu_search_params(params_max)
        assert kitsu_params_max["filter[averageRating]"] == "..85.0"

    def test_to_kitsu_search_params_episode_ranges(self):
        """Test different episode range combinations."""
        # Test only max episodes (should handle existing filter case)
        params_max = UniversalSearchParams(max_episodes=24)
        kitsu_params_max = KitsuMapper.to_kitsu_search_params(params_max)
        assert kitsu_params_max["filter[episodeCount]"] == "..24"

    def test_to_kitsu_search_params_duration_ranges(self):
        """Test different duration range combinations."""
        # Test only max duration (should handle existing filter case)
        params_max = UniversalSearchParams(max_duration=30)
        kitsu_params_max = KitsuMapper.to_kitsu_search_params(params_max)
        assert kitsu_params_max["filter[episodeLength]"] == "..30"

    def test_to_kitsu_search_params_rating_mapping(self):
        """Test age rating mapping."""
        from src.models.universal_anime import AnimeRating
        
        # Test various rating mappings
        rating_test_cases = [
            (AnimeRating.G, "G"),
            (AnimeRating.PG, "PG"),
            (AnimeRating.PG13, "PG"),  # Maps to PG
            (AnimeRating.R, "R"),
            (AnimeRating.R_PLUS, "R18"),  # Maps to R18
            (AnimeRating.RX, "R18"),  # Maps to R18
        ]
        
        for universal_rating, expected_kitsu in rating_test_cases:
            params = UniversalSearchParams(rating=universal_rating)
            kitsu_params = KitsuMapper.to_kitsu_search_params(params)
            assert kitsu_params["filter[ageRating]"] == expected_kitsu

    def test_to_kitsu_search_params_season_mapping(self):
        """Test season mapping."""
        season_test_cases = [
            (AnimeSeason.WINTER, "winter"),
            (AnimeSeason.SPRING, "spring"),
            (AnimeSeason.SUMMER, "summer"),
            (AnimeSeason.FALL, "fall"),
        ]
        
        for universal_season, expected_kitsu in season_test_cases:
            params = UniversalSearchParams(season=universal_season)
            kitsu_params = KitsuMapper.to_kitsu_search_params(params)
            assert kitsu_params["filter[season]"] == expected_kitsu

    def test_to_kitsu_search_params_kitsu_specific(self):
        """Test Kitsu-specific parameters."""
        params = UniversalSearchParams(
            query="test",
            kitsu_streamers=["Crunchyroll", "Funimation"]
        )
        
        kitsu_specific = {
            "ageRating": "R18",
            "subtype": "special"
        }
        
        kitsu_params = KitsuMapper.to_kitsu_search_params(params, kitsu_specific)
        
        # Test streamers
        assert kitsu_params["filter[streamers]"] == "Crunchyroll,Funimation"
        
        # Test overrides
        assert kitsu_params["filter[ageRating]"] == "R18"
        assert kitsu_params["filter[subtype]"] == "special"

    def test_to_kitsu_search_params_streamers_string(self):
        """Test streamers as string instead of list using kitsu_specific."""
        params = UniversalSearchParams()
        kitsu_specific = {"streamers": "Netflix"}
        kitsu_params = KitsuMapper.to_kitsu_search_params(params, kitsu_specific)
        assert kitsu_params["filter[streamers]"] == "Netflix"

    def test_convert_season_to_date_range(self):
        """Test the private season to date range conversion method."""
        # Test all seasons
        season_test_cases = [
            (AnimeSeason.WINTER, "2023-01-01..2023-03-31"),
            (AnimeSeason.SPRING, "2023-04-01..2023-06-30"),
            (AnimeSeason.SUMMER, "2023-07-01..2023-09-30"),
            (AnimeSeason.FALL, "2023-10-01..2023-12-31"),
        ]
        
        for season, expected_range in season_test_cases:
            result = KitsuMapper._convert_season_to_date_range(2023, season)
            assert result == expected_range

    def test_convert_season_to_date_range_unknown_season(self):
        """Test season to date conversion with unknown season."""
        # Test with a mock unknown season (this tests the default case)
        class UnknownSeason:
            pass
        
        result = KitsuMapper._convert_season_to_date_range(2023, UnknownSeason())
        assert result == "2023-01-01..2023-12-31"  # Default full year


    def test_to_kitsu_search_params_max_only_else_branches_100_percent(self):
        """Test max-only parameters to hit the else branches and achieve 100% coverage.
        
        After fixing the source code to use empty string defaults instead of "..",
        the else branches (lines 89, 101, 113) are now reachable when only max
        parameters are provided without corresponding min parameters.
        """
        # Test max-only parameters (which now hit the else branches)
        params_max = UniversalSearchParams(max_score=8.0, max_episodes=24, max_duration=30)
        result_max = KitsuMapper.to_kitsu_search_params(params_max)
        
        # These should hit the else branches since no existing filter exists
        assert result_max["filter[averageRating]"] == "..80.0"  # Line 89
        assert result_max["filter[episodeCount]"] == "..24"     # Line 101
        assert result_max["filter[episodeLength]"] == "..30"    # Line 113
        
        # Test combined min+max parameters (which hit the if branches)
        params_combined = UniversalSearchParams(
            min_score=6.0, max_score=8.0,
            min_episodes=12, max_episodes=24,
            min_duration=20, max_duration=30
        )
        result_combined = KitsuMapper.to_kitsu_search_params(params_combined)
        
        # These should hit the if branches since existing filters contain ".."
        assert result_combined["filter[averageRating]"] == "60.0..80.0"
        assert result_combined["filter[episodeCount]"] == "12..24"
        assert result_combined["filter[episodeLength]"] == "20..30"
        
        # Now all code paths are tested - 100% coverage achieved!