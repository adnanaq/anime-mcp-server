"""Tests for AniDB query parameter mapper.

Following TDD approach - these tests validate the AniDB mapper's 
query parameter conversion functionality.
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

    def test_to_anidb_search_params_basic(self, sample_universal_search_params):
        """Test basic universal to AniDB search parameter conversion."""
        anidb_params = AniDBMapper.to_anidb_search_params(sample_universal_search_params)
        
        # Test basic parameters
        assert anidb_params["query"] == "test anime"
        assert anidb_params["limit"] == 20
        assert anidb_params["type"] == "TV Series"  # AniDB-specific format
        
        # Test score conversion (AniDB uses 0-10 scale like universal)
        assert anidb_params["min_rating"] == 7.0
        assert anidb_params["max_rating"] == 9.0
        
        # Test adult content filter
        assert anidb_params["restricted"] == "false"

    def test_to_anidb_search_params_status_derivation(self):
        """Test status derivation from dates (AniDB doesn't have direct status field)."""
        # AniDB derives status from start/end dates
        params = UniversalSearchParams(status=AnimeStatus.RELEASING)
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        # Should set date filters to derive "airing" status
        assert "start_date" in anidb_params or "end_date" in anidb_params

    def test_to_anidb_search_params_format_mapping(self):
        """Test format mapping from universal to AniDB."""
        format_test_cases = [
            (AnimeFormat.TV, "TV Series"),
            (AnimeFormat.MOVIE, "Movie"),
            (AnimeFormat.SPECIAL, "TV Special"),
            (AnimeFormat.OVA, "OVA"),
            (AnimeFormat.ONA, "Web"),  # AniDB calls ONA "Web"
        ]
        
        for universal_format, expected_anidb in format_test_cases:
            params = UniversalSearchParams(type_format=universal_format)
            anidb_params = AniDBMapper.to_anidb_search_params(params)
            assert anidb_params["type"] == expected_anidb

    def test_to_anidb_search_params_episode_filtering(self):
        """Test episode count filtering."""
        params = UniversalSearchParams(
            min_episodes=12,
            max_episodes=26
        )
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        assert anidb_params["min_episodes"] == 12
        assert anidb_params["max_episodes"] == 26

    def test_to_anidb_search_params_year_filter(self):
        """Test year filtering with AniDB date format."""
        params = UniversalSearchParams(year=2023)
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        # AniDB uses start date for year filtering
        assert "start_date" in anidb_params
        assert "2023" in anidb_params["start_date"]

    def test_to_anidb_search_params_genre_handling(self):
        """Test genre/tag parameter conversion."""
        params = UniversalSearchParams(
            genres=["Action", "Adventure"],
            themes=["School", "Magic"]
        )
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        # AniDB combines genres and themes into tags
        assert "tags" in anidb_params
        tags = anidb_params["tags"]
        assert "Action" in tags
        assert "Adventure" in tags
        assert "School" in tags
        assert "Magic" in tags

    def test_to_anidb_search_params_sort_mapping(self):
        """Test sort parameter mapping."""
        sort_test_cases = [
            ("score", "desc", "rating", "desc"),
            ("score", "asc", "rating", "asc"),
            ("year", "desc", "startdate", "desc"),
            ("title", "asc", "title", "asc"),
            ("episodes", "desc", "episodes", "desc"),
        ]
        
        for sort_by, sort_order, expected_field, expected_order in sort_test_cases:
            params = UniversalSearchParams(sort_by=sort_by, sort_order=sort_order)
            anidb_params = AniDBMapper.to_anidb_search_params(params)
            assert anidb_params["sort_by"] == expected_field
            assert anidb_params["sort_order"] == expected_order

    def test_to_anidb_search_params_minimal(self):
        """Test conversion with minimal parameters."""
        params = UniversalSearchParams()
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        # Should only contain default values
        assert anidb_params["limit"] == 20
        assert anidb_params["restricted"] == "false"  # Default: no adult content
        assert len(anidb_params) == 2  # Only these two fields

    def test_to_anidb_search_params_unsupported_ignored(self):
        """Test that unsupported parameters are gracefully ignored."""
        params = UniversalSearchParams(
            query="test",
            characters=["Character"],  # Not supported in AniDB search
            staff=["Director"],        # Not supported in basic search
            studios=["Studio"],        # Mapped to producers/creators in AniDB
        )
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        # Should include supported params
        assert anidb_params["query"] == "test"
        
        # Should not include unsupported params directly
        assert "characters" not in anidb_params
        assert "staff" not in anidb_params

    def test_to_anidb_search_params_adult_content_handling(self):
        """Test adult content filtering."""
        # Test exclude adult content (default)
        params_sfw = UniversalSearchParams(include_adult=False)
        anidb_params_sfw = AniDBMapper.to_anidb_search_params(params_sfw)
        assert anidb_params_sfw["restricted"] == "false"
        
        # Test include adult content
        params_nsfw = UniversalSearchParams(include_adult=True)
        anidb_params_nsfw = AniDBMapper.to_anidb_search_params(params_nsfw)
        # Should not include restricted filter when allowing adult content
        assert "restricted" not in anidb_params_nsfw

    def test_to_anidb_search_params_tag_combination(self):
        """Test combination of genres and themes into AniDB tags."""
        params = UniversalSearchParams(
            genres=["Action", "Comedy"],
            themes=["School", "Romance"],
            demographics=["Shounen"]
        )
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        # All should be combined into tags array
        assert "tags" in anidb_params
        all_tags = anidb_params["tags"]
        expected_tags = ["Action", "Comedy", "School", "Romance", "Shounen"]
        for tag in expected_tags:
            assert tag in all_tags

    def test_to_anidb_search_params_date_range_handling(self):
        """Test date range parameter conversion."""
        params = UniversalSearchParams(
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        assert anidb_params["start_date"] == "2023-01-01"
        assert anidb_params["end_date"] == "2023-12-31"

    def test_to_anidb_search_params_pagination(self):
        """Test pagination parameter conversion."""
        params = UniversalSearchParams(limit=25, offset=50)
        anidb_params = AniDBMapper.to_anidb_search_params(params)
        
        assert anidb_params["limit"] == 25
        assert anidb_params["offset"] == 50

    def test_bidirectional_consistency(self, sample_universal_search_params):
        """Test that converting to AniDB params preserves critical information."""
        anidb_params = AniDBMapper.to_anidb_search_params(sample_universal_search_params)
        
        # Key parameters should be preserved in some form
        assert "query" in anidb_params  # query preserved
        assert "type" in anidb_params  # format preserved
        assert "min_rating" in anidb_params  # min_score preserved
        assert "tags" in anidb_params  # genres preserved in tags
        
        # Parameters should be in AniDB format
        assert anidb_params["type"] == "TV Series"  # AniDB format
        assert isinstance(anidb_params["min_rating"], float)  # AniDB rating scale
        assert isinstance(anidb_params["tags"], list)  # AniDB tag format