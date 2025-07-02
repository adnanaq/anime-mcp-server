"""Tests for Anime-Planet query parameter mapper.

Following TDD approach - these tests validate the Anime-Planet mapper's 
query parameter conversion functionality.
"""

import pytest
from typing import Dict, Any

from src.integrations.mappers.animeplanet_mapper import AnimePlanetMapper
from src.models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class TestAnimePlanetMapper:
    """Test AnimePlanetMapper query parameter conversion methods."""

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

    def test_to_animeplanet_search_params_basic(self, sample_universal_search_params):
        """Test basic universal to Anime-Planet search parameter conversion."""
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(sample_universal_search_params)
        
        # Test basic parameters
        assert ap_params["name"] == "test anime"  # Anime-Planet uses "name" for search
        assert ap_params["limit"] == 20
        assert ap_params["type"] == "TVSeries"  # Anime-Planet JSON-LD format
        
        # Test score conversion (Anime-Planet likely uses 0-5 or 0-10 scale)
        assert "min_rating" in ap_params
        assert "max_rating" in ap_params

    def test_to_animeplanet_search_params_status_derivation(self):
        """Test status derivation from date patterns (Anime-Planet shows status via date format)."""
        # Anime-Planet shows status through date patterns like "2023 - 2024" vs "2023 - ?"
        params = UniversalSearchParams(status=AnimeStatus.RELEASING)
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
        
        # Should set filters to derive "ongoing" status
        assert "date_pattern" in ap_params or "status_filter" in ap_params

    def test_to_animeplanet_search_params_format_mapping(self):
        """Test format mapping from universal to Anime-Planet."""
        format_test_cases = [
            (AnimeFormat.TV, "TVSeries"),
            (AnimeFormat.MOVIE, "Movie"),
            # Note: Anime-Planet may not support all formats in search
        ]
        
        for universal_format, expected_ap in format_test_cases:
            params = UniversalSearchParams(type_format=universal_format)
            ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
            if "type" in ap_params:  # Only if format is supported
                assert ap_params["type"] == expected_ap

    def test_to_animeplanet_search_params_genre_filtering(self):
        """Test genre parameter conversion."""
        params = UniversalSearchParams(
            genres=["Action", "Adventure"],
            themes=["School"]
        )
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
        
        # Anime-Planet combines genres and themes into tags
        assert "tags" in ap_params
        tags = ap_params["tags"]
        assert "Action" in tags
        assert "Adventure" in tags
        assert "School" in tags

    def test_to_animeplanet_search_params_year_filter(self):
        """Test year filtering."""
        params = UniversalSearchParams(year=2023)
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
        
        assert "year" in ap_params
        assert ap_params["year"] == 2023

    def test_to_animeplanet_search_params_season_filter(self):
        """Test season filtering."""
        params = UniversalSearchParams(
            year=2023,
            season=AnimeSeason.SPRING
        )
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
        
        # Anime-Planet shows season info
        assert "season" in ap_params
        assert ap_params["season"] == "Spring"

    def test_to_animeplanet_search_params_episode_filtering(self):
        """Test episode count filtering."""
        params = UniversalSearchParams(
            min_episodes=12,
            max_episodes=26
        )
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
        
        assert "min_episodes" in ap_params
        assert "max_episodes" in ap_params
        assert ap_params["min_episodes"] == 12
        assert ap_params["max_episodes"] == 26

    def test_to_animeplanet_search_params_sort_mapping(self):
        """Test sort parameter mapping."""
        sort_test_cases = [
            ("score", "desc", "rating", "desc"),
            ("title", "asc", "title", "asc"),
            ("year", "desc", "year", "desc"),
        ]
        
        for sort_by, sort_order, expected_field, expected_order in sort_test_cases:
            params = UniversalSearchParams(sort_by=sort_by, sort_order=sort_order)
            ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
            if "sort_by" in ap_params:  # Only if sorting is supported
                assert ap_params["sort_by"] == expected_field
                assert ap_params["sort_order"] == expected_order

    def test_to_animeplanet_search_params_minimal(self):
        """Test conversion with minimal parameters."""
        params = UniversalSearchParams()
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
        
        # Should only contain default values
        assert ap_params["limit"] == 20
        assert len(ap_params) >= 1  # At least limit

    def test_to_animeplanet_search_params_unsupported_ignored(self):
        """Test that unsupported parameters are gracefully ignored."""
        params = UniversalSearchParams(
            query="test",
            characters=["Character"],  # Not supported in Anime-Planet search
            staff=["Director"],        # May be supported but not in basic search
            demographics=["Shounen"],  # May not be filterable
        )
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
        
        # Should include supported params
        assert ap_params["name"] == "test"
        
        # Unsupported params should be gracefully ignored
        assert "characters" not in ap_params

    def test_to_animeplanet_search_params_score_conversion(self):
        """Test score conversion to Anime-Planet rating scale."""
        params = UniversalSearchParams(
            min_score=7.0,
            max_score=9.0
        )
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
        
        # Should convert scores appropriately
        assert "min_rating" in ap_params
        assert "max_rating" in ap_params
        # Actual conversion depends on Anime-Planet's rating scale

    def test_to_animeplanet_search_params_pagination(self):
        """Test pagination parameter conversion."""
        params = UniversalSearchParams(limit=25, offset=50)
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
        
        assert ap_params["limit"] == 25
        if "offset" in ap_params:
            assert ap_params["offset"] == 50
        elif "page" in ap_params:
            # May use page-based pagination
            assert ap_params["page"] == 3  # (50 / 25) + 1

    def test_to_animeplanet_search_params_studio_filtering(self):
        """Test studio filtering (Anime-Planet has studio info)."""
        params = UniversalSearchParams(studios=["Studio Ghibli", "Mappa"])
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(params)
        
        # Studios should be included if supported
        if "studios" in ap_params:
            studios = ap_params["studios"]
            assert "Studio Ghibli" in studios
            assert "Mappa" in studios

    def test_bidirectional_consistency(self, sample_universal_search_params):
        """Test that converting to Anime-Planet params preserves critical information."""
        ap_params = AnimePlanetMapper.to_animeplanet_search_params(sample_universal_search_params)
        
        # Key parameters should be preserved in some form
        assert "name" in ap_params  # query preserved
        assert "tags" in ap_params or "genres" in ap_params  # genres preserved
        
        # Parameters should be in Anime-Planet format
        assert ap_params["name"] == "test anime"
        if "tags" in ap_params:
            assert isinstance(ap_params["tags"], list)