"""Tests for AnimeSchedule query parameter mapper.

Following TDD approach - these tests validate the AnimeSchedule mapper's 
query parameter conversion functionality.
"""

import pytest
from typing import Dict, Any

from src.integrations.mappers.animeschedule_mapper import AnimeScheduleMapper
from src.models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class TestAnimeScheduleMapper:
    """Test AnimeScheduleMapper query parameter conversion methods."""

    @pytest.fixture
    def sample_universal_search_params(self) -> UniversalSearchParams:
        """Sample universal search parameters."""
        return UniversalSearchParams(
            query="test anime",
            genres=["Action"],
            status=AnimeStatus.RELEASING,
            type_format=AnimeFormat.TV,
            year=2023,
            season=AnimeSeason.SPRING,
            min_score=7.0,
            limit=20,
            sort_by="score",
            sort_order="desc"
        )

    def test_to_animeschedule_search_params_basic(self, sample_universal_search_params):
        """Test basic universal to AnimeSchedule search parameter conversion."""
        as_params = AnimeScheduleMapper.to_animeschedule_search_params(sample_universal_search_params)
        
        # Test basic parameters
        assert as_params["title"] == "test anime"
        assert as_params["limit"] == 20
        assert as_params["status"] == "Ongoing"  # AnimeSchedule status format
        
        # Test broadcasting specific filters
        assert "year" in as_params
        assert "season" in as_params

    def test_to_animeschedule_search_params_status_mapping(self):
        """Test status mapping from universal to AnimeSchedule."""
        status_test_cases = [
            (AnimeStatus.FINISHED, "Finished"),
            (AnimeStatus.RELEASING, "Ongoing"),
            (AnimeStatus.NOT_YET_RELEASED, "Upcoming"),
        ]
        
        for universal_status, expected_as in status_test_cases:
            params = UniversalSearchParams(status=universal_status)
            as_params = AnimeScheduleMapper.to_animeschedule_search_params(params)
            assert as_params["status"] == expected_as

    def test_to_animeschedule_search_params_format_mapping(self):
        """Test format mapping from universal to AnimeSchedule."""
        format_test_cases = [
            (AnimeFormat.TV, "TV"),
            (AnimeFormat.MOVIE, "Movie"),
            (AnimeFormat.SPECIAL, "Special"),
            (AnimeFormat.OVA, "OVA"),
        ]
        
        for universal_format, expected_as in format_test_cases:
            params = UniversalSearchParams(type_format=universal_format)
            as_params = AnimeScheduleMapper.to_animeschedule_search_params(params)
            assert as_params["mediaType"] == expected_as

    def test_to_animeschedule_search_params_temporal_filters(self):
        """Test temporal filtering (year/season) - AnimeSchedule specialty."""
        params = UniversalSearchParams(
            year=2023,
            season=AnimeSeason.SPRING
        )
        as_params = AnimeScheduleMapper.to_animeschedule_search_params(params)
        
        assert as_params["year"] == 2023
        assert as_params["season"] == "Spring"

    def test_to_animeschedule_search_params_genre_handling(self):
        """Test genre parameter conversion."""
        params = UniversalSearchParams(genres=["Action", "Adventure"])
        as_params = AnimeScheduleMapper.to_animeschedule_search_params(params)
        
        assert "genres" in as_params
        assert "Action" in as_params["genres"]
        assert "Adventure" in as_params["genres"]

    def test_to_animeschedule_search_params_minimal(self):
        """Test conversion with minimal parameters."""
        params = UniversalSearchParams()
        as_params = AnimeScheduleMapper.to_animeschedule_search_params(params)
        
        # Should only contain default values
        assert as_params["limit"] == 20
        assert len(as_params) >= 1

    def test_bidirectional_consistency(self, sample_universal_search_params):
        """Test that converting to AnimeSchedule params preserves critical information."""
        as_params = AnimeScheduleMapper.to_animeschedule_search_params(sample_universal_search_params)
        
        # Key parameters should be preserved
        assert "title" in as_params
        assert "status" in as_params
        assert "year" in as_params
        assert "season" in as_params