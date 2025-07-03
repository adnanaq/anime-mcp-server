"""Tests for AniList GraphQL query parameter mapper.

Following TDD approach - these tests validate the AniList mapper's 
query parameter conversion functionality.
"""

import pytest
from typing import Dict, Any

from src.integrations.mappers.anilist_mapper import AniListMapper
from src.models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class TestAniListMapper:
    """Test AniListMapper query parameter conversion methods."""


    @pytest.fixture
    def sample_universal_search_params(self) -> UniversalSearchParams:
        """Sample universal search parameters."""
        return UniversalSearchParams(
            query="test anime",
            genres=["Action"],
            genres_exclude=["Horror"],
            status=AnimeStatus.RELEASING,
            type_format=AnimeFormat.TV,
            year=2023,
            season=AnimeSeason.SPRING,
            min_score=7.0,
            max_score=9.0,
            min_episodes=10,
            max_episodes=30,
            min_duration=20,
            max_duration=30,
            themes=["School"],
            include_adult=False,
            limit=20,
            sort_by="score",
            sort_order="desc"
        )








    def test_to_anilist_search_params_basic(self, sample_universal_search_params):
        """Test basic universal to AniList search parameter conversion."""
        anilist_params = AniListMapper.to_anilist_search_params(sample_universal_search_params)
        
        # Test basic parameters
        assert anilist_params["search"] == "test anime"
        assert anilist_params["genre_in"] == ["Action"]
        assert anilist_params["genre_not_in"] == ["Horror"]
        assert anilist_params["status"] == "RELEASING"
        assert anilist_params["format"] == "TV"
        assert anilist_params["seasonYear"] == 2023
        assert anilist_params["season"] == "SPRING"
        assert anilist_params["perPage"] == 20
        
        # Test score conversion (0-10 to 0-100)
        assert anilist_params["averageScore_greater"] == 70  # 7.0 * 10
        assert anilist_params["averageScore_lesser"] == 90   # 9.0 * 10
        
        # Test episode filters
        assert anilist_params["episodes_greater"] == 10
        assert anilist_params["episodes_lesser"] == 30
        
        # Test duration filters
        assert anilist_params["duration_greater"] == 20
        assert anilist_params["duration_lesser"] == 30
        
        # Test tag filters
        assert anilist_params["tag_in"] == ["School"]
        
        # Test adult content filter
        assert anilist_params["isAdult"] == False
        
        # Test sort conversion
        assert anilist_params["sort"] == ["SCORE_DESC"]

    def test_to_anilist_search_params_sort_mapping(self):
        """Test sort parameter mapping."""
        sort_test_cases = [
            ("score", "desc", ["SCORE_DESC"]),
            ("score", "asc", ["SCORE_ASC"]),
            ("popularity", "desc", ["POPULARITY_DESC"]),
            ("popularity", "asc", ["POPULARITY_ASC"]),
            ("title", "asc", ["TITLE_ROMAJI_ASC"]),
            ("title", "desc", ["TITLE_ROMAJI_DESC"]),
            ("year", "desc", ["START_DATE_DESC"]),
            ("year", "asc", ["START_DATE_ASC"]),
            ("episodes", "desc", ["EPISODES_DESC"]),
            ("duration", "asc", ["DURATION_ASC"]),
        ]
        
        for sort_by, sort_order, expected_sort in sort_test_cases:
            params = UniversalSearchParams(sort_by=sort_by, sort_order=sort_order)
            anilist_params = AniListMapper.to_anilist_search_params(params)
            assert anilist_params.get("sort") == expected_sort

    def test_to_anilist_search_params_status_mapping(self):
        """Test status mapping from universal to AniList."""
        status_test_cases = [
            (AnimeStatus.FINISHED, "FINISHED"),
            (AnimeStatus.RELEASING, "RELEASING"),
            (AnimeStatus.NOT_YET_RELEASED, "NOT_YET_RELEASED"),
            (AnimeStatus.CANCELLED, "CANCELLED"),
            (AnimeStatus.HIATUS, "HIATUS"),
        ]
        
        for universal_status, expected_anilist in status_test_cases:
            params = UniversalSearchParams(status=universal_status)
            anilist_params = AniListMapper.to_anilist_search_params(params)
            assert anilist_params["status"] == expected_anilist

    def test_to_anilist_search_params_format_mapping(self):
        """Test format mapping from universal to AniList."""
        format_test_cases = [
            (AnimeFormat.TV, "TV"),
            (AnimeFormat.MOVIE, "MOVIE"),
            (AnimeFormat.SPECIAL, "SPECIAL"),
            (AnimeFormat.OVA, "OVA"),
            (AnimeFormat.ONA, "ONA"),
            (AnimeFormat.MUSIC, "MUSIC"),
        ]
        
        for universal_format, expected_anilist in format_test_cases:
            params = UniversalSearchParams(type_format=universal_format)
            anilist_params = AniListMapper.to_anilist_search_params(params)
            assert anilist_params["format"] == expected_anilist

    def test_to_anilist_search_params_date_conversion(self):
        """Test date string to FuzzyDateInt conversion."""
        params = UniversalSearchParams(
            start_date="2023-04-15",
            end_date="2023-12-31"
        )
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        assert anilist_params["startDate_greater"] == 20230415
        assert anilist_params["endDate"] == 20231231

    def test_to_anilist_search_params_minimal(self):
        """Test conversion with minimal parameters."""
        params = UniversalSearchParams()
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        # Should only contain perPage and isAdult (default values)
        assert anilist_params["perPage"] == 20
        assert anilist_params["isAdult"] == False
        assert len(anilist_params) == 2  # Only these two fields

    def test_to_anilist_search_params_unsupported_ignored(self):
        """Test that unsupported parameters are gracefully ignored."""
        params = UniversalSearchParams(
            query="test",
            characters=["Naruto"],  # Not supported in AniList search
            producers=["Studio"],   # Would need separate handling
            staff=["Director"],     # Not supported in basic search
        )
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        # Should include supported params
        assert anilist_params["search"] == "test"
        
        # Should not include unsupported params
        assert "characters" not in anilist_params
        assert "producers" not in anilist_params
        assert "staff" not in anilist_params


    def test_bidirectional_consistency(self, sample_universal_search_params):
        """Test that converting to AniList params doesn't lose critical information."""
        anilist_params = AniListMapper.to_anilist_search_params(sample_universal_search_params)
        
        # Key parameters should be preserved in some form
        assert "search" in anilist_params  # query preserved
        assert "status" in anilist_params  # status preserved
        assert "format" in anilist_params  # format preserved
        assert "seasonYear" in anilist_params  # year preserved
        assert "averageScore_greater" in anilist_params  # min_score preserved
        
        # Parameters should be in AniList format
        assert anilist_params["status"] == "RELEASING"  # AniList enum
        assert anilist_params["format"] == "TV"  # AniList enum
        assert isinstance(anilist_params["averageScore_greater"], int)  # AniList score scale

    def test_to_anilist_search_params_min_popularity(self):
        """Test min_popularity mapping to popularity_greater."""
        params = UniversalSearchParams(min_popularity=1000)
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        assert anilist_params["popularity_greater"] == 1000

    def test_to_anilist_search_params_end_date_fixed(self):
        """Test that end_date maps to endDate (not endDate_lesser)."""
        params = UniversalSearchParams(end_date="2023-12-31")
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        assert anilist_params["endDate"] == 20231231
        assert "endDate_lesser" not in anilist_params

    def test_to_anilist_search_params_anilist_specific_basic(self):
        """Test AniList-specific basic parameters."""
        params = UniversalSearchParams(
            anilist_id=123456,
            anilist_id_mal=789,
            anilist_season="SUMMER",
            anilist_season_year=2024,
            anilist_episodes=12,
            anilist_duration=24,
            anilist_is_adult=True
        )
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        assert anilist_params["id"] == 123456
        assert anilist_params["idMal"] == 789
        assert anilist_params["season"] == "SUMMER"
        assert anilist_params["seasonYear"] == 2024
        assert anilist_params["episodes"] == 12
        assert anilist_params["duration"] == 24
        assert anilist_params["isAdult"] == True

    def test_to_anilist_search_params_anilist_specific_arrays(self):
        """Test AniList-specific array parameters."""
        params = UniversalSearchParams(
            anilist_id_in=[100, 200, 300],
            anilist_id_not_in=[400, 500],
            anilist_format_in=["TV", "MOVIE"],
            anilist_status_in=["FINISHED", "RELEASING"],
            anilist_genre_in=["Action", "Adventure"],
            anilist_tag_in=["School", "Magic"]
        )
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        assert anilist_params["id_in"] == [100, 200, 300]
        assert anilist_params["id_not_in"] == [400, 500]
        assert anilist_params["format_in"] == ["TV", "MOVIE"]
        assert anilist_params["status_in"] == ["FINISHED", "RELEASING"]
        assert anilist_params["genre_in"] == ["Action", "Adventure"]
        assert anilist_params["tag_in"] == ["School", "Magic"]

    def test_to_anilist_search_params_anilist_specific_ranges(self):
        """Test AniList-specific range parameters."""
        params = UniversalSearchParams(
            anilist_start_date_greater=20230101,
            anilist_end_date_lesser=20231231,
            anilist_episodes_greater=10,
            anilist_episodes_lesser=50,
            anilist_duration_greater=20,
            anilist_duration_lesser=30,
            anilist_average_score_greater=75,
            anilist_popularity_greater=5000
        )
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        assert anilist_params["startDate_greater"] == 20230101
        assert anilist_params["endDate_lesser"] == 20231231
        assert anilist_params["episodes_greater"] == 10
        assert anilist_params["episodes_lesser"] == 50
        assert anilist_params["duration_greater"] == 20
        assert anilist_params["duration_lesser"] == 30
        assert anilist_params["averageScore_greater"] == 75
        assert anilist_params["popularity_greater"] == 5000

    def test_to_anilist_search_params_anilist_specific_override(self):
        """Test that anilist_specific dict can override universal parameters."""
        universal_params = UniversalSearchParams(query="test")
        anilist_specific = {
            "id": 999999,
            "isAdult": True,
            "genre": "Mecha",
            "tag": "Robot",
            "minimumTagRank": 50,
            "source": "MANGA"
        }
        
        anilist_params = AniListMapper.to_anilist_search_params(universal_params, anilist_specific)
        
        assert anilist_params["search"] == "test"
        assert anilist_params["id"] == 999999
        assert anilist_params["isAdult"] == True
        assert anilist_params["genre"] == "Mecha"
        assert anilist_params["tag"] == "Robot"
        assert anilist_params["minimumTagRank"] == 50
        assert anilist_params["source"] == "MANGA"

    def test_to_anilist_search_params_comprehensive_69_params(self):
        """Test comprehensive AniList parameter mapping with many AniList-specific features."""
        params = UniversalSearchParams(
            query="comprehensive test",
            status=AnimeStatus.FINISHED,
            type_format=AnimeFormat.MOVIE,
            min_score=8.5,
            max_score=10.0,
            min_episodes=1,
            max_episodes=3,
            min_duration=90,
            max_duration=180,
            genres=["Action", "Drama"],
            themes=["War", "Military"],
            end_date="2024-06-30",
            min_popularity=10000,
            include_adult=True,
            limit=50,
            sort_by="score",
            sort_order="desc",
            # AniList-specific parameters
            anilist_id=123456,
            anilist_id_mal=789012,
            anilist_season="FALL",
            anilist_season_year=2023,
            anilist_country_of_origin="JP",
            anilist_source="LIGHT_NOVEL",
            anilist_licensed_by="Crunchyroll",
            anilist_tag_category="Setting",
            anilist_minimum_tag_rank=80,
            anilist_average_score=85,
            anilist_popularity=25000
        )
        
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        # Verify universal mappings
        assert anilist_params["search"] == "comprehensive test"
        assert anilist_params["status"] == "FINISHED"
        assert anilist_params["format"] == "MOVIE"
        assert anilist_params["averageScore_greater"] == 85  # 8.5 * 10
        assert anilist_params["averageScore_lesser"] == 100  # 10.0 * 10
        assert anilist_params["episodes_greater"] == 1
        assert anilist_params["episodes_lesser"] == 3
        assert anilist_params["duration_greater"] == 90
        assert anilist_params["duration_lesser"] == 180
        assert anilist_params["genre_in"] == ["Action", "Drama"]
        assert anilist_params["tag_in"] == ["War", "Military"]
        assert anilist_params["endDate"] == 20240630
        assert anilist_params["popularity_greater"] == 10000
        # Note: AniList mapper only sets isAdult=False when include_adult=False
        # When include_adult=True, it omits the isAdult parameter to include all content
        assert anilist_params["perPage"] == 50
        assert anilist_params["sort"] == ["SCORE_DESC"]
        
        # Verify AniList-specific mappings
        assert anilist_params["id"] == 123456
        assert anilist_params["idMal"] == 789012
        assert anilist_params["season"] == "FALL"
        assert anilist_params["seasonYear"] == 2023
        assert anilist_params["countryOfOrigin"] == "JP"
        assert anilist_params["source"] == "LIGHT_NOVEL"
        assert anilist_params["licensedBy"] == "Crunchyroll"
        assert anilist_params["tagCategory"] == "Setting"
        assert anilist_params["minimumTagRank"] == 80
        assert anilist_params["averageScore"] == 85
        assert anilist_params["popularity"] == 25000

    def test_to_anilist_search_params_pagination(self):
        """Test pagination parameter conversion."""
        params = UniversalSearchParams(limit=25, offset=100)
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        assert anilist_params["perPage"] == 25
        assert anilist_params["page"] == 5  # (100 / 25) + 1 = 5

    def test_to_anilist_search_params_anilist_status_override(self):
        """Test that anilist_specific status overrides universal status."""
        params = UniversalSearchParams(status=AnimeStatus.FINISHED)
        anilist_specific = {"status": "CANCELLED"}
        anilist_params = AniListMapper.to_anilist_search_params(params, anilist_specific)
        
        # anilist_specific should override universal
        assert anilist_params["status"] == "CANCELLED"

    def test_to_anilist_search_params_themes_exclude(self):
        """Test themes_exclude mapping to tag_not_in."""
        params = UniversalSearchParams(themes_exclude=["Ecchi", "Gore"])
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        assert anilist_params["tag_not_in"] == ["Ecchi", "Gore"]

    def test_to_anilist_search_params_invalid_dates(self):
        """Test handling of invalid date formats."""
        params = UniversalSearchParams(
            start_date="invalid-date",
            end_date="2023"  # incomplete date
        )
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        # Invalid dates should be ignored
        assert "startDate_greater" not in anilist_params
        assert "endDate" not in anilist_params

    def test_to_anilist_search_params_date_exception_handling(self):
        """Test date parsing exception handling."""
        # Test ValueError in date parsing
        params = UniversalSearchParams(
            start_date="not-a-number-date",  # Will cause ValueError in int()
            end_date="also-not-a-number"
        )
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        # Should gracefully handle ValueError/IndexError exceptions
        assert "startDate_greater" not in anilist_params
        assert "endDate" not in anilist_params

    def test_to_anilist_search_params_date_index_error_handling(self):
        """Test date parsing IndexError handling."""
        # Test IndexError in date parsing (insufficient parts)
        params = UniversalSearchParams(
            start_date="2023-",  # Will cause IndexError when accessing parts[2]
            end_date="2023-12"   # Will cause IndexError when accessing parts[2]
        )
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        # Should gracefully handle IndexError exceptions
        assert "startDate_greater" not in anilist_params
        assert "endDate" not in anilist_params

    def test_to_anilist_search_params_sort_no_suffix(self):
        """Test sort mapping with only sort_by (no sort_order)."""
        params = UniversalSearchParams(sort_by="score")  # No sort_order provided
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        # When sort_order is None, it should use the base sort without suffix
        assert anilist_params["sort"] == ["SCORE"]

    def test_to_anilist_search_params_comprehensive_anilist_specific(self):
        """Test comprehensive AniList-specific parameter coverage."""
        params = UniversalSearchParams(
            # Basic AniList-specific parameters
            anilist_end_date=20231231,
            anilist_format="SPECIAL",
            anilist_start_date=20230101,
            anilist_chapters=50,
            anilist_volumes=10,
            anilist_genre="Mecha",
            anilist_tag="Robot",
            anilist_on_list=True,
            anilist_licensed_by_id=123,
            anilist_is_licensed=True,
            # Negation filters
            anilist_id_not=999,
            anilist_id_mal_not=888,
            anilist_format_not="MUSIC",
            anilist_average_score_not=50,
            anilist_popularity_not=1000,
            # Array inclusion filters
            anilist_id_mal_in=[100, 200],
            anilist_id_mal_not_in=[300, 400],
            anilist_format_not_in=["OVA", "ONA"],
            anilist_status_not_in=["CANCELLED"],
            anilist_genre_not_in=["Horror"],
            anilist_tag_not_in=["Gore"],
            anilist_tag_category_in=["Setting"],
            anilist_tag_category_not_in=["Theme"],
            anilist_licensed_by_in=["Netflix"],
            anilist_licensed_by_id_in=[1, 2, 3],
            anilist_source_in=["MANGA"],
            anilist_licensed_by_not_in=["Hulu"],
            anilist_source_not_in=["NOVEL"],
            anilist_format_range=["TV", "MOVIE"],
            # Range filters
            anilist_start_date_lesser=20231231,
            anilist_start_date_like="2023%",
            anilist_end_date_greater=20230101,
            anilist_end_date_like="2023%",
            anilist_chapters_greater=10,
            anilist_chapters_lesser=100,
            anilist_volumes_greater=5,
            anilist_volumes_lesser=20,
            anilist_average_score_lesser=90,
            anilist_popularity_lesser=50000,
            # Special sorting
            anilist_sort=["UPDATED_AT_DESC"]
        )
        
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        # Basic AniList-specific parameters
        assert anilist_params["endDate"] == 20231231
        assert anilist_params["format"] == "SPECIAL"
        assert anilist_params["startDate"] == 20230101
        assert anilist_params["chapters"] == 50
        assert anilist_params["volumes"] == 10
        assert anilist_params["genre"] == "Mecha"
        assert anilist_params["tag"] == "Robot"
        assert anilist_params["onList"] == True
        assert anilist_params["licensedById"] == 123
        assert anilist_params["isLicensed"] == True
        
        # Negation filters
        assert anilist_params["id_not"] == 999
        assert anilist_params["idMal_not"] == 888
        assert anilist_params["format_not"] == "MUSIC"
        assert anilist_params["averageScore_not"] == 50
        assert anilist_params["popularity_not"] == 1000
        
        # Array inclusion filters
        assert anilist_params["idMal_in"] == [100, 200]
        assert anilist_params["idMal_not_in"] == [300, 400]
        assert anilist_params["format_not_in"] == ["OVA", "ONA"]
        assert anilist_params["status_not_in"] == ["CANCELLED"]
        assert anilist_params["genre_not_in"] == ["Horror"]
        assert anilist_params["tag_not_in"] == ["Gore"]
        assert anilist_params["tagCategory_in"] == ["Setting"]
        assert anilist_params["tagCategory_not_in"] == ["Theme"]
        assert anilist_params["licensedBy_in"] == ["Netflix"]
        assert anilist_params["licensedById_in"] == [1, 2, 3]
        assert anilist_params["source_in"] == ["MANGA"]
        assert anilist_params["licensedBy_not_in"] == ["Hulu"]
        assert anilist_params["source_not_in"] == ["NOVEL"]
        assert anilist_params["format_range"] == ["TV", "MOVIE"]
        
        # Range filters
        assert anilist_params["startDate_lesser"] == 20231231
        assert anilist_params["startDate_like"] == "2023%"
        assert anilist_params["endDate_greater"] == 20230101
        assert anilist_params["endDate_like"] == "2023%"
        assert anilist_params["chapters_greater"] == 10
        assert anilist_params["chapters_lesser"] == 100
        assert anilist_params["volumes_greater"] == 5
        assert anilist_params["volumes_lesser"] == 20
        assert anilist_params["averageScore_lesser"] == 90
        assert anilist_params["popularity_lesser"] == 50000
        
        # Special sorting
        assert anilist_params["sort"] == ["UPDATED_AT_DESC"]

    def test_to_anilist_search_params_status_not_mapping(self):
        """Test that universal status gets mapped to status_not."""
        params = UniversalSearchParams(status=AnimeStatus.CANCELLED)
        anilist_params = AniListMapper.to_anilist_search_params(params)
        
        # Should map status to both status and status_not
        assert anilist_params["status"] == "CANCELLED"
        assert anilist_params["status_not"] == AnimeStatus.CANCELLED