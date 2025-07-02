"""Tests for universal anime schema models.

Following TDD approach - these tests define the expected behavior 
of the universal schema before implementation refinements.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from src.models.universal_anime import (
    UniversalAnime,
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeRating,
    AnimeSeason,
    UniversalSearchResult,
    UniversalSearchResponse,
)


class TestAnimeStatus:
    """Test AnimeStatus enum and conversion methods."""
    
    def test_status_enum_values(self):
        """Test that all required status values exist."""
        assert AnimeStatus.FINISHED == "FINISHED"
        assert AnimeStatus.RELEASING == "RELEASING"
        assert AnimeStatus.NOT_YET_RELEASED == "NOT_YET_RELEASED"
        assert AnimeStatus.CANCELLED == "CANCELLED"
        assert AnimeStatus.HIATUS == "HIATUS"
    


class TestAnimeFormat:
    """Test AnimeFormat enum and conversion methods."""
    
    def test_format_enum_values(self):
        """Test that all required format values exist."""
        assert AnimeFormat.TV == "TV"
        assert AnimeFormat.TV_SHORT == "TV_SHORT"
        assert AnimeFormat.MOVIE == "MOVIE"
        assert AnimeFormat.OVA == "OVA"
        assert AnimeFormat.ONA == "ONA"
        assert AnimeFormat.SPECIAL == "SPECIAL"
        assert AnimeFormat.MUSIC == "MUSIC"
    


class TestUniversalAnime:
    """Test UniversalAnime model validation and methods."""
    
    def test_minimal_valid_anime(self):
        """Test creating anime with minimal required fields."""
        anime = UniversalAnime(
            id="test_123",
            title="Test Anime",
            type_format=AnimeFormat.TV,
            status=AnimeStatus.RELEASING
        )
        
        assert anime.id == "test_123"
        assert anime.title == "Test Anime"
        assert anime.type_format == AnimeFormat.TV
        assert anime.status == AnimeStatus.RELEASING
        assert anime.genres == []  # Default empty list
        assert anime.studios == []  # Default empty list
        assert anime.platform_ids == {}  # Default empty dict
    
    def test_comprehensive_anime_data(self):
        """Test creating anime with all fields populated."""
        anime = UniversalAnime(
            # Guaranteed properties
            id="comprehensive_456",
            title="Comprehensive Test Anime",
            type_format=AnimeFormat.TV,
            episodes=24,
            status=AnimeStatus.FINISHED,
            genres=["Action", "Adventure"],
            score=8.5,
            image_url="https://example.com/image.jpg",
            image_large="https://example.com/image_large.jpg",
            year=2023,
            synonyms=["Test Anime", "テストアニメ"],
            studios=["Studio Test"],
            
            # High-confidence properties
            description="A comprehensive test anime.",
            url="https://example.com/anime/456",
            score_count=10000,
            title_english="Comprehensive Test Anime",
            title_native="包括的テストアニメ",
            start_date="2023-04-01",
            season=AnimeSeason.SPRING,
            end_date="2023-09-30",
            duration=24,
            
            # Medium-confidence properties
            source="manga",
            rank=100,
            staff=[{"name": "Test Director", "role": "Director"}],
            
            # Additional properties
            characters=[{"name": "Test Character", "role": "Main"}],
            image_small="https://example.com/image_small.jpg",
            rating=AnimeRating.PG13,
            themes=["School", "Friendship"],
            demographics=["Shounen"],
            producers=["Test Producer"],
            popularity=5000,
        )
        
        assert anime.episodes == 24
        assert anime.score == 8.5
        assert anime.season == AnimeSeason.SPRING
        assert len(anime.genres) == 2
        assert len(anime.synonyms) == 2
        assert len(anime.themes) == 2
    
    def test_platform_id_management(self):
        """Test platform ID get/set methods."""
        anime = UniversalAnime(
            id="platform_test",
            title="Platform Test",
            type_format=AnimeFormat.TV,
            status=AnimeStatus.RELEASING
        )
        
        # Test setting platform IDs
        anime.set_platform_id("anilist", 12345)
        anime.set_platform_id("mal", 67890)
        anime.set_platform_id("kitsu", "abc123")
        
        # Test getting platform IDs
        assert anime.get_platform_id("anilist") == 12345
        assert anime.get_platform_id("mal") == 67890
        assert anime.get_platform_id("kitsu") == "abc123"
        assert anime.get_platform_id("nonexistent") is None
        
        # Test platform_ids dict
        assert len(anime.platform_ids) == 3
        assert "anilist" in anime.platform_ids
        assert "mal" in anime.platform_ids
        assert "kitsu" in anime.platform_ids
    
    def test_quality_score_calculation(self):
        """Test data quality score calculation."""
        # Minimal anime should have low quality score
        minimal_anime = UniversalAnime(
            id="minimal",
            title="Minimal",
            type_format=AnimeFormat.TV,
            status=AnimeStatus.RELEASING
        )
        
        minimal_score = minimal_anime.calculate_quality_score()
        assert 0.0 <= minimal_score <= 1.0
        assert minimal_score < 0.5  # Should be low due to missing data
        
        # Comprehensive anime should have high quality score
        comprehensive_anime = UniversalAnime(
            id="comprehensive",
            title="Comprehensive",
            type_format=AnimeFormat.TV,
            status=AnimeStatus.FINISHED,
            genres=["Action"],
            score=8.5,
            image_url="https://example.com/image.jpg",
            image_large="https://example.com/large.jpg",
            year=2023,
            studios=["Studio"],
            description="Description",
            url="https://example.com/anime",
            start_date="2023-01-01",
            duration=24,
            episodes=12,
            synonyms=["Alt Title"],
            themes=["Theme"],
            title_english="Comprehensive English",
            title_native="包括的",
            season=AnimeSeason.WINTER
        )
        
        comprehensive_score = comprehensive_anime.calculate_quality_score()
        assert 0.0 <= comprehensive_score <= 1.0
        assert comprehensive_score > 0.7  # Should be high due to complete data
        assert comprehensive_score > minimal_score
    
    def test_field_validation(self):
        """Test field validation rules."""
        # Test score validation
        with pytest.raises(ValueError):
            UniversalAnime(
                id="invalid_score",
                title="Invalid Score",
                type_format=AnimeFormat.TV,
                status=AnimeStatus.RELEASING,
                score=15.0  # Invalid: > 10
            )
        
        with pytest.raises(ValueError):
            UniversalAnime(
                id="invalid_score2",
                title="Invalid Score 2",
                type_format=AnimeFormat.TV,
                status=AnimeStatus.RELEASING,
                score=-1.0  # Invalid: < 0
            )
        
        # Test year validation
        with pytest.raises(ValueError):
            UniversalAnime(
                id="invalid_year",
                title="Invalid Year",
                type_format=AnimeFormat.TV,
                status=AnimeStatus.RELEASING,
                year=1800  # Invalid: < 1900
            )
        
        # Test episodes validation
        with pytest.raises(ValueError):
            UniversalAnime(
                id="invalid_episodes",
                title="Invalid Episodes",
                type_format=AnimeFormat.TV,
                status=AnimeStatus.RELEASING,
                episodes=-1  # Invalid: < 0
            )


class TestUniversalSearchParams:
    """Test UniversalSearchParams model and validation."""
    
    def test_minimal_search_params(self):
        """Test creating search params with minimal fields."""
        params = UniversalSearchParams()
        
        assert params.query is None
        assert params.limit == 20  # Default value
        assert params.offset == 0  # Default value
        assert params.include_adult == False  # Default value
        assert params.include_unaired == True  # Default value
    
    def test_comprehensive_search_params(self):
        """Test creating search params with all fields."""
        params = UniversalSearchParams(
            # Text search
            query="action anime",
            title="Specific Title",
            title_english="English Title",
            title_native="Native Title",
            
            # Content filters
            genres=["Action", "Adventure"],
            genres_exclude=["Horror"],
            status=AnimeStatus.RELEASING,
            type_format=AnimeFormat.TV,
            rating=AnimeRating.PG13,
            source="manga",
            
            # Temporal filters
            year=2023,
            season=AnimeSeason.SPRING,
            start_date="2023-01-01",
            end_date="2023-12-31",
            
            # Numeric filters
            min_score=7.0,
            max_score=10.0,
            min_episodes=12,
            max_episodes=24,
            min_duration=20,
            max_duration=30,
            
            # Production filters
            studios=["Studio Test"],
            producers=["Producer Test"],
            staff=["Director Test"],
            
            # Content filters
            themes=["School"],
            demographics=["Shounen"],
            characters=["Main Character"],
            
            # Result control
            limit=50,
            offset=10,
            sort_by="score",
            sort_order="desc",
            
            # Platform options
            include_adult=False,
            include_unaired=True,
            preferred_source="anilist",
            require_image=True,
            require_description=True,
        )
        
        assert params.query == "action anime"
        assert len(params.genres) == 2
        assert params.status == AnimeStatus.RELEASING
        assert params.year == 2023
        assert params.min_score == 7.0
        assert params.limit == 50
        assert params.sort_by == "score"
        assert params.preferred_source == "anilist"
    
    def test_search_params_validation(self):
        """Test search params validation rules."""
        # Test invalid sort_order
        with pytest.raises(ValueError):
            UniversalSearchParams(sort_order="invalid")
        
        # Test valid sort_order values
        params_asc = UniversalSearchParams(sort_order="asc")
        params_desc = UniversalSearchParams(sort_order="desc")
        assert params_asc.sort_order == "asc"
        assert params_desc.sort_order == "desc"
        
        # Test score range validation
        with pytest.raises(ValueError):
            UniversalSearchParams(min_score=15.0)  # > 10
        
        with pytest.raises(ValueError):
            UniversalSearchParams(max_score=-1.0)  # < 0
        
        # Test valid scores
        params = UniversalSearchParams(min_score=5.0, max_score=9.0)
        assert params.min_score == 5.0
        assert params.max_score == 9.0
    


class TestUniversalSearchResult:
    """Test UniversalSearchResult model."""
    
    def test_search_result_creation(self):
        """Test creating search result."""
        anime = UniversalAnime(
            id="result_test",
            title="Result Test",
            type_format=AnimeFormat.TV,
            status=AnimeStatus.RELEASING
        )
        
        result = UniversalSearchResult(
            anime=anime,
            relevance_score=0.85,
            source="anilist",
            enrichment_sources=["mal", "kitsu"]
        )
        
        assert result.anime.id == "result_test"
        assert result.relevance_score == 0.85
        assert result.source == "anilist"
        assert len(result.enrichment_sources) == 2


class TestUniversalSearchResponse:
    """Test UniversalSearchResponse model."""
    
    def test_search_response_creation(self):
        """Test creating search response."""
        params = UniversalSearchParams(query="test")
        
        anime = UniversalAnime(
            id="response_test",
            title="Response Test",
            type_format=AnimeFormat.TV,
            status=AnimeStatus.RELEASING
        )
        
        result = UniversalSearchResult(
            anime=anime,
            relevance_score=0.9,
            source="anilist"
        )
        
        response = UniversalSearchResponse(
            query_params=params,
            results=[result],
            total_results=1,
            processing_time_ms=150.5,
            sources_used=["anilist"],
            cache_hit=False
        )
        
        assert response.query_params.query == "test"
        assert len(response.results) == 1
        assert response.total_results == 1
        assert response.processing_time_ms == 150.5
        assert response.sources_used == ["anilist"]
        assert response.cache_hit == False


class TestUniversalSearchParamsPlatformSpecific:
    """Test platform-specific parameters in UniversalSearchParams."""
    
    def test_mal_specific_parameters(self):
        """Test MAL-specific parameter validation and functionality."""
        params = UniversalSearchParams(
            mal_broadcast_day="monday",
            mal_rating="pg_13",
            mal_nsfw="gray",
            mal_source="manga",
            mal_num_list_users=1000,
            mal_num_scoring_users=500,
            mal_created_at="2023-01-01",
            mal_updated_at="2023-12-31",
            mal_average_episode_duration=1440,  # 24 minutes in seconds
            mal_popularity=100,
            mal_rank=50,
            mal_mean=8.5
        )
        
        assert params.mal_broadcast_day == "monday"
        assert params.mal_rating == "pg_13"
        assert params.mal_nsfw == "gray"
        assert params.mal_source == "manga"
        assert params.mal_num_list_users == 1000
        assert params.mal_num_scoring_users == 500
        assert params.mal_created_at == "2023-01-01"
        assert params.mal_updated_at == "2023-12-31"
        assert params.mal_average_episode_duration == 1440
        assert params.mal_popularity == 100
        assert params.mal_rank == 50
        assert params.mal_mean == 8.5

    def test_anilist_specific_parameters(self):
        """Test AniList-specific parameter validation and functionality."""
        params = UniversalSearchParams(
            anilist_id=123456,
            anilist_id_mal=789012,
            anilist_season="FALL",
            anilist_season_year=2023,
            anilist_episodes=24,
            anilist_duration=24,
            anilist_is_adult=True,
            anilist_genre="Action",
            anilist_tag="School",
            anilist_minimum_tag_rank=80,
            anilist_tag_category="Setting",
            anilist_country_of_origin="JP",
            anilist_source="LIGHT_NOVEL",
            anilist_licensed_by="Crunchyroll",
            anilist_average_score=85,
            anilist_popularity=25000,
            # Array parameters
            anilist_id_in=[100, 200, 300],
            anilist_format_in=["TV", "MOVIE"],
            anilist_genre_in=["Action", "Adventure"],
            anilist_tag_in=["School", "Magic"],
            # Range parameters
            anilist_episodes_greater=10,
            anilist_episodes_lesser=50,
            anilist_duration_greater=20,
            anilist_duration_lesser=30,
            anilist_average_score_greater=75,
            anilist_popularity_greater=5000
        )
        
        assert params.anilist_id == 123456
        assert params.anilist_season == "FALL"
        assert params.anilist_is_adult == True
        assert params.anilist_genre == "Action"
        assert params.anilist_country_of_origin == "JP"
        assert params.anilist_source == "LIGHT_NOVEL"
        assert len(params.anilist_id_in) == 3
        assert len(params.anilist_genre_in) == 2
        assert params.anilist_episodes_greater == 10
        assert params.anilist_popularity_greater == 5000

    def test_jikan_specific_parameters(self):
        """Test Jikan-specific parameter validation and functionality."""
        params = UniversalSearchParams(
            jikan_anime_type="movie",
            jikan_sfw=True,
            jikan_genres_exclude=[1, 9, 12],
            jikan_order_by="score",
            jikan_sort="desc",
            jikan_letter="A",
            jikan_page=2,
            jikan_unapproved=False
        )
        
        assert params.jikan_anime_type == "movie"
        assert params.jikan_sfw == True
        assert len(params.jikan_genres_exclude) == 3
        assert params.jikan_order_by == "score"
        assert params.jikan_sort == "desc"
        assert params.jikan_letter == "A"
        assert params.jikan_page == 2
        assert params.jikan_unapproved == False

    def test_platform_specific_validation_errors(self):
        """Test validation errors for platform-specific parameters."""
        # Test invalid MAL broadcast day
        with pytest.raises(ValueError):
            UniversalSearchParams(mal_broadcast_day="invalid_day")
        
        # Test invalid MAL rating
        with pytest.raises(ValueError):
            UniversalSearchParams(mal_rating="invalid_rating")
        
        # Test invalid MAL NSFW value
        with pytest.raises(ValueError):
            UniversalSearchParams(mal_nsfw="invalid_nsfw")
        
        # Test invalid AniList season
        with pytest.raises(ValueError):
            UniversalSearchParams(anilist_season="INVALID_SEASON")
        
        # Test invalid AniList country code
        with pytest.raises(ValueError):
            UniversalSearchParams(anilist_country_of_origin="INVALID")  # Must be 2 letters
        
        # Test invalid AniList source
        with pytest.raises(ValueError):
            UniversalSearchParams(anilist_source="INVALID_SOURCE")
        
        # Test invalid Jikan anime type
        with pytest.raises(ValueError):
            UniversalSearchParams(jikan_anime_type="invalid_type")
        
        # Test invalid Jikan order_by
        with pytest.raises(ValueError):
            UniversalSearchParams(jikan_order_by="invalid_order")
        
        # Test invalid Jikan letter (must be single letter)
        with pytest.raises(ValueError):
            UniversalSearchParams(jikan_letter="AB")

    def test_array_parameter_validation(self):
        """Test validation of array parameters."""
        # Test invalid AniList format array
        with pytest.raises(ValueError):
            UniversalSearchParams(anilist_format_in=["TV", "INVALID_FORMAT"])
        
        # Test invalid AniList status array  
        with pytest.raises(ValueError):
            UniversalSearchParams(anilist_status_in=["FINISHED", "INVALID_STATUS"])
        
        # Test invalid AniList source array
        with pytest.raises(ValueError):
            UniversalSearchParams(anilist_source_in=["MANGA", "INVALID_SOURCE"])
        
        # Test invalid Jikan genre exclude (must be positive integers)
        with pytest.raises(ValueError):
            UniversalSearchParams(jikan_genres_exclude=[1, -5, 10])

    def test_min_duration_validation(self):
        """Test min_duration validation (must be ≥1)."""
        # Valid min_duration
        params = UniversalSearchParams(min_duration=1)
        assert params.min_duration == 1
        
        params = UniversalSearchParams(min_duration=90)
        assert params.min_duration == 90
        
        # Invalid min_duration (must be ≥1)
        with pytest.raises(ValueError):
            UniversalSearchParams(min_duration=0)


class TestUniversalSearchParamsEdgeCases:
    """Test edge cases and boundary conditions for UniversalSearchParams."""
    
    def test_boundary_values(self):
        """Test boundary values for numeric fields."""
        # Test minimum valid values
        params_min = UniversalSearchParams(
            min_score=0.0,
            max_score=0.0,
            min_episodes=0,
            max_episodes=0,
            min_duration=1,  # Minimum is 1
            year=1900,  # Minimum year
            limit=1,    # Minimum limit
            offset=0    # Minimum offset
        )
        
        assert params_min.min_score == 0.0
        assert params_min.min_duration == 1
        assert params_min.year == 1900
        assert params_min.limit == 1
        
        # Test maximum valid values
        params_max = UniversalSearchParams(
            min_score=10.0,
            max_score=10.0,
            year=2030,  # Maximum year
            limit=100   # Maximum limit
        )
        
        assert params_max.min_score == 10.0
        assert params_max.max_score == 10.0
        assert params_max.year == 2030
        assert params_max.limit == 100

    def test_empty_arrays_and_none_values(self):
        """Test handling of empty arrays and None values."""
        params = UniversalSearchParams(
            genres=[],
            genres_exclude=None,
            studios=[],
            producers=None,
            themes=[],
            characters=None
        )
        
        assert params.genres == []
        assert params.genres_exclude is None
        assert params.studios == []
        assert params.producers is None
        assert params.themes == []
        assert params.characters is None

    def test_date_format_edge_cases(self):
        """Test date format validation edge cases."""
        # Valid date formats
        params = UniversalSearchParams(
            start_date="2023-01-01",
            end_date="2023-12-31",
            mal_created_at="2023-06-15",
            mal_updated_at="2023-07-20"
        )
        
        assert params.start_date == "2023-01-01"
        assert params.end_date == "2023-12-31"
        assert params.mal_created_at == "2023-06-15"
        assert params.mal_updated_at == "2023-07-20"

    def test_case_sensitivity_and_normalization(self):
        """Test case sensitivity handling."""
        # Test MAL broadcast day (must already be lowercase due to pattern validation)
        params = UniversalSearchParams(mal_broadcast_day="monday")
        assert params.mal_broadcast_day == "monday"
        
        # Test MAL rating normalization
        params = UniversalSearchParams(mal_rating="PG-13")
        assert params.mal_rating == "pg_13"  # Should be normalized
        
        params = UniversalSearchParams(mal_rating="R+")
        assert params.mal_rating == "r+"  # Should be normalized

    def test_comprehensive_parameter_combination(self):
        """Test using many parameters together without conflicts."""
        params = UniversalSearchParams(
            # Universal parameters
            query="comprehensive test",
            genres=["Action", "Adventure"],
            status=AnimeStatus.FINISHED,
            type_format=AnimeFormat.TV,
            min_score=8.0,
            min_episodes=12,
            min_duration=20,
            year=2023,
            season=AnimeSeason.FALL,
            start_date="2023-10-01",
            end_date="2023-12-31",
            min_popularity=5000,
            studios=["Studio A", "Studio B"],
            producers=["Producer A"],
            themes=["School", "Magic"],
            include_adult=False,
            limit=25,
            sort_by="score",
            sort_order="desc",
            
            # MAL-specific
            mal_broadcast_day="friday",
            mal_rating="pg_13",
            mal_nsfw="white",
            mal_popularity=100,
            
            # AniList-specific
            anilist_id=123456,
            anilist_season="FALL",
            anilist_country_of_origin="JP",
            anilist_source="MANGA",
            anilist_genre_in=["Action", "Drama"],
            anilist_episodes_greater=10,
            anilist_popularity_greater=1000,
            
            # Jikan-specific
            jikan_anime_type="tv",
            jikan_sfw=True,
            jikan_order_by="score",
            jikan_sort="desc"
        )
        
        # Verify all parameters are set correctly
        assert params.query == "comprehensive test"
        assert len(params.genres) == 2
        assert params.min_popularity == 5000
        assert params.mal_broadcast_day == "friday"
        assert params.anilist_id == 123456
        assert params.jikan_anime_type == "tv"
        
        # Verify no conflicts or overrides
        assert params.sort_by == "score"  # Universal sort
        assert params.jikan_order_by == "score"  # Jikan-specific sort
        assert params.include_adult == False  # Universal adult filter
        assert params.jikan_sfw == True  # Jikan-specific SFW filter


class TestUniversalSearchParamsValidators:
    """Test custom validators in UniversalSearchParams."""
    
    def test_sort_order_validator(self):
        """Test sort_order validation and normalization."""
        # Valid sort orders (should be normalized to lowercase)
        params_asc = UniversalSearchParams(sort_order="ASC")
        assert params_asc.sort_order == "asc"
        
        params_desc = UniversalSearchParams(sort_order="DESC")
        assert params_desc.sort_order == "desc"
        
        params_mixed = UniversalSearchParams(sort_order="Asc")
        assert params_mixed.sort_order == "asc"

    def test_mal_rating_validator(self):
        """Test MAL rating validation and conversion."""
        # Test common rating conversions
        params_pg13_1 = UniversalSearchParams(mal_rating="PG-13")
        assert params_pg13_1.mal_rating == "pg_13"
        
        params_pg13_2 = UniversalSearchParams(mal_rating="pg13")
        assert params_pg13_2.mal_rating == "pg_13"
        
        params_rplus = UniversalSearchParams(mal_rating="R+")
        assert params_rplus.mal_rating == "r+"
        
        params_rplus_alt = UniversalSearchParams(mal_rating="r_plus")
        assert params_rplus_alt.mal_rating == "r+"

    def test_anilist_sort_validator(self):
        """Test AniList sort array validation."""
        # Valid sort arrays (using actual valid values from the validator)
        params = UniversalSearchParams(
            anilist_sort=["SCORE_DESC", "POPULARITY_DESC", "TITLE_ROMAJI"]
        )
        assert len(params.anilist_sort) == 3
        assert "SCORE_DESC" in params.anilist_sort
        assert "POPULARITY_DESC" in params.anilist_sort
        
        # Invalid sort value
        with pytest.raises(ValueError):
            UniversalSearchParams(anilist_sort=["INVALID_SORT", "SCORE_DESC"])