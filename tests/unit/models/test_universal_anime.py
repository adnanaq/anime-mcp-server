"""Tests for universal anime schema models.

Following TDD approach - these tests define the expected behavior
of the universal schema before implementation refinements.
"""

import pytest

from src.models.universal_anime import (
    AnimeFormat,
    AnimeRating,
    AnimeSeason,
    AnimeStatus,
    UniversalAnime,
    UniversalSearchParams,
    UniversalSearchResponse,
    UniversalSearchResult,
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
            status=AnimeStatus.RELEASING,
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
            status=AnimeStatus.RELEASING,
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
            status=AnimeStatus.RELEASING,
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
            season=AnimeSeason.WINTER,
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
                score=15.0,  # Invalid: > 10
            )

        with pytest.raises(ValueError):
            UniversalAnime(
                id="invalid_score2",
                title="Invalid Score 2",
                type_format=AnimeFormat.TV,
                status=AnimeStatus.RELEASING,
                score=-1.0,  # Invalid: < 0
            )

        # Test year validation
        with pytest.raises(ValueError):
            UniversalAnime(
                id="invalid_year",
                title="Invalid Year",
                type_format=AnimeFormat.TV,
                status=AnimeStatus.RELEASING,
                year=1800,  # Invalid: < 1900
            )

        # Test episodes validation
        with pytest.raises(ValueError):
            UniversalAnime(
                id="invalid_episodes",
                title="Invalid Episodes",
                type_format=AnimeFormat.TV,
                status=AnimeStatus.RELEASING,
                episodes=-1,  # Invalid: < 0
            )

    def test_platform_ids_validation(self):
        """Test platform_ids field validation - covers lines 158-160."""
        # Test the validator directly to cover the edge case
        from src.models.universal_anime import UniversalAnime

        # Test non-dict input (should return empty dict - line 158-159)
        result = UniversalAnime.validate_platform_ids("not_a_dict")
        assert result == {}

        # Test valid dict input (should pass through - line 160)
        valid_dict = {"anilist": 12345, "mal": 67890}
        result2 = UniversalAnime.validate_platform_ids(valid_dict)
        assert result2 == valid_dict

    def test_quality_score_edge_case(self):
        """Test quality score calculation edge case - covers line 205."""
        # Create an anime with just basic required fields
        anime = UniversalAnime(
            id="quality_edge_case",
            title="Quality Edge Case",
            type_format=AnimeFormat.TV,
            status=AnimeStatus.RELEASING,
        )

        # The quality score calculation should work normally, but we want to test
        # the edge case in the calculate_quality_score method (line 205)
        # Let's run the actual method to see if it works
        score = anime.calculate_quality_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

        # Note: Line 205 (if total_fields == 0: return 0.0) is a safety check
        # that may not be reachable with normal data, but the test confirms
        # the method runs successfully with minimal data


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

    def test_mal_rating_validation(self):
        """Test MAL rating validation - covers lines 457, 480."""
        # Test invalid MAL rating (should raise ValueError - line 477)
        with pytest.raises(ValueError):
            UniversalSearchParams(mal_rating="INVALID_RATING")

        # Test valid MAL rating conversion
        params = UniversalSearchParams(mal_rating="G")  # Valid rating
        assert params.mal_rating == "g"  # Should be converted to lowercase

        # Test MAL rating conversion with mapping
        params2 = UniversalSearchParams(
            mal_rating="PG13"
        )  # Should convert pg13 -> pg_13
        assert params2.mal_rating == "pg_13"

        # Test None case (should return None - line 480)
        params3 = UniversalSearchParams(mal_rating=None)
        assert params3.mal_rating is None

    def test_anilist_validation_edge_cases(self):
        """Test AniList validation edge cases - covers lines 502, 515."""
        # Test valid AniList status (should pass through unchanged - line 502)
        params = UniversalSearchParams(anilist_status_in=["FINISHED", "RELEASING"])
        assert params.anilist_status_in == ["FINISHED", "RELEASING"]

        # Test valid AniList source (should pass through unchanged - line 515)
        params2 = UniversalSearchParams(anilist_source_in=["MANGA", "ORIGINAL"])
        assert params2.anilist_source_in == ["MANGA", "ORIGINAL"]

        # Test invalid AniList status
        with pytest.raises(ValueError):
            UniversalSearchParams(anilist_status_in=["INVALID_STATUS"])

        # Test invalid AniList source
        with pytest.raises(ValueError):
            UniversalSearchParams(anilist_source_in=["INVALID_SOURCE"])


class TestUniversalSearchResult:
    """Test UniversalSearchResult model."""

    def test_search_result_creation(self):
        """Test creating search result."""
        anime = UniversalAnime(
            id="result_test",
            title="Result Test",
            type_format=AnimeFormat.TV,
            status=AnimeStatus.RELEASING,
        )

        result = UniversalSearchResult(
            anime=anime,
            relevance_score=0.85,
            source="anilist",
            enrichment_sources=["mal", "kitsu"],
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
            status=AnimeStatus.RELEASING,
        )

        result = UniversalSearchResult(
            anime=anime, relevance_score=0.9, source="anilist"
        )

        response = UniversalSearchResponse(
            query_params=params,
            results=[result],
            total_results=1,
            processing_time_ms=150.5,
            sources_used=["anilist"],
            cache_hit=False,
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
            mal_broadcast="monday",
            mal_rating="pg_13",
            mal_nsfw="gray",
            mal_source="manga",
            mal_num_list_users=1000,
            mal_num_scoring_users=500,
            mal_created_at="2023-01-01",
            mal_updated_at="2023-12-31",
            mal_start_date="2023-01-01",
            mal_start_season="2023,winter",
        )

        assert params.mal_broadcast == "monday"
        assert params.mal_rating == "pg_13"
        assert params.mal_nsfw == "gray"
        assert params.mal_source == "manga"
        assert params.mal_num_list_users == 1000
        assert params.mal_num_scoring_users == 500
        assert params.mal_created_at == "2023-01-01"
        assert params.mal_updated_at == "2023-12-31"
        assert params.mal_start_date == "2023-01-01"
        assert params.mal_start_season == "2023,winter"

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
            anilist_popularity_greater=5000,
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
        params = UniversalSearchParams(jikan_letter="A", jikan_unapproved=False)

        assert params.jikan_letter == "A"
        assert params.jikan_unapproved == False

    def test_platform_specific_validation_errors(self):
        """Test validation errors for platform-specific parameters."""
        # Test invalid MAL rating
        with pytest.raises(ValueError):
            UniversalSearchParams(mal_rating="invalid_rating")

        # Test invalid MAL NSFW value
        with pytest.raises(ValueError):
            UniversalSearchParams(mal_nsfw="invalid_nsfw")

        # Test invalid MAL source
        with pytest.raises(ValueError):
            UniversalSearchParams(mal_source="invalid_source")

        # Test invalid MAL start season format
        with pytest.raises(ValueError):
            UniversalSearchParams(mal_start_season="invalid_format")

        # Test invalid AniList season
        with pytest.raises(ValueError):
            UniversalSearchParams(anilist_season="INVALID_SEASON")

        # Test invalid AniList country code
        with pytest.raises(ValueError):
            UniversalSearchParams(
                anilist_country_of_origin="INVALID"
            )  # Must be 2 letters

        # Test invalid AniList source
        with pytest.raises(ValueError):
            UniversalSearchParams(anilist_source="INVALID_SOURCE")

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
            limit=1,  # Minimum limit
            offset=0,  # Minimum offset
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
            limit=100,  # Maximum limit
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
            characters=None,
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
            mal_updated_at="2023-07-20",
        )

        assert params.start_date == "2023-01-01"
        assert params.end_date == "2023-12-31"
        assert params.mal_created_at == "2023-06-15"
        assert params.mal_updated_at == "2023-07-20"

    def test_case_sensitivity_and_normalization(self):
        """Test case sensitivity handling."""
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
            mal_broadcast="friday",
            mal_rating="pg_13",
            mal_nsfw="white",
            mal_start_season="2023,fall",
            # AniList-specific
            anilist_id=123456,
            anilist_season="FALL",
            anilist_country_of_origin="JP",
            anilist_source="MANGA",
            anilist_genre_in=["Action", "Drama"],
            anilist_episodes_greater=10,
            anilist_popularity_greater=1000,
            # Jikan-specific
            jikan_letter="A",
            jikan_unapproved=False,
        )

        # Verify all parameters are set correctly
        assert params.query == "comprehensive test"
        assert len(params.genres) == 2
        assert params.min_popularity == 5000
        assert params.mal_broadcast == "friday"
        assert params.anilist_id == 123456
        assert params.jikan_letter == "A"

        # Verify no conflicts or overrides
        assert params.sort_by == "score"  # Universal sort
        assert params.include_adult == False  # Universal adult filter
        assert params.jikan_unapproved == False  # Jikan-specific filter


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


class TestUniversalSearchParamsScoreValidation:
    """Test score validation methods - covers line 457."""

    def test_score_range_validator_direct(self):
        """Test validate_score_range method directly - covers line 457."""
        from src.models.universal_anime import UniversalSearchParams

        # Test valid scores (should pass through unchanged)
        result = UniversalSearchParams.validate_score_range(5.0, None)
        assert result == 5.0

        result = UniversalSearchParams.validate_score_range(None, None)
        assert result is None

        # Test invalid scores - should raise ValueError (line 457)
        with pytest.raises(ValueError, match="Score must be between 0 and 10"):
            UniversalSearchParams.validate_score_range(-1.0, None)

        with pytest.raises(ValueError, match="Score must be between 0 and 10"):
            UniversalSearchParams.validate_score_range(11.0, None)
