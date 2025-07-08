"""Mapper Registry for clean platform routing without if-statement chains.

This registry provides a clean way to route queries to appropriate platform mappers
without massive if-statement chains, following the Registry design pattern.
"""

from typing import Dict, Type, Any, Tuple, Optional
from ..models.universal_anime import UniversalSearchParams
from .mappers import (
    MALMapper,
    JikanMapper, 
    AniListMapper,
    KitsuMapper,
    AniDBMapper,
    AnimePlanetMapper,
    AnimeScheduleMapper,
    AniSearchMapper,
)


class MapperRegistry:
    """Registry for anime platform mappers with intelligent routing."""
    
    # Registry of available mappers
    _mappers: Dict[str, Type] = {
        "mal": MALMapper,
        "jikan": JikanMapper,
        "anilist": AniListMapper, 
        "kitsu": KitsuMapper,
        "anidb": AniDBMapper,
        "animeplanet": AnimePlanetMapper,
        "animeschedule": AnimeScheduleMapper,
        "anisearch": AniSearchMapper,
    }
    
    # Platform capabilities for auto-selection
    _platform_capabilities = {
        "mal": {
            "strengths": ["comprehensive_metadata", "user_engagement_data"],
            "unique_params": [],  # MAL has no unique query parameters - only supports q, limit, offset
            "unique_response_fields": [
                "mal_alternative_titles_field", "mal_my_list_status_field", 
                "mal_num_list_users_field", "mal_num_scoring_users_field", 
                "mal_nsfw_field", "mal_average_episode_duration_field", 
                "mal_start_season_field", "mal_broadcast_field", 
                "mal_main_picture_field", "mal_created_at_field", 
                "mal_updated_at_field"
            ],
            "api_type": "rest",
            "auth_required": True,
            "total_parameters": 3,  # Only q, limit, offset
            "total_response_fields": 26,  # 15 core + 11 MAL-specific
        },
        "anilist": {
            "strengths": ["international_content", "adult_content", "modern_features", "hyper_comprehensive_filtering", "hyper_advanced_search"],
            "unique_params": [
                "anilist_average_score", "anilist_average_score_greater", "anilist_average_score_lesser", 
                "anilist_average_score_not", "anilist_chapters", "anilist_chapters_greater", 
                "anilist_chapters_lesser", "anilist_country_of_origin", "anilist_duration", 
                "anilist_duration_greater", "anilist_duration_lesser", "anilist_end_date", 
                "anilist_end_date_greater", "anilist_end_date_lesser", "anilist_end_date_like", 
                "anilist_episodes", "anilist_episodes_greater", "anilist_episodes_lesser", 
                "anilist_format", "anilist_format_in", "anilist_format_not", "anilist_format_not_in", 
                "anilist_format_range", "anilist_genre", "anilist_genre_in", "anilist_genre_not_in", 
                "anilist_id", "anilist_id_in", "anilist_id_mal", "anilist_id_mal_in", 
                "anilist_id_mal_not", "anilist_id_mal_not_in", "anilist_id_not", "anilist_id_not_in", 
                "anilist_is_adult", "anilist_is_licensed", "anilist_licensed_by", "anilist_licensed_by_id", 
                "anilist_licensed_by_id_in", "anilist_licensed_by_in", "anilist_licensed_by_not_in", 
                "anilist_minimum_tag_rank", "anilist_on_list", "anilist_popularity", 
                "anilist_popularity_greater", "anilist_popularity_lesser", "anilist_popularity_not", 
                "anilist_season", "anilist_season_year", "anilist_sort", "anilist_source", 
                "anilist_source_in", "anilist_source_not_in", "anilist_start_date", 
                "anilist_start_date_greater", "anilist_start_date_lesser", "anilist_start_date_like", 
                "anilist_status", "anilist_status_in", "anilist_status_not", "anilist_status_not_in", 
                "anilist_tag", "anilist_tag_category", "anilist_tag_category_in", 
                "anilist_tag_category_not_in", "anilist_tag_in", "anilist_tag_not_in", 
                "anilist_volumes", "anilist_volumes_greater", "anilist_volumes_lesser"
            ],
            "api_type": "graphql",
            "auth_required": False,
            "total_parameters": 70,
        },
        "kitsu": {
            "strengths": ["comprehensive_metadata", "json_api_standard", "range_syntax_filtering", "streaming_platform_support", "rich_category_system"],
            "supported_universal_params": [
                "query", "status", "type_format", "rating", "min_score", "max_score",
                "min_episodes", "max_episodes", "min_duration", "max_duration", 
                "year", "season", "genres", "sort_by", "sort_order", "limit", "offset"
            ],
            "unique_params": [
                "kitsu_streamers"  # Only truly unique parameter - streaming platform filtering
            ],
            "verified_filters": [
                "filter[text]", "filter[status]", "filter[subtype]", "filter[ageRating]",
                "filter[averageRating]", "filter[episodeCount]", "filter[episodeLength]",
                "filter[seasonYear]", "filter[season]", "filter[categories]", "filter[streamers]"
            ],
            "range_syntax_support": {
                "averageRating": "0-100 scale with .. separator (80.., ..90, 80..90)",
                "episodeCount": "Episode count ranges (12.., ..24, 12..24)",
                "episodeLength": "Duration ranges in minutes (20.., ..30, 20..30)"
            },
            "api_type": "json_api",
            "auth_required": False,
            "total_parameters": 14,  # Verified working parameters
            "verification_status": "comprehensive_api_testing_completed",
        },
        "jikan": {
            "strengths": ["no_auth_required", "mal_compatibility", "advanced_filtering", "comprehensive_sorting", "content_rating_support", "producer_filtering"],
            "supported_universal_params": [
                "query", "status", "type_format", "rating", "min_score", "max_score", 
                "genres", "genres_exclude", "producers", "year", "start_date", "end_date",
                "include_adult", "limit", "offset", "sort_by", "sort_order"
            ],
            "supported_jikan_formats": [
                "TV", "TV_SPECIAL", "MOVIE", "OVA", "ONA", "SPECIAL", "MUSIC", "CM", "PV"
            ],
            "supported_ratings": [
                "g", "pg", "pg13", "r", "r+", "rx"
            ],
            "supported_sort_fields": [
                "score", "popularity", "title", "year", "episodes", "duration", "rank", 
                "mal_id", "scored_by", "members", "favorites", "start_date", "end_date"
            ],
            "unique_params": [
                "jikan_score", "jikan_letter", "jikan_unapproved"
            ],
            "unsupported_features": [
                "episode_range_filtering", "duration_filtering"
            ],
            "api_type": "rest",
            "auth_required": False,
            "total_parameters": 18,  # Updated count including jikan_score parameter
        },
        "anidb": {
            "strengths": ["detailed_episode_data", "comprehensive_staff_credits", "extensive_tag_system"],
            "supported_universal_params": [
                "query"  # Only query supported, mapped to aid parameter
            ],
            "unique_params": [],  # No unique parameters
            "api_type": "rest",
            "auth_required": True,
            "total_parameters": 1,  # Only aid parameter supported
            "limitations": [
                "id_based_lookup_only", "no_search_filtering", "requires_anime_titles_xml", "two_step_process"
            ],
        },
        "animeschedule": {
            "strengths": ["broadcasting_schedules", "timing_data", "comprehensive_filtering", "exclude_options", "multi_platform_ids", "streaming_integration"],
            "supported_universal_params": [
                "query", "status", "type_format", "source", "genres", "genres_exclude", 
                "studios", "year", "season", "episodes", "duration", 
                "min_duration", "max_duration", "sort_by", "limit", "offset"
            ],
            "unique_params": [
                "animeschedule_mt", "animeschedule_st", "animeschedule_streams", "animeschedule_streams_exclude",
                "animeschedule_sources_exclude", "animeschedule_studios_exclude", "animeschedule_media_types_exclude",
                "animeschedule_airing_statuses_exclude", "animeschedule_years_exclude", "animeschedule_seasons_exclude",
                "animeschedule_mal_ids", "animeschedule_anilist_ids", "animeschedule_anidb_ids"
            ],
            "supported_formats": [
                "TV", "Movie", "OVA", "ONA", "Special", "Music", "TV Short", "TV (Chinese)", "ONA (Chinese)"
            ],
            "supported_statuses": [
                "finished", "ongoing", "upcoming"
            ],
            "supported_sources": [
                "Manga", "Light Novel", "Web Manga", "Web Novel", "Novel", "Original", 
                "Video Game", "Visual Novel", "4-koma Manga", "Book", "Music", "Game", "Other"
            ],
            "sort_options": [
                "popularity", "score", "alphabetic", "releaseDate"
            ],
            "exclude_filtering": {
                "genres": True, "studios": True, "sources": True, "media_types": True,
                "airing_statuses": True, "years": True, "seasons": True, "streams": True
            },
            "external_id_support": {
                "mal_ids": True, "anilist_ids": True, "anidb_ids": True
            },
            "api_type": "rest",
            "auth_required": True,
            "total_parameters": 25,  # All 25 parameters verified
            "verification_status": "comprehensive_api_testing_completed",
        },
    }
    
    @classmethod
    def get_mapper(cls, platform: str):
        """Get mapper for specified platform.
        
        Args:
            platform: Platform identifier (mal, anilist, kitsu, etc.)
            
        Returns:
            Mapper class for the platform
            
        Raises:
            ValueError: If platform is not supported
        """
        if platform not in cls._mappers:
            available_platforms = ", ".join(cls._mappers.keys())
            raise ValueError(f"Platform '{platform}' not supported. Available: {available_platforms}")
        
        return cls._mappers[platform]
    
    @classmethod
    def extract_platform_params(cls, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Extract universal and platform-specific parameters from kwargs.
        
        Args:
            **kwargs: Mixed universal and platform-specific parameters
            
        Returns:
            Tuple of (universal_params, platform_specific_params)
            
        Example:
            Input: query="naruto", mal_nsfw="white", anilist_is_adult=False
            Output: (
                {"query": "naruto"}, 
                {"mal": {"nsfw": "white"}, "anilist": {"is_adult": False}}
            )
        """
        universal_params = {}
        platform_specific = {}
        
        for key, value in kwargs.items():
            if value is None:
                continue
                
            # Check if parameter is platform-specific
            platform_found = False
            for platform in cls._mappers.keys():
                prefix = f"{platform}_"
                if key.startswith(prefix):
                    param_name = key[len(prefix):]  # Remove platform prefix
                    if platform not in platform_specific:
                        platform_specific[platform] = {}
                    platform_specific[platform][param_name] = value
                    platform_found = True
                    break
            
            # If not platform-specific, it's universal
            if not platform_found:
                universal_params[key] = value
        
        return universal_params, platform_specific
    
    @classmethod
    def auto_select_platform(cls, universal_params: Dict[str, Any], platform_specific: Dict[str, Dict[str, Any]]) -> str:
        """Automatically select best platform based on query characteristics.
        
        Args:
            universal_params: Universal search parameters
            platform_specific: Platform-specific parameters by platform
            
        Returns:
            Selected platform identifier
        """
        # If platform-specific params provided, use that platform
        if platform_specific:
            # Use the platform with the most specific parameters
            return max(platform_specific.keys(), key=lambda p: len(platform_specific[p]))
        
        # Auto-select based on query characteristics
        query = universal_params.get("query", "").lower()
        
        # Check for parameters that require specific platforms
        
        # Streaming platform filtering → Kitsu (unique feature)
        if any(streamer in query.lower() for streamer in ["crunchyroll", "funimation", "netflix", "hulu"]):
            return "kitsu"
        
        # Range-based filtering (score/episode/duration ranges) → Kitsu (best range syntax support)
        has_range_filters = any([
            universal_params.get("min_score") and universal_params.get("max_score"),
            universal_params.get("min_episodes") and universal_params.get("max_episodes"), 
            universal_params.get("min_duration") and universal_params.get("max_duration")
        ])
        if has_range_filters:
            return "kitsu"
        
        # Advanced AniList-only features → AniList
        if any(param in universal_params for param in ["country_of_origin", "licensed_by", "is_licensed"]):
            return "anilist"
        
        # Adult content queries → AniList (most comprehensive adult content handling)
        if any(term in query for term in ["adult", "mature", "hentai"]):
            return "anilist"
        
        # International content → AniList
        if any(country in query for country in ["korean", "chinese", "korea", "china"]):
            return "anilist"
        
        # Content rating filtering → Jikan or Kitsu (both support rating)
        if universal_params.get("rating"):
            return "jikan"  # Prefer Jikan for broader rating support
        
        # Producer filtering → Jikan (newly added support)
        if universal_params.get("producers"):
            return "jikan"
        
        # Jikan-specific format requests → Jikan
        jikan_formats = ["TV_SPECIAL", "CM", "PV"]
        if universal_params.get("type_format") in jikan_formats:
            return "jikan"
        
        # Broadcast schedule queries → MAL
        if any(day in query for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
            return "mal"
            
        # Default to universal coverage (Jikan - no auth required, enhanced capabilities)
        return "jikan"
    
    @classmethod
    def get_platform_capabilities(cls) -> Dict[str, Dict[str, Any]]:
        """Get capabilities overview for all platforms.
        
        Returns:
            Dictionary of platform capabilities for LLM discovery
        """
        return cls._platform_capabilities.copy()
    
    @classmethod
    def get_supported_platforms(cls) -> list[str]:
        """Get list of all supported platform identifiers."""
        return list(cls._mappers.keys())


class PlatformParameterExtractor:
    """Utility for extracting and validating platform-specific parameters."""
    
    @staticmethod
    def create_universal_params(universal_dict: Dict[str, Any]) -> UniversalSearchParams:
        """Create UniversalSearchParams from dictionary, filtering invalid fields.
        
        Args:
            universal_dict: Dictionary of universal parameters
            
        Returns:
            UniversalSearchParams instance with only valid fields
        """
        # Get valid field names from UniversalSearchParams
        valid_fields = set(UniversalSearchParams.model_fields.keys())
        
        # Filter to only include valid fields
        filtered_params = {k: v for k, v in universal_dict.items() if k in valid_fields}
        
        return UniversalSearchParams(**filtered_params)
    
    @staticmethod
    def validate_platform_params(platform: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate platform-specific parameters.
        
        Args:
            platform: Platform identifier
            params: Platform-specific parameters
            
        Returns:
            Validated parameters
            
        Raises:
            ValueError: If parameters are invalid for the platform
        """
        if platform not in MapperRegistry._mappers:
            raise ValueError(f"Unknown platform: {platform}")
        
        # Platform-specific validation could be added here
        # For now, just return the params as-is
        return params