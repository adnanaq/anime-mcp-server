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
            "strengths": ["broadcast_schedules", "detailed_ratings", "nsfw_filtering", "user_engagement_metrics", "duration_filtering"],
            "unique_params": [
                "mal_broadcast_day", "mal_rating", "mal_nsfw", "mal_source", 
                "mal_num_list_users", "mal_num_scoring_users", "mal_created_at", 
                "mal_updated_at", "mal_average_episode_duration", "mal_broadcast", 
                "mal_main_picture", "mal_start_date", "mal_start_season", 
                "mal_popularity", "mal_rank", "mal_mean"
            ],
            "api_type": "rest",
            "auth_required": True,
            "total_parameters": 16,
        },
        "anilist": {
            "strengths": ["international_content", "adult_content", "modern_features", "comprehensive_filtering", "advanced_search"],
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
            "strengths": ["comprehensive_metadata", "json_api_standard"],
            "unique_params": ["kitsu_age_rating", "kitsu_subtype"],
            "api_type": "json_api",
            "auth_required": False,
            "total_parameters": 2,
        },
        "jikan": {
            "strengths": ["no_auth_required", "mal_compatibility", "advanced_filtering", "comprehensive_sorting"],
            "unique_params": [
                "jikan_anime_type", "jikan_sfw", "jikan_genres_exclude", 
                "jikan_order_by", "jikan_sort", "jikan_letter", "jikan_page", 
                "jikan_unapproved"
            ],
            "api_type": "rest",
            "auth_required": False,
            "total_parameters": 8,
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
        
        # Broadcast schedule queries → MAL
        if any(day in query for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
            return "mal"
        
        # Adult content queries → AniList  
        if any(term in query for term in ["adult", "mature", "r+", "rx"]):
            return "anilist"
        
        # International content → AniList
        if any(country in query for country in ["korean", "chinese", "korea", "china"]):
            return "anilist"
            
        # Default to universal coverage (Jikan - no auth required)
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