"""MAL API v2 query parameter mapper.

This mapper handles the conversion from universal search parameters
to MAL API v2 query parameters, addressing the parameter mapping chaos.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ...models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class MALMapper:
    """Query parameter mapper from universal schema to MAL API v2."""
    
    # Universal to MAL status mappings (for query parameters)
    UNIVERSAL_TO_STATUS = {
        AnimeStatus.FINISHED: "finished_airing",  # Fixed: was "finished" 
        AnimeStatus.RELEASING: "currently_airing",
        AnimeStatus.NOT_YET_RELEASED: "not_yet_aired",
        AnimeStatus.HIATUS: "on_hiatus",  # API accepts this parameter, rare in practice
    }
    
    # Universal to MAL format mappings (for query parameters)
    # Based on actual MAL API v2 media_type values
    UNIVERSAL_TO_FORMAT = {
        AnimeFormat.TV: "tv",
        AnimeFormat.MOVIE: "movie", 
        AnimeFormat.SPECIAL: "special",
        AnimeFormat.OVA: "ova",
        AnimeFormat.ONA: "ona",
        AnimeFormat.MUSIC: "music",
        # Note: MAL also has "tv_special" which could map to SPECIAL
    }
    
    # Universal to MAL season mappings (for query parameters)
    UNIVERSAL_TO_SEASON = {
        AnimeSeason.WINTER: "winter",
        AnimeSeason.SPRING: "spring",
        AnimeSeason.SUMMER: "summer",
        AnimeSeason.FALL: "fall",
    }
    
    
    @classmethod
    def to_mal_search_params(cls, universal_params: UniversalSearchParams, mal_specific: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert universal search parameters to MAL API v2 parameters.
        
        This addresses the parameter mapping chaos by providing a central
        conversion point from universal parameters to MAL API v2 parameters.
        
        Args:
            universal_params: Universal search parameters
            mal_specific: MAL-specific parameters (optional)
            
        Returns:
            Dictionary of MAL API v2 parameters
        """
        if mal_specific is None:
            mal_specific = {}
        mal_params = {}
        
        # Text search
        if universal_params.query:
            mal_params["q"] = universal_params.query
        
        # MAL API v2 does not support filtering by status, format, score, episodes, dates, or adult content
        # These are field parameters only - handle via fields parameter instead
        
        # Handle response field requests using MAL's fields parameter
        requested_fields = []
        
        # Core field parameters
        if getattr(universal_params, 'id_field', None):
            requested_fields.append('id')
            
        if getattr(universal_params, 'title_field', None):
            requested_fields.append('title')
            
        if getattr(universal_params, 'status_field', None):
            requested_fields.append('status')
            
        if getattr(universal_params, 'format_field', None):
            requested_fields.append('media_type')
            
        if getattr(universal_params, 'episodes_field', None):
            requested_fields.append('num_episodes')
            
        if getattr(universal_params, 'score_field', None):
            requested_fields.append('mean')
            
        if getattr(universal_params, 'genres_field', None):
            requested_fields.append('genres')
        if getattr(universal_params, 'start_date_field', None):
            requested_fields.append('start_date')
            
        if getattr(universal_params, 'end_date_field', None):
            requested_fields.append('end_date')
            
        if getattr(universal_params, 'synopsis_field', None):
            requested_fields.append('synopsis')
            
        if getattr(universal_params, 'popularity_field', None):
            requested_fields.append('popularity')
            
        if getattr(universal_params, 'rank_field', None):
            requested_fields.append('rank')
            
        if getattr(universal_params, 'source_field', None):
            requested_fields.append('source')
            
        if getattr(universal_params, 'rating_field', None):
            requested_fields.append('rating')
            
        if getattr(universal_params, 'studios_field', None):
            requested_fields.append('studios')
            
        # MAL-specific field parameters
        if getattr(universal_params, 'mal_alternative_titles_field', None):
            requested_fields.append('alternative_titles')
            
        if getattr(universal_params, 'mal_my_list_status_field', None):
            requested_fields.append('my_list_status')
            
        if getattr(universal_params, 'mal_num_list_users_field', None):
            requested_fields.append('num_list_users')
            
        if getattr(universal_params, 'mal_num_scoring_users_field', None):
            requested_fields.append('num_scoring_users')
            
        if getattr(universal_params, 'mal_nsfw_field', None):
            requested_fields.append('nsfw')
            
        if getattr(universal_params, 'mal_average_episode_duration_field', None):
            requested_fields.append('average_episode_duration')
            
        if getattr(universal_params, 'mal_start_season_field', None):
            requested_fields.append('start_season')
            
        if getattr(universal_params, 'mal_broadcast_field', None):
            requested_fields.append('broadcast')
            
        if getattr(universal_params, 'mal_main_picture_field', None):
            requested_fields.append('main_picture')
            
        if getattr(universal_params, 'mal_created_at_field', None):
            requested_fields.append('created_at')
            
        if getattr(universal_params, 'mal_updated_at_field', None):
            requested_fields.append('updated_at')
            
        # Add fields parameter if any fields were requested
        if requested_fields:
            mal_params['fields'] = ','.join(requested_fields)
        
        # Note: rating, nsfw, source, num_list_users, num_scoring_users, created_at, updated_at 
        # are all field parameters, not query parameters for MAL API v2
        
        # Note: duration, studios, broadcast, main_picture, end_date, rank 
        # are all field parameters, not query parameters for MAL API v2
        
        # Result control
        limit = mal_specific.get("limit") or universal_params.limit
        mal_params["limit"] = limit
        offset = mal_specific.get("offset") or universal_params.offset
        if offset:
            mal_params["offset"] = offset
        
        # Note: popularity, mean, title are field parameters, not query parameters for MAL API v2
        
        return mal_params
    
