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
        
        # Status mapping (prevents the "ongoing" chaos)
        if universal_params.status:
            status_value = cls.UNIVERSAL_TO_STATUS.get(universal_params.status)
            if status_value:
                mal_params["status"] = status_value
        
        # Format mapping (only include if MAL supports this format)
        if universal_params.type_format:
            format_value = cls.UNIVERSAL_TO_FORMAT.get(universal_params.type_format)
            if format_value:
                mal_params["media_type"] = format_value
        
        # Score filters (MAL uses same 0-10 scale)
        min_score = mal_specific.get("min_score") or universal_params.min_score
        if min_score is not None:
            mal_params["min_score"] = min_score
        max_score = mal_specific.get("max_score") or universal_params.max_score
        if max_score is not None:
            mal_params["max_score"] = max_score
        
        # Episode filters
        min_episodes = mal_specific.get("min_episodes") or universal_params.min_episodes
        if min_episodes is not None:
            mal_params["min_episodes"] = min_episodes
        max_episodes = mal_specific.get("max_episodes") or universal_params.max_episodes
        if max_episodes is not None:
            mal_params["max_episodes"] = max_episodes
        
        # Date filtering
        start_date = mal_specific.get("start_date") or universal_params.start_date
        if start_date:
            mal_params["start_date"] = start_date
        
        # Season filtering
        start_season = mal_specific.get("start_season") or universal_params.mal_start_season
        if start_season:
            mal_params["start_season"] = start_season
        
        # Adult content filter
        if not universal_params.include_adult:
            mal_params["nsfw"] = "white"  # MAL: white = SFW content only
        
        # MAL-SPECIFIC PARAMETERS (from platform-specific params or universal schema)
        # Broadcast day
        broadcast_day = mal_specific.get("broadcast_day") or universal_params.mal_broadcast_day
        if broadcast_day:
            mal_params["broadcast_day"] = broadcast_day
        
        # Content rating  
        rating = mal_specific.get("rating") or universal_params.mal_rating
        if rating:
            mal_params["rating"] = rating
        
        # NSFW filtering
        nsfw = mal_specific.get("nsfw") or universal_params.mal_nsfw
        if nsfw:
            mal_params["nsfw"] = nsfw
        
        # Source material
        source = mal_specific.get("source") or universal_params.mal_source or universal_params.source
        if source:
            mal_params["source"] = source
        
        # User engagement filters
        num_list_users = mal_specific.get("num_list_users") or universal_params.mal_num_list_users
        if num_list_users is not None:
            mal_params["num_list_users"] = num_list_users
            
        num_scoring_users = mal_specific.get("num_scoring_users") or universal_params.mal_num_scoring_users
        if num_scoring_users is not None:
            mal_params["num_scoring_users"] = num_scoring_users
        
        # Date filters
        created_at = mal_specific.get("created_at") or universal_params.mal_created_at
        if created_at:
            mal_params["created_at"] = created_at
            
        updated_at = mal_specific.get("updated_at") or universal_params.mal_updated_at
        if updated_at:
            mal_params["updated_at"] = updated_at
        
        # Duration filter (convert minutes to seconds for MAL)
        if universal_params.min_duration is not None:
            mal_params["average_episode_duration"] = universal_params.min_duration * 60
        
        # Studios filter
        if universal_params.studios:
            mal_params["studios"] = ",".join(universal_params.studios)
        
        # Broadcast information
        broadcast = mal_specific.get("broadcast") or universal_params.mal_broadcast
        if broadcast:
            mal_params["broadcast"] = broadcast
        
        # Main picture
        main_picture = mal_specific.get("main_picture") or universal_params.mal_main_picture
        if main_picture:
            mal_params["main_picture"] = main_picture
        
        # End date filter
        end_date = mal_specific.get("end_date") or universal_params.end_date
        if end_date:
            mal_params["end_date"] = end_date
        
        # Additional MAL filters
        rank = mal_specific.get("rank") or universal_params.mal_rank
        if rank is not None:
            mal_params["rank"] = rank
        
        # Result control
        limit = mal_specific.get("limit") or universal_params.limit
        mal_params["limit"] = limit
        offset = mal_specific.get("offset") or universal_params.offset
        if offset:
            mal_params["offset"] = offset
        
        # Universal parameter mappings (with dual-source pattern)
        popularity = mal_specific.get("popularity") or universal_params.min_popularity
        if popularity is not None:
            mal_params["popularity"] = popularity
            
        mean = mal_specific.get("mean") or universal_params.score
        if mean is not None:
            mal_params["mean"] = mean
            
        title = mal_specific.get("title") or universal_params.title
        if title:
            mal_params["title"] = title
        
        return mal_params
    
