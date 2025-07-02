"""Jikan (unofficial MAL) query parameter mapper.

This mapper handles the conversion from universal search parameters
to Jikan API query parameters, addressing the parameter mapping chaos.
"""

from typing import Any, Dict
from ...models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class JikanMapper:
    """Query parameter mapper from universal schema to Jikan API."""
    
    # Universal to Jikan status mappings (for query parameters)
    UNIVERSAL_TO_STATUS = {
        AnimeStatus.FINISHED: "complete",
        AnimeStatus.RELEASING: "airing",
        AnimeStatus.NOT_YET_RELEASED: "upcoming",
    }
    
    # Universal to Jikan format mappings (for query parameters)
    # Only includes formats that Jikan actually supports
    UNIVERSAL_TO_FORMAT = {
        AnimeFormat.TV: "tv",
        AnimeFormat.MOVIE: "movie",
        AnimeFormat.SPECIAL: "special",
        AnimeFormat.OVA: "ova",
        AnimeFormat.ONA: "ona",
        AnimeFormat.MUSIC: "music",
    }
    
    # Universal to Jikan season mappings (for query parameters)
    UNIVERSAL_TO_SEASON = {
        AnimeSeason.WINTER: "winter",
        AnimeSeason.SPRING: "spring",
        AnimeSeason.SUMMER: "summer",
        AnimeSeason.FALL: "fall",
    }
    
    @classmethod
    def to_jikan_search_params(cls, universal_params: UniversalSearchParams, jikan_specific: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert universal search parameters to Jikan API parameters.
        
        This addresses the parameter mapping chaos by providing a central
        conversion point from universal parameters to Jikan API parameters.
        
        Args:
            universal_params: Universal search parameters
            jikan_specific: Jikan-specific parameters (optional)
            
        Returns:
            Dictionary of Jikan API parameters
        """
        if jikan_specific is None:
            jikan_specific = {}
        jikan_params = {}
        
        # Text search
        if universal_params.query:
            jikan_params["q"] = universal_params.query
        
        # Status mapping (prevents the "ongoing" chaos)
        if universal_params.status:
            status_value = cls.UNIVERSAL_TO_STATUS.get(universal_params.status)
            if status_value:
                jikan_params["status"] = status_value
        
        # Format mapping (only include if Jikan supports this format)
        if universal_params.type_format:
            format_value = cls.UNIVERSAL_TO_FORMAT.get(universal_params.type_format)
            if format_value:
                jikan_params["type"] = format_value
        
        # Score filters (Jikan uses same 0-10 scale)
        if universal_params.min_score is not None:
            jikan_params["min_score"] = universal_params.min_score
        if universal_params.max_score is not None:
            jikan_params["max_score"] = universal_params.max_score
        
        # Episode filters
        if universal_params.min_episodes is not None:
            jikan_params["episodes_greater"] = universal_params.min_episodes
        if universal_params.max_episodes is not None:
            jikan_params["episodes_lesser"] = universal_params.max_episodes
        
        # Genre filters (Jikan accepts genre names)
        if universal_params.genres:
            jikan_params["genres"] = ",".join(universal_params.genres)
        if universal_params.genres_exclude:
            jikan_params["genres_exclude"] = ",".join(universal_params.genres_exclude)
        
        # Year filter
        if universal_params.year:
            jikan_params["start_date"] = f"{universal_params.year}-01-01"
        elif universal_params.start_date:
            jikan_params["start_date"] = universal_params.start_date
        
        # End date filter
        if universal_params.end_date:
            jikan_params["end_date"] = universal_params.end_date
        
        # Adult content filter (Jikan expects string 'true'/'false')
        if not universal_params.include_adult:
            jikan_params["sfw"] = "true"
        
        # Result control
        jikan_params["limit"] = universal_params.limit
        if universal_params.offset:
            jikan_params["page"] = (universal_params.offset // universal_params.limit) + 1
        
        # Sort - Jikan has comprehensive sort options
        if universal_params.sort_by:
            sort_mapping = {
                "score": "score",
                "popularity": "popularity",
                "title": "title",
                "year": "start_date",
                "episodes": "episodes",
                "duration": "duration",
                "rank": "rank",
            }
            order_by = sort_mapping.get(universal_params.sort_by)
            if order_by:
                jikan_params["order_by"] = order_by
                if universal_params.sort_order:
                    jikan_params["sort"] = universal_params.sort_order
        
        # JIKAN-SPECIFIC PARAMETERS (only unique features, no duplicates)
        # Anime type (Jikan uses lowercase values)
        anime_type = jikan_specific.get("anime_type") or universal_params.jikan_anime_type
        if anime_type:
            jikan_params["type"] = anime_type.lower()  # Ensure lowercase
        
        # Safe For Work filter (unique to Jikan, expects string)
        sfw = jikan_specific.get("sfw") or universal_params.jikan_sfw
        if sfw is not None:
            jikan_params["sfw"] = "true" if sfw else "false"
        
        # Exclude genres by ID (Jikan-specific format)
        genres_exclude = jikan_specific.get("genres_exclude") or universal_params.jikan_genres_exclude
        if genres_exclude:
            jikan_params["genres_exclude"] = ",".join(map(str, genres_exclude))
        
        # Sorting and ordering (Jikan-specific overrides)
        order_by = jikan_specific.get("order_by") or universal_params.jikan_order_by
        if order_by:
            jikan_params["order_by"] = order_by
            
        sort_direction = jikan_specific.get("sort") or universal_params.jikan_sort
        if sort_direction:
            jikan_params["sort"] = sort_direction
        
        # Letter search (unique to Jikan, but conflicts with 'q' parameter)
        letter = jikan_specific.get("letter") or universal_params.jikan_letter
        if letter and not jikan_params.get("q"):
            jikan_params["letter"] = letter
        
        # Pagination (Jikan-specific)
        page = jikan_specific.get("page") or universal_params.jikan_page
        if page is not None:
            jikan_params["page"] = page
        
        # Special filters (unique to Jikan)
        unapproved = jikan_specific.get("unapproved") or universal_params.jikan_unapproved
        if unapproved is not None:
            jikan_params["unapproved"] = unapproved
        
        return jikan_params