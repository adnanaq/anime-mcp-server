"""AniSearch query parameter mapper.

This mapper handles the conversion from universal search parameters
to AniSearch query parameters. AniSearch is a European anime database
with web scraping characteristics.
"""

from typing import Any, Dict
from ...models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class AniSearchMapper:
    """Query parameter mapper from universal schema to AniSearch."""
    
    # Universal to AniSearch status mappings
    UNIVERSAL_TO_STATUS = {
        AnimeStatus.FINISHED: "COMPLETED",
        AnimeStatus.RELEASING: "ONGOING",
        AnimeStatus.NOT_YET_RELEASED: "UPCOMING",
    }
    
    # Universal to AniSearch format mappings
    UNIVERSAL_TO_TYPE = {
        AnimeFormat.TV: "video.tv_show",
        AnimeFormat.MOVIE: "video.movie",
    }
    
    @classmethod
    def to_anisearch_search_params(cls, universal_params: UniversalSearchParams) -> Dict[str, Any]:
        """Convert universal search parameters to AniSearch parameters.
        
        AniSearch characteristics:
        - European anime database focus
        - Web scraping based with limited API
        - Basic search and filtering capabilities
        - Limited parameter support
        
        Args:
            universal_params: Universal search parameters
            
        Returns:
            Dictionary of AniSearch parameters
        """
        anisearch_params = {}
        
        # Text search
        if universal_params.query:
            anisearch_params["query"] = universal_params.query
        
        # Status mapping
        if universal_params.status:
            status_value = cls.UNIVERSAL_TO_STATUS.get(universal_params.status)
            if status_value:
                anisearch_params["status"] = status_value
        
        # Format mapping
        if universal_params.type_format:
            type_value = cls.UNIVERSAL_TO_TYPE.get(universal_params.type_format)
            if type_value:
                anisearch_params["type"] = type_value
        
        # Genre filters
        if universal_params.genres:
            anisearch_params["genres"] = universal_params.genres
        
        # Score filters
        if universal_params.min_score is not None:
            anisearch_params["min_rating"] = universal_params.min_score
        
        # Year filter
        if universal_params.year:
            anisearch_params["year"] = universal_params.year
        
        # Result control
        anisearch_params["limit"] = universal_params.limit
        if universal_params.offset:
            anisearch_params["offset"] = universal_params.offset
        
        return anisearch_params