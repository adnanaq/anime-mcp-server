"""AniDB query parameter mapper.

This mapper handles the conversion from universal search parameters
to AniDB API query parameters. AniDB has unique features like 
episode-level data and multiple rating types.
"""

from typing import Any, Dict, List, Optional
from ...models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class AniDBMapper:
    """Query parameter mapper from universal schema to AniDB API."""
    
    # Universal to AniDB format mappings (for query parameters)
    UNIVERSAL_TO_TYPE = {
        AnimeFormat.TV: "TV Series",
        AnimeFormat.MOVIE: "Movie",
        AnimeFormat.SPECIAL: "TV Special",
        AnimeFormat.OVA: "OVA",
        AnimeFormat.ONA: "Web",  # AniDB calls ONA "Web"
        # Note: TV_SHORT, MUSIC not specifically supported by AniDB
    }
    
    @classmethod
    def to_anidb_search_params(cls, universal_params: UniversalSearchParams) -> Dict[str, Any]:
        """Convert universal search parameters to AniDB API parameters.
        
        AniDB has unique characteristics:
        - No direct status field (derived from start/end dates)
        - Rich tag system combining genres, themes, demographics
        - Multiple rating types (permanent, temporary, review)
        - Episode-level data support
        
        Args:
            universal_params: Universal search parameters
            
        Returns:
            Dictionary of AniDB API parameters
        """
        anidb_params = {}
        
        # Text search
        if universal_params.query:
            anidb_params["query"] = universal_params.query
        
        # Format mapping
        if universal_params.type_format:
            type_value = cls.UNIVERSAL_TO_TYPE.get(universal_params.type_format)
            if type_value:
                anidb_params["type"] = type_value
        
        # Status handling - AniDB derives status from dates
        if universal_params.status:
            cls._add_status_date_filters(anidb_params, universal_params.status)
        
        # Score filters (AniDB uses 0-10 scale like universal)
        if universal_params.min_score is not None:
            anidb_params["min_rating"] = universal_params.min_score
        if universal_params.max_score is not None:
            anidb_params["max_rating"] = universal_params.max_score
        
        # Episode filters
        if universal_params.min_episodes is not None:
            anidb_params["min_episodes"] = universal_params.min_episodes
        if universal_params.max_episodes is not None:
            anidb_params["max_episodes"] = universal_params.max_episodes
        
        # Date filters
        if universal_params.year:
            anidb_params["start_date"] = f"{universal_params.year}-01-01"
        elif universal_params.start_date:
            anidb_params["start_date"] = universal_params.start_date
        
        if universal_params.end_date:
            anidb_params["end_date"] = universal_params.end_date
        
        # Tag system - combine genres, themes, demographics
        tags = []
        if universal_params.genres:
            tags.extend(universal_params.genres)
        if universal_params.themes:
            tags.extend(universal_params.themes)
        if universal_params.demographics:
            tags.extend(universal_params.demographics)
        
        if tags:
            anidb_params["tags"] = tags
        
        # Adult content filter
        if not universal_params.include_adult:
            anidb_params["restricted"] = "false"
        
        # Result control
        anidb_params["limit"] = universal_params.limit
        if universal_params.offset:
            anidb_params["offset"] = universal_params.offset
        
        # Sort - AniDB has specific sort fields
        if universal_params.sort_by:
            sort_mapping = {
                "score": "rating",
                "year": "startdate",
                "title": "title",
                "episodes": "episodes",
                "popularity": "rating",  # AniDB doesn't have popularity, use rating
            }
            sort_field = sort_mapping.get(universal_params.sort_by)
            if sort_field:
                anidb_params["sort_by"] = sort_field
                if universal_params.sort_order:
                    anidb_params["sort_order"] = universal_params.sort_order
        
        return anidb_params
    
    @classmethod
    def _add_status_date_filters(cls, params: Dict[str, Any], status: AnimeStatus) -> None:
        """Add date filters to derive status since AniDB doesn't have direct status field.
        
        Args:
            params: Parameter dictionary to modify
            status: Universal status to derive from
        """
        from datetime import datetime
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        if status == AnimeStatus.FINISHED:
            # Both start and end dates should be in the past
            params["end_date"] = f"..{current_date}"
        elif status == AnimeStatus.RELEASING:
            # Started but not finished (no end date or future end date)
            params["start_date"] = f"..{current_date}"
            # Could add logic to exclude finished shows
        elif status == AnimeStatus.NOT_YET_RELEASED:
            # Start date in the future
            params["start_date"] = f"{current_date}.."
        # Note: CANCELLED and HIATUS are derived from other metadata in AniDB