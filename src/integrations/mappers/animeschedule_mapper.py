"""AnimeSchedule query parameter mapper.

This mapper handles the conversion from universal search parameters
to AnimeSchedule API parameters. AnimeSchedule focuses on broadcasting
schedules and temporal data.
"""

from typing import Any, Dict
from ...models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class AnimeScheduleMapper:
    """Query parameter mapper from universal schema to AnimeSchedule API."""
    
    # Universal to AnimeSchedule status mappings
    UNIVERSAL_TO_STATUS = {
        AnimeStatus.FINISHED: "Finished",
        AnimeStatus.RELEASING: "Ongoing",
        AnimeStatus.NOT_YET_RELEASED: "Upcoming",
    }
    
    # Universal to AnimeSchedule format mappings
    UNIVERSAL_TO_MEDIA_TYPE = {
        AnimeFormat.TV: "TV",
        AnimeFormat.MOVIE: "Movie",
        AnimeFormat.SPECIAL: "Special",
        AnimeFormat.OVA: "OVA",
    }
    
    # Universal to AnimeSchedule season mappings
    UNIVERSAL_TO_SEASON = {
        AnimeSeason.WINTER: "Winter",
        AnimeSeason.SPRING: "Spring",
        AnimeSeason.SUMMER: "Summer",
        AnimeSeason.FALL: "Fall",
    }
    
    @classmethod
    def to_animeschedule_search_params(cls, universal_params: UniversalSearchParams) -> Dict[str, Any]:
        """Convert universal search parameters to AnimeSchedule API parameters.
        
        AnimeSchedule specializes in:
        - Broadcasting schedules and temporal data
        - Season/year filtering
        - Status tracking for airing shows
        - Limited but focused parameter set
        
        Args:
            universal_params: Universal search parameters
            
        Returns:
            Dictionary of AnimeSchedule API parameters
        """
        as_params = {}
        
        # Text search
        if universal_params.query:
            as_params["title"] = universal_params.query
        
        # Status mapping
        if universal_params.status:
            status_value = cls.UNIVERSAL_TO_STATUS.get(universal_params.status)
            if status_value:
                as_params["status"] = status_value
        
        # Format mapping
        if universal_params.type_format:
            media_type = cls.UNIVERSAL_TO_MEDIA_TYPE.get(universal_params.type_format)
            if media_type:
                as_params["mediaType"] = media_type
        
        # Temporal filters (AnimeSchedule specialty)
        if universal_params.year:
            as_params["year"] = universal_params.year
        
        if universal_params.season:
            season_value = cls.UNIVERSAL_TO_SEASON.get(universal_params.season)
            if season_value:
                as_params["season"] = season_value
        
        # Genre filters
        if universal_params.genres:
            as_params["genres"] = universal_params.genres
        
        # Score filters (if supported)
        if universal_params.min_score is not None:
            as_params["minScore"] = universal_params.min_score
        
        # Result control
        as_params["limit"] = universal_params.limit
        if universal_params.offset:
            as_params["offset"] = universal_params.offset
        
        return as_params