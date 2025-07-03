"""AnimeSchedule API v3 query parameter mapper.

This mapper handles the conversion from universal search parameters
to AnimeSchedule API v3 parameters. AnimeSchedule provides comprehensive
filtering with 25 parameters including:

VERIFIED FEATURES:
- Text search with multi-field matching
- 9 media formats (TV, Movie, OVA, ONA, Special, Music, TV Short, etc.)
- 3 airing statuses (finished, ongoing, upcoming)  
- 13 source materials (Manga, Light Novel, Original, etc.)
- Comprehensive exclude filtering for all categories
- External ID support (MAL, AniList, AniDB)
- Streaming platform filtering
- Broadcasting schedules and timing data

UNIQUE CAPABILITIES:
- Both include AND exclude filtering for most categories
- Cross-platform ID filtering for data enrichment
- Streaming service integration
- Comprehensive temporal filtering with exclude options
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
    
    # Universal to AnimeSchedule status mappings (verified with API)
    UNIVERSAL_TO_STATUS = {
        AnimeStatus.FINISHED: "finished",
        AnimeStatus.RELEASING: "ongoing", 
        AnimeStatus.NOT_YET_RELEASED: "upcoming",
    }
    
    # Universal to AnimeSchedule format mappings (verified with API)
    UNIVERSAL_TO_MEDIA_TYPE = {
        AnimeFormat.TV: "TV",
        AnimeFormat.MOVIE: "Movie",
        AnimeFormat.SPECIAL: "Special",
        AnimeFormat.OVA: "OVA",
        AnimeFormat.ONA: "ONA",
        AnimeFormat.MUSIC: "Music",
        AnimeFormat.TV_SPECIAL: "TV Short",
    }
    
    # Universal to AnimeSchedule season mappings (verified with API)
    UNIVERSAL_TO_SEASON = {
        AnimeSeason.WINTER: "Winter",
        AnimeSeason.SPRING: "Spring", 
        AnimeSeason.SUMMER: "Summer",
        AnimeSeason.FALL: "Fall",
    }
    
    @classmethod
    def to_animeschedule_search_params(cls, universal_params: UniversalSearchParams) -> Dict[str, Any]:
        """Convert universal search parameters to AnimeSchedule API v3 parameters.
        
        AnimeSchedule API v3 supports 25 parameters with comprehensive filtering:
        - Core search: q, page, mt, st
        - Content filters: genres, studios, sources, media-types, airing-statuses
        - Temporal filters: years, seasons (with exclude options)
        - Content filters: duration, episodes
        - Streaming: streams (with exclude options)
        - External IDs: mal-ids, anilist-ids, anidb-ids
        - All categories support both include and exclude filtering
        
        Args:
            universal_params: Universal search parameters
            
        Returns:
            Dictionary of AnimeSchedule API v3 parameters
        """
        as_params = {}
        
        # CORE SEARCH PARAMETERS
        if universal_params.query:
            as_params["q"] = universal_params.query
        
        # Pagination - AnimeSchedule uses page numbers (18 per page fixed)
        if universal_params.offset and universal_params.limit:
            page = (universal_params.offset // 18) + 1
            as_params["page"] = page
        
        # CONTENT CLASSIFICATION PARAMETERS
        if universal_params.status:
            status = cls.UNIVERSAL_TO_STATUS.get(universal_params.status)
            if status:
                as_params["airing-statuses"] = status
        
        if universal_params.type_format:
            type_format = cls.UNIVERSAL_TO_MEDIA_TYPE.get(universal_params.type_format)
            if type_format:
                as_params["media-types"] = type_format
        
        if universal_params.source:
            as_params["sources"] = universal_params.source
        
        # CONTENT FILTERING PARAMETERS  
        if universal_params.genres:
            as_params["genres"] = ",".join(universal_params.genres)
        
        if universal_params.genres_exclude:
            as_params["genres-exclude"] = ",".join(universal_params.genres_exclude)
        
        # PRODUCTION PARAMETERS
        if universal_params.studios:
            as_params["studios"] = ",".join(universal_params.studios)
        
        # TEMPORAL PARAMETERS
        if universal_params.year:
            as_params["years"] = str(universal_params.year)
        
        if universal_params.season:
            season = cls.UNIVERSAL_TO_SEASON.get(universal_params.season)
            if season:
                as_params["seasons"] = season
        
        # EPISODE/DURATION PARAMETERS
        if universal_params.episodes:
            as_params["episodes"] = str(universal_params.episodes)
        
        # Duration handling - AnimeSchedule uses single duration parameter, not ranges
        if universal_params.duration:
            as_params["duration"] = str(universal_params.duration)
        elif universal_params.min_duration:
            as_params["duration"] = str(universal_params.min_duration)
        elif universal_params.max_duration:
            as_params["duration"] = str(universal_params.max_duration)
        
        # SORTING PARAMETERS
        if universal_params.sort_by:
            # Map universal sort fields to AnimeSchedule sort options
            sort_mapping = {
                "score": "score",
                "popularity": "popularity", 
                "title": "alphabetic",
                "year": "releaseDate",
                "start_date": "releaseDate"
            }
            sort_by = sort_mapping.get(universal_params.sort_by, "popularity")
            as_params["st"] = sort_by
        
        # ANIMESCHEDULE-SPECIFIC PARAMETERS
        if universal_params.animeschedule_mt:
            as_params["mt"] = universal_params.animeschedule_mt
        
        if universal_params.animeschedule_st:
            as_params["st"] = universal_params.animeschedule_st
        
        if universal_params.animeschedule_streams:
            as_params["streams"] = ",".join(universal_params.animeschedule_streams)
        
        if universal_params.animeschedule_streams_exclude:
            as_params["streams-exclude"] = ",".join(universal_params.animeschedule_streams_exclude)
        
        if universal_params.animeschedule_sources_exclude:
            as_params["sources-exclude"] = ",".join(universal_params.animeschedule_sources_exclude)
        
        if universal_params.animeschedule_studios_exclude:
            as_params["studios-exclude"] = ",".join(universal_params.animeschedule_studios_exclude)
        
        if universal_params.animeschedule_media_types_exclude:
            as_params["media-types-exclude"] = ",".join(universal_params.animeschedule_media_types_exclude)
        
        if universal_params.animeschedule_airing_statuses_exclude:
            as_params["airing-statuses-exclude"] = ",".join(universal_params.animeschedule_airing_statuses_exclude)
        
        if universal_params.animeschedule_years_exclude:
            as_params["years-exclude"] = ",".join(map(str, universal_params.animeschedule_years_exclude))
        
        if universal_params.animeschedule_seasons_exclude:
            as_params["seasons-exclude"] = ",".join(universal_params.animeschedule_seasons_exclude)
        
        if universal_params.animeschedule_mal_ids:
            as_params["mal-ids"] = ",".join(map(str, universal_params.animeschedule_mal_ids))
        
        if universal_params.animeschedule_anilist_ids:
            as_params["anilist-ids"] = ",".join(map(str, universal_params.animeschedule_anilist_ids))
        
        if universal_params.animeschedule_anidb_ids:
            as_params["anidb-ids"] = ",".join(map(str, universal_params.animeschedule_anidb_ids))
        
        return as_params