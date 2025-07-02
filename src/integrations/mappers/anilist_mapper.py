"""AniList GraphQL query parameter mapper.

This mapper handles the conversion from universal search parameters
to AniList GraphQL variables, addressing the parameter mapping chaos.
"""

from typing import Any, Dict
from ...models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class AniListMapper:
    """Query parameter mapper from universal schema to AniList GraphQL."""
    
    # Universal to AniList status mappings (for query parameters)
    UNIVERSAL_TO_STATUS = {
        AnimeStatus.FINISHED: "FINISHED",
        AnimeStatus.RELEASING: "RELEASING",
        AnimeStatus.NOT_YET_RELEASED: "NOT_YET_RELEASED",
        AnimeStatus.CANCELLED: "CANCELLED",
        AnimeStatus.HIATUS: "HIATUS",
    }
    
    # Universal to AniList format mappings (for query parameters)
    UNIVERSAL_TO_FORMAT = {
        AnimeFormat.TV: "TV",
        AnimeFormat.TV_SHORT: "TV_SHORT",
        AnimeFormat.MOVIE: "MOVIE",
        AnimeFormat.SPECIAL: "SPECIAL",
        AnimeFormat.OVA: "OVA",
        AnimeFormat.ONA: "ONA",
        AnimeFormat.MUSIC: "MUSIC",
    }
    
    # Universal to AniList season mappings (for query parameters)
    UNIVERSAL_TO_SEASON = {
        AnimeSeason.WINTER: "WINTER",
        AnimeSeason.SPRING: "SPRING",
        AnimeSeason.SUMMER: "SUMMER",
        AnimeSeason.FALL: "FALL",
    }
    
    @classmethod
    def to_anilist_search_params(cls, universal_params: UniversalSearchParams, anilist_specific: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert universal search parameters to AniList GraphQL variables.
        
        This addresses the parameter mapping chaos by providing a central
        conversion point from universal parameters to AniList GraphQL variables.
        
        Args:
            universal_params: Universal search parameters
            
        Returns:
            Dictionary of AniList GraphQL variables
        """
        if anilist_specific is None:
            anilist_specific = {}
        anilist_params = {}
        
        # Text search
        if universal_params.query:
            anilist_params["search"] = universal_params.query
        
        # Status mapping
        if universal_params.status:
            status_value = cls.UNIVERSAL_TO_STATUS.get(universal_params.status)
            if status_value:
                anilist_params["status"] = status_value
        
        # AniList-specific status override
        status = anilist_specific.get("status")
        if status:
            anilist_params["status"] = status
        
        # Format mapping (AniList supports all formats including TV_SHORT)
        if universal_params.type_format:
            format_value = cls.UNIVERSAL_TO_FORMAT.get(universal_params.type_format)
            if format_value:
                anilist_params["format"] = format_value
        
        # Score filters (AniList uses 0-100 scale, convert from 0-10)
        if universal_params.min_score is not None:
            anilist_params["averageScore_greater"] = int(universal_params.min_score * 10)
        if universal_params.max_score is not None:
            anilist_params["averageScore_lesser"] = int(universal_params.max_score * 10)
        
        # Episode filters
        if universal_params.min_episodes is not None:
            anilist_params["episodes_greater"] = universal_params.min_episodes
        if universal_params.max_episodes is not None:
            anilist_params["episodes_lesser"] = universal_params.max_episodes
        
        # Duration filters (AniList duration is in minutes)
        if universal_params.min_duration is not None:
            anilist_params["duration_greater"] = universal_params.min_duration
        if universal_params.max_duration is not None:
            anilist_params["duration_lesser"] = universal_params.max_duration
        
        # Genre filters (AniList accepts genre arrays)
        if universal_params.genres:
            anilist_params["genre_in"] = universal_params.genres
        if universal_params.genres_exclude:
            anilist_params["genre_not_in"] = universal_params.genres_exclude
        
        # Theme/Tag filters (AniList-specific feature)
        if universal_params.themes:
            anilist_params["tag_in"] = universal_params.themes
        if universal_params.themes_exclude:
            anilist_params["tag_not_in"] = universal_params.themes_exclude
        
        # Year and season filters
        if universal_params.year:
            anilist_params["seasonYear"] = universal_params.year
        if universal_params.season:
            anilist_params["season"] = cls.UNIVERSAL_TO_SEASON.get(universal_params.season)
        
        # Start date filter
        if universal_params.start_date:
            # Convert YYYY-MM-DD to AniList fuzzy date format
            try:
                date_parts = universal_params.start_date.split("-")
                if len(date_parts) >= 3:
                    year = int(date_parts[0])
                    month = int(date_parts[1])
                    day = int(date_parts[2])
                    anilist_params["startDate_greater"] = year * 10000 + month * 100 + day
            except (ValueError, IndexError):
                pass
        
        # End date filter
        if universal_params.end_date:
            # Convert YYYY-MM-DD to AniList fuzzy date format
            try:
                date_parts = universal_params.end_date.split("-")
                if len(date_parts) >= 3:
                    year = int(date_parts[0])
                    month = int(date_parts[1])
                    day = int(date_parts[2])
                    anilist_params["endDate"] = year * 10000 + month * 100 + day
            except (ValueError, IndexError):
                pass
        
        # Popularity filter
        if universal_params.min_popularity is not None:
            anilist_params["popularity_greater"] = universal_params.min_popularity
        
        # Adult content filter
        if not universal_params.include_adult:
            anilist_params["isAdult"] = False
        
        # Result control
        anilist_params["perPage"] = universal_params.limit
        if universal_params.offset:
            anilist_params["page"] = (universal_params.offset // universal_params.limit) + 1
        
        # Sort - AniList has comprehensive sort options
        if universal_params.sort_by:
            sort_base = {
                "score": "SCORE",
                "popularity": "POPULARITY", 
                "title": "TITLE_ROMAJI",
                "year": "START_DATE",
                "episodes": "EPISODES",
                "duration": "DURATION",
            }
            base_sort = sort_base.get(universal_params.sort_by)
            if base_sort:
                # Add direction suffix
                if universal_params.sort_order == "desc":
                    sort_value = f"{base_sort}_DESC"
                elif universal_params.sort_order == "asc":
                    sort_value = f"{base_sort}_ASC"
                else:
                    sort_value = base_sort  # Default no suffix
                anilist_params["sort"] = [sort_value]
        
        # ANILIST-SPECIFIC PARAMETERS (All 69+ GraphQL parameters)
        # Basic filters
        # New basic filter parameters  
        id = anilist_specific.get("id") or universal_params.anilist_id
        if id is not None:
            anilist_params["id"] = id
            
        endDate = anilist_specific.get("endDate") or universal_params.anilist_end_date
        if endDate is not None:
            anilist_params["endDate"] = endDate
            
        format = anilist_specific.get("format") or universal_params.anilist_format
        if format:
            anilist_params["format"] = format
            
            
        id_mal = anilist_specific.get("idMal") or universal_params.anilist_id_mal
        if id_mal is not None:
            anilist_params["idMal"] = id_mal
            
        start_date = anilist_specific.get("startDate") or universal_params.anilist_start_date
        if start_date is not None:
            anilist_params["startDate"] = start_date
            
        season_specific = anilist_specific.get("season") or universal_params.anilist_season
        if season_specific:
            anilist_params["season"] = season_specific
            
        season_year = anilist_specific.get("seasonYear") or universal_params.anilist_season_year
        if season_year is not None:
            anilist_params["seasonYear"] = season_year
            
        episodes_exact = anilist_specific.get("episodes") or universal_params.anilist_episodes
        if episodes_exact is not None:
            anilist_params["episodes"] = episodes_exact
            
        duration_exact = anilist_specific.get("duration") or universal_params.anilist_duration
        if duration_exact is not None:
            anilist_params["duration"] = duration_exact
            
        chapters = anilist_specific.get("chapters") or universal_params.anilist_chapters
        if chapters is not None:
            anilist_params["chapters"] = chapters
            
        volumes = anilist_specific.get("volumes") or universal_params.anilist_volumes
        if volumes is not None:
            anilist_params["volumes"] = volumes
            
        is_adult_specific = anilist_specific.get("isAdult") or universal_params.anilist_is_adult
        if is_adult_specific is not None:
            anilist_params["isAdult"] = is_adult_specific
            
        genre_single = anilist_specific.get("genre") or universal_params.anilist_genre
        if genre_single:
            anilist_params["genre"] = genre_single
            
        tag_single = anilist_specific.get("tag") or universal_params.anilist_tag
        if tag_single:
            anilist_params["tag"] = tag_single
            
        min_tag_rank = anilist_specific.get("minimumTagRank") or universal_params.anilist_minimum_tag_rank
        if min_tag_rank is not None:
            anilist_params["minimumTagRank"] = min_tag_rank
            
        tag_category = anilist_specific.get("tagCategory") or universal_params.anilist_tag_category
        if tag_category:
            anilist_params["tagCategory"] = tag_category
            
        on_list = anilist_specific.get("onList") or universal_params.anilist_on_list
        if on_list is not None:
            anilist_params["onList"] = on_list
            
        licensed_by = anilist_specific.get("licensedBy") or universal_params.anilist_licensed_by
        if licensed_by:
            anilist_params["licensedBy"] = licensed_by
            
        licensed_by_id = anilist_specific.get("licensedById") or universal_params.anilist_licensed_by_id
        if licensed_by_id is not None:
            anilist_params["licensedById"] = licensed_by_id
            
        avg_score_exact = anilist_specific.get("averageScore") or universal_params.anilist_average_score
        if avg_score_exact is not None:
            anilist_params["averageScore"] = avg_score_exact
            
        popularity_exact = anilist_specific.get("popularity") or universal_params.anilist_popularity
        if popularity_exact is not None:
            anilist_params["popularity"] = popularity_exact
            
        source_specific = anilist_specific.get("source") or universal_params.anilist_source
        if source_specific:
            anilist_params["source"] = source_specific
            
        country = anilist_specific.get("countryOfOrigin") or universal_params.anilist_country_of_origin
        if country:
            anilist_params["countryOfOrigin"] = country
            
        is_licensed = anilist_specific.get("isLicensed") or universal_params.anilist_is_licensed
        if is_licensed is not None:
            anilist_params["isLicensed"] = is_licensed
        
        # Negation filters
        id_not = anilist_specific.get("id_not") or universal_params.anilist_id_not
        if id_not is not None:
            anilist_params["id_not"] = id_not
            
        id_mal_not = anilist_specific.get("idMal_not") or universal_params.anilist_id_mal_not
        if id_mal_not is not None:
            anilist_params["idMal_not"] = id_mal_not
            
        format_not = anilist_specific.get("format_not") or universal_params.anilist_format_not
        if format_not:
            anilist_params["format_not"] = format_not
            
        if universal_params.status:
            anilist_params["status_not"] = universal_params.status
            
        avg_score_not = anilist_specific.get("averageScore_not") or universal_params.anilist_average_score_not
        if avg_score_not is not None:
            anilist_params["averageScore_not"] = avg_score_not
            
        popularity_not = anilist_specific.get("popularity_not") or universal_params.anilist_popularity_not
        if popularity_not is not None:
            anilist_params["popularity_not"] = popularity_not
        
        # Array inclusion filters
        id_in = anilist_specific.get("id_in") or universal_params.anilist_id_in
        if id_in:
            anilist_params["id_in"] = id_in
            
        id_not_in = anilist_specific.get("id_not_in") or universal_params.anilist_id_not_in
        if id_not_in:
            anilist_params["id_not_in"] = id_not_in
            
        id_mal_in = anilist_specific.get("idMal_in") or universal_params.anilist_id_mal_in
        if id_mal_in:
            anilist_params["idMal_in"] = id_mal_in
            
        id_mal_not_in = anilist_specific.get("idMal_not_in") or universal_params.anilist_id_mal_not_in
        if id_mal_not_in:
            anilist_params["idMal_not_in"] = id_mal_not_in
            
        format_in = anilist_specific.get("format_in") or universal_params.anilist_format_in
        if format_in:
            anilist_params["format_in"] = format_in
            
        format_not_in = anilist_specific.get("format_not_in") or universal_params.anilist_format_not_in
        if format_not_in:
            anilist_params["format_not_in"] = format_not_in
            
        status_in = anilist_specific.get("status_in") or universal_params.anilist_status_in
        if status_in:
            anilist_params["status_in"] = status_in
            
        status_not_in = anilist_specific.get("status_not_in") or universal_params.anilist_status_not_in
        if status_not_in:
            anilist_params["status_not_in"] = status_not_in
            
        genre_in_specific = anilist_specific.get("genre_in") or universal_params.anilist_genre_in
        if genre_in_specific:
            anilist_params["genre_in"] = genre_in_specific
            
        genre_not_in = anilist_specific.get("genre_not_in") or universal_params.anilist_genre_not_in
        if genre_not_in:
            anilist_params["genre_not_in"] = genre_not_in
            
        tag_in_specific = anilist_specific.get("tag_in") or universal_params.anilist_tag_in
        if tag_in_specific:
            anilist_params["tag_in"] = tag_in_specific
            
        tag_not_in = anilist_specific.get("tag_not_in") or universal_params.anilist_tag_not_in
        if tag_not_in:
            anilist_params["tag_not_in"] = tag_not_in
            
        tag_category_in = anilist_specific.get("tagCategory_in") or universal_params.anilist_tag_category_in
        if tag_category_in:
            anilist_params["tagCategory_in"] = tag_category_in
            
        tag_category_not_in = anilist_specific.get("tagCategory_not_in") or universal_params.anilist_tag_category_not_in
        if tag_category_not_in:
            anilist_params["tagCategory_not_in"] = tag_category_not_in
            
        licensed_by_in = anilist_specific.get("licensedBy_in") or universal_params.anilist_licensed_by_in
        if licensed_by_in:
            anilist_params["licensedBy_in"] = licensed_by_in
            
        licensed_by_id_in = anilist_specific.get("licensedById_in") or universal_params.anilist_licensed_by_id_in
        if licensed_by_id_in:
            anilist_params["licensedById_in"] = licensed_by_id_in
            
        source_in = anilist_specific.get("source_in") or universal_params.anilist_source_in
        if source_in:
            anilist_params["source_in"] = source_in
            
        # New array filters
        licensed_by_not_in = anilist_specific.get("licensedBy_not_in") or universal_params.anilist_licensed_by_not_in
        if licensed_by_not_in:
            anilist_params["licensedBy_not_in"] = licensed_by_not_in
            
        source_not_in = anilist_specific.get("source_not_in") or universal_params.anilist_source_not_in
        if source_not_in:
            anilist_params["source_not_in"] = source_not_in
            
        format_range = anilist_specific.get("format_range") or universal_params.anilist_format_range
        if format_range:
            anilist_params["format_range"] = format_range
        
        # Range filters
        start_date_greater = anilist_specific.get("startDate_greater") or universal_params.anilist_start_date_greater
        if start_date_greater is not None:
            anilist_params["startDate_greater"] = start_date_greater
            
        start_date_lesser = anilist_specific.get("startDate_lesser") or universal_params.anilist_start_date_lesser
        if start_date_lesser is not None:
            anilist_params["startDate_lesser"] = start_date_lesser
            
        start_date_like = anilist_specific.get("startDate_like") or universal_params.anilist_start_date_like
        if start_date_like:
            anilist_params["startDate_like"] = start_date_like
            
        end_date_greater = anilist_specific.get("endDate_greater") or universal_params.anilist_end_date_greater
        if end_date_greater is not None:
            anilist_params["endDate_greater"] = end_date_greater
            
        end_date_lesser = anilist_specific.get("endDate_lesser") or universal_params.anilist_end_date_lesser
        if end_date_lesser is not None:
            anilist_params["endDate_lesser"] = end_date_lesser
            
        end_date_like = anilist_specific.get("endDate_like") or universal_params.anilist_end_date_like
        if end_date_like:
            anilist_params["endDate_like"] = end_date_like
            
        episodes_greater = anilist_specific.get("episodes_greater") or universal_params.anilist_episodes_greater
        if episodes_greater is not None:
            anilist_params["episodes_greater"] = episodes_greater
            
        episodes_lesser = anilist_specific.get("episodes_lesser") or universal_params.anilist_episodes_lesser
        if episodes_lesser is not None:
            anilist_params["episodes_lesser"] = episodes_lesser
            
        duration_greater = anilist_specific.get("duration_greater") or universal_params.anilist_duration_greater
        if duration_greater is not None:
            anilist_params["duration_greater"] = duration_greater
            
        duration_lesser = anilist_specific.get("duration_lesser") or universal_params.anilist_duration_lesser
        if duration_lesser is not None:
            anilist_params["duration_lesser"] = duration_lesser
            
        chapters_greater = anilist_specific.get("chapters_greater") or universal_params.anilist_chapters_greater
        if chapters_greater is not None:
            anilist_params["chapters_greater"] = chapters_greater
            
        chapters_lesser = anilist_specific.get("chapters_lesser") or universal_params.anilist_chapters_lesser
        if chapters_lesser is not None:
            anilist_params["chapters_lesser"] = chapters_lesser
            
        volumes_greater = anilist_specific.get("volumes_greater") or universal_params.anilist_volumes_greater
        if volumes_greater is not None:
            anilist_params["volumes_greater"] = volumes_greater
            
        volumes_lesser = anilist_specific.get("volumes_lesser") or universal_params.anilist_volumes_lesser
        if volumes_lesser is not None:
            anilist_params["volumes_lesser"] = volumes_lesser
            
        avg_score_greater = anilist_specific.get("averageScore_greater") or universal_params.anilist_average_score_greater
        if avg_score_greater is not None:
            anilist_params["averageScore_greater"] = avg_score_greater
            
        avg_score_lesser = anilist_specific.get("averageScore_lesser") or universal_params.anilist_average_score_lesser
        if avg_score_lesser is not None:
            anilist_params["averageScore_lesser"] = avg_score_lesser
            
        popularity_greater = anilist_specific.get("popularity_greater") or universal_params.anilist_popularity_greater
        if popularity_greater is not None:
            anilist_params["popularity_greater"] = popularity_greater
            
        popularity_lesser = anilist_specific.get("popularity_lesser") or universal_params.anilist_popularity_lesser
        if popularity_lesser is not None:
            anilist_params["popularity_lesser"] = popularity_lesser
        
        # Special sorting (AniList-specific override)
        sort_specific = anilist_specific.get("sort") or universal_params.anilist_sort
        if sort_specific:
            anilist_params["sort"] = sort_specific
        
        return anilist_params