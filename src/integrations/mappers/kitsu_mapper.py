"""Kitsu JSON:API query parameter mapper.

This mapper handles the conversion from universal search parameters
to Kitsu API query parameters. Kitsu uses JSON:API standard with 
comprehensive metadata and category support.
"""

from typing import Any, Dict
from ...models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class KitsuMapper:
    """Query parameter mapper from universal schema to Kitsu JSON:API."""
    
    # Universal to Kitsu status mappings (for query parameters)
    UNIVERSAL_TO_STATUS = {
        AnimeStatus.FINISHED: "finished",
        AnimeStatus.RELEASING: "current",
        AnimeStatus.NOT_YET_RELEASED: "upcoming",
    }
    
    # Universal to Kitsu subtype mappings (for query parameters)
    UNIVERSAL_TO_SUBTYPE = {
        AnimeFormat.TV: "TV",
        AnimeFormat.MOVIE: "movie",
        AnimeFormat.SPECIAL: "special",
        AnimeFormat.OVA: "OVA",
        AnimeFormat.ONA: "ONA",
        AnimeFormat.MUSIC: "music",
    }
    
    @classmethod
    def to_kitsu_search_params(cls, universal_params: UniversalSearchParams) -> Dict[str, Any]:
        """Convert universal search parameters to Kitsu JSON:API parameters.
        
        Kitsu uses JSON:API standard with filter[attribute] syntax for
        query parameters and comprehensive category/production filtering.
        
        Args:
            universal_params: Universal search parameters
            
        Returns:
            Dictionary of Kitsu API filter parameters
        """
        kitsu_params = {}
        
        # Text search - Kitsu supports text filtering
        if universal_params.query:
            kitsu_params["filter[text]"] = universal_params.query
        
        # Status mapping
        if universal_params.status:
            status_value = cls.UNIVERSAL_TO_STATUS.get(universal_params.status)
            if status_value:
                kitsu_params["filter[status]"] = status_value
        
        # Format/subtype mapping
        if universal_params.type_format:
            subtype_value = cls.UNIVERSAL_TO_SUBTYPE.get(universal_params.type_format)
            if subtype_value:
                kitsu_params["filter[subtype]"] = subtype_value
        
        # Score filters (Kitsu uses 1-5 star rating, convert from 0-10)
        if universal_params.min_score is not None:
            # Convert 0-10 to 1-5 stars (multiply by 0.5, add 0.5 for rounding)
            kitsu_rating = max(1, min(5, (universal_params.min_score / 2) + 0.5))
            kitsu_params["filter[averageRating]"] = f"{kitsu_rating:.1f}.."
        if universal_params.max_score is not None:
            kitsu_rating = max(1, min(5, (universal_params.max_score / 2) + 0.5))
            existing_filter = kitsu_params.get("filter[averageRating]", "..")
            if ".." in existing_filter:
                min_part = existing_filter.split("..")[0]
                kitsu_params["filter[averageRating]"] = f"{min_part}..{kitsu_rating:.1f}"
            else:
                kitsu_params["filter[averageRating]"] = f"..{kitsu_rating:.1f}"
        
        # Episode filters
        if universal_params.min_episodes is not None:
            kitsu_params["filter[episodeCount]"] = f"{universal_params.min_episodes}.."
        if universal_params.max_episodes is not None:
            existing_filter = kitsu_params.get("filter[episodeCount]", "..")
            if ".." in existing_filter:
                min_part = existing_filter.split("..")[0]
                kitsu_params["filter[episodeCount]"] = f"{min_part}..{universal_params.max_episodes}"
            else:
                kitsu_params["filter[episodeCount]"] = f"..{universal_params.max_episodes}"
        
        # Duration filters (Kitsu stores episode length in minutes)
        if universal_params.min_duration is not None:
            kitsu_params["filter[episodeLength]"] = f"{universal_params.min_duration}.."
        if universal_params.max_duration is not None:
            existing_filter = kitsu_params.get("filter[episodeLength]", "..")
            if ".." in existing_filter:
                min_part = existing_filter.split("..")[0]
                kitsu_params["filter[episodeLength]"] = f"{min_part}..{universal_params.max_duration}"
            else:
                kitsu_params["filter[episodeLength]"] = f"..{universal_params.max_duration}"
        
        # Year filter (Kitsu filters by startDate year)
        if universal_params.year:
            # Format as YYYY-01-01..YYYY-12-31 range
            kitsu_params["filter[startDate]"] = f"{universal_params.year}-01-01..{universal_params.year}-12-31"
        elif universal_params.start_date:
            kitsu_params["filter[startDate]"] = f"{universal_params.start_date}.."
        
        # End date filter
        if universal_params.end_date:
            existing_filter = kitsu_params.get("filter[startDate]", "..")
            if ".." in existing_filter:
                min_part = existing_filter.split("..")[0]
                kitsu_params["filter[startDate]"] = f"{min_part}..{universal_params.end_date}"
            else:
                kitsu_params["filter[startDate]"] = f"..{universal_params.end_date}"
        
        # Categories (genres) - Kitsu has rich category system
        if universal_params.genres:
            # Kitsu supports category filtering via relationships
            kitsu_params["filter[categories]"] = ",".join(universal_params.genres)
        
        # Adult content filter
        if not universal_params.include_adult:
            kitsu_params["filter[nsfw]"] = "false"
        
        # Result control
        kitsu_params["page[limit]"] = universal_params.limit
        if universal_params.offset:
            # Kitsu uses page-based pagination
            page_number = (universal_params.offset // universal_params.limit) + 1
            kitsu_params["page[number]"] = page_number
        
        # Sort - Kitsu has comprehensive sort options
        if universal_params.sort_by:
            sort_mapping = {
                "score": "averageRating",
                "popularity": "popularityRank",
                "title": "canonicalTitle",
                "year": "startDate",
                "episodes": "episodeCount",
                "duration": "episodeLength",
                "rank": "ratingRank",
            }
            sort_field = sort_mapping.get(universal_params.sort_by)
            if sort_field:
                if universal_params.sort_order == "desc":
                    kitsu_params["sort"] = f"-{sort_field}"
                else:
                    kitsu_params["sort"] = sort_field
        
        # Include related data for comprehensive responses
        includes = []
        if universal_params.genres:
            includes.append("categories")
        if universal_params.studios:
            includes.append("animeProductions.producer")
        if includes:
            kitsu_params["include"] = ",".join(includes)
        
        return kitsu_params
    
    @classmethod
    def _convert_season_to_date_range(cls, year: int, season: AnimeSeason) -> str:
        """Convert year and season to Kitsu date range filter.
        
        Args:
            year: Year (e.g., 2023)
            season: Universal season enum
            
        Returns:
            Date range string in YYYY-MM-DD..YYYY-MM-DD format
        """
        season_to_months = {
            AnimeSeason.WINTER: ("01-01", "03-31"),
            AnimeSeason.SPRING: ("04-01", "06-30"), 
            AnimeSeason.SUMMER: ("07-01", "09-30"),
            AnimeSeason.FALL: ("10-01", "12-31"),
        }
        
        start_month, end_month = season_to_months.get(season, ("01-01", "12-31"))
        return f"{year}-{start_month}..{year}-{end_month}"