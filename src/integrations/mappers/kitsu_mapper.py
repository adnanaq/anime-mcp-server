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
    def to_kitsu_search_params(cls, universal_params: UniversalSearchParams, kitsu_specific: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert universal search parameters to Kitsu JSON:API parameters.
        
        Kitsu uses JSON:API standard with filter[attribute] syntax for
        query parameters and comprehensive category/production filtering.
        
        Based on comprehensive API verification - 14 working parameters confirmed.
        
        Args:
            universal_params: Universal search parameters
            kitsu_specific: Kitsu-specific parameters override
            
        Returns:
            Dictionary of Kitsu API filter parameters
        """
        if kitsu_specific is None:
            kitsu_specific = {}
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
        
        # Score filters (Kitsu uses 0-100 scale, same as universal 0-10 scale * 10)
        # Kitsu Range Syntax: Uses ".." separator for ranges
        # - "80.." = >= 80 (minimum score)
        # - "..90" = <= 90 (maximum score) 
        # - "80..90" = between 80 and 90 (both min and max)
        if universal_params.min_score is not None:
            # Convert 0-10 to 0-100 scale  
            kitsu_rating = universal_params.min_score * 10
            kitsu_params["filter[averageRating]"] = f"{kitsu_rating}.."
        if universal_params.max_score is not None:
            kitsu_rating = universal_params.max_score * 10
            existing_filter = kitsu_params.get("filter[averageRating]", "..")
            if ".." in existing_filter:
                min_part = existing_filter.split("..")[0]
                kitsu_params["filter[averageRating]"] = f"{min_part}..{kitsu_rating}"
            else:
                kitsu_params["filter[averageRating]"] = f"..{kitsu_rating}"
        
        # Episode filters - Uses same Kitsu range syntax as score filters
        # Examples: "12.." (>=12 episodes), "..24" (<=24 episodes), "12..24" (12-24 episodes)
        if universal_params.min_episodes is not None:
            kitsu_params["filter[episodeCount]"] = f"{universal_params.min_episodes}.."
        if universal_params.max_episodes is not None:
            existing_filter = kitsu_params.get("filter[episodeCount]", "..")
            if ".." in existing_filter:
                min_part = existing_filter.split("..")[0]
                kitsu_params["filter[episodeCount]"] = f"{min_part}..{universal_params.max_episodes}"
            else:
                kitsu_params["filter[episodeCount]"] = f"..{universal_params.max_episodes}"
        
        # Duration filters - Kitsu stores episode length in minutes, uses range syntax  
        # Examples: "20.." (>=20 min), "..30" (<=30 min), "20..30" (20-30 min episodes)
        if universal_params.min_duration is not None:
            kitsu_params["filter[episodeLength]"] = f"{universal_params.min_duration}.."
        if universal_params.max_duration is not None:
            existing_filter = kitsu_params.get("filter[episodeLength]", "..")
            if ".." in existing_filter:
                min_part = existing_filter.split("..")[0]
                kitsu_params["filter[episodeLength]"] = f"{min_part}..{universal_params.max_duration}"
            else:
                kitsu_params["filter[episodeLength]"] = f"..{universal_params.max_duration}"
        
        # NOTE: filter[startDate] is NOT SUPPORTED by Kitsu API (returns "Filter not allowed")
        # Use filter[seasonYear] and filter[season] instead for temporal filtering
        
        # Categories (genres) - Kitsu has rich category system
        if universal_params.genres:
            # Kitsu supports category filtering via relationships
            kitsu_params["filter[categories]"] = ",".join(universal_params.genres)
        
        # Season filters (Kitsu native support)
        if universal_params.season:
            # Map universal season to Kitsu season values
            season_mapping = {
                "WINTER": "winter",
                "SPRING": "spring", 
                "SUMMER": "summer",
                "FALL": "fall"
            }
            season_value = season_mapping.get(universal_params.season.value)
            if season_value:
                kitsu_params["filter[season]"] = season_value
        
        # Season year filter (Kitsu native support)
        if universal_params.year:
            kitsu_params["filter[seasonYear]"] = universal_params.year
        
        # Age rating filter (Kitsu native support)
        if universal_params.rating:
            # Map universal rating to Kitsu age rating values
            rating_mapping = {
                "G": "G",
                "PG": "PG", 
                "PG13": "PG", # Map PG13 to PG for Kitsu
                "R": "R",
                "R_PLUS": "R18", # Map R+ to R18 for Kitsu  
                "RX": "R18"  # Map RX to R18 for Kitsu
            }
            rating_value = rating_mapping.get(universal_params.rating.value)
            if rating_value:
                kitsu_params["filter[ageRating]"] = rating_value
        
        # KITSU-SPECIFIC PARAMETERS
        
        # Streaming platforms (Kitsu unique feature)
        streamers = kitsu_specific.get("streamers") or universal_params.kitsu_streamers
        if streamers:
            kitsu_params["filter[streamers]"] = ",".join(streamers) if isinstance(streamers, list) else streamers
        
        # Age rating override (use kitsu_specific only since universal rating is handled above)
        age_rating_override = kitsu_specific.get("ageRating")
        if age_rating_override:
            kitsu_params["filter[ageRating]"] = age_rating_override
            
        # Subtype override (use kitsu_specific only since universal type_format is handled above)
        subtype_override = kitsu_specific.get("subtype") 
        if subtype_override:
            kitsu_params["filter[subtype]"] = subtype_override
        
        # Result control
        kitsu_params["page[limit]"] = universal_params.limit
        if universal_params.offset:
            # Kitsu uses page-based pagination
            page_number = (universal_params.offset // universal_params.limit) + 1
            kitsu_params["page[number]"] = page_number
        
        # Sort - Kitsu has comprehensive sort options (verified working)
        if universal_params.sort_by:
            sort_mapping = {
                "score": "averageRating",
                "popularity": "popularityRank", 
                "title": "canonicalTitle",
                "year": "startDate",
                "episodes": "episodeCount",
                "duration": "episodeLength", 
                "rank": "ratingRank",
                # Additional verified Kitsu sort fields
                "id": "id",
                "created_at": "createdAt",
                "updated_at": "updatedAt"
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