"""Anime-Planet query parameter mapper.

This mapper handles the conversion from universal search parameters
to Anime-Planet search parameters. Anime-Planet uses web scraping
with JSON-LD structured data and unique status derivation.
"""

from typing import Any, Dict, List, Optional
from ...models.universal_anime import (
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeSeason,
)


class AnimePlanetMapper:
    """Query parameter mapper from universal schema to Anime-Planet."""
    
    # Universal to Anime-Planet format mappings (limited support)
    UNIVERSAL_TO_TYPE = {
        AnimeFormat.TV: "TVSeries",
        AnimeFormat.MOVIE: "Movie",
        # Note: Anime-Planet may not support all formats in search filters
    }
    
    # Universal to Anime-Planet season mappings
    UNIVERSAL_TO_SEASON = {
        AnimeSeason.WINTER: "Winter",
        AnimeSeason.SPRING: "Spring",
        AnimeSeason.SUMMER: "Summer",
        AnimeSeason.FALL: "Fall",
    }
    
    @classmethod
    def to_animeplanet_search_params(cls, universal_params: UniversalSearchParams) -> Dict[str, Any]:
        """Convert universal search parameters to Anime-Planet search parameters.
        
        Anime-Planet characteristics:
        - Uses "name" for title search instead of "query"
        - Limited filter support (mainly tags, year, basic info)
        - Status derived from date patterns ("2023 - ?" vs "2023 - 2024")
        - JSON-LD structured data for results
        - Web scraping based, so limited API-style filtering
        
        Args:
            universal_params: Universal search parameters
            
        Returns:
            Dictionary of Anime-Planet search parameters
        """
        ap_params = {}
        
        # Text search - Anime-Planet uses "name" field
        if universal_params.query:
            ap_params["name"] = universal_params.query
        
        # Format mapping (limited support)
        if universal_params.type_format:
            type_value = cls.UNIVERSAL_TO_TYPE.get(universal_params.type_format)
            if type_value:
                ap_params["type"] = type_value
        
        # Status handling - derive from date patterns
        if universal_params.status:
            cls._add_status_filters(ap_params, universal_params.status)
        
        # Score filters (convert to Anime-Planet's rating scale)
        if universal_params.min_score is not None:
            # Anime-Planet uses 5-star rating, convert from 0-10
            ap_rating = (universal_params.min_score / 2.0)
            ap_params["min_rating"] = max(0, min(5, ap_rating))
        if universal_params.max_score is not None:
            ap_rating = (universal_params.max_score / 2.0)
            ap_params["max_rating"] = max(0, min(5, ap_rating))
        
        # Episode filters
        if universal_params.min_episodes is not None:
            ap_params["min_episodes"] = universal_params.min_episodes
        if universal_params.max_episodes is not None:
            ap_params["max_episodes"] = universal_params.max_episodes
        
        # Year filter
        if universal_params.year:
            ap_params["year"] = universal_params.year
        
        # Season filter
        if universal_params.season:
            season_value = cls.UNIVERSAL_TO_SEASON.get(universal_params.season)
            if season_value:
                ap_params["season"] = season_value
        
        # Tags - combine genres and themes (Anime-Planet has tag system)
        tags = []
        if universal_params.genres:
            tags.extend(universal_params.genres)
        if universal_params.themes:
            tags.extend(universal_params.themes)
        
        if tags:
            ap_params["tags"] = tags
        
        # Studios (Anime-Planet has studio information)
        if universal_params.studios:
            ap_params["studios"] = universal_params.studios
        
        # Result control
        ap_params["limit"] = universal_params.limit
        if universal_params.offset:
            # Anime-Planet may use page-based pagination
            page_number = (universal_params.offset // universal_params.limit) + 1
            ap_params["page"] = page_number
        
        # Sort - limited sort options on Anime-Planet
        if universal_params.sort_by:
            sort_mapping = {
                "score": "rating",
                "title": "title", 
                "year": "year",
                # Note: Anime-Planet has limited sort options
            }
            sort_field = sort_mapping.get(universal_params.sort_by)
            if sort_field:
                ap_params["sort_by"] = sort_field
                if universal_params.sort_order:
                    ap_params["sort_order"] = universal_params.sort_order
        
        return ap_params
    
    @classmethod
    def _add_status_filters(cls, params: Dict[str, Any], status: AnimeStatus) -> None:
        """Add filters to derive status from date patterns.
        
        Anime-Planet shows status through date formats:
        - "2023 - 2024" (finished)
        - "2023 - ?" (ongoing)
        - Future dates (upcoming)
        
        Args:
            params: Parameter dictionary to modify
            status: Universal status to derive from
        """
        from datetime import datetime
        
        current_year = datetime.now().year
        
        if status == AnimeStatus.FINISHED:
            # Look for shows with complete date ranges
            params["status_filter"] = "completed"
        elif status == AnimeStatus.RELEASING:
            # Look for shows with ongoing pattern ("YYYY - ?")
            params["status_filter"] = "ongoing"  
        elif status == AnimeStatus.NOT_YET_RELEASED:
            # Look for shows with future start dates
            params["status_filter"] = "upcoming"
        
        # Alternative: use date pattern matching
        params["date_pattern"] = cls._get_date_pattern_for_status(status)
    
    @classmethod
    def _get_date_pattern_for_status(cls, status: AnimeStatus) -> str:
        """Get date pattern to match for given status.
        
        Args:
            status: Universal status
            
        Returns:
            Date pattern string for Anime-Planet filtering
        """
        if status == AnimeStatus.FINISHED:
            return "completed_range"  # "YYYY - YYYY" pattern
        elif status == AnimeStatus.RELEASING:
            return "ongoing_range"    # "YYYY - ?" pattern  
        elif status == AnimeStatus.NOT_YET_RELEASED:
            return "future_start"     # Future year start
        else:
            return "any"