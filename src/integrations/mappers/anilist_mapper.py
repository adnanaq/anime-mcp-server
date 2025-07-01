"""AniList universal schema mapper for bidirectional data conversion.

This mapper handles the conversion between AniList-specific data formats 
and the universal anime schema, addressing the mapping chaos identified
in our architectural analysis.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ...models.universal_anime import (
    UniversalAnime,
    UniversalSearchParams,
    AnimeStatus,
    AnimeFormat,
    AnimeRating,
    AnimeSeason,
)


class AniListMapper:
    """Bidirectional mapper between AniList data and universal schema."""
    
    # AniList-specific status mappings
    STATUS_TO_UNIVERSAL = {
        "FINISHED": AnimeStatus.FINISHED,
        "RELEASING": AnimeStatus.RELEASING, 
        "NOT_YET_RELEASED": AnimeStatus.NOT_YET_RELEASED,
        "CANCELLED": AnimeStatus.CANCELLED,
        "HIATUS": AnimeStatus.HIATUS,
    }
    
    UNIVERSAL_TO_STATUS = {v: k for k, v in STATUS_TO_UNIVERSAL.items()}
    
    # AniList-specific format mappings
    FORMAT_TO_UNIVERSAL = {
        "TV": AnimeFormat.TV,
        "TV_SHORT": AnimeFormat.TV,
        "MOVIE": AnimeFormat.MOVIE,
        "SPECIAL": AnimeFormat.SPECIAL,
        "OVA": AnimeFormat.OVA,
        "ONA": AnimeFormat.ONA,
        "MUSIC": AnimeFormat.MUSIC,
    }
    
    UNIVERSAL_TO_FORMAT = {
        AnimeFormat.TV: "TV",
        AnimeFormat.MOVIE: "MOVIE", 
        AnimeFormat.SPECIAL: "SPECIAL",
        AnimeFormat.OVA: "OVA",
        AnimeFormat.ONA: "ONA",
        AnimeFormat.MUSIC: "MUSIC",
    }
    
    # AniList season mappings
    SEASON_TO_UNIVERSAL = {
        "WINTER": AnimeSeason.WINTER,
        "SPRING": AnimeSeason.SPRING,
        "SUMMER": AnimeSeason.SUMMER,
        "FALL": AnimeSeason.FALL,
    }
    
    UNIVERSAL_TO_SEASON = {v: k for k, v in SEASON_TO_UNIVERSAL.items()}
    
    @classmethod
    def to_universal_anime(cls, anilist_data: Dict[str, Any]) -> UniversalAnime:
        """Convert AniList anime data to universal schema.
        
        Args:
            anilist_data: Raw AniList GraphQL response data
            
        Returns:
            UniversalAnime instance with mapped data
        """
        # Extract basic information
        anime_id = str(anilist_data.get("id", ""))
        title = cls._extract_title(anilist_data.get("title", {}))
        
        # Map status and format using our enum mappings
        status_str = anilist_data.get("status", "NOT_YET_RELEASED")
        status = cls.STATUS_TO_UNIVERSAL.get(status_str, AnimeStatus.NOT_YET_RELEASED)
        
        format_str = anilist_data.get("format", "TV")
        type_format = cls.FORMAT_TO_UNIVERSAL.get(format_str, AnimeFormat.TV)
        
        # Extract season information
        season = None
        season_str = anilist_data.get("season")
        if season_str:
            season = cls.SEASON_TO_UNIVERSAL.get(season_str)
        
        # Extract dates
        start_date = cls._format_anilist_date(anilist_data.get("startDate"))
        end_date = cls._format_anilist_date(anilist_data.get("endDate"))
        
        # Extract images
        cover_image = anilist_data.get("coverImage", {})
        image_url = cover_image.get("medium")
        image_large = cover_image.get("large") 
        
        # Extract studios
        studios_data = anilist_data.get("studios", {}).get("nodes", [])
        studios = [
            studio["name"] for studio in studios_data 
            if studio.get("isAnimationStudio", True)
        ]
        
        # Extract genres and tags
        genres = anilist_data.get("genres", [])
        tags_data = anilist_data.get("tags", [])
        themes = [tag["name"] for tag in tags_data if tag.get("category") in ["Theme", "Setting"]]
        
        # Extract title variants
        title_data = anilist_data.get("title", {})
        title_english = title_data.get("english")
        title_native = title_data.get("native")
        
        # Build synonyms from alternative titles
        synonyms = []
        if title_data.get("romaji") and title_data.get("romaji") != title:
            synonyms.append(title_data.get("romaji"))
        if title_english and title_english != title:
            synonyms.append(title_english)
        if title_native and title_native != title:
            synonyms.append(title_native)
        
        # Extract numeric values safely
        episodes = anilist_data.get("episodes")
        duration = anilist_data.get("duration")
        year = anilist_data.get("seasonYear")
        score = anilist_data.get("averageScore")
        if score:
            score = score / 10.0  # Convert from 0-100 to 0-10 scale
        
        popularity = anilist_data.get("popularity")
        favourites = anilist_data.get("favourites")
        
        # URL
        url = anilist_data.get("siteUrl")
        
        # Description (clean HTML if present)
        description = anilist_data.get("description")
        if description:
            description = cls._clean_html_description(description)
        
        # Source material
        source = anilist_data.get("source")
        if source:
            source = source.lower().replace("_", " ")
        
        # Create UniversalAnime instance
        universal_anime = UniversalAnime(
            # GUARANTEED UNIVERSAL PROPERTIES
            id=anime_id,
            title=title,
            type_format=type_format,
            episodes=episodes,
            status=status,
            genres=genres,
            score=score,
            image_url=image_url,
            image_large=image_large,
            year=year,
            synonyms=synonyms,
            studios=studios,
            
            # HIGH-CONFIDENCE PROPERTIES
            description=description,
            url=url,
            score_count=None,  # AniList doesn't provide this directly
            title_english=title_english,
            title_native=title_native,
            start_date=start_date,
            season=season,
            end_date=end_date,
            duration=duration,
            
            # MEDIUM-CONFIDENCE PROPERTIES
            source=source,
            rank=None,  # Would need separate query for ranking
            staff=[],   # Would need separate query for staff
            
            # ADDITIONAL PROPERTIES
            characters=[],  # Would need separate query for characters
            image_small=cover_image.get("medium"),  # Use medium as small
            rating=None,  # AniList doesn't provide MPAA-style ratings
            themes=themes,
            demographics=[],  # Would need tag analysis
            producers=[],   # Would need separate query 
            popularity=popularity,
        )
        
        # Set platform ID
        universal_anime.set_platform_id("anilist", int(anime_id))
        
        # Calculate quality score
        universal_anime.data_quality_score = universal_anime.calculate_quality_score()
        universal_anime.last_updated = datetime.utcnow()
        universal_anime.source_priority = ["anilist"]
        
        return universal_anime
    
    @classmethod
    def to_anilist_search_params(cls, universal_params: UniversalSearchParams) -> Dict[str, Any]:
        """Convert universal search parameters to AniList-specific parameters.
        
        This addresses the parameter mapping chaos by providing a central
        conversion point from universal parameters to AniList GraphQL variables.
        
        Args:
            universal_params: Universal search parameters
            
        Returns:
            Dictionary of AniList GraphQL variables
        """
        anilist_params = {}
        
        # Text search
        if universal_params.query:
            anilist_params["search"] = universal_params.query
        
        # Genre filters
        if universal_params.genres:
            anilist_params["genre_in"] = universal_params.genres
        if universal_params.genres_exclude:
            anilist_params["genre_not_in"] = universal_params.genres_exclude
        
        # Status mapping (prevents the "ongoing" chaos)
        if universal_params.status:
            anilist_params["status"] = cls.UNIVERSAL_TO_STATUS.get(universal_params.status)
        
        # Format mapping
        if universal_params.type_format:
            anilist_params["format"] = cls.UNIVERSAL_TO_FORMAT.get(universal_params.type_format)
        
        # Temporal filters
        if universal_params.year:
            anilist_params["seasonYear"] = universal_params.year
        if universal_params.season:
            anilist_params["season"] = cls.UNIVERSAL_TO_SEASON.get(universal_params.season)
        
        # Date range filters (convert to FuzzyDateInt)
        if universal_params.start_date:
            anilist_params["startDate_greater"] = cls._parse_fuzzy_date(universal_params.start_date)
        if universal_params.end_date:
            anilist_params["endDate_lesser"] = cls._parse_fuzzy_date(universal_params.end_date)
        
        # Numeric filters (convert score from 0-10 to 0-100)
        if universal_params.min_score:
            anilist_params["averageScore_greater"] = int(universal_params.min_score * 10)
        if universal_params.max_score:
            anilist_params["averageScore_lesser"] = int(universal_params.max_score * 10)
        
        if universal_params.min_episodes:
            anilist_params["episodes_greater"] = universal_params.min_episodes
        if universal_params.max_episodes:
            anilist_params["episodes_lesser"] = universal_params.max_episodes
        
        if universal_params.min_duration:
            anilist_params["duration_greater"] = universal_params.min_duration
        if universal_params.max_duration:
            anilist_params["duration_lesser"] = universal_params.max_duration
        
        if universal_params.min_popularity:
            anilist_params["popularity_greater"] = universal_params.min_popularity
        if universal_params.max_popularity:
            anilist_params["popularity_lesser"] = universal_params.max_popularity
        
        # Tag filters (themes map to tags)
        if universal_params.themes:
            anilist_params["tag_in"] = universal_params.themes
        
        # Adult content filter
        if not universal_params.include_adult:
            anilist_params["isAdult"] = False
        
        # Result control
        anilist_params["perPage"] = universal_params.limit
        
        # Sort mapping
        if universal_params.sort_by:
            sort_mapping = {
                "score": "SCORE_DESC",
                "popularity": "POPULARITY_DESC", 
                "title": "TITLE_ROMAJI",
                "year": "START_DATE_DESC",
                "episodes": "EPISODES_DESC",
                "duration": "DURATION_DESC",
                "start_date": "START_DATE_DESC",
            }
            sort_field = sort_mapping.get(universal_params.sort_by)
            if sort_field:
                if universal_params.sort_order == "asc":
                    sort_field = sort_field.replace("_DESC", "_ASC")
                anilist_params["sort"] = [sort_field]
        
        return anilist_params
    
    @classmethod
    def _extract_title(cls, title_data: Dict[str, Any]) -> str:
        """Extract the best available title from AniList title object.
        
        Priority: English > Romaji > Native
        """
        if title_data.get("english"):
            return title_data["english"]
        elif title_data.get("romaji"):
            return title_data["romaji"]
        elif title_data.get("native"):
            return title_data["native"]
        else:
            return "Unknown Title"
    
    @classmethod
    def _format_anilist_date(cls, date_data: Optional[Dict[str, Any]]) -> Optional[str]:
        """Convert AniList date object to ISO 8601 string.
        
        Args:
            date_data: AniList date object with year, month, day
            
        Returns:
            ISO 8601 date string (YYYY-MM-DD) or None
        """
        if not date_data:
            return None
        
        year = date_data.get("year")
        month = date_data.get("month")
        day = date_data.get("day")
        
        if not year:
            return None
        
        # Build date string with available components
        if month and day:
            return f"{year:04d}-{month:02d}-{day:02d}"
        elif month:
            return f"{year:04d}-{month:02d}-01"
        else:
            return f"{year:04d}-01-01"
    
    @classmethod
    def _parse_fuzzy_date(cls, date_string: str) -> Optional[int]:
        """Parse ISO date string to AniList FuzzyDateInt format (YYYYMMDD)."""
        if not date_string:
            return None
            
        try:
            if '-' in date_string:
                parts = date_string.split('-')
                if len(parts) >= 3:
                    year, month, day = parts[:3]
                    return int(f"{year:0>4}{month:0>2}{day:0>2}")
                elif len(parts) == 2:
                    year, month = parts
                    return int(f"{year:0>4}{month:0>2}01")
                elif len(parts) == 1:
                    year = parts[0]
                    return int(f"{year:0>4}0101")
            elif len(date_string) == 4 and date_string.isdigit():
                return int(f"{date_string}0101")
        except (ValueError, IndexError):
            pass
            
        return None
    
    @classmethod
    def _clean_html_description(cls, description: str) -> str:
        """Clean HTML tags and formatting from AniList descriptions."""
        if not description:
            return ""
        
        # Simple HTML tag removal (could be enhanced with proper HTML parser)
        import re
        
        # Remove HTML tags
        description = re.sub(r'<[^>]+>', '', description)
        
        # Replace HTML entities
        description = description.replace('&lt;', '<')
        description = description.replace('&gt;', '>')
        description = description.replace('&amp;', '&')
        description = description.replace('&quot;', '"')
        description = description.replace('&#039;', "'")
        description = description.replace('<br>', '\n')
        
        # Clean up whitespace
        description = re.sub(r'\n\s*\n', '\n\n', description)
        description = description.strip()
        
        return description