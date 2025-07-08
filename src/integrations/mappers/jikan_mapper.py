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
    AnimeRating,
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
    # Includes all Jikan-supported formats including unique ones
    UNIVERSAL_TO_FORMAT = {
        AnimeFormat.TV: "tv",
        AnimeFormat.TV_SPECIAL: "tv_special",  # Jikan-specific
        AnimeFormat.MOVIE: "movie",
        AnimeFormat.SPECIAL: "special",
        AnimeFormat.OVA: "ova",
        AnimeFormat.ONA: "ona",
        AnimeFormat.MUSIC: "music",
        AnimeFormat.CM: "cm",                  # Jikan-specific
        AnimeFormat.PV: "pv",                  # Jikan-specific
    }
    
    # Universal to Jikan season mappings (for query parameters)
    UNIVERSAL_TO_SEASON = {
        AnimeSeason.WINTER: "winter",
        AnimeSeason.SPRING: "spring",
        AnimeSeason.SUMMER: "summer",
        AnimeSeason.FALL: "fall",
    }
    
    # Universal to Jikan rating mappings (content ratings)
    UNIVERSAL_TO_RATING = {
        AnimeRating.G: "g",        # All Ages
        AnimeRating.PG: "pg",      # Children
        AnimeRating.PG13: "pg13",  # Teens 13 or older
        AnimeRating.R: "r",        # 17+ (violence & profanity)
        AnimeRating.R_PLUS: "r+",  # Mild Nudity
        AnimeRating.RX: "rx",      # Hentai
    }
    
    # Genre name to Jikan ID mapping (Jikan API requires genre IDs, not names)
    # Based on Jikan API v4 genres endpoint
    GENRE_NAME_TO_ID = {
        "action": "1",
        "adventure": "2", 
        "avant garde": "5",
        "award winning": "46",
        "boys love": "28",
        "comedy": "4",
        "drama": "8",
        "ecchi": "9",
        "fantasy": "10",
        "girls love": "26",
        "gourmet": "47",
        "horror": "14",
        "mystery": "7",
        "romance": "22",
        "sci-fi": "24",
        "slice of life": "36",
        "sports": "30",
        "supernatural": "37",
        "suspense": "41",
        "thriller": "41",  # Maps to Suspense
        "shounen": "27",
        "seinen": "42",
        "shoujo": "25",
        "josei": "43",
        "kids": "15",
        "historical": "13",
        "military": "38",
        "music": "19",
        "parody": "20",
        "psychological": "40",
        "school": "23",
        "space": "29",
        "super power": "31",
        "vampire": "32",
        "mecha": "18",
        "martial arts": "17",
        "samurai": "21",
        "game": "11",
        "dementia": "5",  # Maps to Avant Garde
        "demons": "6",
        "harem": "35",
        "magic": "16",
        "cars": "3",
        "workplace": "48",
        "iyashikei": "63",
        "cgdct": "52",  # Cute Girls Doing Cute Things
        "organized crime": "68",
        "otaku culture": "69",
        "racing": "3",   # Maps to Cars
        "reverse harem": "49",
        "love polygon": "49",  # Maps to Reverse Harem
        "idols (female)": "39",
        "idols (male)": "56",
        "strategy game": "11",  # Maps to Game
        "team sports": "30",    # Maps to Sports
        "combat sports": "30",  # Maps to Sports
        "adult cast": "50",
        "anthropomorphic": "51",
        "crossdressing": "44",
        "delinquents": "55",
        "gag humor": "57",
        "gore": "58",
        "high stakes game": "59",
        "isekai": "62",
        "love triangle": "49",  # Maps to Reverse Harem
        "mahou shoujo": "66",
        "mythology": "6",    # Maps to Demons
        "pets": "51",        # Maps to Anthropomorphic
        "reincarnation": "72",
        "time travel": "78",
        "urban fantasy": "10",  # Maps to Fantasy
        "villainess": "74",
        "showbiz": "75",
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
        
        # Exact score match (Jikan-specific)
        score = jikan_specific.get("score") or universal_params.jikan_score
        if score is not None:
            jikan_params["score"] = score
        
        # NOTE: Episode range filtering is NOT supported by Jikan API v4
        # Removed episodes_greater/episodes_lesser as per documentation verification
        
        # Genre filters (Jikan requires genre IDs, not names)
        if universal_params.genres:
            genre_ids = []
            for genre_name in universal_params.genres:
                genre_id = cls.GENRE_NAME_TO_ID.get(genre_name.lower())
                if genre_id:
                    genre_ids.append(genre_id)
                # If genre not found, skip it (graceful degradation)
            if genre_ids:
                jikan_params["genres"] = ",".join(genre_ids)
        
        if universal_params.genres_exclude:
            exclude_ids = []
            for genre_name in universal_params.genres_exclude:
                genre_id = cls.GENRE_NAME_TO_ID.get(genre_name.lower())
                if genre_id:
                    exclude_ids.append(genre_id)
            if exclude_ids:
                jikan_params["genres_exclude"] = ",".join(exclude_ids)
        
        # Year filter
        if universal_params.year:
            jikan_params["start_date"] = f"{universal_params.year}-01-01"
        elif universal_params.start_date:
            jikan_params["start_date"] = universal_params.start_date
        
        # End date filter
        if universal_params.end_date:
            jikan_params["end_date"] = universal_params.end_date
        
        # Content rating filter (NEW - from mapping document)
        if universal_params.rating:
            rating_value = cls.UNIVERSAL_TO_RATING.get(universal_params.rating)
            if rating_value:
                jikan_params["rating"] = rating_value
        
        # Producer filter (NEW - from mapping document)
        if universal_params.producers:
            # WARNING: Jikan API requires producer IDs (integers), not names
            # TODO: Implement producer name-to-ID mapping or document this limitation
            try:
                # Try to validate that all producers are numeric IDs
                producer_ids = []
                for producer in universal_params.producers:
                    if producer.isdigit():
                        producer_ids.append(producer)
                    else:
                        # Skip non-numeric producer names for now
                        # TODO: Add producer name-to-ID mapping
                        continue
                
                if producer_ids:
                    jikan_params["producers"] = ",".join(producer_ids)
            except Exception:
                # If validation fails, skip producer filtering
                pass
        
        # Adult content filter (Jikan expects string 'true'/'false')
        if not universal_params.include_adult:
            jikan_params["sfw"] = "true"
        
        # Result control
        jikan_params["limit"] = universal_params.limit
        if universal_params.offset:
            jikan_params["page"] = (universal_params.offset // universal_params.limit) + 1
        
        # Sort - Jikan has comprehensive sort options (UPDATED with complete list)
        if universal_params.sort_by:
            sort_mapping = {
                "score": "score",
                "popularity": "popularity",
                "title": "title",
                "year": "start_date",
                "episodes": "episodes",
                "duration": "duration",
                "rank": "rank",
                # Additional Jikan-specific sort options from documentation
                "mal_id": "mal_id",
                "scored_by": "scored_by",
                "members": "members",
                "favorites": "favorites",
                "start_date": "start_date",
                "end_date": "end_date",
            }
            order_by = sort_mapping.get(universal_params.sort_by)
            if order_by:
                jikan_params["order_by"] = order_by
                if universal_params.sort_order:
                    jikan_params["sort"] = universal_params.sort_order
        
        # JIKAN-SPECIFIC PARAMETERS (only unique features, no duplicates)
        # Anime type from jikan_specific dict only (legacy support)
        anime_type = jikan_specific.get("anime_type")
        if anime_type and "type" not in jikan_params:
            jikan_params["type"] = anime_type.lower()  # Ensure lowercase
        
        # Safe For Work filter from jikan_specific dict only (but don't override universal include_adult)
        sfw = jikan_specific.get("sfw")
        if sfw is not None and "sfw" not in jikan_params:
            jikan_params["sfw"] = "true" if sfw else "false"
        
        # Exclude genres by ID from jikan_specific dict only
        genres_exclude = jikan_specific.get("genres_exclude")
        if genres_exclude:
            jikan_params["genres_exclude"] = ",".join(map(str, genres_exclude))
        
        # Sorting and ordering from jikan_specific dict only
        order_by = jikan_specific.get("order_by")
        if order_by:
            jikan_params["order_by"] = order_by
            
        sort_direction = jikan_specific.get("sort")
        if sort_direction:
            jikan_params["sort"] = sort_direction
        
        # Letter search (unique to Jikan, but conflicts with 'q' parameter)
        letter = jikan_specific.get("letter") or universal_params.jikan_letter
        if letter and not jikan_params.get("q"):
            jikan_params["letter"] = letter
        
        # Pagination (from jikan_specific dict only, but don't override universal offset conversion)
        page = jikan_specific.get("page")
        if page is not None and "page" not in jikan_params:
            jikan_params["page"] = page
        
        # Special filters (unique to Jikan)
        unapproved = jikan_specific.get("unapproved") or universal_params.jikan_unapproved
        if unapproved is not None:
            jikan_params["unapproved"] = unapproved
        
        return jikan_params