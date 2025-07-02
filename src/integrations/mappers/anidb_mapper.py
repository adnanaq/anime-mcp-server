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
        
        AniDB API limitations (verified by testing):
        - Only supports `aid` parameter for ID-based lookup
        - No search/filter parameters supported (limit, offset, type, status, etc.)
        - Requires anime-titles.xml for title-to-ID mapping
        - Use two-step process: title lookup → data retrieval
        
        Args:
            universal_params: Universal search parameters
            
        Returns:
            Dictionary of AniDB API parameters (only `aid` supported)
        """
        anidb_params = {}
        
        # Only parameter that works with AniDB API
        # Map universal query to AniDB's aid parameter (ID-based lookup)
        if universal_params.query:
            # Assume query contains anime ID for direct lookup
            try:
                # Strip whitespace and convert to int
                aid = int(universal_params.query.strip())
                # Only accept positive IDs
                if aid > 0:
                    anidb_params["aid"] = aid
            except (ValueError, TypeError):
                # Query is not a numeric ID - AniDB doesn't support text search
                # Would need title-to-ID lookup from anime-titles.xml first
                pass
        
        # All other parameters are not supported by AniDB API
        # Implementation note: Use anime-titles.xml for search functionality
        
        return anidb_params
    
