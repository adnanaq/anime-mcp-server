"""
Jikan (MAL Unofficial API) specific MCP tools.

Comprehensive tools for Jikan v4 API with rich MAL data access without requiring API keys.
Jikan provides extensive MAL data including user statistics, recommendations, and detailed metadata.
"""

from typing import Dict, List, Literal, Optional, Any, Union
from fastmcp import FastMCP
from mcp.server.fastmcp import Context

from ...integrations.clients.jikan_client import JikanClient
from ...integrations.mappers.jikan_mapper import JikanMapper
from ...models.universal_anime import UniversalSearchParams, AnimeStatus, AnimeFormat, AnimeRating
from ...config import get_settings

# Initialize components
settings = get_settings()
jikan_client = JikanClient()
jikan_mapper = JikanMapper()

# Create FastMCP instance for tools
mcp = FastMCP("Jikan Tools")


@mcp.tool(
    name="search_anime_jikan",
    description="Search anime using Jikan (MAL unofficial API) with extensive filtering options",
    annotations={
        "title": "Jikan Anime Search",
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
async def search_anime_jikan(
    # Basic search
    query: Optional[str] = None,
    
    # Content filtering
    type: Optional[Literal["tv", "movie", "ova", "special", "ona", "music", "cm", "pv", "tv_special"]] = None,
    status: Optional[Literal["airing", "complete", "upcoming"]] = None,
    rating: Optional[Literal["g", "pg", "pg13", "r17", "r", "rx"]] = None,
    
    # Score and popularity filtering
    score: Optional[int] = None,  # Exact score match (0-10)
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    
    # Temporal filtering
    start_date: Optional[str] = None,  # YYYY-MM-DD
    end_date: Optional[str] = None,    # YYYY-MM-DD
    
    # Genre filtering (Jikan uses genre IDs)
    genres: Optional[List[int]] = None,
    genres_exclude: Optional[List[int]] = None,
    
    # Producer/Studio filtering
    producers: Optional[List[int]] = None,
    
    # Advanced filtering
    order_by: Optional[Literal["mal_id", "title", "type", "rating", "start_date", "end_date", "episodes", "score", "scored_by", "rank", "popularity", "members", "favorites"]] = None,
    sort: Optional[Literal["desc", "asc"]] = "desc",
    sfw: bool = True,
    
    # Jikan-specific parameters
    letter: Optional[str] = None,  # Return entries starting with letter (A-Z)
    unapproved: Optional[bool] = None,  # Include unapproved entries
    
    # Pagination
    limit: int = 25,
    page: int = 1,
    
    ctx: Optional[Context] = None
) -> List[Dict[str, Any]]:
    """
    Search anime using Jikan (MAL unofficial API) with comprehensive filtering.
    
    Jikan Specializations:
    - No API key required (unofficial MAL API)
    - Rich MAL community data (members, favorites, scored_by)
    - Comprehensive genre system with IDs
    - Producer and studio filtering by ID
    - Advanced sorting and ranking options
    - User statistics and community engagement metrics
    
    Args:
        query: Search query for anime titles
        type: Media type filter (tv, movie, ova, special, ona, music, cm, pv, tv_special)
        status: Airing status (airing, complete, upcoming)
        rating: Age rating (g, pg, pg13, r17, r, rx)
        score: Exact score match (0-10)
        min_score/max_score: Score range filtering (1-10 scale)
        start_date/end_date: Date range filtering (YYYY, YYYY-MM, YYYY-MM-DD)
        genres: Include specific genre IDs (MAL genre system)
        genres_exclude: Exclude specific genre IDs
        producers: Filter by producer/studio IDs
        order_by: Sort field (score, popularity, members, etc.)
        sort: Sort direction (desc, asc)
        sfw: Safe for work content only
        letter: Return entries starting with letter (A-Z, conflicts with query)
        unapproved: Include unapproved MAL entries (unique to Jikan)
        limit: Results per page (max: 25)
        page: Page number for pagination
        
    Returns:
        List of anime with comprehensive MAL data and community metrics
    """
    if ctx:
        await ctx.info(f"Searching Jikan (MAL) with {limit} results per page")
    
    try:
        # Create UniversalSearchParams with direct string inputs (Union types handle both enum and string)
        universal_params = UniversalSearchParams(
            query=query,
            limit=min(limit, 25),  # Jikan max limit
            offset=(page - 1) * limit if page > 1 else 0,
            status=status,           # Pass LLM string directly ("airing", "complete", "upcoming")
            type_format=type,        # Pass LLM string directly ("tv", "movie", "ova", etc.)
            rating=rating,           # Pass LLM string directly ("g", "pg", "pg13", etc.)
            min_score=min_score,
            max_score=max_score,
            start_date=start_date,
            end_date=end_date,
            genres=[str(g) for g in genres] if genres else None,
            genres_exclude=[str(g) for g in genres_exclude] if genres_exclude else None,
            producers=[str(p) for p in producers] if producers else None,
            sort_by=order_by,
            sort_order=sort,
            include_adult=not sfw,
            jikan_score=score,
            jikan_letter=letter,
            jikan_unapproved=unapproved
        )
        
        # No jikan_specific parameters needed - all handled by universal_params now
        jikan_specific = {}
        
        # Use mapper to convert parameters
        jikan_params = jikan_mapper.to_jikan_search_params(universal_params, jikan_specific)
        
        # Execute search and return raw Jikan results directly
        raw_results = await jikan_client.search_anime(**jikan_params)
        
        # Add source attribution to each result
        for result in raw_results:
            if isinstance(result, dict):
                result["source_platform"] = "jikan"

        if ctx:
            await ctx.info(f"Found {len(raw_results)} anime via Jikan")
            
        return raw_results
        
    except Exception as e:
        error_msg = f"Jikan search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="get_anime_jikan",
    description="Get detailed anime information from Jikan by MAL ID with comprehensive data",
    annotations={
        "title": "Jikan Anime Details",
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
async def get_anime_jikan(
    mal_id: int,
    include_characters: bool = False,
    include_staff: bool = False,
    include_episodes: bool = False,
    include_videos: bool = False,
    include_pictures: bool = False,
    include_statistics: bool = True,
    include_recommendations: bool = False,
    include_reviews: bool = False,
    ctx: Optional[Context] = None
) -> Optional[Dict[str, Any]]:
    """
    Get detailed anime information from Jikan by MAL ID.
    
    Provides comprehensive MAL data including:
    - Community statistics and engagement metrics
    - Character and staff information
    - Episode details and videos
    - User recommendations and reviews
    - Complete image galleries
    - Related anime and adaptations
    
    Args:
        mal_id: MyAnimeList anime ID
        include_characters: Include character information (default: False)
        include_staff: Include staff information (default: False)
        include_episodes: Include episode details (default: False)
        include_videos: Include promotional videos (default: False)
        include_pictures: Include image gallery (default: False)
        include_statistics: Include user statistics (default: True)
        include_recommendations: Include user recommendations (default: False)
        include_reviews: Include user reviews (default: False)
        
    Returns:
        Comprehensive anime information with extensive MAL data
    """
    if ctx:
        await ctx.info(f"Fetching Jikan anime details for MAL ID: {mal_id}")
    
    try:
        # Get base anime data
        raw_result = await jikan_client.get_anime_by_id(mal_id)
        
        if not raw_result:
            if ctx:
                await ctx.info(f"Anime with MAL ID {mal_id} not found via Jikan")
            return None
        
        # Convert base data
        universal_anime = jikan_mapper.to_universal_anime(raw_result)
        
        # Build comprehensive response
        result = {
            "id": universal_anime.id,
            "title": universal_anime.title,
            "title_english": raw_result.get("title_english"),
            "title_japanese": raw_result.get("title_japanese"),
            "title_synonyms": raw_result.get("title_synonyms", []),
            "synopsis": universal_anime.description,
            "background": raw_result.get("background"),
            
            # Basic information
            "type": universal_anime.type_format,
            "episodes": universal_anime.episodes,
            "status": universal_anime.status,
            "aired": {
                "from": universal_anime.start_date,
                "to": universal_anime.end_date,
                "prop": raw_result.get("aired", {}).get("prop", {}),
                "string": raw_result.get("aired", {}).get("string")
            },
            "duration": raw_result.get("duration"),
            "rating": raw_result.get("rating"),
            "source": raw_result.get("source"),
            "season": raw_result.get("season"),
            "year": universal_anime.year,
            
            # Community metrics
            "score": universal_anime.score,
            "scored_by": raw_result.get("scored_by"),
            "rank": raw_result.get("rank"),
            "popularity": raw_result.get("popularity"),
            "members": raw_result.get("members"),
            "favorites": raw_result.get("favorites"),
            
            # Content classification
            "genres": [
                {
                    "mal_id": genre.get("mal_id"),
                    "type": genre.get("type"),
                    "name": genre.get("name"),
                    "url": genre.get("url")
                }
                for genre in raw_result.get("genres", [])
            ],
            "themes": [
                {
                    "mal_id": theme.get("mal_id"),
                    "type": theme.get("type"),
                    "name": theme.get("name"),
                    "url": theme.get("url")
                }
                for theme in raw_result.get("themes", [])
            ],
            "demographics": [
                {
                    "mal_id": demo.get("mal_id"),
                    "type": demo.get("type"),
                    "name": demo.get("name"),
                    "url": demo.get("url")
                }
                for demo in raw_result.get("demographics", [])
            ],
            
            # Production information
            "producers": [
                {
                    "mal_id": prod.get("mal_id"),
                    "type": prod.get("type"),
                    "name": prod.get("name"),
                    "url": prod.get("url")
                }
                for prod in raw_result.get("producers", [])
            ],
            "licensors": [
                {
                    "mal_id": lic.get("mal_id"),
                    "type": lic.get("type"),
                    "name": lic.get("name"),
                    "url": lic.get("url")
                }
                for lic in raw_result.get("licensors", [])
            ],
            "studios": [
                {
                    "mal_id": studio.get("mal_id"),
                    "type": studio.get("type"),
                    "name": studio.get("name"),
                    "url": studio.get("url")
                }
                for studio in raw_result.get("studios", [])
            ],
            
            # Media content
            "images": {
                "jpg": raw_result.get("images", {}).get("jpg", {}),
                "webp": raw_result.get("images", {}).get("webp", {})
            },
            "trailer": raw_result.get("trailer", {}),
            
            # Broadcast information
            "broadcast": raw_result.get("broadcast", {}),
            
            # Relations
            "relations": raw_result.get("relations", []),
            
            # External links
            "external": raw_result.get("external", []),
            "streaming": raw_result.get("streaming", []),
            
            # Jikan-specific metadata
            "mal_id": mal_id,
            "jikan_url": raw_result.get("url"),
            "approved": raw_result.get("approved"),
            
            # Optional detailed data (will be populated below if requested)
            "characters": [],
            "staff": [],
            "episodes": [],
            "videos": {},
            "pictures": [],
            "statistics": {},
            "recommendations": [],
            "reviews": [],
            
            # Source attribution
            "source_platform": "jikan",
            "data_quality_score": universal_anime.data_quality_score
        }
        
        # Fetch optional data based on parameters
        if include_characters:
            try:
                characters_data = await jikan_client.get_anime_characters(mal_id)
                result["characters"] = characters_data.get("data", [])
            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to fetch characters: {str(e)}")
        
        if include_staff:
            try:
                staff_data = await jikan_client.get_anime_staff(mal_id)
                result["staff"] = staff_data.get("data", [])
            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to fetch staff: {str(e)}")
        
        if include_episodes:
            try:
                episodes_data = await jikan_client.get_anime_episodes(mal_id)
                result["episodes"] = episodes_data.get("data", [])
            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to fetch episodes: {str(e)}")
        
        if include_videos:
            try:
                videos_data = await jikan_client.get_anime_videos(mal_id)
                result["videos"] = videos_data.get("data", {})
            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to fetch videos: {str(e)}")
        
        if include_pictures:
            try:
                pictures_data = await jikan_client.get_anime_pictures(mal_id)
                result["pictures"] = pictures_data.get("data", [])
            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to fetch pictures: {str(e)}")
        
        if include_statistics:
            try:
                stats_data = await jikan_client.get_anime_statistics(mal_id)
                result["statistics"] = stats_data.get("data", {})
            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to fetch statistics: {str(e)}")
        
        if include_recommendations:
            try:
                rec_data = await jikan_client.get_anime_recommendations(mal_id)
                result["recommendations"] = rec_data.get("data", [])
            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to fetch recommendations: {str(e)}")
        
        if include_reviews:
            try:
                reviews_data = await jikan_client.get_anime_reviews(mal_id)
                result["reviews"] = reviews_data.get("data", [])
            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to fetch reviews: {str(e)}")
        
        if ctx:
            await ctx.info(f"Retrieved comprehensive Jikan data for '{result['title']}'")
            
        return result
        
    except Exception as e:
        error_msg = f"Failed to get Jikan anime {mal_id}: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)


@mcp.tool(
    name="get_jikan_seasonal",
    description="Get seasonal anime from Jikan with comprehensive filtering",
    annotations={
        "title": "Jikan Seasonal Anime",
        "readOnlyHint": True,
        "idempotentHint": True,
    }
)
async def get_jikan_seasonal(
    year: int,
    season: Literal["winter", "spring", "summer", "fall"],
    filter: Optional[Literal["tv", "movie", "ova", "special", "ona"]] = None,
    sfw: bool = True,
    unapproved: bool = False,
    continuing: bool = False,
    limit: int = 25,
    page: int = 1,
    ctx: Optional[Context] = None
) -> List[Dict[str, Any]]:
    """
    Get seasonal anime from Jikan with filtering options.
    
    Args:
        year: Year for the season
        season: Season (winter, spring, summer, fall)
        filter: Media type filter (tv, movie, ova, special, ona)
        sfw: Safe for work content only
        unapproved: Include unapproved entries
        continuing: Include continuing series from previous seasons
        limit: Results per page (max: 25)
        page: Page number
        
    Returns:
        List of seasonal anime with comprehensive MAL data
    """
    if ctx:
        await ctx.info(f"Fetching Jikan seasonal anime for {season.title()} {year}")
    
    try:
        params = {
            "year": year,
            "season": season,
            "sfw": sfw,
            "unapproved": unapproved,
            "continuing": continuing,
            "limit": min(limit, 25),
            "page": page
        }
        
        if filter:
            params["filter"] = filter
        
        raw_results = await jikan_client.get_seasonal_anime(params)
        
        results = []
        for raw_result in raw_results:
            try:
                universal_anime = jikan_mapper.to_universal_anime(raw_result)
                
                result = {
                    "id": universal_anime.id,
                    "title": universal_anime.title,
                    "type": universal_anime.type_format,
                    "episodes": universal_anime.episodes,
                    "score": universal_anime.score,
                    "year": universal_anime.year,
                    "status": universal_anime.status,
                    "genres": [genre.get("name") for genre in raw_result.get("genres", [])],
                    "studios": [studio.get("name") for studio in raw_result.get("studios", [])],
                    "synopsis": universal_anime.description,
                    "image_url": universal_anime.image_url,
                    
                    # Jikan seasonal data
                    "mal_id": raw_result.get("mal_id"),
                    "jikan_score": raw_result.get("score"),
                    "jikan_members": raw_result.get("members"),
                    "jikan_favorites": raw_result.get("favorites"),
                    "season": season,
                    "season_year": year,
                    "continuing": raw_result.get("continuing", False),
                    
                    "source_platform": "jikan",
                    "data_quality_score": universal_anime.data_quality_score
                }
                
                results.append(result)
                
            except Exception as e:
                if ctx:
                    await ctx.error(f"Failed to process seasonal result: {str(e)}")
                continue
        
        if ctx:
            await ctx.info(f"Found {len(results)} seasonal anime")
            
        return results
        
    except Exception as e:
        error_msg = f"Jikan seasonal search failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)