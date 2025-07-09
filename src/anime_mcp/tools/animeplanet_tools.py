"""
AnimePlanet specific MCP tools.

Comprehensive tools for AnimePlanet web scraping with rich metadata extraction.
AnimePlanet specializes in Western community ratings, comprehensive tagging, and detailed reviews.
"""

from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from mcp.server.fastmcp import Context

from ...config import get_settings
from ...integrations.scrapers.extractors.anime_planet_scraper import AnimePlanetScraper
from ...models.structured_responses import (
    AnimeType,
    BasicAnimeResult,
)

# Initialize components
settings = get_settings()
animeplanet_scraper = AnimePlanetScraper()

# Create FastMCP instance for tools
mcp = FastMCP("AnimePlanet Tools")


@mcp.tool(
    name="search_anime_animeplanet",
    description="Search anime on AnimePlanet with rich metadata from hover tooltips",
    annotations={
        "title": "AnimePlanet Anime Search",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def search_anime_animeplanet(
    # Basic search
    query: str,
    limit: int = 10,
    ctx: Optional[Context] = None,
) -> List[Dict[str, Any]]:
    """
    Search anime on AnimePlanet with comprehensive metadata extraction.

    AnimePlanet provides unique Western community perspective with rich tooltip data including:
    - Community ratings and tags
    - Alternative titles (Japanese/English)
    - Studio information and production details
    - Comprehensive synopsis and episode counts
    - Western-focused tagging system

    Args:
        query: Search query for anime titles
        limit: Maximum number of results to return (default: 10, max: 50)

    Returns:
        List of anime with rich metadata from AnimePlanet tooltips
    """
    if ctx:
        await ctx.info(f"Searching AnimePlanet for '{query}' with limit {limit}")

    try:
        # Execute search with AnimePlanet scraper
        raw_results = await animeplanet_scraper.search_anime(query, limit=min(limit, 50))

        if not raw_results:
            if ctx:
                await ctx.info("No results found on AnimePlanet")
            return []

        # Convert to structured response format
        anime_results = []
        for result in raw_results:
            if isinstance(result, dict):
                # Convert raw AnimePlanet data to BasicAnimeResult
                anime_result = BasicAnimeResult(
                    id=result.get("animeplanet_id", ""),
                    title=result.get("title", ""),
                    score=result.get("rating") if result.get("rating") else None,
                    year=result.get("year") if result.get("year") else None,
                    type=(
                        AnimeType(result.get("type", "tv").lower())
                        if result.get("type")
                        else None
                    ),
                    synopsis=result.get("synopsis", ""),
                    url=result.get("url", ""),
                    source_platform="animeplanet",
                )

                # Add AnimePlanet-specific enrichment data
                enriched_result = anime_result.model_dump()
                enriched_result.update({
                    "animeplanet_data": {
                        "slug": result.get("slug"),
                        "alt_title": result.get("alt_title"),
                        "episodes": result.get("episodes"),
                        "studios": result.get("studios", []),
                        "tags": result.get("tags", []),
                        "rating": result.get("rating"),
                        "total_episodes": result.get("total_episodes"),
                        "content_type": result.get("content_type"),
                        "domain": result.get("domain"),
                    }
                })

                anime_results.append(enriched_result)

        if ctx:
            await ctx.info(f"Retrieved {len(anime_results)} results from AnimePlanet")

        return anime_results

    except Exception as e:
        if ctx:
            await ctx.error(f"AnimePlanet search failed: {str(e)}")
        return []


@mcp.tool(
    name="get_anime_animeplanet",
    description="Get comprehensive anime details from AnimePlanet by slug",
    annotations={
        "title": "AnimePlanet Anime Details",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def get_anime_animeplanet(
    slug: str,
    ctx: Optional[Context] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive anime information from AnimePlanet by slug.

    Provides detailed AnimePlanet data including:
    - Community ratings and score breakdowns
    - Comprehensive tagging and genre information
    - Alternative titles and translations
    - Studio and production information
    - Related anime and franchise connections
    - Western community reviews and recommendations

    Args:
        slug: AnimePlanet anime slug (e.g., "naruto", "attack-on-titan")

    Returns:
        Comprehensive anime information with AnimePlanet community data
    """
    if ctx:
        await ctx.info(f"Fetching AnimePlanet anime details for slug: {slug}")

    try:
        # Get comprehensive anime data
        raw_result = await animeplanet_scraper.get_anime_by_slug(slug)

        if not raw_result:
            if ctx:
                await ctx.info(f"Anime with slug '{slug}' not found on AnimePlanet")
            return None

        # Extract title and basic data
        primary_title = raw_result.get("title", "Unknown")

        # Map type to AnimeType
        anime_type = None
        animeplanet_type = raw_result.get("type")
        if animeplanet_type:
            type_mapping = {
                "TV": AnimeType.TV,
                "Movie": AnimeType.MOVIE,
                "OVA": AnimeType.OVA,
                "ONA": AnimeType.ONA,
                "Special": AnimeType.SPECIAL,
                "Music": AnimeType.MUSIC,
            }
            anime_type = type_mapping.get(animeplanet_type.upper())

        # Create structured result
        result = {
            "id": raw_result.get("animeplanet_id", ""),
            "title": primary_title,
            "alt_title": raw_result.get("title_native") or raw_result.get("title_english"),
            "synopsis": raw_result.get("synopsis", ""),
            "type": anime_type.value if anime_type else None,
            "episodes": raw_result.get("episodes"),
            "year": raw_result.get("year"),
            "score": raw_result.get("score"),
            "rating": raw_result.get("rating"),
            "status": raw_result.get("status"),
            "url": f"https://www.anime-planet.com/anime/{slug}",
            "source_platform": "animeplanet",
            "animeplanet_data": {
                "slug": slug,
                "domain": raw_result.get("domain"),
                "tags": raw_result.get("tags", []),
                "studios": raw_result.get("studios", []),
                "synonyms": raw_result.get("synonyms", []),
                "start_date": raw_result.get("start_date"),
                "end_date": raw_result.get("end_date"),
                "season": raw_result.get("season"),
                "rank": raw_result.get("rank"),
                "score_count": raw_result.get("score_count"),
                "related_anime": raw_result.get("related_anime", []),
                "related_count": raw_result.get("related_count", 0),
                "characters": raw_result.get("characters", []),
                "voice_actors": raw_result.get("voice_actors", []),
                "directors": raw_result.get("directors", []),
                "music_composers": raw_result.get("music_composers", []),
                "json_ld": raw_result.get("json_ld"),
                "structured_metadata": raw_result.get("structured_metadata"),
            }
        }

        if ctx:
            await ctx.info(f"Retrieved comprehensive AnimePlanet data for '{result['title']}'")

        return result

    except Exception as e:
        if ctx:
            await ctx.error(f"AnimePlanet details fetch failed: {str(e)}")
        return None




# Tool registration function
def register_animeplanet_tools(mcp_instance):
    """Register AnimePlanet tools with MCP server."""
    mcp_instance.mount(mcp)
    return 2  # Number of tools registered


# Export for easy access
__all__ = [
    "search_anime_animeplanet",
    "get_anime_animeplanet", 
    "register_animeplanet_tools",
    "mcp",
]