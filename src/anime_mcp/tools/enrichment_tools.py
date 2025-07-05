"""
MCP tools for cross-platform anime data enrichment and correlation.

Provides high-level MCP tools that use the cross-platform enrichment system
to handle complex queries requiring data from multiple anime platforms.
"""

from typing import Dict, Any, List, Optional
import logging

from fastmcp import FastMCP
from mcp.server.fastmcp import Context

from ...langgraph.enrichment_tools import CrossPlatformEnrichment
from ...models.universal_anime import UniversalSearchParams

logger = logging.getLogger(__name__)

# Initialize FastMCP instance (will be imported by main server)
mcp = FastMCP()

# Global enrichment instance
enrichment_system: Optional[CrossPlatformEnrichment] = None


def get_enrichment_system() -> CrossPlatformEnrichment:
    """Get or create the global enrichment system instance."""
    global enrichment_system
    if enrichment_system is None:
        enrichment_system = CrossPlatformEnrichment()
    return enrichment_system


@mcp.tool(
    name="compare_anime_ratings_cross_platform",
    description="Compare anime ratings across multiple platforms (MAL, AniList, Kitsu, etc.)",
    annotations={
        "title": "Cross-Platform Rating Comparison",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def compare_anime_ratings_cross_platform(
    anime_title: str,
    platforms: List[str] = ["mal", "anilist", "kitsu"],
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Compare anime ratings across multiple platforms with detailed statistics.

    Handles complex queries like:
    "Compare Death Note's ratings across MAL, AniList, and Anime-Planet"

    Args:
        anime_title: Name of the anime to compare ratings for
        platforms: List of platforms to compare (default: ["mal", "anilist", "kitsu"])

    Returns:
        Detailed rating comparison with statistics and platform-specific data
    """
    if ctx:
        await ctx.info(
            f"Comparing ratings for '{anime_title}' across {len(platforms)} platforms"
        )

    logger.info(f"MCP rating comparison: '{anime_title}' across {platforms}")

    try:
        enrichment = get_enrichment_system()
        result = await enrichment.compare_anime_ratings(
            anime_title=anime_title, comparison_platforms=platforms
        )

        if ctx:
            rating_count = len(result.get("rating_details", {}))
            await ctx.info(
                f"Successfully compared ratings from {rating_count} platforms"
            )

        logger.info(f"Rating comparison completed for '{anime_title}'")
        return result

    except Exception as e:
        error_msg = f"Rating comparison failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        logger.error(f"Rating comparison error: {e}")
        raise RuntimeError(error_msg)


@mcp.tool(
    name="get_cross_platform_anime_data",
    description="Enrich anime data by combining information from multiple platforms",
    annotations={
        "title": "Cross-Platform Data Enrichment",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def get_cross_platform_anime_data(
    anime_title: str,
    platforms: List[str] = ["mal", "anilist", "animeschedule", "kitsu"],
    include_ratings: bool = True,
    include_streaming: bool = True,
    include_schedule: bool = True,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Enrich anime data by fetching and correlating information from multiple platforms.

    Provides comprehensive anime information by combining:
    - Rating data from different communities
    - Streaming availability across platforms
    - Broadcast schedules and timing
    - Cross-platform correlation and linking

    Args:
        anime_title: Name of the anime to enrich
        platforms: List of platforms to query for data
        include_ratings: Include rating comparisons across platforms
        include_streaming: Include streaming availability information
        include_schedule: Include broadcast schedule data

    Returns:
        Comprehensive enriched anime data with cross-platform correlation
    """
    if ctx:
        await ctx.info(
            f"Enriching data for '{anime_title}' from {len(platforms)} platforms"
        )

    logger.info(f"MCP data enrichment: '{anime_title}' from {platforms}")

    try:
        enrichment = get_enrichment_system()
        result = await enrichment.enrich_anime_cross_platform(
            reference_anime=anime_title,
            platforms=platforms,
            include_ratings=include_ratings,
            include_streaming=include_streaming,
            include_schedule=include_schedule,
        )

        if ctx:
            platforms_found = len(
                result.get("correlation_results", {}).get("matches", {})
            )
            await ctx.info(
                f"Successfully enriched data from {platforms_found} platforms"
            )

        logger.info(f"Data enrichment completed for '{anime_title}'")
        return result

    except Exception as e:
        error_msg = f"Data enrichment failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        logger.error(f"Data enrichment error: {e}")
        raise RuntimeError(error_msg)


@mcp.tool(
    name="correlate_anime_across_platforms",
    description="Find and link the same anime across multiple platforms with confidence scoring",
    annotations={
        "title": "Cross-Platform Anime Correlation",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def correlate_anime_across_platforms(
    anime_title: str,
    platforms: List[str] = ["mal", "anilist", "kitsu", "jikan"],
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Correlate and link the same anime across multiple platforms.

    Identifies the same anime across different platforms using title matching,
    metadata correlation, and confidence scoring. Useful for cross-platform
    data verification and discrepancy detection.

    Args:
        anime_title: Name of the anime to correlate
        platforms: List of platforms to search for correlation

    Returns:
        Correlation results with platform-specific IDs, confidence scores, and discrepancies
    """
    if ctx:
        await ctx.info(f"Correlating '{anime_title}' across {len(platforms)} platforms")

    logger.info(f"MCP anime correlation: '{anime_title}' across {platforms}")

    try:
        enrichment = get_enrichment_system()
        result = await enrichment.correlate_anime_across_platforms(
            anime_title=anime_title, correlation_platforms=platforms
        )

        if ctx:
            confidence = result.get("correlation_confidence", 0.0)
            matches = len(result.get("platform_matches", {}))
            await ctx.info(
                f"Correlation completed: {matches} matches with {confidence:.1%} confidence"
            )

        logger.info(f"Anime correlation completed for '{anime_title}'")
        return result

    except Exception as e:
        error_msg = f"Anime correlation failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        logger.error(f"Anime correlation error: {e}")
        raise RuntimeError(error_msg)


@mcp.tool(
    name="get_streaming_availability_multi_platform",
    description="Aggregate streaming availability information from multiple platforms",
    annotations={
        "title": "Multi-Platform Streaming Availability",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def get_streaming_availability_multi_platform(
    anime_title: str,
    platforms: List[str] = ["kitsu", "animeschedule"],
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Get comprehensive streaming availability by aggregating data from multiple platforms.

    Combines streaming information from platforms that specialize in availability data
    to provide the most comprehensive view of where an anime can be watched.

    Args:
        anime_title: Name of the anime to check streaming for
        platforms: List of platforms to query for streaming data

    Returns:
        Aggregated streaming availability with platform lists and regional information
    """
    if ctx:
        await ctx.info(
            f"Checking streaming availability for '{anime_title}' across {len(platforms)} platforms"
        )

    logger.info(f"MCP streaming check: '{anime_title}' from {platforms}")

    try:
        enrichment = get_enrichment_system()
        result = await enrichment.get_cross_platform_streaming_info(
            anime_title=anime_title, target_platforms=platforms
        )

        if ctx:
            total_platforms = result.get("availability_summary", {}).get(
                "total_platforms", 0
            )
            await ctx.info(
                f"Found streaming availability on {total_platforms} platforms"
            )

        logger.info(f"Streaming availability check completed for '{anime_title}'")
        return result

    except Exception as e:
        error_msg = f"Streaming availability check failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        logger.error(f"Streaming availability error: {e}")
        raise RuntimeError(error_msg)


@mcp.tool(
    name="detect_platform_discrepancies",
    description="Detect discrepancies in anime data between different platforms",
    annotations={
        "title": "Cross-Platform Discrepancy Detection",
        "readOnlyHint": True,
        "idempotentHint": True,
    },
)
async def detect_platform_discrepancies(
    anime_title: str,
    comparison_fields: List[str] = ["title", "year", "episodes", "rating"],
    platforms: List[str] = ["mal", "anilist", "kitsu"],
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Detect discrepancies in anime data between platforms.

    Handles queries like:
    "Find discrepancies in scores between Eastern (MAL) and Western (Anime-Planet) sites"
    "Check if this anime has different episode counts on different platforms"

    Args:
        anime_title: Name of the anime to check for discrepancies
        comparison_fields: Fields to compare across platforms
        platforms: List of platforms to compare

    Returns:
        Detailed discrepancy analysis with field-by-field comparison
    """
    if ctx:
        await ctx.info(
            f"Checking for discrepancies in '{anime_title}' across {len(platforms)} platforms"
        )

    logger.info(
        f"MCP discrepancy detection: '{anime_title}' comparing {comparison_fields}"
    )

    try:
        enrichment = get_enrichment_system()

        # Get correlated data across platforms
        correlation_result = await enrichment.correlate_anime_across_platforms(
            anime_title=anime_title, correlation_platforms=platforms
        )

        # Analyze discrepancies
        discrepancies = {
            "anime_title": anime_title,
            "platforms_compared": platforms,
            "fields_analyzed": comparison_fields,
            "discrepancies_found": [],
            "consistency_score": 0.0,
            "analysis_timestamp": correlation_result.get("correlation_timestamp"),
        }

        platform_matches = correlation_result.get("platform_matches", {})

        if len(platform_matches) >= 2:
            # Compare each field across platforms
            for field in comparison_fields:
                field_values = {}
                for platform, anime_data in platform_matches.items():
                    if field in anime_data:
                        field_values[platform] = anime_data[field]

                # Detect discrepancies in this field
                if len(field_values) >= 2:
                    unique_values = list(set(field_values.values()))
                    if len(unique_values) > 1:
                        discrepancies["discrepancies_found"].append(
                            {
                                "field": field,
                                "platform_values": field_values,
                                "unique_values": unique_values,
                                "discrepancy_type": "value_mismatch",
                            }
                        )

            # Calculate consistency score
            total_fields = len(comparison_fields)
            consistent_fields = total_fields - len(discrepancies["discrepancies_found"])
            discrepancies["consistency_score"] = (
                consistent_fields / total_fields if total_fields > 0 else 1.0
            )

        if ctx:
            discrepancy_count = len(discrepancies["discrepancies_found"])
            consistency = discrepancies["consistency_score"]
            await ctx.info(
                f"Found {discrepancy_count} discrepancies, {consistency:.1%} consistency"
            )

        logger.info(f"Discrepancy detection completed for '{anime_title}'")
        return discrepancies

    except Exception as e:
        error_msg = f"Discrepancy detection failed: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        logger.error(f"Discrepancy detection error: {e}")
        raise RuntimeError(error_msg)
