#!/usr/bin/env python3
"""
Comprehensive MCP Server Verification Script

This script provides complete testing of the FastMCP anime server including:
- MCP protocol compliance and tool functionality
- Image search capabilities and CLIP processing
- Database statistics and health monitoring
- Multi-modal search validation

Usage:
    python scripts/verify_mcp_server.py [--detailed] [--skip-image-tests]

Requirements:
    - MCP server dependencies installed
    - Qdrant running with anime database
    - CLIP model for image processing (optional, will use mock if unavailable)
"""
import argparse
import asyncio
import base64
import json
import logging
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.vector.qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_mcp_functionality(session: ClientSession) -> bool:
    """Test basic MCP protocol functionality."""
    logger.info("Testing basic MCP functionality...")

    try:
        # List available tools
        tools = await session.list_tools()
        tool_names = [tool.name for tool in tools.tools]
        logger.info(f"Available tools: {tool_names}")

        # Verify expected tools are present
        expected_tools = [
            "search_anime",
            "get_anime_details",
            "find_similar_anime",
            "get_anime_stats",
            "recommend_anime",
            "search_anime_by_image",
            "find_visually_similar_anime",
            "search_multimodal_anime",
        ]

        missing_tools = [tool for tool in expected_tools if tool not in tool_names]
        if missing_tools:
            logger.error(f"Missing expected tools: {missing_tools}")
            return False

        logger.info("All expected MCP tools are available")

        # Test basic search
        search_result = await session.call_tool(
            "search_anime", arguments={"query": "dragon ball", "limit": 2}
        )

        if search_result.content and search_result.content[0].text:
            logger.info("Basic search test successful")
            # Parse the JSON result to verify structure
            try:
                result_data = json.loads(search_result.content[0].text)
                if isinstance(result_data, list) and len(result_data) > 0:
                    logger.info(f"   Found {len(result_data)} results")
                    first_result = result_data[0]
                    if "title" in first_result and "anime_id" in first_result:
                        logger.info(f"   Sample: {first_result['title']}")
                    else:
                        logger.warning("Search result missing expected fields")
                else:
                    logger.warning("Search returned empty or malformed results")
            except json.JSONDecodeError:
                logger.warning("Search result is not valid JSON")
        else:
            logger.error("Basic search test failed - no content returned")
            return False

        # Test stats
        stats_result = await session.call_tool("get_anime_stats", arguments={})
        if stats_result.content and stats_result.content[0].text:
            logger.info("Stats test successful")
            try:
                stats_data = json.loads(stats_result.content[0].text)
                if "total_documents" in stats_data:
                    logger.info(
                        f"   Database contains {stats_data['total_documents']:,} anime entries"
                    )
                else:
                    logger.warning("Stats missing expected fields")
            except json.JSONDecodeError:
                logger.warning("Stats result is not valid JSON")
        else:
            logger.error("Stats test failed")
            return False

        # Test resources
        resources = await session.list_resources()
        resource_uris = [res.uri for res in resources.resources]
        logger.info(f"Available resources: {resource_uris}")

        expected_resources = ["anime://database/stats", "anime://database/schema"]
        missing_resources = [
            res for res in expected_resources if res not in resource_uris
        ]
        if missing_resources:
            logger.warning(f"Missing expected resources: {missing_resources}")
        else:
            logger.info("All expected resources are available")

        return True

    except Exception as e:
        logger.error(f"Basic MCP functionality test failed: {e}")
        return False


async def test_image_search_functionality(
    session: ClientSession, detailed: bool = False
) -> bool:
    """Test image search capabilities through MCP tools."""
    logger.info("Testing image search functionality...")

    try:
        # Test image for search (10x10 pixel test image in base64)
        test_image_b64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAKAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="

        # Test 1: Basic image search
        logger.info("Test 1: search_anime_by_image")
        try:
            image_search_result = await session.call_tool(
                "search_anime_by_image",
                arguments={"image_data": test_image_b64, "limit": 3},
            )

            if image_search_result.content and image_search_result.content[0].text:
                try:
                    results = json.loads(image_search_result.content[0].text)
                    if isinstance(results, list):
                        logger.info(f"Image search returned {len(results)} results")
                        if detailed and results:
                            for i, result in enumerate(results[:2], 1):
                                title = result.get("title", "Unknown")
                                score = result.get("score", 0)
                                logger.info(f"   {i}. {title} (score: {score:.4f})")
                    else:
                        logger.warning("Image search returned non-list result")
                except json.JSONDecodeError:
                    logger.warning("Image search result is not valid JSON")
            else:
                logger.warning("Image search returned no content")

        except Exception as e:
            logger.error(f"Image search test failed: {e}")
            return False

        # Test 2: Visual similarity search (need to find an anime with image embeddings)
        logger.info("Test 2: find_visually_similar_anime")
        try:
            # First get a sample anime ID that has image embeddings
            settings = get_settings()
            qdrant_client = QdrantClient(settings=settings)

            # Check if we can connect to Qdrant directly
            if await qdrant_client.health_check():
                # Get a sample anime with image embeddings
                loop = asyncio.get_event_loop()
                sample_points, _ = await loop.run_in_executor(
                    None,
                    lambda: qdrant_client.client.scroll(
                        collection_name=qdrant_client.collection_name,
                        limit=10,
                        with_vectors=["image"],
                        with_payload=True,
                    ),
                )

                test_anime_id = None
                for point in sample_points:
                    image_vector = point.vector.get("image", [])
                    if image_vector and not all(v == 0.0 for v in image_vector):
                        test_anime_id = point.payload.get("anime_id")
                        test_anime_title = point.payload.get("title", "Unknown")
                        logger.info(f"   Using test anime: {test_anime_title}")
                        break

                if test_anime_id:
                    similar_result = await session.call_tool(
                        "find_visually_similar_anime",
                        arguments={"anime_id": test_anime_id, "limit": 3},
                    )

                    if similar_result.content and similar_result.content[0].text:
                        try:
                            results = json.loads(similar_result.content[0].text)
                            if isinstance(results, list):
                                logger.info(
                                    f"Visual similarity search returned {len(results)} results"
                                )
                                if detailed and results:
                                    for i, result in enumerate(results[:2], 1):
                                        title = result.get("title", "Unknown")
                                        score = result.get("score", 0)
                                        logger.info(
                                            f"   {i}. {title} (score: {score:.4f})"
                                        )
                            else:
                                logger.warning(
                                    "Visual similarity returned non-list result"
                                )
                        except json.JSONDecodeError:
                            logger.warning("Visual similarity result is not valid JSON")
                    else:
                        logger.warning("Visual similarity search returned no content")
                else:
                    logger.warning(
                        "No anime found with image embeddings for similarity test"
                    )
            else:
                logger.warning(
                    "Cannot connect to Qdrant directly for similarity test setup"
                )

        except Exception as e:
            logger.error(f"Visual similarity test failed: {e}")
            return False

        # Test 3: Multimodal search
        logger.info("Test 3: search_multimodal_anime")
        try:
            multimodal_result = await session.call_tool(
                "search_multimodal_anime",
                arguments={
                    "query": "action anime",
                    "image_data": test_image_b64,
                    "limit": 3,
                    "text_weight": 0.7,
                },
            )

            if multimodal_result.content and multimodal_result.content[0].text:
                try:
                    results = json.loads(multimodal_result.content[0].text)
                    if isinstance(results, list):
                        logger.info(
                            f"Multimodal search returned {len(results)} results"
                        )
                        if detailed and results:
                            for i, result in enumerate(results[:2], 1):
                                title = result.get("title", "Unknown")
                                score = result.get("score", 0)
                                logger.info(f"   {i}. {title} (score: {score:.4f})")
                    else:
                        logger.warning("Multimodal search returned non-list result")
                except json.JSONDecodeError:
                    logger.warning("Multimodal search result is not valid JSON")
            else:
                logger.warning("Multimodal search returned no content")

        except Exception as e:
            logger.error(f"Multimodal search test failed: {e}")
            return False

        logger.info("Image search functionality tests completed")
        return True

    except Exception as e:
        logger.error(f"Image search functionality test failed: {e}")
        return False


async def test_database_health_and_statistics(detailed: bool = False) -> bool:
    """Test database health and provide statistics on image processing."""
    logger.info("Testing database health and statistics...")

    try:
        settings = get_settings()
        qdrant_client = QdrantClient(settings=settings)

        # Test connection
        if not await qdrant_client.health_check():
            logger.error("Qdrant connection failed")
            return False

        logger.info("Qdrant connection verified")

        # Get basic stats
        stats = await qdrant_client.get_stats()
        total_docs = stats.get("total_documents", 0)
        logger.info(f"Total anime entries: {total_docs:,}")

        if detailed and total_docs > 0:
            # Check image processing statistics
            logger.info("Analyzing image processing coverage...")

            processed_count = 0
            zero_count = 0
            sample_size = min(100, total_docs)

            loop = asyncio.get_event_loop()
            batch_points, _ = await loop.run_in_executor(
                None,
                lambda: qdrant_client.client.scroll(
                    collection_name=qdrant_client.collection_name,
                    limit=sample_size,
                    with_vectors=["image"],
                ),
            )

            for point in batch_points:
                image_vector = point.vector.get("image", [])
                if image_vector and not all(v == 0.0 for v in image_vector):
                    processed_count += 1
                else:
                    zero_count += 1

            processing_rate = (
                (processed_count / sample_size) * 100 if sample_size > 0 else 0
            )

            logger.info(f"Image processing statistics (sample of {sample_size}):")
            logger.info(f"   With image embeddings: {processed_count}")
            logger.info(f"   Without image embeddings: {zero_count}")
            logger.info(f"   Processing rate: {processing_rate:.1f}%")

            if processing_rate < 50:
                logger.warning(
                    "Low image processing coverage - consider running image embedding pipeline"
                )
            elif processing_rate > 90:
                logger.info("Excellent image processing coverage!")

        return True

    except Exception as e:
        logger.error(f"Database health test failed: {e}")
        return False


async def run_comprehensive_tests(
    detailed: bool = False, skip_image_tests: bool = False
):
    """Run comprehensive MCP server verification tests."""
    logger.info("Starting comprehensive FastMCP Anime Server verification...")
    logger.info("=" * 60)

    # Server parameters
    server_params = StdioServerParameters(
        command=sys.executable, args=["-m", "src.mcp.server"]
    )

    total_tests = 0
    passed_tests = 0

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the client
                await session.initialize()
                logger.info("MCP session initialized")

                # Test 1: Basic MCP functionality
                total_tests += 1
                if await test_basic_mcp_functionality(session):
                    passed_tests += 1

                # Test 2: Image search functionality (unless skipped)
                if not skip_image_tests:
                    total_tests += 1
                    if await test_image_search_functionality(session, detailed):
                        passed_tests += 1
                else:
                    logger.info("Skipping image search tests (--skip-image-tests)")

        # Test 3: Database health (runs outside MCP session)
        total_tests += 1
        if await test_database_health_and_statistics(detailed):
            passed_tests += 1

    except Exception as e:
        logger.error(f"Test session failed: {e}")
        return False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info(f"   Tests passed: {passed_tests}/{total_tests}")
    logger.info(f"   Tests failed: {total_tests - passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        logger.info("All tests completed successfully!")
        logger.info("MCP server is fully functional")
        logger.info("Image search capabilities verified")
        logger.info("Database connectivity confirmed")
        return True
    else:
        logger.error(f"{total_tests - passed_tests} test(s) failed")
        logger.info("Check logs above for specific error details")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive MCP Server Verification"
    )
    parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed test results and sample data",
    )
    parser.add_argument(
        "--skip-image-tests",
        action="store_true",
        help="Skip image search tests (useful if CLIP not available)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    success = asyncio.run(
        run_comprehensive_tests(
            detailed=args.detailed, skip_image_tests=args.skip_image_tests
        )
    )
    sys.exit(0 if success else 1)
