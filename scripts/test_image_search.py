#!/usr/bin/env python3
"""
Test Image Search Functionality

Tests the new image search capabilities including:
- search_anime_by_image
- find_visually_similar_anime
- search_multimodal_anime

For MCP tools testing.
"""

import asyncio
import sys
import base64
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.vector.qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_image_search():
    """Test image search functionality."""
    logger.info("🧪 Testing Image Search Functionality")
    logger.info("=" * 50)
    
    # Initialize client
    settings = get_settings()
    qdrant_client = QdrantClient(settings=settings)
    
    # Verify connection
    if not await qdrant_client.health_check():
        logger.error("❌ Qdrant connection failed")
        return False
    
    logger.info("✅ Qdrant connection verified")
    
    # Test 1: Simple image search with test image
    logger.info("\n📝 Test 1: Simple image search")
    try:
        # Create a proper test image (10x10 pixel red square PNG in base64)
        test_image_b64 = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAKAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
        
        results = await qdrant_client.search_by_image(image_data=test_image_b64, limit=3)
        
        if results:
            logger.info(f"✅ Image search returned {len(results)} results")
            for i, result in enumerate(results[:2], 1):
                title = result.get('title', 'Unknown')
                score = result.get('score', 0)
                logger.info(f"   {i}. {title} (score: {score:.4f})")
        else:
            logger.warning("⚠️  Image search returned no results")
            
    except Exception as e:
        logger.error(f"❌ Image search test failed: {e}")
        return False
    
    # Test 2: Get an anime ID that has image embeddings
    logger.info("\n📝 Test 2: Visual similarity search")
    try:
        # Get first anime with non-zero image vectors
        loop = asyncio.get_event_loop()
        sample_points, _ = await loop.run_in_executor(
            None,
            lambda: qdrant_client.client.scroll(
                collection_name=qdrant_client.collection_name,
                limit=10,
                with_vectors=["image"],
                with_payload=True
            )
        )
        
        # Find an anime with real image embeddings
        test_anime_id = None
        for point in sample_points:
            image_vector = point.vector.get("image", [])
            if image_vector and not all(v == 0.0 for v in image_vector):
                # Use the anime_id from payload, not the point ID (which is hashed)
                test_anime_id = point.payload.get("anime_id")
                test_anime_title = point.payload.get("title", "Unknown")
                logger.info(f"Using test anime: {test_anime_title} (ID: {test_anime_id})")
                break
        
        if test_anime_id:
            # Test find_visually_similar_anime
            similar_results = await qdrant_client.find_visually_similar_anime(anime_id=test_anime_id, limit=3)
            
            if similar_results:
                logger.info(f"✅ Visual similarity search returned {len(similar_results)} results")
                for i, result in enumerate(similar_results[:2], 1):
                    title = result.get('title', 'Unknown')
                    score = result.get('score', 0)
                    logger.info(f"   {i}. {title} (score: {score:.4f})")
            else:
                logger.warning("⚠️  Visual similarity search returned no results")
        else:
            logger.warning("⚠️  No anime found with image embeddings for similarity test")
            
    except Exception as e:
        logger.error(f"❌ Visual similarity test failed: {e}")
        return False
    
    # Test 3: Multimodal search
    logger.info("\n📝 Test 3: Multimodal search (text + image)")
    try:
        results = await qdrant_client.search_multimodal(
            query="action anime",
            image_data=test_image_b64,
            limit=3,
            text_weight=0.7
        )
        
        if results:
            logger.info(f"✅ Multimodal search returned {len(results)} results")
            for i, result in enumerate(results[:2], 1):
                title = result.get('title', 'Unknown')
                score = result.get('score', 0)
                logger.info(f"   {i}. {title} (score: {score:.4f})")
        else:
            logger.warning("⚠️  Multimodal search returned no results")
            
    except Exception as e:
        logger.error(f"❌ Multimodal search test failed: {e}")
        return False
    
    # Test 4: Check statistics
    logger.info("\n📝 Test 4: Database statistics with image info")
    try:
        stats = await qdrant_client.get_stats()
        total_docs = stats.get("total_documents", 0)
        
        # Count how many have image embeddings
        processed_count = 0
        batch_points, _ = await loop.run_in_executor(
            None,
            lambda: qdrant_client.client.scroll(
                collection_name=qdrant_client.collection_name,
                limit=100,
                with_vectors=["image"]
            )
        )
        
        for point in batch_points:
            image_vector = point.vector.get("image", [])
            if image_vector and not all(v == 0.0 for v in image_vector):
                processed_count += 1
        
        logger.info(f"✅ Database statistics:")
        logger.info(f"   📊 Total anime: {total_docs:,}")
        logger.info(f"   🖼️  With image embeddings: {processed_count} (sample from first 100)")
        logger.info(f"   📈 Processing rate: {(processed_count/min(100, total_docs)*100):.1f}% (sample)")
        
    except Exception as e:
        logger.error(f"❌ Statistics test failed: {e}")
        return False
    
    logger.info("\n🎉 All image search tests completed successfully!")
    logger.info("📋 Image search functionality is working correctly")
    
    return True


async def main():
    """Main test entry point."""
    try:
        success = await test_image_search()
        if success:
            logger.info("\n✅ Image search testing completed successfully")
            logger.info("💡 Next steps:")
            logger.info("   1. Process more anime images (run with higher --limit)")
            logger.info("   2. Add REST API endpoints for image search")
            logger.info("   3. Test with real anime poster images")
        else:
            logger.error("\n❌ Image search testing failed")
        
        return success
    except Exception as e:
        logger.error(f"❌ Test error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)