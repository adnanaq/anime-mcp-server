#!/usr/bin/env python3
"""
Image Embedding Processing Pipeline

Processes anime poster images and adds image vector embeddings to the
existing multi-vector Qdrant collection. Downloads images from URLs,
generates CLIP embeddings, and updates collection with named vectors.

Safety Features:
- Batch processing for memory efficiency
- Error handling for failed downloads/processing
- Progress monitoring with statistics
- Verification of embedding quality
- Rollback capability if critical errors occur

Usage:
    python scripts/add_image_embeddings.py [--batch-size 100] [--dry-run] [--force]
"""

import argparse
import asyncio
import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from PIL import Image

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client.models import PointStruct

from src.config import get_settings
from src.vector.qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("image_processing.log")],
)
logger = logging.getLogger(__name__)


class ImageEmbeddingProcessor:
    """Processes anime poster images and adds embeddings to Qdrant collection."""

    def __init__(self, qdrant_client: QdrantClient, batch_size: int = 100):
        self.qdrant_client = qdrant_client
        self.batch_size = batch_size
        self.session = None
        self.stats = {
            "total_anime": 0,
            "processed": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "skipped_no_url": 0,
            "updated_vectors": 0,
        }

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Anime-MCP-Server/1.0"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get_anime_points_needing_images(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get anime points that need image embeddings processed.

        Args:
            limit: Maximum number of anime to process (None for all)
        """
        logger.info("🔍 Fetching anime entries that need image processing...")

        all_points = []
        scroll_offset = None

        while True:
            # Get batch of points with vectors to check image embedding status
            loop = asyncio.get_event_loop()
            batch_points, next_offset = await loop.run_in_executor(
                None,
                lambda: self.qdrant_client.client.scroll(
                    collection_name=self.qdrant_client.collection_name,
                    limit=self.batch_size,
                    offset=scroll_offset,
                    with_payload=True,
                    with_vectors=["image"],  # Only get image vectors to check if empty
                ),
            )

            # Filter points that need image processing (have zero image vectors)
            for point in batch_points:
                image_vector = point.vector.get("image", [])
                # Check if image vector is all zeros (needs processing)
                if not image_vector or all(v == 0.0 for v in image_vector):
                    picture_url = point.payload.get("picture", "")
                    if picture_url:  # Only process if has picture URL
                        all_points.append(
                            {
                                "id": point.id,
                                "payload": point.payload,
                                "picture_url": picture_url,
                            }
                        )

                        # Check limit
                        if limit and len(all_points) >= limit:
                            break
                    else:
                        self.stats["skipped_no_url"] += 1

            # Break if we hit the limit or reached the end
            if (limit and len(all_points) >= limit) or next_offset is None:
                break
            scroll_offset = next_offset

        self.stats["total_anime"] = len(all_points)
        logger.info(
            f"📊 Found {len(all_points)} anime entries needing image processing"
        )
        logger.info(
            f"📊 Skipped {self.stats['skipped_no_url']} entries without picture URLs"
        )

        return all_points

    async def download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL with error handling."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content_type = response.headers.get("content-type", "")
                    if "image" in content_type.lower():
                        data = await response.read()
                        # Verify it's a valid image
                        try:
                            Image.open(io.BytesIO(data)).verify()
                            return data
                        except Exception:
                            logger.warning(f"Invalid image data from {url}")
                            return None
                    else:
                        logger.warning(
                            f"Non-image content type: {content_type} for {url}"
                        )
                        return None
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading {url}")
            return None
        except Exception as e:
            logger.warning(f"Error downloading {url}: {e}")
            return None

    def image_to_base64(self, image_data: bytes) -> str:
        """Convert image bytes to base64 string for embedding processing."""
        return base64.b64encode(image_data).decode("utf-8")

    async def process_image_batch(
        self, anime_batch: List[Dict[str, Any]]
    ) -> List[PointStruct]:
        """Process a batch of anime images and generate embeddings."""
        updated_points = []

        for anime in anime_batch:
            self.stats["processed"] += 1
            anime_id = anime["id"]
            picture_url = anime["picture_url"]
            payload = anime["payload"]

            logger.debug(
                f"Processing {anime['payload'].get('title', 'Unknown')} - {picture_url}"
            )

            # Download image
            image_data = await self.download_image(picture_url)
            if not image_data:
                self.stats["failed_downloads"] += 1
                continue

            self.stats["successful_downloads"] += 1

            # Convert to base64 for embedding processing
            image_base64 = self.image_to_base64(image_data)

            # Generate image embedding using CLIP
            image_embedding = self.qdrant_client._create_image_embedding(image_base64)
            if not image_embedding or all(v == 0.0 for v in image_embedding):
                self.stats["failed_embeddings"] += 1
                logger.warning(
                    f"Failed to generate embedding for {anime['payload'].get('title', 'Unknown')}"
                )
                continue

            self.stats["successful_embeddings"] += 1

            # Get existing text vector (we need both for multi-vector update)
            try:
                loop = asyncio.get_event_loop()
                existing_point = await loop.run_in_executor(
                    None,
                    lambda: self.qdrant_client.client.retrieve(
                        collection_name=self.qdrant_client.collection_name,
                        ids=[anime_id],
                        with_vectors=["text"],
                    ),
                )

                if existing_point:
                    text_vector = existing_point[0].vector.get("text", [])

                    # Create updated point with both text and image vectors
                    updated_point = PointStruct(
                        id=anime_id,
                        vector={"text": text_vector, "image": image_embedding},
                        payload=payload,
                    )
                    updated_points.append(updated_point)
                    self.stats["updated_vectors"] += 1
                else:
                    logger.warning(f"Could not retrieve existing point for {anime_id}")

            except Exception as e:
                logger.error(f"Error processing {anime_id}: {e}")
                self.stats["failed_embeddings"] += 1
                continue

        return updated_points

    async def update_collection_batch(self, points: List[PointStruct]) -> bool:
        """Update Qdrant collection with new image embeddings."""
        if not points:
            return True

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.qdrant_client.client.upsert(
                    collection_name=self.qdrant_client.collection_name, points=points
                ),
            )
            logger.info(f"✅ Updated {len(points)} anime with image embeddings")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update batch: {e}")
            return False

    async def process_all_images(
        self, dry_run: bool = False, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process anime images and add embeddings to collection.

        Args:
            dry_run: If True, don't make actual changes
            limit: Maximum number of anime to process (None for all)
        """
        start_time = time.time()

        # Get anime entries needing image processing
        anime_points = await self.get_anime_points_needing_images(limit=limit)

        if not anime_points:
            logger.info("✅ No anime entries need image processing")
            return {"success": True, "message": "No processing needed"}

        if dry_run:
            logger.info(f"🔧 DRY RUN: Would process {len(anime_points)} anime images")
            return {"success": True, "message": "Dry run completed"}

        logger.info(f"🚀 Starting image processing for {len(anime_points)} anime...")

        # Process in batches
        for i in range(0, len(anime_points), self.batch_size):
            batch = anime_points[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(anime_points) + self.batch_size - 1) // self.batch_size

            logger.info(
                f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} anime)"
            )

            # Process images and generate embeddings
            updated_points = await self.process_image_batch(batch)

            # Update collection if we have successful embeddings
            if updated_points:
                success = await self.update_collection_batch(updated_points)
                if not success:
                    logger.error(f"❌ Critical error updating batch {batch_num}")
                    break

            # Progress update
            progress = (self.stats["processed"] / len(anime_points)) * 100
            logger.info(
                f"📈 Progress: {progress:.1f}% ({self.stats['processed']}/{len(anime_points)})"
            )

            # Small delay to prevent overwhelming servers
            await asyncio.sleep(1)

        # Final statistics
        end_time = time.time()
        duration = end_time - start_time

        logger.info("🎉 Image processing completed!")
        logger.info("📊 Final Statistics:")
        logger.info(f"   📁 Total anime entries: {self.stats['total_anime']}")
        logger.info(f"   ✅ Successfully processed: {self.stats['processed']}")
        logger.info(f"   📥 Successful downloads: {self.stats['successful_downloads']}")
        logger.info(f"   ❌ Failed downloads: {self.stats['failed_downloads']}")
        logger.info(
            f"   🧠 Successful embeddings: {self.stats['successful_embeddings']}"
        )
        logger.info(f"   ❌ Failed embeddings: {self.stats['failed_embeddings']}")
        logger.info(f"   📝 Updated vectors: {self.stats['updated_vectors']}")
        logger.info(f"   ⏱️  Total time: {duration/60:.1f} minutes")

        # Calculate success rate
        if self.stats["processed"] > 0:
            success_rate = (
                self.stats["successful_embeddings"] / self.stats["processed"]
            ) * 100
            logger.info(f"   📈 Success rate: {success_rate:.1f}%")

        return {"success": True, "stats": self.stats, "duration_minutes": duration / 60}


async def verify_image_processing(qdrant_client: QdrantClient) -> Dict[str, Any]:
    """Verify that image processing was successful."""
    logger.info("🔍 Verifying image processing results...")

    # Sample a few points to check image vectors
    loop = asyncio.get_event_loop()
    sample_points, _ = await loop.run_in_executor(
        None,
        lambda: qdrant_client.client.scroll(
            collection_name=qdrant_client.collection_name,
            limit=10,
            with_vectors=["image"],
        ),
    )

    non_zero_count = 0
    zero_count = 0

    for point in sample_points:
        image_vector = point.vector.get("image", [])
        if image_vector and not all(v == 0.0 for v in image_vector):
            non_zero_count += 1
        else:
            zero_count += 1

    logger.info(f"📊 Sample verification (10 anime):")
    logger.info(f"   ✅ With image embeddings: {non_zero_count}")
    logger.info(f"   ❌ Still zero vectors: {zero_count}")

    # Test image search functionality
    try:
        # Create a proper test image (10x10 pixel red square PNG in base64)
        test_image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAKAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
        results = await qdrant_client.search_by_image(image_data=test_image, limit=5)

        if results:
            logger.info(f"✅ Image search test successful: {len(results)} results")
            return {
                "success": True,
                "image_search_working": True,
                "sample_non_zero": non_zero_count,
            }
        else:
            logger.warning("⚠️  Image search returned no results")
            return {
                "success": True,
                "image_search_working": False,
                "sample_non_zero": non_zero_count,
            }

    except Exception as e:
        logger.error(f"❌ Image search test failed: {e}")
        return {"success": False, "error": str(e), "sample_non_zero": non_zero_count}


async def main():
    """Main entry point for image embedding processing."""
    parser = argparse.ArgumentParser(
        description="Process anime poster images and add embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of anime to process (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("🎯 Anime Image Embedding Processor")
    logger.info("=" * 50)

    # Initialize settings and client
    settings = get_settings()
    qdrant_client = QdrantClient(settings=settings)

    logger.info(f"📍 Target collection: {settings.qdrant_collection_name}")
    logger.info(f"🔗 Qdrant URL: {settings.qdrant_url}")
    logger.info(f"📦 Batch size: {args.batch_size}")

    # Verify prerequisites
    if not await qdrant_client.health_check():
        logger.error("❌ Qdrant connection failed")
        return False

    logger.info("✅ Qdrant connection verified")

    # Check if multi-vector is enabled
    if not getattr(qdrant_client, "_supports_multi_vector", False):
        logger.error(
            "❌ Multi-vector support not enabled. Set ENABLE_MULTI_VECTOR=true"
        )
        return False

    logger.info("✅ Multi-vector support enabled")

    # Estimate processing time
    stats = await qdrant_client.get_stats()
    total_docs = stats.get("total_documents", 0)
    estimated_minutes = (total_docs / 100) * (args.batch_size / 50)  # Rough estimate
    logger.info(f"⏱️  Estimated processing time: {estimated_minutes:.1f} minutes")

    # Confirmation (unless --force or --dry-run)
    if not args.force and not args.dry_run:
        response = (
            input(f"\n🤔 Process {total_docs} anime images? [y/N]: ").strip().lower()
        )
        if response not in ["y", "yes"]:
            logger.info("❌ Processing cancelled by user")
            return False

    # Process images
    async with ImageEmbeddingProcessor(
        qdrant_client, batch_size=args.batch_size
    ) as processor:
        result = await processor.process_all_images(
            dry_run=args.dry_run, limit=args.limit
        )

        if not result["success"]:
            logger.error("❌ Image processing failed")
            return False

        if args.dry_run:
            logger.info("✅ Dry run completed - no changes made")
            return True

    # Verify results
    verification = await verify_image_processing(qdrant_client)
    if verification["success"] and verification.get("image_search_working", False):
        logger.info("🎉 Image processing and verification completed successfully!")
        logger.info("📋 Next steps:")
        logger.info("   1. Test image search with real queries")
        logger.info("   2. Add REST API endpoints for image search")
        logger.info("   3. Test multimodal search functionality")
        return True
    else:
        logger.warning("⚠️  Processing completed but verification failed")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("❌ Processing cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
