#!/usr/bin/env python3
"""
Production Migration Script: Single-Vector to Multi-Vector Collection

This script safely migrates the existing anime_database collection from
single-vector (text only) to multi-vector (text + image) configuration.

Safety Features:
- Creates backup before migration
- Preserves all existing data
- Rollback capability on failure
- Progress monitoring
- Zero downtime (collection recreated with same name)

Usage:
    python scripts/migrate_to_multivector.py [--dry-run] [--force]
"""

import asyncio
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_settings
from src.vector.qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("migration.log")
    ]
)
logger = logging.getLogger(__name__)


async def check_prerequisites(qdrant_client: QdrantClient) -> bool:
    """Check if migration can proceed safely."""
    logger.info("🔍 Checking migration prerequisites...")
    
    # Check Qdrant connection
    if not await qdrant_client.health_check():
        logger.error("❌ Qdrant connection failed")
        return False
    logger.info("✅ Qdrant connection healthy")
    
    # Check collection exists
    stats = await qdrant_client.get_stats()
    if stats.get("error"):
        logger.error(f"❌ Collection access failed: {stats.get('error')}")
        return False
    
    total_docs = stats.get("total_documents", 0)
    logger.info(f"✅ Collection accessible with {total_docs:,} documents")
    
    # Check collection is single-vector
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        collection_info = await loop.run_in_executor(
            None,
            lambda: qdrant_client.client.get_collection(qdrant_client.collection_name)
        )
        
        vectors_config = collection_info.config.params.vectors
        if isinstance(vectors_config, dict):
            logger.warning("⚠️  Collection already appears to be multi-vector")
            if "text" in vectors_config and "image" in vectors_config:
                logger.info("✅ Collection already migrated to multi-vector")
                return False  # No migration needed
        
        logger.info("✅ Collection confirmed as single-vector (ready for migration)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to inspect collection: {e}")
        return False


async def estimate_migration_time(qdrant_client: QdrantClient) -> float:
    """Estimate migration duration."""
    stats = await qdrant_client.get_stats()
    document_count = stats.get("total_documents", 0)
    
    # Estimate: ~100 documents per second processing
    estimated_seconds = max(document_count / 100, 60)
    return estimated_seconds


async def run_migration(qdrant_client: QdrantClient, dry_run: bool = False) -> bool:
    """Execute the migration process."""
    
    if dry_run:
        logger.info("🔧 DRY RUN MODE - No changes will be made")
        return True
    
    logger.info("🚀 Starting collection migration to multi-vector...")
    
    try:
        # Run the migration
        result = await qdrant_client.migrate_to_multi_vector()
        
        if result["migration_successful"]:
            logger.info(f"✅ Migration completed successfully!")
            logger.info(f"   📊 Preserved vectors: {result['preserved_vectors']:,}")
            logger.info(f"   💾 Backup collection: {result.get('backup_collection', 'N/A')}")
            
            # Verify post-migration
            stats = await qdrant_client.get_stats()
            final_count = stats.get("total_documents", 0)
            
            if final_count == result["preserved_vectors"]:
                logger.info(f"✅ Migration verification passed: {final_count:,} documents preserved")
                return True
            else:
                logger.error(f"❌ Migration verification failed: expected {result['preserved_vectors']}, got {final_count}")
                return False
                
        else:
            logger.error(f"❌ Migration failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Migration error: {e}")
        return False


async def verify_migration(qdrant_client: QdrantClient) -> bool:
    """Verify migration was successful."""
    logger.info("🔍 Verifying migration results...")
    
    try:
        # Check collection structure
        loop = asyncio.get_event_loop()
        collection_info = await loop.run_in_executor(
            None,
            lambda: qdrant_client.client.get_collection(qdrant_client.collection_name)
        )
        
        vectors_config = collection_info.config.params.vectors
        
        if isinstance(vectors_config, dict):
            if "text" in vectors_config and "image" in vectors_config:
                text_size = vectors_config["text"].size
                image_size = vectors_config["image"].size
                logger.info(f"✅ Multi-vector structure confirmed:")
                logger.info(f"   📝 Text vectors: {text_size} dimensions") 
                logger.info(f"   🖼️  Image vectors: {image_size} dimensions")
                
                # Test a sample search
                results = await qdrant_client.search("test search", limit=1)
                if results:
                    logger.info("✅ Text search functionality verified")
                    return True
                else:
                    logger.warning("⚠️  Text search returned no results (collection may be empty)")
                    return True
            else:
                logger.error("❌ Multi-vector structure missing required vectors")
                return False
        else:
            logger.error("❌ Collection still appears to be single-vector")
            return False
            
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False


async def main():
    """Main migration script entry point."""
    parser = argparse.ArgumentParser(description="Migrate anime collection to multi-vector")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🎯 Anime Collection Migration Tool")
    logger.info("=" * 50)
    
    # Initialize settings and client
    settings = get_settings()
    qdrant_client = QdrantClient(settings=settings)
    
    logger.info(f"📍 Target collection: {settings.qdrant_collection_name}")
    logger.info(f"🔗 Qdrant URL: {settings.qdrant_url}")
    
    # Step 1: Prerequisites check
    if not await check_prerequisites(qdrant_client):
        logger.error("❌ Prerequisites check failed - aborting migration")
        return False
    
    # Step 2: Estimate time
    estimated_time = await estimate_migration_time(qdrant_client)
    logger.info(f"⏱️  Estimated migration time: {estimated_time/60:.1f} minutes")
    
    # Step 3: Confirmation (unless --force)
    if not args.force and not args.dry_run:
        response = input("\n🤔 Proceed with migration? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            logger.info("❌ Migration cancelled by user")
            return False
    
    # Step 4: Run migration
    success = await run_migration(qdrant_client, dry_run=args.dry_run)
    
    if not success:
        logger.error("❌ Migration failed - check logs for details")
        return False
    
    if args.dry_run:
        logger.info("✅ Dry run completed - no changes made")
        return True
    
    # Step 5: Verify results
    if not await verify_migration(qdrant_client):
        logger.error("❌ Migration verification failed")
        return False
    
    logger.info("🎉 Migration completed successfully!")
    logger.info("📋 Next steps:")
    logger.info("   1. Run image processing: python scripts/add_image_embeddings.py")
    logger.info("   2. Test image search functionality")
    logger.info("   3. Add REST API endpoints for image search")
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("❌ Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)