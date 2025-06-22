#!/usr/bin/env python3
"""
Download and process anime-offline-database for vector indexing
"""
import asyncio
import sys
import os
sys.path.append('src')

from src.services.data_service import AnimeDataService
from src.vector.qdrant_client import QdrantClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Download anime data and index in Qdrant"""
    
    # Initialize services
    data_service = AnimeDataService()
    qdrant_client = QdrantClient()
    
    # Create data directory
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    try:
        # Step 1: Download anime database
        logger.info("🚀 Starting anime database download...")
        raw_data = await data_service.download_anime_database()
        
        # Save raw data
        import json
        with open("data/raw/anime-offline-database.json", "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
        logger.info("💾 Saved raw database to data/raw/")
        
        # Step 2: Process data for vector indexing
        logger.info("🔄 Processing anime data for vector indexing...")
        processed_data = await data_service.process_all_anime(raw_data)
        
        # Save processed data
        with open("data/processed/anime-vectors.json", "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        logger.info("💾 Saved processed data to data/processed/")
        
        # Step 3: Index in Qdrant
        logger.info("📚 Indexing data in Qdrant vector database...")
        
        # Ensure collection exists (already done in constructor)
        # await qdrant_client.create_collection()  # Already handled in __init__
        
        # Index documents (tensor_fields not needed for Qdrant FastEmbed)
        success = await qdrant_client.add_documents(processed_data)
        
        if success:
            logger.info("✅ Successfully indexed all anime data!")
            
            # Get stats
            stats = await qdrant_client.get_stats()
            logger.info(f"📊 Database stats: {stats['total_anime']} anime entries indexed")
            
        else:
            logger.error("❌ Failed to index anime data")
            
    except Exception as e:
        logger.error(f"❌ Error during data processing: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())