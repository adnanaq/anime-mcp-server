# src/services/update_service.py - Automated Update Service
import asyncio
import json
import logging
import hashlib
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
from pathlib import Path
from dotenv import load_dotenv

from .data_service import AnimeDataService
from ..vector.qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class UpdateService:
    """Handles automated updates of anime database"""
    
    def __init__(self, qdrant_client=None):
        self.data_service = AnimeDataService()
        # Use provided client or create new one from environment
        if qdrant_client:
            self.qdrant_client = qdrant_client
            logger.info(f"🔧 UpdateService using provided QdrantClient")
        else:
            # Use Docker service name when running in container
            qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
            collection_name = os.getenv("QDRANT_COLLECTION_NAME", "anime_database")
            logger.info(f"🔧 UpdateService creating new QdrantClient with URL: {qdrant_url}")
            self.qdrant_client = QdrantClient(url=qdrant_url, collection_name=collection_name)
        self.data_dir = Path("data")
        self.metadata_file = self.data_dir / "update_metadata.json"
        
    async def check_for_updates(self) -> bool:
        """Check if anime-offline-database has updates"""
        try:
            # Download latest data
            logger.info("🔍 Checking for database updates...")
            latest_data = await self.data_service.download_anime_database()
            
            # Calculate content hash
            content_str = json.dumps(latest_data, sort_keys=True)
            latest_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            # Load previous metadata
            metadata = self.load_metadata()
            previous_hash = metadata.get('content_hash')
            
            if latest_hash != previous_hash:
                logger.info(f"📈 Update detected! Hash changed: {previous_hash[:8]} → {latest_hash[:8]}")
                
                # Save new data
                raw_file = self.data_dir / "raw" / "anime-offline-database-latest.json"
                raw_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(raw_file, 'w', encoding='utf-8') as f:
                    json.dump(latest_data, f, ensure_ascii=False, indent=2)
                
                # Update metadata
                metadata.update({
                    'content_hash': latest_hash,
                    'last_check': datetime.utcnow().isoformat(),
                    'last_update': datetime.utcnow().isoformat(),
                    'entry_count': len(latest_data.get('data', [])),
                    'update_available': True
                })
                self.save_metadata(metadata)
                return True
            else:
                logger.info("✅ No updates available")
                metadata['last_check'] = datetime.utcnow().isoformat()
                self.save_metadata(metadata)
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking for updates: {e}")
            return False
    
    async def perform_incremental_update(self) -> bool:
        """Perform incremental update by comparing old vs new data"""
        try:
            logger.info("🔄 Starting incremental update...")
            
            # Load current and new data
            current_data = self.load_current_data()
            new_data = self.load_latest_data()
            
            if not current_data or not new_data:
                logger.warning("⚠️ Data files missing, performing full update")
                return await self.perform_full_update()
            
            # Find differences
            changes = self.compare_datasets(current_data, new_data)
            
            if not any(changes.values()):
                logger.info("✅ No changes detected")
                return True
            
            logger.info(f"📊 Changes detected: {len(changes['added'])} added, "
                       f"{len(changes['modified'])} modified, {len(changes['removed'])} removed")
            
            # Process only changed entries
            updated_entries = []
            
            # Process new and modified entries
            for anime_data in changes['added'] + changes['modified']:
                processed = self.data_service.process_anime_entry(anime_data)
                if processed:
                    updated_entries.append(processed)
            
            # Remove old entries from vector DB
            if changes['removed']:
                await self.remove_entries(changes['removed'])
            
            # Add/update entries in vector DB
            if updated_entries:
                tensor_fields = ["embedding_text", "search_text"]
                await self.qdrant_client.add_documents(updated_entries, tensor_fields)
                logger.info(f"✅ Updated {len(updated_entries)} entries in vector database")
            
            # Update processed data file
            await self.update_processed_data(new_data)
            
            # Update metadata
            metadata = self.load_metadata()
            metadata.update({
                'last_incremental_update': datetime.utcnow().isoformat(),
                'update_available': False,
                'last_changes': {
                    'added': len(changes['added']),
                    'modified': len(changes['modified']),
                    'removed': len(changes['removed'])
                }
            })
            self.save_metadata(metadata)
            
            logger.info("🎉 Incremental update completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Incremental update failed: {e}")
            return False
    
    async def perform_full_update(self) -> bool:
        """Perform full database refresh"""
        try:
            logger.info("🔄 Starting full database update...")
            
            # Download fresh data
            raw_data = await self.data_service.download_anime_database()
            
            # Process all data
            processed_data = await self.data_service.process_all_anime(raw_data)
            
            # Clear existing index
            await self.qdrant_client.clear_index()
            
            # Re-index everything
            tensor_fields = ["embedding_text", "search_text"]
            await self.qdrant_client.add_documents(processed_data, tensor_fields)
            
            # Save processed data
            processed_file = self.data_dir / "processed" / "anime-vectors.json"
            processed_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            # Update metadata
            content_str = json.dumps(raw_data, sort_keys=True)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            metadata = {
                'content_hash': content_hash,
                'last_full_update': datetime.utcnow().isoformat(),
                'last_check': datetime.utcnow().isoformat(),
                'entry_count': len(processed_data),
                'update_available': False
            }
            self.save_metadata(metadata)
            
            logger.info(f"🎉 Full update completed! Indexed {len(processed_data)} entries")
            return True
            
        except Exception as e:
            logger.error(f"❌ Full update failed: {e}")
            return False
    
    async def schedule_weekly_update(self):
        """Run weekly update check (for cron job)"""
        logger.info("⏰ Starting scheduled weekly update check...")
        
        has_updates = await self.check_for_updates()
        
        if has_updates:
            success = await self.perform_incremental_update()
            if success:
                logger.info("✅ Weekly update completed successfully!")
            else:
                logger.error("❌ Weekly update failed")
        else:
            logger.info("✅ No updates needed this week")
    
    def compare_datasets(self, old_data: Dict, new_data: Dict) -> Dict[str, List]:
        """Compare two datasets and return differences"""
        old_entries = {self._get_entry_key(entry): entry for entry in old_data.get('data', [])}
        new_entries = {self._get_entry_key(entry): entry for entry in new_data.get('data', [])}
        
        old_keys = set(old_entries.keys())
        new_keys = set(new_entries.keys())
        
        added = [new_entries[key] for key in new_keys - old_keys]
        removed = [old_entries[key] for key in old_keys - new_keys]
        
        # Check for modifications in common entries
        modified = []
        for key in old_keys & new_keys:
            if self._entry_hash(old_entries[key]) != self._entry_hash(new_entries[key]):
                modified.append(new_entries[key])
        
        return {
            'added': added,
            'modified': modified,
            'removed': removed
        }
    
    def _get_entry_key(self, entry: Dict) -> str:
        """Generate unique key for anime entry"""
        return f"{entry.get('title', '')}_{entry.get('sources', [''])[0]}"
    
    def _entry_hash(self, entry: Dict) -> str:
        """Generate hash for anime entry content"""
        # Use key fields that might change
        key_fields = ['title', 'synopsis', 'episodes', 'status', 'tags', 'studios']
        content = {k: entry.get(k) for k in key_fields}
        return hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()
    
    async def remove_entries(self, entries: List[Dict]):
        """Remove entries from vector database"""
        # TODO: Implement entry removal from Qdrant
        # This would require tracking anime_ids and using delete operations
        logger.info(f"🗑️ Would remove {len(entries)} entries (not yet implemented)")
    
    def load_metadata(self) -> Dict:
        """Load update metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self, metadata: Dict):
        """Save update metadata"""
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load_current_data(self) -> Dict:
        """Load current anime data"""
        current_file = self.data_dir / "raw" / "anime-offline-database.json"
        if current_file.exists():
            with open(current_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def load_latest_data(self) -> Dict:
        """Load latest downloaded data"""
        latest_file = self.data_dir / "raw" / "anime-offline-database-latest.json"
        if latest_file.exists():
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    async def update_processed_data(self, new_data: Dict):
        """Update processed data with new dataset"""
        processed_data = await self.data_service.process_all_anime(new_data)
        
        processed_file = self.data_dir / "processed" / "anime-vectors.json"
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)