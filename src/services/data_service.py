# src/services/data_service.py - Anime Data Processing Service
import aiohttp
import asyncio
import json
import logging
import time
from typing import List, Dict, Any
from ..models.anime import AnimeEntry
import hashlib
import re

logger = logging.getLogger(__name__)

class AnimeDataService:
    """Service for downloading and processing anime data"""
    
    def __init__(self):
        self.anime_db_url = "https://raw.githubusercontent.com/manami-project/anime-offline-database/master/anime-offline-database-minified.json"
        
        # Platform configurations for ID extraction
        self.platform_configs = {
            # Numeric ID platforms with /anime/ path
            'myanimelist': {
                'domain': 'myanimelist.net',
                'pattern': re.compile(r'/anime/(\d+)'),
                'id_type': 'numeric'
            },
            'anilist': {
                'domain': 'anilist.co', 
                'pattern': re.compile(r'/anime/(\d+)'),
                'id_type': 'numeric'
            },
            'kitsu': {
                'domain': 'kitsu.app',  # Fixed: database uses kitsu.app not kitsu.io
                'pattern': re.compile(r'/anime/(\d+)'),
                'id_type': 'numeric'
            },
            'anidb': {
                'domain': 'anidb.net',
                'pattern': re.compile(r'/anime/(\d+)'),
                'id_type': 'numeric'
            },
            'anisearch': {
                'domain': 'anisearch.com',
                'pattern': re.compile(r'/anime/(\d+)'),
                'id_type': 'numeric'
            },
            'simkl': {
                'domain': 'simkl.com',
                'pattern': re.compile(r'/anime/(\d+)'),
                'id_type': 'numeric'
            },
            'livechart': {
                'domain': 'livechart.me',
                'pattern': re.compile(r'/anime/(\d+)'),
                'id_type': 'numeric'
            },
            # Query parameter platforms
            'animenewsnetwork': {
                'domain': 'animenewsnetwork.com',
                'pattern': re.compile(r'id=(\d+)'),
                'id_type': 'numeric'
            },
            # Special patterns
            'animeplanet': {
                'domain': 'anime-planet.com',
                'pattern': re.compile(r'/anime/([^/?]+)'),
                'id_type': 'slug'
            },
            'notify': {
                'domain': 'notify.moe',
                'pattern': re.compile(r'/anime/([A-Za-z0-9_-]+)'),
                'id_type': 'alphanumeric'
            },
            'animecountdown': {
                'domain': 'animecountdown.com',
                'pattern': re.compile(r'/(\d+)'),
                'id_type': 'numeric'
            }
        }
    
    async def download_anime_database(self) -> Dict[str, Any]:
        """Download the latest anime offline database"""
        try:
            logger.info("📥 Downloading anime-offline-database...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.anime_db_url) as response:
                    if response.status == 200:
                        # Get text first, then parse as JSON to handle content-type issues
                        text_content = await response.text()
                        data = json.loads(text_content)
                        logger.info(f"✅ Downloaded {len(data.get('data', []))} anime entries")
                        return data
                    else:
                        raise Exception(f"Failed to download database: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"❌ Failed to download anime database: {e}")
            raise
    
    def process_anime_entry(self, raw_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw anime entry for vector database"""
        try:
            # Create AnimeEntry from raw data
            anime = AnimeEntry(**raw_entry)
            
            # Generate unique anime_id
            anime_id = self._generate_anime_id(anime.title, anime.sources)
            
            # Extract year and season
            year, season = self._extract_year_season(anime.animeSeason)
            
            # Create embedding text for vector search
            embedding_text = self._create_embedding_text(anime)
            
            # Create search text for indexing
            search_text = self._create_search_text(anime)
            
            # Extract IDs from all available sources
            platform_ids = self._extract_all_platform_ids(anime.sources)
            
            # Create processed document for Qdrant
            processed_doc = {
                "anime_id": anime_id,
                "title": anime.title,
                "synopsis": anime.synopsis or "",
                "type": anime.type,
                "episodes": anime.episodes,
                "status": anime.status,
                "tags": anime.tags,
                "studios": anime.studios,
                "producers": anime.producers,
                "synonyms": anime.synonyms,
                "picture": anime.picture,
                "thumbnail": anime.thumbnail,
                "year": year,
                "season": season,
                "sources": anime.sources,
                
                # Platform IDs for cross-referencing
                **platform_ids,
                
                # Text fields for vector embedding
                "embedding_text": embedding_text,
                "search_text": search_text,
                
                # Metadata
                "data_quality_score": self._calculate_quality_score(anime)
            }
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"❌ Failed to process anime entry: {e}")
            return None
    
    async def process_all_anime(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process all anime entries with optimized async processing"""
        anime_list = raw_data.get("data", [])
        
        logger.info(f"🔄 Processing {len(anime_list)} anime entries...")
        start_time = time.time()
        
        # Process in larger batches with async processing
        batch_size = 2000  # Increased batch size
        max_concurrent_workers = 8  # Process multiple entries concurrently
        semaphore = asyncio.Semaphore(max_concurrent_workers)
        
        async def process_entry_async(entry):
            """Process single entry asynchronously"""
            async with semaphore:
                return await asyncio.to_thread(self.process_anime_entry, entry)
        
        async def process_batch_async(batch, batch_num):
            """Process a batch of entries concurrently"""
            batch_start = time.time()
            
            # Create tasks for concurrent processing
            tasks = [process_entry_async(entry) for entry in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            batch_processed = []
            error_count = 0
            for result in results:
                if isinstance(result, dict):
                    batch_processed.append(result)
                elif result is not None:  # Exception occurred
                    error_count += 1
            
            batch_duration = time.time() - batch_start
            entries_per_second = len(batch) / batch_duration if batch_duration > 0 else 0
            
            logger.info(f"📝 Batch {batch_num}: {len(batch_processed)}/{len(batch)} entries ({entries_per_second:.1f} entries/s)")
            if error_count > 0:
                logger.warning(f"⚠️ Batch {batch_num} had {error_count} processing errors")
            
            return batch_processed
        
        # Process all batches
        all_processed = []
        batch_tasks = []
        
        for i in range(0, len(anime_list), batch_size):
            batch = anime_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            task = process_batch_async(batch, batch_num)
            batch_tasks.append(task)
        
        # Execute all batch tasks concurrently
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results
        for batch_result in batch_results:
            all_processed.extend(batch_result)
        
        end_time = time.time()
        duration = end_time - start_time
        total_entries_per_second = len(all_processed) / duration if duration > 0 else 0
        
        logger.info(f"✅ Processed {len(all_processed)}/{len(anime_list)} entries in {duration:.2f}s ({total_entries_per_second:.1f} entries/s)")
        return all_processed
    
    def _generate_anime_id(self, title: str, sources: List[str]) -> str:
        """Generate unique anime ID"""
        # Use title + first source for uniqueness
        unique_str = f"{title}_{sources[0] if sources else ''}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:12]
    
    def _extract_year_season(self, anime_season: Dict[str, Any]) -> tuple:
        """Extract year and season from animeSeason"""
        if not anime_season:
            return None, None
        
        year = anime_season.get("year")
        season = anime_season.get("season", "").lower() if anime_season.get("season") else None
        
        return year, season
    
    def _create_embedding_text(self, anime: AnimeEntry) -> str:
        """Create rich text for vector embedding"""
        text_parts = [
            anime.title,
            anime.synopsis or "",
            " ".join(anime.synonyms),
            " ".join(anime.tags),
            " ".join(anime.studios),
            anime.type
        ]
        
        return " ".join(filter(None, text_parts))
    
    def _create_search_text(self, anime: AnimeEntry) -> str:
        """Create text optimized for search indexing"""
        return f"{anime.title} {' '.join(anime.synonyms)} {' '.join(anime.tags)}"
    
    def _extract_all_platform_ids(self, sources: List[str]) -> Dict[str, Any]:
        """Extract IDs from all available platforms"""
        platform_ids = {}
        
        for source in sources:
            for platform_name, config in self.platform_configs.items():
                if config['domain'] in source:
                    match = config['pattern'].search(source)
                    if match:
                        extracted_id = match.group(1)
                        
                        # Convert to appropriate type
                        if config['id_type'] == 'numeric':
                            try:
                                extracted_id = int(extracted_id)
                            except ValueError:
                                continue
                        
                        # Store with platform-specific key
                        platform_ids[f"{platform_name}_id"] = extracted_id
                        break  # Found match for this source, move to next
        
        return platform_ids
    
    def _calculate_quality_score(self, anime: AnimeEntry) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.0
        
        # Title and basic info
        if anime.title: score += 0.2
        if anime.type: score += 0.1
        if anime.episodes > 0: score += 0.1
        
        # Rich metadata
        if anime.synopsis: score += 0.2
        if anime.tags: score += 0.15
        if anime.studios: score += 0.1
        if anime.picture: score += 0.05
        
        # Sources and cross-references
        if len(anime.sources) >= 2: score += 0.1
        
        return min(score, 1.0)