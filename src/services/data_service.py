# src/services/data_service.py - Anime Data Processing Service
import asyncio
import base64
import hashlib
import io
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

import aiohttp
from PIL import Image

from ..config import Settings, get_settings
from ..exceptions import (
    DataProcessingError,
    handle_exception_safely,
)
from ..models.anime import AnimeEntry

logger = logging.getLogger(__name__)


class ProcessingConfig:
    """Configuration for anime data processing operations."""

    def __init__(
        self, batch_size: int, max_concurrent_batches: int, processing_timeout: int
    ):
        """Initialize processing configuration.

        Args:
            batch_size: Number of entries per batch
            max_concurrent_batches: Maximum concurrent batch operations
            processing_timeout: Timeout for processing operations in seconds
        """
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.processing_timeout = processing_timeout


class AnimeDataService:
    """Service for downloading and processing anime data with centralized configuration."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize data service with configuration.

        Args:
            settings: Optional settings instance. If not provided, uses default settings.
        """
        self.settings = settings or get_settings()
        self.anime_db_url = self.settings.anime_database_url
        self._http_session = None

        # Platform configurations for ID extraction
        self.platform_configs = {
            # Numeric ID platforms with /anime/ path
            "myanimelist": {
                "domain": "myanimelist.net",
                "pattern": re.compile(r"/anime/(\d+)"),
                "id_type": "numeric",
            },
            "anilist": {
                "domain": "anilist.co",
                "pattern": re.compile(r"/anime/(\d+)"),
                "id_type": "numeric",
            },
            "kitsu": {
                "domain": "kitsu.app",  # Fixed: database uses kitsu.app not kitsu.io
                "pattern": re.compile(r"/anime/(\d+)"),
                "id_type": "numeric",
            },
            "anidb": {
                "domain": "anidb.net",
                "pattern": re.compile(r"/anime/(\d+)"),
                "id_type": "numeric",
            },
            "anisearch": {
                "domain": "anisearch.com",
                "pattern": re.compile(r"/anime/(\d+)"),
                "id_type": "numeric",
            },
            "simkl": {
                "domain": "simkl.com",
                "pattern": re.compile(r"/anime/(\d+)"),
                "id_type": "numeric",
            },
            "livechart": {
                "domain": "livechart.me",
                "pattern": re.compile(r"/anime/(\d+)"),
                "id_type": "numeric",
            },
            # Query parameter platforms
            "animenewsnetwork": {
                "domain": "animenewsnetwork.com",
                "pattern": re.compile(r"id=(\d+)"),
                "id_type": "numeric",
            },
            # Special patterns
            "animeplanet": {
                "domain": "anime-planet.com",
                "pattern": re.compile(r"/anime/([^/?]+)"),
                "id_type": "slug",
            },
            "notify": {
                "domain": "notify.moe",
                "pattern": re.compile(r"/anime/([A-Za-z0-9_-]+)"),
                "id_type": "alphanumeric",
            },
            "animecountdown": {
                "domain": "animecountdown.com",
                "pattern": re.compile(r"/(\d+)"),
                "id_type": "numeric",
            },
        }

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session for image downloads."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "Anime-MCP-Server/1.0"},
            )
        return self._http_session

    async def _close_http_session(self):
        """Close HTTP session if open."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None

    async def _download_image(self, url: str) -> Optional[str]:
        """Download image from URL and return base64 encoded data.

        Args:
            url: Image URL to download

        Returns:
            Base64 encoded image data or None if failed
        """
        if not url:
            return None

        try:
            session = await self._get_http_session()
            async with session.get(url) as response:
                if response.status == 200:
                    content_type = response.headers.get("content-type", "")
                    if "image" in content_type.lower():
                        image_data = await response.read()

                        # Verify it's a valid image
                        try:
                            Image.open(io.BytesIO(image_data)).verify()
                            return base64.b64encode(image_data).decode("utf-8")
                        except Exception:
                            logger.debug(f"Invalid image data from {url}")
                            return None
                    else:
                        logger.debug(
                            f"Non-image content type: {content_type} for {url}"
                        )
                        return None
                else:
                    logger.debug(f"HTTP {response.status} for {url}")
                    return None
        except asyncio.TimeoutError:
            logger.debug(f"Timeout downloading {url}")
            return None
        except Exception as e:
            logger.debug(f"Error downloading {url}: {e}")
            return None

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
                        logger.info(
                            f"✅ Downloaded {len(data.get('data', []))} anime entries"
                        )
                        return data
                    else:
                        raise Exception(
                            f"Failed to download database: HTTP {response.status}"
                        )

        except Exception as e:
            logger.error(f"❌ Failed to download anime database: {e}")
            raise

    async def process_anime_entry(self, raw_entry: Dict[str, Any]) -> Dict[str, Any]:
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

            # Download images for embedding processing
            picture_data = (
                await self._download_image(anime.picture) if anime.picture else None
            )
            thumbnail_data = (
                await self._download_image(anime.thumbnail) if anime.thumbnail else None
            )

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
                "relatedAnime": anime.relatedAnime,
                "picture": anime.picture,
                "thumbnail": anime.thumbnail,
                "year": year,
                "season": season,
                "sources": anime.sources,
                # Score data from offline database
                "score": anime.score,
                # Platform IDs for cross-referencing
                **platform_ids,
                # Text fields for vector embedding
                "embedding_text": embedding_text,
                "search_text": search_text,
                # Image data for multi-vector embedding - both picture and thumbnail
                "picture_data": picture_data,
                "thumbnail_data": thumbnail_data,
                # Metadata
                "data_quality_score": self._calculate_quality_score(anime),
            }

            return processed_doc

        except Exception as e:
            logger.error(f"❌ Failed to process anime entry: {e}")
            return None

    @handle_exception_safely
    async def process_all_anime(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process all anime entries with optimized async processing.

        Args:
            raw_data: Raw anime database data containing 'data' list

        Returns:
            List of processed anime entries

        Raises:
            DataProcessingError: When processing fails
        """
        anime_list = raw_data.get("data", [])
        if not anime_list:
            raise DataProcessingError("No anime data found in raw_data")

        logger.info(f"🔄 Processing {len(anime_list)} anime entries...")
        start_time = time.time()

        try:
            # Create processing configuration
            processing_config = self._create_processing_config()

            # Create processing batches
            batches = self._create_batches(anime_list, processing_config.batch_size)

            # Process all batches concurrently
            batch_results = await self._process_batches_concurrently(
                batches, processing_config
            )

            # Aggregate and analyze results
            all_processed = self._aggregate_results(batch_results)

            # Log final metrics
            self._log_processing_metrics(
                start_time, len(anime_list), len(all_processed)
            )

            return all_processed

        except Exception as e:
            logger.error(f"Failed to process anime data: {e}")
            raise DataProcessingError(f"Anime processing failed: {str(e)}")
        finally:
            # Ensure HTTP session is closed
            await self._close_http_session()

    def _create_processing_config(self) -> "ProcessingConfig":
        """Create processing configuration from settings."""
        return ProcessingConfig(
            batch_size=self.settings.batch_size,
            max_concurrent_batches=self.settings.max_concurrent_batches,
            processing_timeout=self.settings.processing_timeout,
        )

    def _create_batches(
        self, anime_list: List[Dict[str, Any]], batch_size: int
    ) -> List[List[Dict[str, Any]]]:
        """Split anime list into processing batches.

        Args:
            anime_list: List of anime entries to process
            batch_size: Size of each batch

        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(anime_list), batch_size):
            batch = anime_list[i : i + batch_size]
            batches.append(batch)
        return batches

    async def _process_batches_concurrently(
        self, batches: List[List[Dict[str, Any]]], config: "ProcessingConfig"
    ) -> List[List[Dict[str, Any]]]:
        """Process multiple batches concurrently.

        Args:
            batches: List of anime entry batches
            config: Processing configuration

        Returns:
            List of processed batch results
        """
        # Create semaphore for concurrent batch processing
        semaphore = asyncio.Semaphore(config.max_concurrent_batches)

        async def process_single_batch(
            batch: List[Dict[str, Any]], batch_num: int
        ) -> List[Dict[str, Any]]:
            """Process a single batch with concurrency control."""
            async with semaphore:
                return await self._process_batch(batch, batch_num, config)

        # Create tasks for all batches
        batch_tasks = [
            process_single_batch(batch, batch_num + 1)
            for batch_num, batch in enumerate(batches)
        ]

        # Execute all batch tasks with timeout
        try:
            batch_results = await asyncio.wait_for(
                asyncio.gather(*batch_tasks, return_exceptions=True),
                timeout=config.processing_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Batch processing timed out after {config.processing_timeout}s"
            )
            raise DataProcessingError(
                f"Processing timed out after {config.processing_timeout} seconds"
            )

        # Handle any batch processing exceptions
        successful_results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i+1} failed: {result}")
                # Continue with other batches rather than failing completely
                successful_results.append([])
            else:
                successful_results.append(result)

        return successful_results

    async def _process_batch(
        self, batch: List[Dict[str, Any]], batch_num: int, config: "ProcessingConfig"
    ) -> List[Dict[str, Any]]:
        """Process a single batch of anime entries.

        Args:
            batch: Batch of anime entries to process
            batch_num: Batch number for logging
            config: Processing configuration

        Returns:
            List of successfully processed anime entries
        """
        batch_start = time.time()

        # Create tasks for concurrent entry processing within batch
        entry_semaphore = asyncio.Semaphore(
            min(8, len(batch))
        )  # Limit concurrent entries per batch

        async def process_entry_async(
            entry: Dict[str, Any],
        ) -> Optional[Dict[str, Any]]:
            """Process single entry asynchronously with error handling."""
            async with entry_semaphore:
                try:
                    # Process entry directly (now async)
                    return await self.process_anime_entry(entry)
                except Exception as e:
                    logger.debug(
                        f"Failed to process entry {entry.get('title', 'unknown')}: {e}"
                    )
                    return None

        # Process all entries in the batch concurrently
        tasks = [process_entry_async(entry) for entry in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results and count errors
        batch_processed = []
        error_count = 0

        for result in results:
            if isinstance(result, dict):
                batch_processed.append(result)
            elif isinstance(result, Exception):
                error_count += 1
            elif result is None:
                error_count += 1

        # Log batch metrics
        self._log_batch_metrics(
            batch_num, batch_start, len(batch), len(batch_processed), error_count
        )

        return batch_processed

    def _aggregate_results(
        self, batch_results: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Aggregate results from all processed batches.

        Args:
            batch_results: List of batch processing results

        Returns:
            Flattened list of all processed anime entries
        """
        all_processed = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                all_processed.extend(batch_result)
        return all_processed

    def _log_batch_metrics(
        self,
        batch_num: int,
        batch_start: float,
        total_entries: int,
        processed_entries: int,
        error_count: int,
    ) -> None:
        """Log metrics for a completed batch.

        Args:
            batch_num: Batch number
            batch_start: Batch start time
            total_entries: Total entries in batch
            processed_entries: Successfully processed entries
            error_count: Number of processing errors
        """
        batch_duration = time.time() - batch_start
        entries_per_second = total_entries / batch_duration if batch_duration > 0 else 0

        logger.info(
            f"📝 Batch {batch_num}: {processed_entries}/{total_entries} entries "
            f"({entries_per_second:.1f} entries/s)"
        )

        if error_count > 0:
            error_rate = (error_count / total_entries) * 100
            logger.warning(
                f"⚠️ Batch {batch_num} had {error_count} errors ({error_rate:.1f}% error rate)"
            )

    def _log_processing_metrics(
        self, start_time: float, total_entries: int, processed_entries: int
    ) -> None:
        """Log final processing metrics.

        Args:
            start_time: Processing start time
            total_entries: Total entries to process
            processed_entries: Successfully processed entries
        """
        duration = time.time() - start_time
        total_entries_per_second = processed_entries / duration if duration > 0 else 0
        success_rate = (
            (processed_entries / total_entries) * 100 if total_entries > 0 else 0
        )

        logger.info(
            f"✅ Processed {processed_entries}/{total_entries} entries "
            f"({success_rate:.1f}% success rate) in {duration:.2f}s "
            f"({total_entries_per_second:.1f} entries/s)"
        )

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
        season = (
            anime_season.get("season", "").lower()
            if anime_season.get("season")
            else None
        )

        return year, season

    def _create_embedding_text(self, anime: AnimeEntry) -> str:
        """Create rich text for vector embedding"""
        text_parts = [
            anime.title,
            anime.synopsis or "",
            " ".join(anime.synonyms),
            " ".join(anime.tags),
            " ".join(anime.studios),
            anime.type,
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
                if config["domain"] in source:
                    match = config["pattern"].search(source)
                    if match:
                        extracted_id = match.group(1)

                        # Convert to appropriate type
                        if config["id_type"] == "numeric":
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
        if anime.title:
            score += 0.2
        if anime.type:
            score += 0.1
        if anime.episodes > 0:
            score += 0.1

        # Rich metadata
        if anime.synopsis:
            score += 0.2
        if anime.tags:
            score += 0.15
        if anime.studios:
            score += 0.1
        if anime.picture:
            score += 0.05

        # Sources and cross-references
        if len(anime.sources) >= 2:
            score += 0.1

        return min(score, 1.0)
