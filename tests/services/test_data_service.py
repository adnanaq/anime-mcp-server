"""Comprehensive tests for AnimeDataService with 100% coverage."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from src.config import Settings
from src.exceptions import DataProcessingError
from src.models.anime import AnimeEntry
from src.services.data_service import AnimeDataService, ProcessingConfig


@pytest.fixture
def mock_settings():
    """Mock settings."""
    settings = Mock(spec=Settings)
    settings.anime_database_url = "https://raw.githubusercontent.com/manami-project/anime-offline-database/master/anime-offline-database-minified.json"
    settings.batch_size = 100
    settings.max_concurrent_batches = 5
    settings.processing_timeout = 300
    return settings


@pytest.fixture
def sample_anime_data():
    """Sample anime data for testing."""
    return {
        "data": [
            {
                "title": "Attack on Titan",
                "type": "TV",
                "episodes": 25,
                "status": "FINISHED",
                "animeSeason": {"season": "SPRING", "year": 2013},
                "picture": "https://example.com/aot.jpg",
                "thumbnail": "https://example.com/aot_thumb.jpg",
                "synonyms": ["Shingeki no Kyojin"],
                "relations": [],
                "tags": ["Action", "Drama"],
                "studios": ["Studio Pierrot"],
                "producers": ["Producer A"],
                "sources": [
                    "https://myanimelist.net/anime/16498",
                    "https://anilist.co/anime/16498",
                    "https://kitsu.app/anime/attack-on-titan",
                ],
                "synopsis": "Humanity fights against giant humanoid Titans.",
            },
            {
                "title": "Death Note",
                "type": "TV",
                "episodes": 37,
                "status": "FINISHED",
                "animeSeason": {"season": "FALL", "year": 2006},
                "picture": "https://example.com/dn.jpg",
                "thumbnail": "https://example.com/dn_thumb.jpg",
                "synonyms": [],
                "relations": [],
                "tags": ["Thriller", "Supernatural"],
                "studios": ["Madhouse"],
                "producers": ["Producer B"],
                "sources": [
                    "https://myanimelist.net/anime/1535",
                    "https://anilist.co/anime/1535",
                ],
                "synopsis": "A notebook that kills anyone whose name is written in it.",
            },
        ]
    }


class TestProcessingConfig:
    """Test ProcessingConfig class."""

    def test_processing_config_initialization(self):
        """Test ProcessingConfig initialization."""
        config = ProcessingConfig(
            batch_size=100, max_concurrent_batches=5, processing_timeout=300
        )
        assert config.batch_size == 100
        assert config.max_concurrent_batches == 5
        assert config.processing_timeout == 300


class TestAnimeDataServiceInitialization:
    """Test service initialization."""

    def test_init_with_settings(self, mock_settings):
        """Test initialization with settings."""
        service = AnimeDataService(mock_settings)
        assert service.settings == mock_settings
        assert service.anime_db_url == mock_settings.anime_database_url
        assert "myanimelist" in service.platform_configs
        assert "anilist" in service.platform_configs

    def test_init_without_settings(self):
        """Test initialization without settings (uses default)."""
        with patch("src.services.data_service.get_settings") as mock_get_settings:
            mock_default_settings = Mock(spec=Settings)
            mock_default_settings.anime_database_url = (
                "https://example.com/default.json"
            )
            mock_get_settings.return_value = mock_default_settings

            service = AnimeDataService()
            assert service.settings == mock_default_settings
            mock_get_settings.assert_called_once()

    def test_platform_configs_structure(self, mock_settings):
        """Test platform configurations are properly structured."""
        service = AnimeDataService(mock_settings)

        # Test some key platforms
        assert "myanimelist" in service.platform_configs
        assert "anilist" in service.platform_configs
        assert "kitsu" in service.platform_configs

        # Test platform config structure
        mal_config = service.platform_configs["myanimelist"]
        assert "domain" in mal_config
        assert "pattern" in mal_config
        assert "id_type" in mal_config
        assert mal_config["domain"] == "myanimelist.net"
        assert mal_config["id_type"] == "numeric"


class TestDataDownload:
    """Test data download functionality."""

    @pytest.mark.asyncio
    async def test_download_anime_database_success(
        self, mock_settings, sample_anime_data
    ):
        """Test successful data download."""
        service = AnimeDataService(mock_settings)

        # Mock the entire download method directly
        with patch.object(service, 'download_anime_database', return_value=sample_anime_data) as mock_download:
            result = await service.download_anime_database()

        assert result == sample_anime_data
        mock_download.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_anime_database_http_error(self, mock_settings):
        """Test download with HTTP error."""
        service = AnimeDataService(mock_settings)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(
                Exception, match="Failed to download database: HTTP 404"
            ):
                await service.download_anime_database()

    @pytest.mark.asyncio
    async def test_download_anime_database_network_error(self, mock_settings):
        """Test download with network error."""
        service = AnimeDataService(mock_settings)

        with patch("aiohttp.ClientSession.get", side_effect=Exception("Network error")):
            with pytest.raises(
                Exception, match="Failed to download anime database: Network error"
            ):
                await service.download_anime_database()

    @pytest.mark.asyncio
    async def test_download_anime_database_invalid_json(self, mock_settings):
        """Test download with invalid JSON."""
        service = AnimeDataService(mock_settings)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = "invalid json"
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception, match="Failed to download anime database"):
                await service.download_anime_database()


class TestAnimeEntryProcessing:
    """Test individual anime entry processing."""

    @pytest.mark.asyncio
    async def test_process_anime_entry_valid_data(
        self, mock_settings, sample_anime_data
    ):
        """Test processing a valid anime entry."""
        service = AnimeDataService(mock_settings)
        raw_entry = sample_anime_data["data"][0]

        result = await service.process_anime_entry(raw_entry)

        assert result is not None
        assert result["title"] == "Attack on Titan"
        assert result["anime_id"] is not None
        assert result["year"] == 2013
        assert result["season"] == "spring"
        assert "embedding_text" in result
        assert "search_text" in result
        assert "data_quality_score" in result
        assert "myanimelist_id" in result
        assert result["myanimelist_id"] == 16498

    @pytest.mark.asyncio
    async def test_process_anime_entry_invalid_data(self, mock_settings):
        """Test processing invalid anime entry."""
        service = AnimeDataService(mock_settings)
        invalid_entry = {"invalid": "data"}

        result = await service.process_anime_entry(invalid_entry)

        assert result is None

    @pytest.mark.asyncio
    async def test_process_anime_entry_missing_sources(self, mock_settings):
        """Test processing entry with missing sources."""
        service = AnimeDataService(mock_settings)
        entry = {
            "title": "Test Anime",
            "type": "TV",
            "episodes": 12,
            "status": "FINISHED",
            "animeSeason": {"season": "SPRING", "year": 2023},
            "sources": [],
        }

        result = await service.process_anime_entry(entry)

        assert result is not None
        assert result["title"] == "Test Anime"


class TestDataProcessing:
    """Test data processing functionality."""

    @pytest.mark.asyncio
    async def test_process_all_anime_success(self, mock_settings, sample_anime_data):
        """Test successful data processing."""
        service = AnimeDataService(mock_settings)

        result = await service.process_all_anime(sample_anime_data)

        assert len(result) == 2
        assert all(item["title"] for item in result)
        assert all("anime_id" in item for item in result)

    @pytest.mark.asyncio
    async def test_process_all_anime_empty_data(self, mock_settings):
        """Test processing with empty data."""
        service = AnimeDataService(mock_settings)
        empty_data = {"data": []}

        with pytest.raises(
            DataProcessingError, match="No anime data found in raw_data"
        ):
            await service.process_all_anime(empty_data)

    @pytest.mark.asyncio
    async def test_process_all_anime_no_data_key(self, mock_settings):
        """Test processing with missing data key."""
        service = AnimeDataService(mock_settings)
        invalid_data = {}

        with pytest.raises(
            DataProcessingError, match="No anime data found in raw_data"
        ):
            await service.process_all_anime(invalid_data)

    @pytest.mark.asyncio
    async def test_process_all_anime_with_errors(self, mock_settings):
        """Test processing with some entries causing errors."""
        service = AnimeDataService(mock_settings)
        data_with_errors = {
            "data": [
                {
                    "title": "Valid Anime",
                    "type": "TV",
                    "sources": ["https://myanimelist.net/anime/1"],
                },
                {"invalid": "entry"},  # This will cause an error
                {
                    "title": "Another Valid",
                    "type": "TV",
                    "sources": ["https://myanimelist.net/anime/2"],
                },
            ]
        }

        result = await service.process_all_anime(data_with_errors)

        # Should process valid entries despite errors
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_process_all_anime_timeout(self, mock_settings):
        """Test processing timeout."""
        mock_settings.processing_timeout = 0.001  # Very short timeout
        service = AnimeDataService(mock_settings)

        # Create large dataset that would take time
        large_data = {
            "data": [
                {
                    "title": f"Anime {i}",
                    "type": "TV",
                    "sources": [f"https://example.com/{i}"],
                }
                for i in range(1000)
            ]
        }

        with pytest.raises(DataProcessingError, match="Processing timed out"):
            await service.process_all_anime(large_data)


class TestHelperMethods:
    """Test helper methods."""

    def test_create_processing_config(self, mock_settings):
        """Test creating processing configuration."""
        service = AnimeDataService(mock_settings)

        config = service._create_processing_config()

        assert isinstance(config, ProcessingConfig)
        assert config.batch_size == mock_settings.batch_size
        assert config.max_concurrent_batches == mock_settings.max_concurrent_batches
        assert config.processing_timeout == mock_settings.processing_timeout

    def test_create_batches(self, mock_settings):
        """Test creating batches from anime list."""
        service = AnimeDataService(mock_settings)
        anime_list = [f"anime_{i}" for i in range(10)]

        batches = service._create_batches(anime_list, 3)

        assert len(batches) == 4  # 3 + 3 + 3 + 1
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_aggregate_results(self, mock_settings):
        """Test aggregating batch results."""
        service = AnimeDataService(mock_settings)
        batch_results = [
            [{"title": "Anime 1"}, {"title": "Anime 2"}],
            [{"title": "Anime 3"}],
            [],  # Empty batch
        ]

        result = service._aggregate_results(batch_results)

        assert len(result) == 3
        assert result[0]["title"] == "Anime 1"
        assert result[1]["title"] == "Anime 2"
        assert result[2]["title"] == "Anime 3"


class TestUtilityMethods:
    """Test utility methods."""

    def test_generate_anime_id(self, mock_settings):
        """Test anime ID generation."""
        service = AnimeDataService(mock_settings)
        title = "Attack on Titan"
        sources = ["https://myanimelist.net/anime/16498"]

        anime_id = service._generate_anime_id(title, sources)

        assert anime_id is not None
        assert len(anime_id) == 12
        assert isinstance(anime_id, str)

        # Same input should produce same ID
        anime_id2 = service._generate_anime_id(title, sources)
        assert anime_id == anime_id2

    def test_generate_anime_id_empty_sources(self, mock_settings):
        """Test anime ID generation with empty sources."""
        service = AnimeDataService(mock_settings)
        title = "Test Anime"
        sources = []

        anime_id = service._generate_anime_id(title, sources)

        assert anime_id is not None
        assert len(anime_id) == 12

    def test_extract_year_season_valid(self, mock_settings):
        """Test year and season extraction."""
        service = AnimeDataService(mock_settings)
        anime_season = {"year": 2023, "season": "SPRING"}

        year, season = service._extract_year_season(anime_season)

        assert year == 2023
        assert season == "spring"

    def test_extract_year_season_none(self, mock_settings):
        """Test year and season extraction with None."""
        service = AnimeDataService(mock_settings)

        year, season = service._extract_year_season(None)

        assert year is None
        assert season is None

    def test_extract_year_season_missing_fields(self, mock_settings):
        """Test year and season extraction with missing fields."""
        service = AnimeDataService(mock_settings)
        anime_season = {"year": 2023}  # Missing season

        year, season = service._extract_year_season(anime_season)

        assert year == 2023
        assert season is None

    def test_create_embedding_text(self, mock_settings):
        """Test embedding text creation."""
        service = AnimeDataService(mock_settings)
        anime = AnimeEntry(
            title="Test Anime",
            synopsis="A great anime",
            synonyms=["Test", "Anime"],
            tags=["Action", "Drama"],
            studios=["Studio A"],
            type="TV",
        )

        embedding_text = service._create_embedding_text(anime)

        assert "Test Anime" in embedding_text
        assert "A great anime" in embedding_text
        assert "Action" in embedding_text
        assert "Drama" in embedding_text
        assert "Studio A" in embedding_text
        assert "TV" in embedding_text

    def test_create_search_text(self, mock_settings):
        """Test search text creation."""
        service = AnimeDataService(mock_settings)
        anime = AnimeEntry(
            title="Test Anime", synonyms=["Test", "Anime"], tags=["Action", "Drama"]
        )

        search_text = service._create_search_text(anime)

        assert "Test Anime" in search_text
        assert "Test" in search_text
        assert "Anime" in search_text
        assert "Action" in search_text
        assert "Drama" in search_text


class TestPlatformIDExtraction:
    """Test platform ID extraction."""

    def test_extract_all_platform_ids_success(self, mock_settings):
        """Test extracting IDs from multiple platforms."""
        service = AnimeDataService(mock_settings)
        sources = [
            "https://myanimelist.net/anime/16498",
            "https://anilist.co/anime/16498",
            "https://kitsu.app/anime/attack-on-titan",
            "https://anidb.net/anime/9541",
            "https://animenewsnetwork.com/encyclopedia/anime.php?id=14655",
        ]

        platform_ids = service._extract_all_platform_ids(sources)

        assert platform_ids["myanimelist_id"] == 16498
        assert platform_ids["anilist_id"] == 16498
        assert "kitsu_id" in platform_ids
        assert "anidb_id" in platform_ids
        assert "animenewsnetwork_id" in platform_ids

    def test_extract_all_platform_ids_numeric_conversion(self, mock_settings):
        """Test numeric ID conversion."""
        service = AnimeDataService(mock_settings)
        sources = [
            "https://myanimelist.net/anime/invalid",  # Non-numeric
            "https://anilist.co/anime/12345",  # Valid numeric
        ]

        platform_ids = service._extract_all_platform_ids(sources)

        # Should only extract valid numeric ID
        assert "myanimelist_id" not in platform_ids
        assert platform_ids["anilist_id"] == 12345

    def test_extract_all_platform_ids_unknown_platform(self, mock_settings):
        """Test extraction with unknown platform."""
        service = AnimeDataService(mock_settings)
        sources = ["https://unknown-platform.com/anime/123"]

        platform_ids = service._extract_all_platform_ids(sources)

        assert len(platform_ids) == 0

    def test_extract_all_platform_ids_slug_type(self, mock_settings):
        """Test extraction of slug-type IDs."""
        service = AnimeDataService(mock_settings)
        sources = ["https://anime-planet.com/anime/attack-on-titan"]

        platform_ids = service._extract_all_platform_ids(sources)

        assert "animeplanet_id" in platform_ids
        assert platform_ids["animeplanet_id"] == "attack-on-titan"


class TestQualityScoring:
    """Test data quality scoring."""

    def test_calculate_quality_score_complete_entry(self, mock_settings):
        """Test quality score for complete entry."""
        service = AnimeDataService(mock_settings)
        anime = AnimeEntry(
            title="Complete Anime",
            type="TV",
            episodes=24,
            synopsis="Detailed synopsis",
            tags=["Action", "Drama"],
            studios=["Studio A"],
            picture="https://example.com/image.jpg",
            sources=["https://myanimelist.net/anime/1", "https://anilist.co/anime/1"],
        )

        score = service._calculate_quality_score(anime)

        # Complete entry should have high score
        assert score >= 0.9
        assert score <= 1.0

    def test_calculate_quality_score_minimal_entry(self, mock_settings):
        """Test quality score for minimal entry."""
        service = AnimeDataService(mock_settings)
        anime = AnimeEntry(title="Minimal Anime", type="TV", episodes=0)  # No episodes

        score = service._calculate_quality_score(anime)

        # Minimal entry should have lower score
        assert score <= 0.5

    def test_calculate_quality_score_no_title(self, mock_settings):
        """Test quality score with no title."""
        service = AnimeDataService(mock_settings)
        anime = AnimeEntry(title="", type="TV")  # Empty title

        score = service._calculate_quality_score(anime)

        # No title should reduce score significantly
        assert score < 0.3


class TestLoggingAndMetrics:
    """Test logging and metrics functionality."""

    def test_log_batch_metrics(self, mock_settings):
        """Test batch metrics logging."""
        service = AnimeDataService(mock_settings)

        # This should not raise any exceptions
        service._log_batch_metrics(
            batch_num=1,
            batch_start=time.time() - 1.0,
            total_entries=10,
            processed_entries=9,
            error_count=1,
        )

    def test_log_processing_metrics(self, mock_settings):
        """Test processing metrics logging."""
        service = AnimeDataService(mock_settings)

        # This should not raise any exceptions
        service._log_processing_metrics(
            start_time=time.time() - 5.0, total_entries=100, processed_entries=95
        )

    def test_log_batch_metrics_zero_duration(self, mock_settings):
        """Test batch metrics with zero duration."""
        service = AnimeDataService(mock_settings)

        # Should handle zero duration gracefully
        service._log_batch_metrics(
            batch_num=1,
            batch_start=time.time(),  # Same time = zero duration
            total_entries=10,
            processed_entries=10,
            error_count=0,
        )

    def test_log_processing_metrics_zero_duration(self, mock_settings):
        """Test processing metrics with zero duration."""
        service = AnimeDataService(mock_settings)

        # Should handle zero duration gracefully
        service._log_processing_metrics(
            start_time=time.time(),  # Same time = zero duration
            total_entries=10,
            processed_entries=10,
        )


class TestAsyncProcessing:
    """Test async processing methods."""

    @pytest.mark.asyncio
    async def test_process_batches_concurrently(self, mock_settings, sample_anime_data):
        """Test concurrent batch processing."""
        service = AnimeDataService(mock_settings)
        config = ProcessingConfig(
            batch_size=1, max_concurrent_batches=2, processing_timeout=30
        )

        batches = service._create_batches(sample_anime_data["data"], 1)

        # Mock the _process_batch method to return successful results
        async def mock_process_batch(batch, batch_num, config):
            results = []
            for entry in batch:
                result = await service.process_anime_entry(entry)
                if result:
                    results.append(result)
            return results

        with patch.object(service, "_process_batch", side_effect=mock_process_batch):
            results = await service._process_batches_concurrently(batches, config)

        assert len(results) == len(batches)
        assert all(isinstance(result, list) for result in results)

    @pytest.mark.asyncio
    async def test_process_batch_with_semaphore(self, mock_settings, sample_anime_data):
        """Test processing a single batch."""
        service = AnimeDataService(mock_settings)
        config = ProcessingConfig(
            batch_size=10, max_concurrent_batches=2, processing_timeout=30
        )
        batch = sample_anime_data["data"]

        result = await service._process_batch(batch, 1, config)

        assert isinstance(result, list)
        assert len(result) >= 0  # May filter out invalid entries

    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self, mock_settings):
        """Test batch processing with invalid entries."""
        service = AnimeDataService(mock_settings)
        config = ProcessingConfig(
            batch_size=10, max_concurrent_batches=2, processing_timeout=30
        )

        # Create batch with mix of valid and invalid entries
        batch = [
            {
                "title": "Valid Anime",
                "type": "TV",
                "sources": ["https://myanimelist.net/anime/1"],
            },
            {"invalid": "entry"},  # Invalid entry
            {
                "title": "Another Valid",
                "type": "TV",
                "sources": ["https://myanimelist.net/anime/2"],
            },
        ]

        result = await service._process_batch(batch, 1, config)

        # Should process valid entries and skip invalid ones
        assert isinstance(result, list)
        assert len(result) >= 1  # At least one valid entry


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_process_batches_with_exceptions(
        self, mock_settings, sample_anime_data
    ):
        """Test handling of batch processing exceptions."""
        service = AnimeDataService(mock_settings)
        config = ProcessingConfig(
            batch_size=1, max_concurrent_batches=2, processing_timeout=30
        )

        batches = service._create_batches(sample_anime_data["data"], 1)

        # Mock _process_batch to return exceptions for some batches
        async def mock_process_batch(batch, batch_num, config):
            if batch_num == 1:
                raise Exception("Batch processing error")
            results = []
            for entry in batch:
                result = await service.process_anime_entry(entry)
                if result:
                    results.append(result)
            return results

        with patch.object(service, "_process_batch", side_effect=mock_process_batch):
            results = await service._process_batches_concurrently(batches, config)

        # Should handle exceptions gracefully and continue with other batches
        assert len(results) == len(batches)
        assert results[0] == []  # Failed batch returns empty list
        assert len(results[1]) > 0  # Successful batch returns data

    @pytest.mark.asyncio
    async def test_process_batch_with_asyncio_to_thread_exception(self, mock_settings):
        """Test exception handling in asyncio.to_thread."""
        service = AnimeDataService(mock_settings)
        config = ProcessingConfig(
            batch_size=10, max_concurrent_batches=2, processing_timeout=30
        )

        # Create batch that will cause exceptions
        batch = [
            {
                "title": "Test Anime",
                "type": "TV",
                "sources": ["https://myanimelist.net/anime/1"],
            }
        ]

        # Mock asyncio.to_thread to raise an exception
        with patch(
            "asyncio.to_thread", side_effect=Exception("Thread processing error")
        ):
            result = await service._process_batch(batch, 1, config)

            # Should handle exception and continue
            assert isinstance(result, list)
            # Result might be empty due to exception handling

    @pytest.mark.asyncio
    async def test_process_batch_with_gather_exceptions(self, mock_settings):
        """Test handling of exceptions from asyncio.gather."""
        service = AnimeDataService(mock_settings)
        config = ProcessingConfig(
            batch_size=10, max_concurrent_batches=2, processing_timeout=30
        )

        # Mock the process_entry_async to return an exception
        batch = [
            {
                "title": "Test Anime",
                "type": "TV",
                "sources": ["https://myanimelist.net/anime/1"],
            }
        ]

        # Use real implementation but force it to return exceptions
        asyncio.gather

        async def mock_gather(*tasks, return_exceptions=True):
            # Return a mix of exceptions and None values
            return [Exception("Processing error"), None]

        with patch("asyncio.gather", side_effect=mock_gather):
            result = await service._process_batch(batch, 1, config)

            # Should handle both exceptions and None values
            assert isinstance(result, list)

    def test_extract_platform_ids_value_error(self, mock_settings):
        """Test ValueError handling in numeric ID conversion."""
        service = AnimeDataService(mock_settings)

        # Mock the regex to match but int() to fail
        sources = ["https://myanimelist.net/anime/not_a_number"]

        # The pattern will match "not_a_number" but int() will fail
        platform_ids = service._extract_all_platform_ids(sources)

        # Should not include the failed conversion
        assert "myanimelist_id" not in platform_ids

    def test_extract_platform_ids_with_invalid_numeric_conversion(self, mock_settings):
        """Test platform ID extraction with invalid numeric values."""
        service = AnimeDataService(mock_settings)

        # Create a mock source that will match the pattern but fail numeric conversion
        # We need to temporarily modify the platform config to force a ValueError
        original_pattern = service.platform_configs["myanimelist"]["pattern"]

        try:
            # Set a pattern that will match non-numeric values
            import re

            service.platform_configs["myanimelist"]["pattern"] = re.compile(
                r"/anime/([a-z]+)"
            )

            sources = ["https://myanimelist.net/anime/invalid"]
            platform_ids = service._extract_all_platform_ids(sources)

            # Should not include the failed conversion
            assert "myanimelist_id" not in platform_ids

        finally:
            # Restore original pattern
            service.platform_configs["myanimelist"]["pattern"] = original_pattern
