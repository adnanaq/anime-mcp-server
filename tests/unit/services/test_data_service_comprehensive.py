"""Comprehensive tests for AnimeDataService with 100% coverage."""

import json
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.anime import AnimeEntry
from src.services.data_service import AnimeDataService


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = Mock()
    client.add_documents = AsyncMock()
    client.clear_index = AsyncMock()
    client.get_stats = AsyncMock()
    client.health_check = AsyncMock()
    return client


@pytest.fixture
def mock_settings():
    """Mock settings."""
    settings = Mock()
    settings.anime_data_url = "https://example.com/anime-offline-database.json"
    settings.data_cache_dir = "/tmp/anime_cache"
    settings.batch_size = 100
    settings.max_retries = 3
    return settings


@pytest.fixture
def sample_anime_data():
    """Sample anime data for testing."""
    return [
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
            "sources": [
                "https://myanimelist.net/anime/16498",
                "https://anilist.co/anime/16498",
            ],
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
            "sources": [
                "https://myanimelist.net/anime/1535",
                "https://anilist.co/anime/1535",
            ],
        },
    ]


class TestAnimeDataServiceInitialization:
    """Test service initialization."""

    def test_init_with_valid_params(self, mock_qdrant_client, mock_settings):
        """Test initialization with valid parameters."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        assert service.qdrant_client == mock_qdrant_client
        assert service.settings == mock_settings
        assert service.session is not None

    def test_init_without_client(self, mock_settings):
        """Test initialization without client."""
        with pytest.raises(ValueError, match="Qdrant client is required"):
            AnimeDataService(None, mock_settings)

    def test_init_without_settings(self, mock_qdrant_client):
        """Test initialization without settings."""
        with pytest.raises(ValueError, match="Settings are required"):
            AnimeDataService(mock_qdrant_client, None)


class TestDataDownload:
    """Test data download functionality."""

    @pytest.mark.asyncio
    async def test_download_anime_data_success(
        self, mock_qdrant_client, mock_settings, sample_anime_data
    ):
        """Test successful data download."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"data": sample_anime_data}
            mock_response.headers = {"content-length": "1024"}
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await service.download_anime_data()

        assert result["status"] == "success"
        assert result["downloaded_entries"] == 2
        assert "download_time_seconds" in result
        assert "file_size_mb" in result

    @pytest.mark.asyncio
    async def test_download_anime_data_http_error(
        self, mock_qdrant_client, mock_settings
    ):
        """Test download with HTTP error."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.text.return_value = "Not Found"
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception, match="HTTP 404"):
                await service.download_anime_data()

    @pytest.mark.asyncio
    async def test_download_anime_data_network_error(
        self, mock_qdrant_client, mock_settings
    ):
        """Test download with network error."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        with patch("aiohttp.ClientSession.get", side_effect=Exception("Network error")):
            with pytest.raises(Exception, match="Network error"):
                await service.download_anime_data()

    @pytest.mark.asyncio
    async def test_download_anime_data_invalid_json(
        self, mock_qdrant_client, mock_settings
    ):
        """Test download with invalid JSON."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.headers = {"content-length": "1024"}
            mock_get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(Exception, match="Failed to parse JSON"):
                await service.download_anime_data()

    @pytest.mark.asyncio
    async def test_download_anime_data_with_retries(
        self, mock_qdrant_client, mock_settings, sample_anime_data
    ):
        """Test download with retries on failure."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        with patch("aiohttp.ClientSession.get") as mock_get:
            # First call fails, second succeeds
            mock_response_fail = AsyncMock()
            mock_response_fail.status = 500
            mock_response_fail.text.return_value = "Server Error"

            mock_response_success = AsyncMock()
            mock_response_success.status = 200
            mock_response_success.json.return_value = {"data": sample_anime_data}
            mock_response_success.headers = {"content-length": "1024"}

            mock_get.return_value.__aenter__.side_effect = [
                mock_response_fail,
                mock_response_success,
            ]

            result = await service.download_anime_data()

        assert result["status"] == "success"
        assert mock_get.call_count == 2


class TestDataProcessing:
    """Test data processing functionality."""

    @pytest.mark.asyncio
    async def test_process_raw_anime_data_success(
        self, mock_qdrant_client, mock_settings, sample_anime_data
    ):
        """Test successful data processing."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        service.raw_anime_data = sample_anime_data

        with patch.object(
            service, "_process_anime_batch", return_value={"processed": 2, "skipped": 0}
        ):
            result = await service.process_raw_anime_data()

        assert result["status"] == "success"
        assert result["processed_entries"] == 2
        assert "processing_time_seconds" in result

    @pytest.mark.asyncio
    async def test_process_raw_anime_data_no_data(
        self, mock_qdrant_client, mock_settings
    ):
        """Test processing when no raw data available."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        service.raw_anime_data = None

        with pytest.raises(Exception, match="No raw anime data available"):
            await service.process_raw_anime_data()

    @pytest.mark.asyncio
    async def test_process_raw_anime_data_empty_data(
        self, mock_qdrant_client, mock_settings
    ):
        """Test processing with empty data."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        service.raw_anime_data = []

        result = await service.process_raw_anime_data()

        assert result["status"] == "success"
        assert result["processed_entries"] == 0

    @pytest.mark.asyncio
    async def test_process_anime_batch_success(
        self, mock_qdrant_client, mock_settings, sample_anime_data
    ):
        """Test batch processing success."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        mock_qdrant_client.add_documents.return_value = None

        result = await service._process_anime_batch(sample_anime_data)

        assert result["processed"] == 2
        assert result["skipped"] == 0
        mock_qdrant_client.add_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_anime_batch_with_invalid_entries(
        self, mock_qdrant_client, mock_settings
    ):
        """Test batch processing with invalid entries."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        invalid_data = [
            {"title": "Valid Anime", "type": "TV", "episodes": 12},
            {"title": "", "type": "TV"},  # Invalid: empty title
            {"type": "TV", "episodes": 12},  # Invalid: missing title
        ]

        result = await service._process_anime_batch(invalid_data)

        assert result["processed"] == 1
        assert result["skipped"] == 2

    def test_create_anime_entry_valid(
        self, mock_qdrant_client, mock_settings, sample_anime_data
    ):
        """Test anime entry creation with valid data."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        anime_dict = sample_anime_data[0]

        entry = service._create_anime_entry(anime_dict)

        assert isinstance(entry, AnimeEntry)
        assert entry.title == "Attack on Titan"
        assert entry.type == "TV"
        assert entry.episodes == 25
        assert entry.year == 2013
        assert len(entry.tags) == 2

    def test_create_anime_entry_missing_required_fields(
        self, mock_qdrant_client, mock_settings
    ):
        """Test anime entry creation with missing required fields."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        invalid_data = {"type": "TV", "episodes": 12}  # Missing title

        entry = service._create_anime_entry(invalid_data)
        assert entry is None

    def test_create_anime_entry_with_defaults(self, mock_qdrant_client, mock_settings):
        """Test anime entry creation with default values."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        minimal_data = {
            "title": "Minimal Anime",
            "type": "TV",
            "sources": ["https://myanimelist.net/anime/1"],
        }

        entry = service._create_anime_entry(minimal_data)

        assert entry.title == "Minimal Anime"
        assert entry.episodes == 0  # Default value
        assert entry.year is None
        assert entry.tags == []
        assert entry.studios == []

    def test_extract_platform_ids_success(self, mock_qdrant_client, mock_settings):
        """Test platform ID extraction."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        sources = [
            "https://myanimelist.net/anime/16498",
            "https://anilist.co/anime/16498",
            "https://kitsu.io/anime/attack-on-titan",
            "https://anidb.net/anime/9541",
        ]

        platform_ids = service._extract_platform_ids(sources)

        assert platform_ids["myanimelist_id"] == 16498
        assert platform_ids["anilist_id"] == 16498
        assert "kitsu_id" in platform_ids
        assert "anidb_id" in platform_ids

    def test_extract_platform_ids_invalid_urls(self, mock_qdrant_client, mock_settings):
        """Test platform ID extraction with invalid URLs."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        sources = [
            "https://example.com/anime/123",  # Unknown platform
            "invalid-url",  # Invalid URL
            "https://myanimelist.net/anime/invalid",  # Invalid ID
        ]

        platform_ids = service._extract_platform_ids(sources)

        # Should return empty dict for unknown/invalid sources
        assert len(platform_ids) == 0

    def test_calculate_data_quality_score_high_quality(
        self, mock_qdrant_client, mock_settings
    ):
        """Test data quality calculation for high quality entry."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        entry = AnimeEntry(
            anime_id="test",
            title="High Quality Anime",
            synopsis="Detailed synopsis here",
            type="TV",
            episodes=24,
            year=2023,
            season="SPRING",
            tags=["Action", "Drama", "Fantasy"],
            studios=["Studio A", "Studio B"],
            picture="https://example.com/image.jpg",
            platform_ids={"myanimelist_id": 123, "anilist_id": 123},
        )

        score = service._calculate_data_quality_score(entry)

        assert score >= 0.8  # High quality should have high score

    def test_calculate_data_quality_score_low_quality(
        self, mock_qdrant_client, mock_settings
    ):
        """Test data quality calculation for low quality entry."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        entry = AnimeEntry(anime_id="test", title="Minimal Anime", type="TV")

        score = service._calculate_data_quality_score(entry)

        assert score < 0.5  # Low quality should have low score

    def test_generate_embedding_text(
        self, mock_qdrant_client, mock_settings, sample_anime_data
    ):
        """Test embedding text generation."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        entry = service._create_anime_entry(sample_anime_data[0])

        embedding_text = service._generate_embedding_text(entry)

        assert "Attack on Titan" in embedding_text
        assert "Action" in embedding_text
        assert "Drama" in embedding_text
        assert str(entry.year) in embedding_text


class TestStatsAndMonitoring:
    """Test stats and monitoring functionality."""

    @pytest.mark.asyncio
    async def test_get_processing_stats_success(
        self, mock_qdrant_client, mock_settings
    ):
        """Test getting processing stats."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        mock_qdrant_client.get_stats.return_value = {
            "total_documents": 38894,
            "collection_name": "anime_database",
        }

        result = await service.get_processing_stats()

        assert result["total_entries"] == 38894
        assert result["collection_name"] == "anime_database"
        assert "last_processing_time" in result

    @pytest.mark.asyncio
    async def test_get_processing_stats_client_error(
        self, mock_qdrant_client, mock_settings
    ):
        """Test getting stats with client error."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        mock_qdrant_client.get_stats.side_effect = Exception("Stats error")

        with pytest.raises(Exception, match="Stats error"):
            await service.get_processing_stats()


class TestDataClearing:
    """Test data clearing functionality."""

    @pytest.mark.asyncio
    async def test_clear_all_data_success(self, mock_qdrant_client, mock_settings):
        """Test successful data clearing."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        mock_qdrant_client.clear_index.return_value = None

        result = await service.clear_all_data()

        assert result["status"] == "success"
        assert "cleared_entries" in result
        mock_qdrant_client.clear_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_all_data_client_error(self, mock_qdrant_client, mock_settings):
        """Test data clearing with client error."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        mock_qdrant_client.clear_index.side_effect = Exception("Clear error")

        with pytest.raises(Exception, match="Clear error"):
            await service.clear_all_data()


class TestUpdateOperations:
    """Test update operations."""

    @pytest.mark.asyncio
    async def test_check_for_updates_success(
        self, mock_qdrant_client, mock_settings, sample_anime_data
    ):
        """Test checking for updates."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        with patch.object(service, "download_anime_data") as mock_download:
            mock_download.return_value = {
                "status": "success",
                "downloaded_entries": len(sample_anime_data),
            }
            service.raw_anime_data = sample_anime_data

            result = await service.check_for_updates()

        assert "updates_available" in result
        assert "new_entries" in result
        assert "analysis" in result

    @pytest.mark.asyncio
    async def test_perform_incremental_update_success(
        self, mock_qdrant_client, mock_settings
    ):
        """Test incremental update."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        with (
            patch.object(service, "check_for_updates") as mock_check,
            patch.object(service, "process_raw_anime_data") as mock_process,
        ):
            mock_check.return_value = {"updates_available": True, "new_entries": 50}
            mock_process.return_value = {"status": "success", "processed_entries": 50}

            result = await service.perform_incremental_update()

        assert result["status"] == "success"
        assert result["update_type"] == "incremental"

    @pytest.mark.asyncio
    async def test_perform_full_update_success(self, mock_qdrant_client, mock_settings):
        """Test full update."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        with (
            patch.object(service, "download_anime_data") as mock_download,
            patch.object(service, "clear_all_data") as mock_clear,
            patch.object(service, "process_raw_anime_data") as mock_process,
        ):
            mock_download.return_value = {
                "status": "success",
                "downloaded_entries": 38894,
            }
            mock_clear.return_value = {"status": "success"}
            mock_process.return_value = {
                "status": "success",
                "processed_entries": 38894,
            }

            result = await service.perform_full_update()

        assert result["status"] == "success"
        assert result["update_type"] == "full"

    @pytest.mark.asyncio
    async def test_get_update_status(self, mock_qdrant_client, mock_settings):
        """Test getting update status."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        result = await service.get_update_status()

        assert "status" in result
        assert "last_update" in result
        assert "next_scheduled_update" in result

    @pytest.mark.asyncio
    async def test_schedule_weekly_updates(self, mock_qdrant_client, mock_settings):
        """Test scheduling weekly updates."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        result = await service.schedule_weekly_updates()

        assert result["status"] == "scheduled"
        assert result["frequency"] == "weekly"

    @pytest.mark.asyncio
    async def test_analyze_update_schedule(self, mock_qdrant_client, mock_settings):
        """Test update schedule analysis."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        result = await service.analyze_update_schedule()

        assert "recommended_frequency" in result
        assert "optimal_time" in result
        assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_check_update_safety(self, mock_qdrant_client, mock_settings):
        """Test update safety check."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        mock_qdrant_client.health_check.return_value = True

        result = await service.check_update_safety()

        assert "safe_to_update" in result
        assert "risk_level" in result
        assert "warnings" in result

    @pytest.mark.asyncio
    async def test_smart_update_with_analysis(self, mock_qdrant_client, mock_settings):
        """Test smart update with analysis."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        with (
            patch.object(service, "check_update_safety") as mock_safety,
            patch.object(service, "perform_incremental_update") as mock_update,
        ):
            mock_safety.return_value = {"safe_to_update": True, "risk_level": "low"}
            mock_update.return_value = {"status": "success", "updated_entries": 50}

            result = await service.smart_update_with_analysis()

        assert result["status"] == "success"
        assert "safety_analysis" in result


class TestFileCaching:
    """Test file caching functionality."""

    def test_cache_file_operations(self, mock_qdrant_client, mock_settings):
        """Test file caching operations."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        # Test cache directory creation
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_settings.data_cache_dir = temp_dir
            cache_file = service._get_cache_file_path("test.json")

            assert temp_dir in cache_file
            assert cache_file.endswith("test.json")

    def test_data_validation(self, mock_qdrant_client, mock_settings):
        """Test data validation methods."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        # Test valid anime data
        valid_data = {
            "title": "Test Anime",
            "type": "TV",
            "episodes": 12,
            "sources": ["https://myanimelist.net/anime/1"],
        }
        assert service._validate_anime_data(valid_data) is True

        # Test invalid anime data
        invalid_data = {"type": "TV"}  # Missing title
        assert service._validate_anime_data(invalid_data) is False


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_process_anime_batch_client_error(
        self, mock_qdrant_client, mock_settings, sample_anime_data
    ):
        """Test batch processing with client error."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        mock_qdrant_client.add_documents.side_effect = Exception("Client error")

        with pytest.raises(Exception, match="Client error"):
            await service._process_anime_batch(sample_anime_data)

    @pytest.mark.asyncio
    async def test_download_with_timeout(self, mock_qdrant_client, mock_settings):
        """Test download with timeout."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        with patch("aiohttp.ClientSession.get", side_effect=Exception("Timeout")):
            with pytest.raises(Exception, match="Timeout"):
                await service.download_anime_data()

    def test_malformed_platform_urls(self, mock_qdrant_client, mock_settings):
        """Test handling of malformed platform URLs."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        malformed_sources = [
            "not-a-url",
            "https://",
            "https://unknown-platform.com/anime/123",
            "https://myanimelist.net/anime/",  # Missing ID
        ]

        platform_ids = service._extract_platform_ids(malformed_sources)
        assert len(platform_ids) == 0

    def test_anime_entry_creation_edge_cases(self, mock_qdrant_client, mock_settings):
        """Test anime entry creation edge cases."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        # Test with None values
        data_with_nones = {
            "title": "Test Anime",
            "type": "TV",
            "episodes": None,
            "animeSeason": None,
            "tags": None,
            "sources": ["https://myanimelist.net/anime/1"],
        }

        entry = service._create_anime_entry(data_with_nones)
        assert entry is not None
        assert entry.episodes == 0  # Should handle None episodes
        assert entry.year is None
        assert entry.tags == []

    @pytest.mark.asyncio
    async def test_concurrent_processing_safety(
        self, mock_qdrant_client, mock_settings, sample_anime_data
    ):
        """Test concurrent processing safety."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        service.raw_anime_data = sample_anime_data * 10  # Larger dataset

        # Mock successful batch processing
        mock_qdrant_client.add_documents.return_value = None

        # This should not raise any concurrency issues
        result = await service.process_raw_anime_data()
        assert result["status"] == "success"


class TestPerformanceOptimizations:
    """Test performance optimization features."""

    @pytest.mark.asyncio
    async def test_batch_size_optimization(
        self, mock_qdrant_client, mock_settings, sample_anime_data
    ):
        """Test batch size optimization."""
        mock_settings.batch_size = 1  # Small batch size for testing
        service = AnimeDataService(mock_qdrant_client, mock_settings)
        service.raw_anime_data = sample_anime_data

        with patch.object(
            service, "_process_anime_batch", wraps=service._process_anime_batch
        ) as mock_batch:
            await service.process_raw_anime_data()

            # Should be called once per anime (batch size = 1)
            assert mock_batch.call_count == 2

    def test_memory_efficiency(self, mock_qdrant_client, mock_settings):
        """Test memory-efficient processing."""
        service = AnimeDataService(mock_qdrant_client, mock_settings)

        # Create large dataset
        large_dataset = []
        for i in range(1000):
            large_dataset.append(
                {
                    "title": f"Anime {i}",
                    "type": "TV",
                    "episodes": 12,
                    "sources": [f"https://myanimelist.net/anime/{i}"],
                }
            )

        service.raw_anime_data = large_dataset

        # Should not cause memory issues
        batches = list(service._create_batches(large_dataset, 100))
        assert len(batches) == 10
        assert len(batches[0]) == 100
