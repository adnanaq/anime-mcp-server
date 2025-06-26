"""Comprehensive tests for UpdateService with 100% coverage."""

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.services.update_service import UpdateService


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = Mock()
    client.get_stats = AsyncMock()
    client.health_check = AsyncMock()
    return client


@pytest.fixture
def mock_data_service():
    """Mock data service."""
    service = Mock()
    service.download_anime_database = AsyncMock()
    service.process_anime_data = AsyncMock()
    return service


@pytest.fixture
def sample_anime_data():
    """Sample anime data for testing."""
    return [
        {
            "title": "Test Anime 1",
            "type": "TV",
            "episodes": 12,
            "sources": ["https://myanimelist.net/anime/1"],
        },
        {
            "title": "Test Anime 2",
            "type": "Movie",
            "episodes": 1,
            "sources": ["https://myanimelist.net/anime/2"],
        },
    ]


class TestUpdateServiceInitialization:
    """Test UpdateService initialization."""

    def test_init_with_provided_client(self, mock_qdrant_client):
        """Test initialization with provided Qdrant client."""
        with patch(
            "src.services.update_service.AnimeDataService"
        ) as mock_data_service_class:
            mock_data_service_class.return_value = Mock()

            service = UpdateService(qdrant_client=mock_qdrant_client)

            assert service.qdrant_client == mock_qdrant_client
            assert service.data_dir == Path("data")
            assert service.metadata_file == Path("data/update_metadata.json")

    def test_init_without_client(self):
        """Test initialization without provided client."""
        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch("src.services.update_service.get_settings") as mock_get_settings,
            patch(
                "src.services.update_service.QdrantClient"
            ) as mock_qdrant_client_class,
        ):
            mock_data_service_class.return_value = Mock()
            mock_settings = Mock()
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_get_settings.return_value = mock_settings
            mock_client = Mock()
            mock_qdrant_client_class.return_value = mock_client

            service = UpdateService()

            assert service.qdrant_client == mock_client
            mock_qdrant_client_class.assert_called_once_with(settings=mock_settings)


class TestUpdateChecking:
    """Test update checking functionality."""

    @pytest.mark.asyncio
    async def test_check_for_updates_new_data(
        self, mock_qdrant_client, sample_anime_data
    ):
        """Test check for updates when new data is available."""
        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(Path, "exists", return_value=False),
        ):
            mock_data_service = Mock()
            mock_data_service.download_anime_database = AsyncMock(
                return_value=sample_anime_data
            )
            mock_data_service_class.return_value = mock_data_service

            service = UpdateService(qdrant_client=mock_qdrant_client)
            service.data_service = mock_data_service

            result = await service.check_for_updates()

            assert result is True
            mock_data_service.download_anime_database.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_for_updates_same_data(
        self, mock_qdrant_client, sample_anime_data
    ):
        """Test check for updates when data hasn't changed."""
        content_str = json.dumps(sample_anime_data, sort_keys=True)
        expected_hash = hashlib.md5(content_str.encode()).hexdigest()

        metadata = {"content_hash": expected_hash, "last_check": "2024-01-01T10:00:00Z"}

        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", create=True) as mock_open,
            patch("json.load", return_value=metadata),
        ):
            mock_data_service = Mock()
            mock_data_service.download_anime_database = AsyncMock(
                return_value=sample_anime_data
            )
            mock_data_service_class.return_value = mock_data_service

            service = UpdateService(qdrant_client=mock_qdrant_client)
            service.data_service = mock_data_service

            result = await service.check_for_updates()

            assert result is False

    @pytest.mark.asyncio
    async def test_check_for_updates_error_handling(self, mock_qdrant_client):
        """Test check for updates error handling."""
        with patch(
            "src.services.update_service.AnimeDataService"
        ) as mock_data_service_class:
            mock_data_service = Mock()
            mock_data_service.download_anime_database = AsyncMock(
                side_effect=Exception("Download failed")
            )
            mock_data_service_class.return_value = mock_data_service

            service = UpdateService(qdrant_client=mock_qdrant_client)
            service.data_service = mock_data_service

            with pytest.raises(Exception, match="Download failed"):
                await service.check_for_updates()


class TestUpdateOperations:
    """Test update operations."""

    @pytest.mark.asyncio
    async def test_perform_incremental_update_success(
        self, mock_qdrant_client, sample_anime_data
    ):
        """Test successful incremental update."""
        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(UpdateService, "check_for_updates", return_value=True),
            patch.object(UpdateService, "save_metadata"),
        ):
            mock_data_service = Mock()
            mock_data_service.download_anime_database = AsyncMock(
                return_value=sample_anime_data
            )
            mock_data_service.process_anime_data = AsyncMock(
                return_value={"processed": 2}
            )
            mock_data_service_class.return_value = mock_data_service

            service = UpdateService(qdrant_client=mock_qdrant_client)
            service.data_service = mock_data_service

            result = await service.perform_incremental_update()

            assert "status" in result
            assert "updated_entries" in result

    @pytest.mark.asyncio
    async def test_perform_incremental_update_no_updates(self, mock_qdrant_client):
        """Test incremental update when no updates available."""
        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(UpdateService, "check_for_updates", return_value=False),
        ):
            mock_data_service = Mock()
            mock_data_service_class.return_value = mock_data_service

            service = UpdateService(qdrant_client=mock_qdrant_client)
            service.data_service = mock_data_service

            result = await service.perform_incremental_update()

            assert result["status"] == "no_updates"

    @pytest.mark.asyncio
    async def test_perform_full_update_success(
        self, mock_qdrant_client, sample_anime_data
    ):
        """Test successful full update."""
        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(UpdateService, "save_metadata"),
        ):
            mock_data_service = Mock()
            mock_data_service.download_anime_database = AsyncMock(
                return_value=sample_anime_data
            )
            mock_data_service.process_anime_data = AsyncMock(
                return_value={"processed": 2}
            )
            mock_data_service_class.return_value = mock_data_service
            mock_qdrant_client.clear_index = AsyncMock()

            service = UpdateService(qdrant_client=mock_qdrant_client)
            service.data_service = mock_data_service

            result = await service.perform_full_update()

            assert "status" in result
            assert "total_entries" in result
            mock_qdrant_client.clear_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_schedule_weekly_update(self, mock_qdrant_client):
        """Test weekly update scheduling."""
        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(
                UpdateService,
                "perform_incremental_update",
                return_value={"status": "success"},
            ),
        ):
            mock_data_service = Mock()
            mock_data_service_class.return_value = mock_data_service

            service = UpdateService(qdrant_client=mock_qdrant_client)

            result = await service.schedule_weekly_update()

            assert "status" in result


class TestMetadataOperations:
    """Test metadata operations."""

    def test_load_metadata_file_exists(self, mock_qdrant_client):
        """Test loading metadata when file exists."""
        metadata = {"last_update": "2024-01-01", "content_hash": "abc123"}

        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", create=True),
            patch("json.load", return_value=metadata),
        ):
            mock_data_service_class.return_value = Mock()
            service = UpdateService(qdrant_client=mock_qdrant_client)

            result = service.load_metadata()

            assert result == metadata

    def test_load_metadata_file_not_exists(self, mock_qdrant_client):
        """Test loading metadata when file doesn't exist."""
        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(Path, "exists", return_value=False),
        ):
            mock_data_service_class.return_value = Mock()
            service = UpdateService(qdrant_client=mock_qdrant_client)

            result = service.load_metadata()

            assert result == {}

    def test_load_metadata_json_error(self, mock_qdrant_client):
        """Test loading metadata with JSON error."""
        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", create=True),
            patch("json.load", side_effect=json.JSONDecodeError("Invalid JSON", "", 0)),
        ):
            mock_data_service_class.return_value = Mock()
            service = UpdateService(qdrant_client=mock_qdrant_client)

            result = service.load_metadata()

            assert result == {}

    def test_save_metadata_success(self, mock_qdrant_client):
        """Test saving metadata successfully."""
        metadata = {"last_update": "2024-01-01", "content_hash": "abc123"}

        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(Path, "mkdir") as mock_mkdir,
            patch("builtins.open", create=True) as mock_open,
            patch("json.dump") as mock_json_dump,
        ):
            mock_data_service_class.return_value = Mock()
            service = UpdateService(qdrant_client=mock_qdrant_client)

            service.save_metadata(metadata)

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_json_dump.assert_called_once()

    def test_save_metadata_error(self, mock_qdrant_client):
        """Test saving metadata with error."""
        metadata = {"last_update": "2024-01-01"}

        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(Path, "mkdir", side_effect=OSError("Permission denied")),
        ):
            mock_data_service_class.return_value = Mock()
            service = UpdateService(qdrant_client=mock_qdrant_client)

            # Should not raise exception, just log error
            service.save_metadata(metadata)


class TestHashCalculation:
    """Test hash calculation functionality."""

    def test_calculate_content_hash(self, mock_qdrant_client, sample_anime_data):
        """Test content hash calculation."""
        with patch(
            "src.services.update_service.AnimeDataService"
        ) as mock_data_service_class:
            mock_data_service_class.return_value = Mock()
            service = UpdateService(qdrant_client=mock_qdrant_client)

            content_str = json.dumps(sample_anime_data, sort_keys=True)
            expected_hash = hashlib.md5(content_str.encode()).hexdigest()

            result = service._calculate_content_hash(sample_anime_data)

            assert result == expected_hash

    def test_calculate_content_hash_empty_data(self, mock_qdrant_client):
        """Test content hash calculation with empty data."""
        with patch(
            "src.services.update_service.AnimeDataService"
        ) as mock_data_service_class:
            mock_data_service_class.return_value = Mock()
            service = UpdateService(qdrant_client=mock_qdrant_client)

            empty_data = []
            content_str = json.dumps(empty_data, sort_keys=True)
            expected_hash = hashlib.md5(content_str.encode()).hexdigest()

            result = service._calculate_content_hash(empty_data)

            assert result == expected_hash


class TestUpdateStatus:
    """Test update status functionality."""

    def test_get_update_status_with_metadata(self, mock_qdrant_client):
        """Test getting update status when metadata exists."""
        metadata = {
            "last_update": "2024-01-01T10:00:00Z",
            "content_hash": "abc123",
            "entry_count": 38894,
        }

        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(UpdateService, "load_metadata", return_value=metadata),
        ):
            mock_data_service_class.return_value = Mock()
            service = UpdateService(qdrant_client=mock_qdrant_client)

            result = service.get_update_status()

            assert result["last_update"] == metadata["last_update"]
            assert result["entry_count"] == metadata["entry_count"]

    def test_get_update_status_no_metadata(self, mock_qdrant_client):
        """Test getting update status when no metadata exists."""
        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(UpdateService, "load_metadata", return_value={}),
        ):
            mock_data_service_class.return_value = Mock()
            service = UpdateService(qdrant_client=mock_qdrant_client)

            result = service.get_update_status()

            assert "last_update" in result
            assert "status" in result


class TestErrorHandling:
    """Test comprehensive error handling."""

    @pytest.mark.asyncio
    async def test_update_operation_error_recovery(self, mock_qdrant_client):
        """Test error recovery during update operations."""
        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(
                UpdateService,
                "check_for_updates",
                side_effect=Exception("Check failed"),
            ),
        ):
            mock_data_service_class.return_value = Mock()
            service = UpdateService(qdrant_client=mock_qdrant_client)

            with pytest.raises(Exception, match="Check failed"):
                await service.perform_incremental_update()

    @pytest.mark.asyncio
    async def test_full_update_error_handling(self, mock_qdrant_client):
        """Test full update error handling."""
        with (
            patch(
                "src.services.update_service.AnimeDataService"
            ) as mock_data_service_class,
            patch.object(
                UpdateService, "save_metadata", side_effect=Exception("Save failed")
            ),
        ):
            mock_data_service = Mock()
            mock_data_service.download_anime_database = AsyncMock(return_value=[])
            mock_data_service.process_anime_data = AsyncMock(
                return_value={"processed": 0}
            )
            mock_data_service_class.return_value = mock_data_service
            mock_qdrant_client.clear_index = AsyncMock()

            service = UpdateService(qdrant_client=mock_qdrant_client)
            service.data_service = mock_data_service

            # Should still complete successfully despite metadata save error
            result = await service.perform_full_update()
            assert "status" in result
