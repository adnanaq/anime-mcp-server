"""Comprehensive tests for UpdateService with 100% coverage."""

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

from src.services.update_service import UpdateService


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = Mock()
    client.collection_name = "anime_database"
    client.client = Mock()
    client.clear_index = AsyncMock()
    client.add_documents = AsyncMock()
    client._generate_point_id = Mock(side_effect=lambda x: f"point_{x}")
    return client


@pytest.fixture
def mock_data_service():
    """Mock data service."""
    service = Mock()
    service.download_anime_database = AsyncMock()
    service.process_anime_entry = Mock()
    service.process_all_anime = AsyncMock()
    return service


@pytest.fixture
def sample_anime_data():
    """Sample anime data for testing."""
    return {
        "data": [
            {
                "title": "Test Anime 1",
                "type": "TV",
                "episodes": 12,
                "sources": ["https://myanimelist.net/anime/1"],
                "anime_id": "anime_1",
            },
            {
                "title": "Test Anime 2",
                "type": "Movie",
                "episodes": 1,
                "sources": ["https://myanimelist.net/anime/2"],
                "anime_id": "anime_2",
            },
        ]
    }


class TestUpdateServiceInitialization:
    """Test UpdateService initialization."""

    @patch("src.services.update_service.AnimeDataService")
    def test_init_with_provided_client(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test initialization with provided Qdrant client."""
        mock_data_service_class.return_value = Mock()

        service = UpdateService(qdrant_client=mock_qdrant_client)

        assert service.qdrant_client == mock_qdrant_client
        assert service.data_dir == Path("data")
        assert service.metadata_file == Path("data/update_metadata.json")

    @patch("src.services.update_service.QdrantClient")
    @patch("src.config.get_settings")
    @patch("src.services.update_service.AnimeDataService")
    def test_init_without_client(
        self, mock_data_service_class, mock_get_settings, mock_qdrant_client_class
    ):
        """Test initialization without provided client."""
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
    @patch("src.services.update_service.AnimeDataService")
    async def test_check_for_updates_new_data(
        self, mock_data_service_class, mock_qdrant_client, sample_anime_data
    ):
        """Test check for updates when new data is available."""
        mock_data_service = Mock()
        mock_data_service.download_anime_database = AsyncMock(
            return_value=sample_anime_data
        )
        mock_data_service_class.return_value = mock_data_service

        service = UpdateService(qdrant_client=mock_qdrant_client)

        # Mock file operations and data service method separately
        with (
            patch.object(service, "load_metadata", return_value={}),
            patch.object(service, "save_metadata") as mock_save,
            patch("builtins.open", mock_open()),
            patch("pathlib.Path.mkdir"),
            patch("json.dump"),
        ):
            # Mock the specific method call that actually gets called
            with patch.object(
                service.data_service,
                "download_anime_database",
                new_callable=AsyncMock,
                return_value=sample_anime_data,
            ):
                result = await service.check_for_updates()

                assert result is True
                mock_save.assert_called()

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_check_for_updates_same_data(
        self, mock_data_service_class, mock_qdrant_client, sample_anime_data
    ):
        """Test check for updates when data hasn't changed."""
        content_str = json.dumps(sample_anime_data, sort_keys=True)
        expected_hash = hashlib.md5(content_str.encode()).hexdigest()

        mock_data_service = Mock()
        mock_data_service.download_anime_database = AsyncMock(
            return_value=sample_anime_data
        )
        mock_data_service_class.return_value = mock_data_service

        service = UpdateService(qdrant_client=mock_qdrant_client)
        service.data_service = mock_data_service

        # Mock metadata with same hash
        with (
            patch.object(
                service, "load_metadata", return_value={"content_hash": expected_hash}
            ),
            patch.object(service, "save_metadata") as mock_save,
        ):
            result = await service.check_for_updates()

            assert result is False
            mock_save.assert_called()

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_check_for_updates_error_handling(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test check for updates error handling."""
        mock_data_service = Mock()
        mock_data_service.download_anime_database = AsyncMock(
            side_effect=Exception("Download failed")
        )
        mock_data_service_class.return_value = mock_data_service

        service = UpdateService(qdrant_client=mock_qdrant_client)
        service.data_service = mock_data_service

        result = await service.check_for_updates()
        assert result is False


class TestIncrementalUpdate:
    """Test incremental update functionality."""

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_perform_incremental_update_success(
        self, mock_data_service_class, mock_qdrant_client, sample_anime_data
    ):
        """Test successful incremental update."""
        mock_data_service = Mock()
        mock_data_service.process_anime_entry = Mock(return_value={"anime_id": "test"})
        mock_data_service_class.return_value = mock_data_service

        service = UpdateService(qdrant_client=mock_qdrant_client)
        service.data_service = mock_data_service

        # Mock different data to trigger changes
        old_data = {
            "data": [{"title": "Old Anime", "sources": ["url1"], "anime_id": "old"}]
        }
        new_data = sample_anime_data

        with (
            patch.object(service, "load_current_data", return_value=old_data),
            patch.object(service, "load_latest_data", return_value=new_data),
            patch.object(service, "remove_entries", return_value=True),
            patch.object(service, "update_processed_data"),
            patch.object(service, "load_metadata", return_value={}),
            patch.object(service, "save_metadata"),
        ):
            result = await service.perform_incremental_update()

            assert result is True
            mock_qdrant_client.add_documents.assert_called()

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_perform_incremental_update_no_data(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test incremental update when data files missing."""
        mock_data_service_class.return_value = Mock()

        service = UpdateService(qdrant_client=mock_qdrant_client)

        with (
            patch.object(service, "load_current_data", return_value=None),
            patch.object(service, "load_latest_data", return_value=None),
            patch.object(service, "perform_full_update", return_value=True),
        ):
            result = await service.perform_incremental_update()

            assert result is True

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_perform_incremental_update_no_changes(
        self, mock_data_service_class, mock_qdrant_client, sample_anime_data
    ):
        """Test incremental update when no changes detected."""
        mock_data_service_class.return_value = Mock()

        service = UpdateService(qdrant_client=mock_qdrant_client)

        with (
            patch.object(service, "load_current_data", return_value=sample_anime_data),
            patch.object(service, "load_latest_data", return_value=sample_anime_data),
        ):
            result = await service.perform_incremental_update()

            assert result is True

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_perform_incremental_update_error(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test incremental update error handling."""
        mock_data_service_class.return_value = Mock()

        service = UpdateService(qdrant_client=mock_qdrant_client)

        with patch.object(
            service, "load_current_data", side_effect=Exception("File error")
        ):
            result = await service.perform_incremental_update()

            assert result is False


class TestFullUpdate:
    """Test full update functionality."""

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_perform_full_update_success(
        self, mock_data_service_class, mock_qdrant_client, sample_anime_data
    ):
        """Test successful full update."""
        mock_data_service = Mock()
        mock_data_service.download_anime_database = AsyncMock(
            return_value=sample_anime_data
        )
        mock_data_service.process_all_anime = AsyncMock(
            return_value=[{"anime_id": "test"}]
        )
        mock_data_service_class.return_value = mock_data_service

        service = UpdateService(qdrant_client=mock_qdrant_client)
        service.data_service = mock_data_service

        with (
            patch("builtins.open", mock_open()),
            patch.object(Path, "mkdir"),
            patch("json.dump"),
            patch.object(service, "save_metadata"),
        ):
            result = await service.perform_full_update()

            assert result is True
            mock_qdrant_client.clear_index.assert_called_once()
            mock_qdrant_client.add_documents.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_perform_full_update_error(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test full update error handling."""
        mock_data_service = Mock()
        mock_data_service.download_anime_database = AsyncMock(
            side_effect=Exception("Download failed")
        )
        mock_data_service_class.return_value = mock_data_service

        service = UpdateService(qdrant_client=mock_qdrant_client)
        service.data_service = mock_data_service

        result = await service.perform_full_update()
        assert result is False


class TestScheduledUpdate:
    """Test scheduled update functionality."""

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_schedule_weekly_update_with_updates(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test weekly update when updates are available."""
        mock_data_service_class.return_value = Mock()

        service = UpdateService(qdrant_client=mock_qdrant_client)

        with (
            patch.object(service, "check_for_updates", return_value=True),
            patch.object(service, "perform_incremental_update", return_value=True),
        ):
            await service.schedule_weekly_update()

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_schedule_weekly_update_no_updates(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test weekly update when no updates available."""
        mock_data_service_class.return_value = Mock()

        service = UpdateService(qdrant_client=mock_qdrant_client)

        with patch.object(service, "check_for_updates", return_value=False):
            await service.schedule_weekly_update()

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_schedule_weekly_update_with_failed_update(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test weekly update when update fails."""
        mock_data_service_class.return_value = Mock()

        service = UpdateService(qdrant_client=mock_qdrant_client)

        with (
            patch.object(service, "check_for_updates", return_value=True),
            patch.object(service, "perform_incremental_update", return_value=False),
        ):
            await service.schedule_weekly_update()


class TestDataComparison:
    """Test data comparison functionality."""

    @patch("src.services.update_service.AnimeDataService")
    def test_compare_datasets_added_entries(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test comparing datasets with added entries."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        old_data = {"data": [{"title": "Old Anime", "sources": ["url1"]}]}
        new_data = {
            "data": [
                {"title": "Old Anime", "sources": ["url1"]},
                {"title": "New Anime", "sources": ["url2"]},
            ]
        }

        changes = service.compare_datasets(old_data, new_data)

        assert len(changes["added"]) == 1
        assert len(changes["removed"]) == 0
        assert changes["added"][0]["title"] == "New Anime"

    @patch("src.services.update_service.AnimeDataService")
    def test_compare_datasets_removed_entries(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test comparing datasets with removed entries."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        old_data = {
            "data": [
                {"title": "Old Anime", "sources": ["url1"]},
                {"title": "Removed Anime", "sources": ["url2"]},
            ]
        }
        new_data = {"data": [{"title": "Old Anime", "sources": ["url1"]}]}

        changes = service.compare_datasets(old_data, new_data)

        assert len(changes["added"]) == 0
        assert len(changes["removed"]) == 1
        assert changes["removed"][0]["title"] == "Removed Anime"

    @patch("src.services.update_service.AnimeDataService")
    def test_compare_datasets_modified_entries(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test comparing datasets with modified entries."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        old_data = {"data": [{"title": "Anime", "sources": ["url1"], "episodes": 12}]}
        new_data = {"data": [{"title": "Anime", "sources": ["url1"], "episodes": 24}]}

        changes = service.compare_datasets(old_data, new_data)

        assert len(changes["added"]) == 0
        assert len(changes["removed"]) == 0
        assert len(changes["modified"]) == 1


class TestUtilityMethods:
    """Test utility methods."""

    @patch("src.services.update_service.AnimeDataService")
    def test_get_entry_key(self, mock_data_service_class, mock_qdrant_client):
        """Test entry key generation."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        entry = {"title": "Test Anime", "sources": ["https://example.com/1"]}
        key = service._get_entry_key(entry)

        assert key == "Test Anime_https://example.com/1"

    @patch("src.services.update_service.AnimeDataService")
    def test_entry_hash(self, mock_data_service_class, mock_qdrant_client):
        """Test entry hash generation."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        entry = {"title": "Test Anime", "episodes": 12, "status": "finished"}
        hash1 = service._entry_hash(entry)
        hash2 = service._entry_hash(entry)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length


class TestFileOperations:
    """Test file operation methods."""

    @patch("src.services.update_service.AnimeDataService")
    def test_load_metadata_exists(self, mock_data_service_class, mock_qdrant_client):
        """Test loading metadata when file exists."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        metadata = {"last_update": "2024-01-01", "content_hash": "abc123"}

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(metadata))),
            patch("json.load", return_value=metadata),
        ):
            result = service.load_metadata()

            assert result == metadata

    @patch("src.services.update_service.AnimeDataService")
    def test_load_metadata_not_exists(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test loading metadata when file doesn't exist."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        with patch("pathlib.Path.exists", return_value=False):
            result = service.load_metadata()

            assert result == {}

    @patch("src.services.update_service.AnimeDataService")
    def test_save_metadata(self, mock_data_service_class, mock_qdrant_client):
        """Test saving metadata."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        metadata = {"last_update": "2024-01-01"}

        with (
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump") as mock_json_dump,
        ):
            service.save_metadata(metadata)

            mock_json_dump.assert_called_once()

    @patch("src.services.update_service.AnimeDataService")
    def test_load_current_data(self, mock_data_service_class, mock_qdrant_client):
        """Test loading current data."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        data = {"data": [{"title": "Test"}]}
        service.data_dir / "raw" / "anime-offline-database.json"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open()),
            patch("json.load", return_value=data),
        ):
            result = service.load_current_data()

            assert result == data

    @patch("src.services.update_service.AnimeDataService")
    def test_load_current_data_not_exists(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test loading current data when file doesn't exist."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        service.data_dir / "raw" / "anime-offline-database.json"

        with patch("pathlib.Path.exists", return_value=False):
            result = service.load_current_data()

            assert result == {}

    @patch("src.services.update_service.AnimeDataService")
    def test_load_latest_data(self, mock_data_service_class, mock_qdrant_client):
        """Test loading latest data."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        data = {"data": [{"title": "Test"}]}
        service.data_dir / "raw" / "anime-offline-database-latest.json"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open()),
            patch("json.load", return_value=data),
        ):
            result = service.load_latest_data()

            assert result == data

    @patch("src.services.update_service.AnimeDataService")
    def test_load_latest_data_not_exists(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test loading latest data when file doesn't exist."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        service.data_dir / "raw" / "anime-offline-database-latest.json"

        with patch("pathlib.Path.exists", return_value=False):
            result = service.load_latest_data()

            assert result == {}


class TestEntryRemoval:
    """Test entry removal functionality."""

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_remove_entries_success(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test successful entry removal."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        entries = [
            {"anime_id": "anime_1", "title": "Test 1"},
            {"anime_id": "anime_2", "title": "Test 2"},
        ]

        # Mock successful deletion
        mock_delete_result = Mock()
        mock_delete_result.status = "completed"
        mock_qdrant_client.client.delete.return_value = mock_delete_result

        result = await service.remove_entries(entries)

        assert result is True
        mock_qdrant_client.client.delete.assert_called()

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_remove_entries_empty_list(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test removing empty entry list."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        result = await service.remove_entries([])

        assert result is True

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_remove_entries_missing_anime_id(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test removing entries without anime_id."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        entries = [{"title": "Test", "sources": ["url1"]}]  # No anime_id

        result = await service.remove_entries(entries)

        assert result is False

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_remove_entries_client_error(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test entry removal with client error."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        entries = [{"anime_id": "anime_1", "title": "Test"}]
        mock_qdrant_client.client.delete.side_effect = Exception("Delete failed")
        mock_qdrant_client._generate_point_id.return_value = "point_id"

        result = await service.remove_entries(entries)
        # Returns False when success rate < 80% (0% in this case)
        assert result is False

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_remove_entries_partial_failure(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test entry removal with partial failure but high success rate."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        # Create 10 entries to trigger batch processing
        entries = [{"anime_id": f"anime_{i}", "title": f"Test {i}"} for i in range(10)]

        # Mock first batch success, second batch failure
        mock_delete_result_success = Mock()
        mock_delete_result_success.status = "completed"
        mock_delete_result_failure = Mock()
        mock_delete_result_failure.status = "failed"

        mock_qdrant_client.client.delete.side_effect = [
            mock_delete_result_success
        ] * 8 + [mock_delete_result_failure] * 2
        mock_qdrant_client._generate_point_id.side_effect = lambda x: f"point_{x}"

        result = await service.remove_entries(entries)
        # Should return True as 80% success rate still passes
        assert result is True

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_remove_entries_batch_status_failure(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test entry removal when batch delete returns non-completed status."""
        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        entries = [{"anime_id": "anime_1", "title": "Test"}]

        # Mock delete result with non-completed status
        mock_delete_result = Mock()
        mock_delete_result.status = "failed"
        mock_qdrant_client.client.delete.return_value = mock_delete_result
        mock_qdrant_client._generate_point_id.return_value = "point_1"

        result = await service.remove_entries(entries)
        # Should return False as 0% success rate (< 80%)
        assert result is False

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_remove_entries_critical_exception(
        self, mock_data_service_class, mock_qdrant_client
    ):
        """Test entry removal with critical exception that should raise VectorDatabaseError."""
        from src.exceptions import VectorDatabaseError

        mock_data_service_class.return_value = Mock()
        service = UpdateService(qdrant_client=mock_qdrant_client)

        # Mock range() to raise an exception before batch processing starts
        with patch("builtins.range", side_effect=Exception("Critical system error")):
            entries = [{"anime_id": "anime_1", "title": "Test"}]

            with pytest.raises(VectorDatabaseError):
                await service.remove_entries(entries)


class TestUpdateProcessedData:
    """Test update processed data functionality."""

    @pytest.mark.asyncio
    @patch("src.services.update_service.AnimeDataService")
    async def test_update_processed_data(
        self, mock_data_service_class, mock_qdrant_client, sample_anime_data
    ):
        """Test updating processed data."""
        mock_data_service = Mock()
        mock_data_service.process_all_anime = AsyncMock(
            return_value=[{"anime_id": "test"}]
        )
        mock_data_service_class.return_value = mock_data_service

        service = UpdateService(qdrant_client=mock_qdrant_client)
        service.data_service = mock_data_service

        with patch("builtins.open", mock_open()), patch("json.dump"):
            await service.update_processed_data(sample_anime_data)

            mock_data_service.process_all_anime.assert_called_once_with(
                sample_anime_data
            )
