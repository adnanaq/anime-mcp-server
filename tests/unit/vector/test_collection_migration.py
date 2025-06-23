"""Tests for Qdrant collection migration to multi-vector support.

This module tests the migration process from single-vector (text-only) to 
multi-vector (text + image) collections while preserving all existing data.

Following TDD approach: tests written first to define expected behavior.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from qdrant_client.models import VectorParams, Distance
from src.vector.qdrant_client import QdrantClient
from src.config import Settings


class TestCollectionMigration:
    """Test collection migration from single to multi-vector."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = MagicMock(spec=Settings)
        settings.qdrant_url = "http://localhost:6333"
        settings.qdrant_collection_name = "test_anime_database"
        settings.qdrant_vector_size = 384
        settings.qdrant_distance_metric = "cosine"
        settings.fastembed_model = "BAAI/bge-small-en-v1.5"
        settings.enable_multi_vector = True
        settings.image_vector_size = 512
        settings.clip_model = "ViT-B/32"
        return settings

    @pytest.fixture
    def mock_qdrant_sdk(self):
        """Mock Qdrant SDK client."""
        with patch('src.vector.qdrant_client.QdrantSDK') as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def qdrant_client(self, mock_settings, mock_qdrant_sdk):
        """QdrantClient instance with mocked dependencies."""
        return QdrantClient(settings=mock_settings)

    @pytest.mark.asyncio
    async def test_migration_preserves_existing_data(self, qdrant_client, mock_qdrant_sdk):
        """Test that migration preserves all existing text vectors."""
        # Setup: Mock existing single-vector collection
        mock_qdrant_sdk.collection_exists.return_value = True
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors = MagicMock()
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors.size = 384
        
        # Mock existing points in collection  
        mock_points = [
            MagicMock(id="anime1", vector=[0.1] * 384, payload={"title": "Test Anime 1"}),
            MagicMock(id="anime2", vector=[0.2] * 384, payload={"title": "Test Anime 2"})
        ]
        mock_qdrant_sdk.scroll.return_value = (mock_points, None)
        
        # Mock stats for verification
        qdrant_client.get_stats = AsyncMock(return_value={"total_documents": 2})
        
        # Execute migration
        result = await qdrant_client.migrate_to_multi_vector()
        
        # Verify: All existing data preserved
        assert result["preserved_vectors"] == 2
        assert result["migration_successful"] is True
        assert "backup_collection" in result
        
        # Verify collection recreation with multi-vector config
        expected_vectors_config = {
            "text": VectorParams(size=384, distance=Distance.COSINE),
            "image": VectorParams(size=512, distance=Distance.COSINE)
        }
        mock_qdrant_sdk.recreate_collection.assert_called_once()
        call_args = mock_qdrant_sdk.recreate_collection.call_args
        assert call_args[1]["vectors_config"] == expected_vectors_config

    @pytest.mark.asyncio 
    async def test_migration_creates_backup_collection(self, qdrant_client, mock_qdrant_sdk):
        """Test that migration creates backup before proceeding."""
        # Setup: Mock collection exists
        mock_qdrant_sdk.collection_exists.return_value = True
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors = MagicMock()
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors.size = 384
        mock_qdrant_sdk.scroll.return_value = ([], None)  # Empty collection
        qdrant_client.get_stats = AsyncMock(return_value={"total_documents": 0})
        
        # Execute migration  
        await qdrant_client.migrate_to_multi_vector()
        
        # Verify: Backup collection created
        backup_calls = [call for call in mock_qdrant_sdk.create_collection.call_args_list 
                       if 'backup' in str(call)]
        assert len(backup_calls) >= 1

    @pytest.mark.asyncio
    async def test_migration_rollback_on_failure(self, qdrant_client, mock_qdrant_sdk):
        """Test that migration rolls back on failure."""
        # Setup: Mock migration failure
        mock_qdrant_sdk.collection_exists.return_value = True
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors = MagicMock()
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors.size = 384
        mock_qdrant_sdk.scroll.return_value = ([], None)
        mock_qdrant_sdk.recreate_collection.side_effect = Exception("Migration failed")
        
        # Execute migration (should handle gracefully)
        result = await qdrant_client.migrate_to_multi_vector()
        
        # Verify: Rollback occurred
        assert result["migration_successful"] is False
        assert "error" in result
        # Original collection should be restored from backup
        mock_qdrant_sdk.delete_collection.assert_called()  # Delete failed collection
        # Backup should be renamed back to original

    @pytest.mark.asyncio
    async def test_migration_validates_prerequisites(self, qdrant_client, mock_qdrant_sdk):
        """Test that migration validates prerequisites before starting."""
        # Setup: Mock missing collection
        mock_qdrant_sdk.collection_exists.return_value = False
        
        # Execute migration
        result = await qdrant_client.migrate_to_multi_vector()
        
        # Verify: Migration failed with proper error
        assert result["migration_successful"] is False
        assert "error" in result
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_migration_handles_empty_collection(self, qdrant_client, mock_qdrant_sdk):
        """Test migration of empty collection."""
        # Setup: Mock empty collection
        mock_qdrant_sdk.collection_exists.return_value = True
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors = MagicMock()
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors.size = 384
        mock_qdrant_sdk.scroll.return_value = ([], None)  # No points
        qdrant_client.get_stats = AsyncMock(return_value={"total_documents": 0})
        
        # Execute migration
        result = await qdrant_client.migrate_to_multi_vector()
        
        # Verify: Migration succeeds with zero vectors
        assert result["migration_successful"] is True
        assert result["preserved_vectors"] == 0

    @pytest.mark.asyncio
    async def test_migration_batch_processing(self, qdrant_client, mock_qdrant_sdk):
        """Test that migration processes large collections in batches."""
        # Setup: Mock large collection
        mock_qdrant_sdk.collection_exists.return_value = True
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors = MagicMock()
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors.size = 384
        
        mock_points = [MagicMock(id=f"anime{i}", vector=[0.1] * 384, payload={"title": f"Anime {i}"}) 
                      for i in range(2500)]  # More than batch size
        
        # Mock scroll to return batches
        def scroll_side_effect(collection_name, limit, **kwargs):
            offset = kwargs.get('offset', None)
            start = offset if offset is not None else 0
            end = min(start + limit, len(mock_points))
            batch = mock_points[start:end]
            next_offset = end if end < len(mock_points) else None
            return batch, next_offset
            
        mock_qdrant_sdk.scroll.side_effect = scroll_side_effect
        qdrant_client.get_stats = AsyncMock(return_value={"total_documents": 2500})
        
        # Execute migration
        result = await qdrant_client.migrate_to_multi_vector()
        
        # Verify: All points migrated in batches
        assert result["preserved_vectors"] == 2500
        # Verify multiple upsert calls (batching)
        assert mock_qdrant_sdk.upsert.call_count > 1

    @pytest.mark.asyncio
    async def test_migration_preserves_metadata(self, qdrant_client, mock_qdrant_sdk):
        """Test that migration preserves all metadata and payload."""
        # Setup: Mock points with rich metadata
        mock_qdrant_sdk.collection_exists.return_value = True
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors = MagicMock()
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors.size = 384
        
        mock_points = [
            MagicMock(
                id="anime1", 
                vector=[0.1] * 384,
                payload={
                    "title": "Test Anime",
                    "synopsis": "Test synopsis",
                    "tags": ["action", "adventure"],
                    "studios": ["Test Studio"],
                    "year": 2023,
                    "myanimelist_id": 12345,
                    "data_quality_score": 0.85
                }
            )
        ]
        mock_qdrant_sdk.scroll.return_value = (mock_points, None)
        qdrant_client.get_stats = AsyncMock(return_value={"total_documents": 1})
        
        # Execute migration
        result = await qdrant_client.migrate_to_multi_vector()
        
        # Verify: Metadata preserved in upsert
        upsert_calls = mock_qdrant_sdk.upsert.call_args_list
        assert len(upsert_calls) > 0
        
        # Check that original payload is preserved
        points = upsert_calls[0][1]["points"]
        assert points[0].payload["title"] == "Test Anime"
        assert points[0].payload["myanimelist_id"] == 12345
        assert points[0].payload["data_quality_score"] == 0.85

    @pytest.mark.asyncio
    async def test_async_migration_performance(self, qdrant_client, mock_qdrant_sdk):
        """Test that migration completes within reasonable time."""
        import time
        
        # Setup: Mock moderate size collection
        mock_qdrant_sdk.collection_exists.return_value = True
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors = MagicMock()
        mock_qdrant_sdk.get_collection.return_value.config.params.vectors.size = 384
        mock_qdrant_sdk.scroll.return_value = ([], None)  # Empty for speed
        qdrant_client.get_stats = AsyncMock(return_value={"total_documents": 0})
        
        # Execute migration with timing
        start_time = time.time()
        result = await qdrant_client.migrate_to_multi_vector_async()
        end_time = time.time()
        
        # Verify: Migration completes quickly (< 30 seconds for mocked operations)
        migration_time = end_time - start_time
        assert migration_time < 30.0
        assert result["migration_successful"] is True


class TestMigrationUtilities:
    """Test utility functions for migration process."""

    @pytest.fixture
    def qdrant_client(self, mock_settings, mock_qdrant_sdk):
        """QdrantClient instance for utility testing."""
        return QdrantClient(settings=mock_settings)

    def test_generate_backup_collection_name(self, qdrant_client):
        """Test backup collection name generation."""
        backup_name = qdrant_client._generate_backup_collection_name("anime_database")
        
        # Verify: Backup name includes timestamp and original name
        assert "anime_database" in backup_name
        assert "backup" in backup_name
        assert len(backup_name.split("_")) >= 3  # name_backup_timestamp

    def test_validate_collection_compatibility(self, qdrant_client, mock_qdrant_sdk):
        """Test collection compatibility validation."""
        # Setup: Mock compatible collection (single vector, correct size)
        mock_collection = MagicMock()
        mock_collection.config.params.vectors.size = 384
        mock_collection.config.params.vectors.distance = Distance.COSINE
        mock_qdrant_sdk.get_collection.return_value = mock_collection
        
        # Execute validation
        is_compatible = qdrant_client._validate_collection_for_migration("test_collection")
        
        # Verify: Collection is compatible
        assert is_compatible is True

    def test_validate_incompatible_collection(self, qdrant_client, mock_qdrant_sdk):
        """Test validation fails for incompatible collection."""
        # Setup: Mock incompatible collection (wrong vector size)
        mock_collection = MagicMock()
        mock_collection.config.params.vectors.size = 512  # Wrong size
        mock_qdrant_sdk.get_collection.return_value = mock_collection
        
        # Execute validation  
        is_compatible = qdrant_client._validate_collection_for_migration("test_collection")
        
        # Verify: Collection is not compatible
        assert is_compatible is False

    def test_estimate_migration_time(self, qdrant_client, mock_qdrant_sdk):
        """Test migration time estimation."""
        # Setup: Mock collection with known size
        mock_qdrant_sdk.count.return_value.count = 10000
        
        # Execute estimation
        estimated_time = qdrant_client._estimate_migration_time()
        
        # Verify: Reasonable time estimate (should be > 0)
        assert estimated_time > 0
        assert estimated_time < 3600  # Should be less than 1 hour

    def test_check_disk_space_requirements(self, qdrant_client):
        """Test disk space requirement checking."""
        # Execute disk space check
        has_space = qdrant_client._check_disk_space_requirements(collection_size_mb=1000)
        
        # Verify: Returns boolean (implementation may vary)
        assert isinstance(has_space, bool)


class TestMigrationIntegration:
    """Integration tests for the complete migration process."""

    @pytest.fixture
    def settings_with_multi_vector(self):
        """Settings with multi-vector enabled."""
        settings = Settings()
        settings.enable_multi_vector = True
        settings.image_vector_size = 512
        settings.clip_model = "ViT-B/32"
        return settings

    @pytest.mark.integration
    def test_end_to_end_migration_simulation(self, settings_with_multi_vector):
        """Test complete migration process simulation."""
        # This test would require actual Qdrant instance
        # For now, we'll skip it and mark as integration test
        pytest.skip("Requires actual Qdrant instance for integration testing")

    def test_migration_idempotency(self, qdrant_client, mock_qdrant_sdk):
        """Test that running migration twice doesn't cause issues."""
        # Setup: Mock already migrated collection (multi-vector)
        mock_collection = MagicMock()
        mock_collection.config.params.vectors = {
            "text": VectorParams(size=384, distance=Distance.COSINE),
            "image": VectorParams(size=512, distance=Distance.COSINE)
        }
        mock_qdrant_sdk.get_collection.return_value = mock_collection
        
        # Execute migration on already-migrated collection
        result = qdrant_client.migrate_to_multi_vector()
        
        # Verify: Migration detects existing multi-vector setup
        assert result["already_migrated"] is True
        assert result["migration_successful"] is True
        # Should not recreate collection
        mock_qdrant_sdk.recreate_collection.assert_not_called()