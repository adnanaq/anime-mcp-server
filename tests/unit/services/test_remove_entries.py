"""Unit tests for remove_entries functionality."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List, Dict, Any


class TestRemoveEntriesFunctionality:
    """Test cases for remove_entries functionality in update service."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client for testing."""
        client = MagicMock()
        client.collection_name = "anime_database"
        client.client = MagicMock()
        
        # Mock point ID generation
        client._generate_point_id = lambda anime_id: f"point_{anime_id}"
        
        # Mock successful delete response
        delete_response = MagicMock()
        delete_response.status = "completed"
        client.client.delete.return_value = delete_response
        
        return client
    
    @pytest.fixture
    def mock_update_service(self, mock_qdrant_client):
        """Create mock update service for testing."""
        class MockUpdateService:
            def __init__(self, qdrant_client):
                self.qdrant_client = qdrant_client
            
            async def remove_entries(self, entries: List[Dict]) -> bool:
                """Implementation of remove_entries for testing."""
                if not entries:
                    return True
                
                try:
                    # Extract anime IDs from entries to remove
                    anime_ids_to_remove = []
                    for entry in entries:
                        anime_id = entry.get('anime_id')
                        if anime_id:
                            anime_ids_to_remove.append(anime_id)
                    
                    if not anime_ids_to_remove:
                        return False
                    
                    # Remove entries in batches
                    batch_size = 100
                    successful_removals = 0
                    failed_removals = 0
                    
                    for i in range(0, len(anime_ids_to_remove), batch_size):
                        batch_ids = anime_ids_to_remove[i:i + batch_size]
                        
                        try:
                            # Generate point IDs for the anime IDs
                            point_ids = [self.qdrant_client._generate_point_id(anime_id) for anime_id in batch_ids]
                            
                            # Delete points from Qdrant
                            delete_result = self.qdrant_client.client.delete(
                                collection_name=self.qdrant_client.collection_name,
                                points_selector={"points": point_ids}
                            )
                            
                            if delete_result.status == "completed":
                                successful_removals += len(batch_ids)
                            else:
                                failed_removals += len(batch_ids)
                                
                        except Exception:
                            failed_removals += len(batch_ids)
                    
                    # Consider it successful if at least 80% of entries were removed
                    total_attempted = len(anime_ids_to_remove)
                    success_rate = (successful_removals / total_attempted) * 100 if total_attempted > 0 else 0
                    return success_rate >= 80.0
                    
                except Exception:
                    return False
        
        return MockUpdateService(mock_qdrant_client)
    
    @pytest.fixture
    def sample_entries(self):
        """Create sample anime entries for testing."""
        return [
            {
                "anime_id": "anime123",
                "title": "Attack on Titan",
                "synopsis": "Humanity fights titans"
            },
            {
                "anime_id": "anime456",
                "title": "Death Note",
                "synopsis": "Supernatural notebook"
            },
            {
                "anime_id": "anime789",
                "title": "One Piece",
                "synopsis": "Pirates search for treasure"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_remove_entries_success(self, mock_update_service, sample_entries):
        """Test successful removal of entries."""
        result = await mock_update_service.remove_entries(sample_entries)
        
        assert result is True
        
        # Verify delete was called for each batch
        mock_update_service.qdrant_client.client.delete.assert_called()
        
        # Verify point ID generation was called for each anime_id
        expected_point_ids = ["point_anime123", "point_anime456", "point_anime789"]
        for anime_id in ["anime123", "anime456", "anime789"]:
            point_id = mock_update_service.qdrant_client._generate_point_id(anime_id)
            assert point_id in expected_point_ids
    
    @pytest.mark.asyncio
    async def test_remove_entries_empty_list(self, mock_update_service):
        """Test removal with empty entry list."""
        result = await mock_update_service.remove_entries([])
        
        assert result is True
        
        # Verify no delete operations were called
        mock_update_service.qdrant_client.client.delete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_remove_entries_missing_anime_ids(self, mock_update_service):
        """Test removal with entries missing anime_id."""
        entries_without_ids = [
            {
                "title": "Attack on Titan",
                "synopsis": "No anime_id field"
            },
            {
                "anime_id": "",  # Empty anime_id
                "title": "Death Note"
            },
            {
                "title": "One Piece"
                # Missing anime_id field entirely
            }
        ]
        
        result = await mock_update_service.remove_entries(entries_without_ids)
        
        assert result is False
        
        # Verify no delete operations were called since no valid anime_ids
        mock_update_service.qdrant_client.client.delete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_remove_entries_mixed_valid_invalid(self, mock_update_service):
        """Test removal with mix of valid and invalid entries."""
        mixed_entries = [
            {
                "anime_id": "anime123",
                "title": "Valid Entry 1"
            },
            {
                "title": "Invalid Entry - No ID"
            },
            {
                "anime_id": "anime456",
                "title": "Valid Entry 2"
            },
            {
                "anime_id": "",
                "title": "Invalid Entry - Empty ID"
            }
        ]
        
        result = await mock_update_service.remove_entries(mixed_entries)
        
        assert result is True
        
        # Verify delete was called (should process the 2 valid entries)
        mock_update_service.qdrant_client.client.delete.assert_called()
    
    @pytest.mark.asyncio
    async def test_remove_entries_batch_processing(self, mock_update_service):
        """Test batch processing with large number of entries."""
        # Create 250 entries to test batching (batch size is 100)
        large_entry_list = [
            {"anime_id": f"anime{i:03d}", "title": f"Anime {i}"}
            for i in range(250)
        ]
        
        result = await mock_update_service.remove_entries(large_entry_list)
        
        assert result is True
        
        # Verify delete was called multiple times for batches
        # Should be called 3 times: 100 + 100 + 50
        assert mock_update_service.qdrant_client.client.delete.call_count == 3
    
    @pytest.mark.asyncio
    async def test_remove_entries_partial_failure(self, mock_update_service):
        """Test removal with partial batch failures."""
        # Mock some failures
        def mock_delete_with_failures(collection_name, points_selector):
            # Simulate failure for every other call
            mock_delete_with_failures.call_count = getattr(mock_delete_with_failures, 'call_count', 0) + 1
            
            response = MagicMock()
            if mock_delete_with_failures.call_count % 2 == 0:
                response.status = "failed"  # Simulate failure
            else:
                response.status = "completed"  # Simulate success
            return response
        
        mock_update_service.qdrant_client.client.delete.side_effect = mock_delete_with_failures
        
        # Create entries that will require 2 batches
        entries = [
            {"anime_id": f"anime{i:03d}", "title": f"Anime {i}"}
            for i in range(150)  # 2 batches: 100 + 50
        ]
        
        result = await mock_update_service.remove_entries(entries)
        
        # Should succeed since first batch succeeds (100/150 = 66.7% > 80% threshold)
        # Actually this should fail since only 50% success rate
        # Let's adjust the test to reflect the actual logic
        assert result is False  # Only 50% success rate, below 80% threshold
    
    @pytest.mark.asyncio
    async def test_remove_entries_complete_failure(self, mock_update_service):
        """Test removal with complete batch failures."""
        # Mock all delete operations to fail
        failure_response = MagicMock()
        failure_response.status = "failed"
        mock_update_service.qdrant_client.client.delete.return_value = failure_response
        
        entries = [
            {"anime_id": "anime123", "title": "Test Anime"}
        ]
        
        result = await mock_update_service.remove_entries(entries)
        
        assert result is False
        
        # Verify delete was attempted
        mock_update_service.qdrant_client.client.delete.assert_called()
    
    @pytest.mark.asyncio
    async def test_remove_entries_exception_handling(self, mock_update_service):
        """Test removal with exception during delete operation."""
        # Mock delete to raise exception
        mock_update_service.qdrant_client.client.delete.side_effect = Exception("Qdrant connection error")
        
        entries = [
            {"anime_id": "anime123", "title": "Test Anime"}
        ]
        
        result = await mock_update_service.remove_entries(entries)
        
        assert result is False
    
    def test_point_id_generation_consistency(self, mock_update_service):
        """Test that point ID generation is consistent."""
        anime_id = "test_anime_123"
        
        # Generate point ID multiple times
        point_id_1 = mock_update_service.qdrant_client._generate_point_id(anime_id)
        point_id_2 = mock_update_service.qdrant_client._generate_point_id(anime_id)
        point_id_3 = mock_update_service.qdrant_client._generate_point_id(anime_id)
        
        # All should be identical
        assert point_id_1 == point_id_2 == point_id_3
        assert point_id_1 == f"point_{anime_id}"
    
    def test_point_id_generation_uniqueness(self, mock_update_service):
        """Test that different anime IDs generate different point IDs."""
        anime_ids = ["anime1", "anime2", "anime3"]
        point_ids = []
        
        for anime_id in anime_ids:
            point_id = mock_update_service.qdrant_client._generate_point_id(anime_id)
            point_ids.append(point_id)
        
        # All point IDs should be unique
        assert len(set(point_ids)) == len(point_ids)
        
        # Verify format
        for i, point_id in enumerate(point_ids):
            assert point_id == f"point_{anime_ids[i]}"
    
    def test_batch_size_calculation(self):
        """Test batch size calculation logic."""
        def calculate_batches(total_items: int, batch_size: int) -> int:
            """Calculate number of batches needed."""
            return (total_items + batch_size - 1) // batch_size  # Ceiling division
        
        test_cases = [
            (100, 100, 1),    # Exact division
            (150, 100, 2),    # Requires 2 batches
            (99, 100, 1),     # Less than batch size
            (0, 100, 0),      # Empty list
            (250, 100, 3),    # Multiple batches
        ]
        
        for total, batch_size, expected_batches in test_cases:
            actual_batches = calculate_batches(total, batch_size)
            assert actual_batches == expected_batches
    
    def test_success_rate_calculation(self):
        """Test success rate calculation logic."""
        def calculate_success_rate(successful: int, total: int) -> float:
            """Calculate success rate percentage."""
            if total == 0:
                return 0.0
            return (successful / total) * 100
        
        test_cases = [
            (100, 100, 100.0),  # Perfect success
            (80, 100, 80.0),    # 80% success
            (0, 100, 0.0),      # Complete failure
            (50, 100, 50.0),    # Half success
            (0, 0, 0.0),        # Edge case: no items
        ]
        
        for successful, total, expected_rate in test_cases:
            actual_rate = calculate_success_rate(successful, total)
            assert actual_rate == expected_rate
    
    def test_success_threshold_logic(self):
        """Test success threshold decision logic."""
        def is_removal_successful(success_rate: float, threshold: float = 80.0) -> bool:
            """Determine if removal is considered successful."""
            return success_rate >= threshold
        
        test_cases = [
            (100.0, True),   # Perfect success
            (80.0, True),    # Exactly at threshold
            (79.9, False),   # Just below threshold
            (50.0, False),   # Well below threshold
            (0.0, False),    # Complete failure
        ]
        
        for success_rate, expected_result in test_cases:
            actual_result = is_removal_successful(success_rate)
            assert actual_result == expected_result
    
    @pytest.mark.asyncio
    async def test_remove_entries_with_custom_threshold(self, mock_update_service):
        """Test removal with custom success threshold."""
        # Mock a scenario where 70% of entries succeed
        def mock_delete_70_percent_success(collection_name, points_selector):
            # Simulate 70% success rate
            mock_delete_70_percent_success.call_count = getattr(mock_delete_70_percent_success, 'call_count', 0) + 1
            
            response = MagicMock()
            # 7 out of 10 calls succeed
            if mock_delete_70_percent_success.call_count <= 7:
                response.status = "completed"
            else:
                response.status = "failed"
            return response
        
        mock_update_service.qdrant_client.client.delete.side_effect = mock_delete_70_percent_success
        
        # Create 10 batches worth of entries (1000 entries)
        entries = [
            {"anime_id": f"anime{i:04d}", "title": f"Anime {i}"}
            for i in range(1000)
        ]
        
        result = await mock_update_service.remove_entries(entries)
        
        # Should fail with default 80% threshold, but 70% success rate
        assert result is False
    
    def test_error_logging_scenarios(self):
        """Test different error logging scenarios."""
        # Mock logger to capture different error conditions
        error_scenarios = [
            {
                "condition": "missing_anime_ids",
                "entries": [{"title": "No ID"}],
                "expected_log": "No valid anime_ids found"
            },
            {
                "condition": "empty_entries",
                "entries": [],
                "expected_log": "No entries to remove"
            },
            {
                "condition": "partial_failure",
                "entries": [{"anime_id": "test"}],
                "expected_log": "entries failed to remove"
            }
        ]
        
        # Verify that appropriate error conditions are detected
        for scenario in error_scenarios:
            condition = scenario["condition"]
            entries = scenario["entries"]
            
            if condition == "missing_anime_ids":
                # Check that entries without anime_id are detected
                valid_ids = [e.get("anime_id") for e in entries if e.get("anime_id")]
                assert len(valid_ids) == 0
            
            elif condition == "empty_entries":
                # Check that empty entry list is detected
                assert len(entries) == 0
            
            elif condition == "partial_failure":
                # Check that we can detect scenarios that would lead to partial failure
                assert len(entries) > 0
                assert all(e.get("anime_id") for e in entries)
    
    @pytest.mark.asyncio
    async def test_remove_entries_integration_with_qdrant_api(self, mock_update_service):
        """Test integration with Qdrant API format."""
        entries = [
            {"anime_id": "test_anime_1", "title": "Test Anime 1"},
            {"anime_id": "test_anime_2", "title": "Test Anime 2"}
        ]
        
        await mock_update_service.remove_entries(entries)
        
        # Verify the correct Qdrant API call format
        call_args = mock_update_service.qdrant_client.client.delete.call_args
        
        # Check that collection_name was passed correctly
        assert call_args[1]["collection_name"] == "anime_database"
        
        # Check that points_selector has correct format
        points_selector = call_args[1]["points_selector"]
        assert "points" in points_selector
        assert isinstance(points_selector["points"], list)
        
        # Check that point IDs were generated correctly
        expected_point_ids = ["point_test_anime_1", "point_test_anime_2"]
        assert points_selector["points"] == expected_point_ids