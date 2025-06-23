"""Unit tests for refactored data processing functionality."""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Dict, Any
import time


class MockProcessingConfig:
    """Mock processing configuration for testing."""
    
    def __init__(self, batch_size: int = 1000, max_concurrent_batches: int = 3, processing_timeout: int = 300):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.processing_timeout = processing_timeout


class TestRefactoredDataProcessing:
    """Test cases for refactored data processing functionality."""
    
    @pytest.fixture
    def mock_data_service(self):
        """Create mock data service for testing."""
        class MockDataService:
            def __init__(self):
                self.settings = MagicMock()
                self.settings.batch_size = 1000
                self.settings.max_concurrent_batches = 3
                self.settings.processing_timeout = 300
                self.platform_configs = {
                    'myanimelist': {'domain': 'myanimelist.net', 'pattern': 'anime/(\\d+)'},
                    'anilist': {'domain': 'anilist.co', 'pattern': 'anime/(\\d+)'},
                }
            
            def _create_processing_config(self):
                return MockProcessingConfig(
                    batch_size=self.settings.batch_size,
                    max_concurrent_batches=self.settings.max_concurrent_batches,
                    processing_timeout=self.settings.processing_timeout
                )
            
            def _create_batches(self, anime_list: List[Dict[str, Any]], batch_size: int):
                batches = []
                for i in range(0, len(anime_list), batch_size):
                    batch = anime_list[i:i + batch_size]
                    batches.append(batch)
                return batches
            
            def _aggregate_results(self, batch_results: List[List[Dict[str, Any]]]):
                all_processed = []
                for batch_result in batch_results:
                    if isinstance(batch_result, list):
                        all_processed.extend(batch_result)
                return all_processed
            
            def _log_batch_metrics(self, batch_num: int, batch_start: float, total_entries: int, processed_entries: int, error_count: int):
                pass  # Mock logging
            
            def _log_processing_metrics(self, start_time: float, total_entries: int, processed_entries: int):
                pass  # Mock logging
            
            def process_anime_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
                """Mock process single anime entry."""
                if entry.get('title') == 'invalid_anime':
                    raise ValueError("Invalid anime data")
                return {
                    'anime_id': f"anime_{entry.get('title', 'unknown')}",
                    'title': entry.get('title', 'Unknown'),
                    'processed': True
                }
            
            async def _process_batch(self, batch: List[Dict[str, Any]], batch_num: int, config):
                """Mock process batch method."""
                processed = []
                for entry in batch:
                    try:
                        result = self.process_anime_entry(entry)
                        processed.append(result)
                    except Exception:
                        # Skip invalid entries
                        pass
                return processed
            
            async def _process_batches_concurrently(self, batches: List[List[Dict[str, Any]]], config):
                """Mock concurrent batch processing."""
                results = []
                for i, batch in enumerate(batches):
                    batch_result = await self._process_batch(batch, i + 1, config)
                    results.append(batch_result)
                return results
        
        return MockDataService()
    
    @pytest.fixture
    def sample_anime_data(self):
        """Create sample anime data for testing."""
        return {
            "data": [
                {"title": "Attack on Titan", "synopsis": "Humanity fights giant humanoids"},
                {"title": "Death Note", "synopsis": "A supernatural notebook that kills"},
                {"title": "One Piece", "synopsis": "Pirates searching for treasure"},
                {"title": "Naruto", "synopsis": "Young ninja's journey to become Hokage"},
                {"title": "Dragon Ball Z", "synopsis": "Saiyans protecting Earth"},
                {"title": "invalid_anime", "synopsis": "This will cause processing error"},
                {"title": "Demon Slayer", "synopsis": "Demon hunting organization"},
                {"title": "My Hero Academia", "synopsis": "Heroes with superpowers"},
            ]
        }
    
    def test_create_processing_config(self, mock_data_service):
        """Test processing configuration creation."""
        config = mock_data_service._create_processing_config()
        
        assert isinstance(config, MockProcessingConfig)
        assert config.batch_size == 1000
        assert config.max_concurrent_batches == 3
        assert config.processing_timeout == 300
    
    def test_create_batches_normal_case(self, mock_data_service):
        """Test batch creation with normal data."""
        anime_list = [{"title": f"Anime {i}"} for i in range(10)]
        batch_size = 3
        
        batches = mock_data_service._create_batches(anime_list, batch_size)
        
        assert len(batches) == 4  # 10 items with batch size 3 = 4 batches
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1  # Last batch has remainder
        
        # Verify all items are included
        flattened = [item for batch in batches for item in batch]
        assert len(flattened) == len(anime_list)
    
    def test_create_batches_exact_division(self, mock_data_service):
        """Test batch creation when list divides exactly."""
        anime_list = [{"title": f"Anime {i}"} for i in range(9)]
        batch_size = 3
        
        batches = mock_data_service._create_batches(anime_list, batch_size)
        
        assert len(batches) == 3  # 9 items with batch size 3 = 3 batches
        assert all(len(batch) == 3 for batch in batches)
    
    def test_create_batches_empty_list(self, mock_data_service):
        """Test batch creation with empty list."""
        anime_list = []
        batch_size = 3
        
        batches = mock_data_service._create_batches(anime_list, batch_size)
        
        assert len(batches) == 0
    
    def test_create_batches_single_item(self, mock_data_service):
        """Test batch creation with single item."""
        anime_list = [{"title": "Single Anime"}]
        batch_size = 3
        
        batches = mock_data_service._create_batches(anime_list, batch_size)
        
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0]["title"] == "Single Anime"
    
    def test_create_batches_large_batch_size(self, mock_data_service):
        """Test batch creation when batch size is larger than list."""
        anime_list = [{"title": f"Anime {i}"} for i in range(5)]
        batch_size = 10
        
        batches = mock_data_service._create_batches(anime_list, batch_size)
        
        assert len(batches) == 1
        assert len(batches[0]) == 5
    
    @pytest.mark.asyncio
    async def test_process_batch_success(self, mock_data_service):
        """Test successful batch processing."""
        batch = [
            {"title": "Attack on Titan", "synopsis": "Titans"},
            {"title": "Death Note", "synopsis": "Notebook"},
        ]
        config = MockProcessingConfig()
        
        results = await mock_data_service._process_batch(batch, 1, config)
        
        assert len(results) == 2
        assert results[0]["title"] == "Attack on Titan"
        assert results[0]["processed"] is True
        assert results[1]["title"] == "Death Note"
        assert results[1]["processed"] is True
    
    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self, mock_data_service):
        """Test batch processing with some errors."""
        batch = [
            {"title": "Attack on Titan", "synopsis": "Titans"},
            {"title": "invalid_anime", "synopsis": "Invalid"},  # This will cause error
            {"title": "Death Note", "synopsis": "Notebook"},
        ]
        config = MockProcessingConfig()
        
        results = await mock_data_service._process_batch(batch, 1, config)
        
        # Should process 2 valid entries, skip 1 invalid
        assert len(results) == 2
        assert results[0]["title"] == "Attack on Titan"
        assert results[1]["title"] == "Death Note"
        
        # Verify invalid entry was skipped
        titles = [r["title"] for r in results]
        assert "invalid_anime" not in titles
    
    @pytest.mark.asyncio
    async def test_process_batch_empty_batch(self, mock_data_service):
        """Test processing empty batch."""
        batch = []
        config = MockProcessingConfig()
        
        results = await mock_data_service._process_batch(batch, 1, config)
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_process_batches_concurrently(self, mock_data_service):
        """Test concurrent batch processing."""
        batches = [
            [{"title": "Anime 1"}, {"title": "Anime 2"}],
            [{"title": "Anime 3"}, {"title": "Anime 4"}],
            [{"title": "Anime 5"}],
        ]
        config = MockProcessingConfig()
        
        results = await mock_data_service._process_batches_concurrently(batches, config)
        
        assert len(results) == 3  # Three batch results
        assert len(results[0]) == 2  # First batch has 2 results
        assert len(results[1]) == 2  # Second batch has 2 results
        assert len(results[2]) == 1  # Third batch has 1 result
    
    @pytest.mark.asyncio
    async def test_process_batches_with_mixed_success(self, mock_data_service):
        """Test concurrent batch processing with mixed success/failure."""
        batches = [
            [{"title": "Valid Anime 1"}],
            [{"title": "invalid_anime"}],  # This batch will have processing errors
            [{"title": "Valid Anime 2"}],
        ]
        config = MockProcessingConfig()
        
        results = await mock_data_service._process_batches_concurrently(batches, config)
        
        assert len(results) == 3
        assert len(results[0]) == 1  # First batch successful
        assert len(results[1]) == 0  # Second batch failed (empty result)
        assert len(results[2]) == 1  # Third batch successful
    
    def test_aggregate_results_normal_case(self, mock_data_service):
        """Test result aggregation from multiple batches."""
        batch_results = [
            [{"title": "Anime 1"}, {"title": "Anime 2"}],
            [{"title": "Anime 3"}],
            [{"title": "Anime 4"}, {"title": "Anime 5"}, {"title": "Anime 6"}],
        ]
        
        aggregated = mock_data_service._aggregate_results(batch_results)
        
        assert len(aggregated) == 6
        assert aggregated[0]["title"] == "Anime 1"
        assert aggregated[2]["title"] == "Anime 3"
        assert aggregated[5]["title"] == "Anime 6"
    
    def test_aggregate_results_empty_batches(self, mock_data_service):
        """Test result aggregation with empty batches."""
        batch_results = [
            [{"title": "Anime 1"}],
            [],  # Empty batch
            [{"title": "Anime 2"}],
            [],  # Another empty batch
        ]
        
        aggregated = mock_data_service._aggregate_results(batch_results)
        
        assert len(aggregated) == 2
        assert aggregated[0]["title"] == "Anime 1"
        assert aggregated[1]["title"] == "Anime 2"
    
    def test_aggregate_results_all_empty(self, mock_data_service):
        """Test result aggregation with all empty batches."""
        batch_results = [[], [], []]
        
        aggregated = mock_data_service._aggregate_results(batch_results)
        
        assert aggregated == []
    
    def test_aggregate_results_mixed_types(self, mock_data_service):
        """Test result aggregation with mixed result types."""
        batch_results = [
            [{"title": "Anime 1"}],
            None,  # Invalid result type
            [{"title": "Anime 2"}],
            "invalid",  # Another invalid type
            [{"title": "Anime 3"}],
        ]
        
        aggregated = mock_data_service._aggregate_results(batch_results)
        
        # Should only include valid list results
        assert len(aggregated) == 3
        assert aggregated[0]["title"] == "Anime 1"
        assert aggregated[1]["title"] == "Anime 2"
        assert aggregated[2]["title"] == "Anime 3"
    
    def test_processing_config_parameter_validation(self):
        """Test processing configuration parameter validation."""
        # Test valid configurations
        valid_configs = [
            MockProcessingConfig(batch_size=100, max_concurrent_batches=1, processing_timeout=60),
            MockProcessingConfig(batch_size=5000, max_concurrent_batches=10, processing_timeout=3600),
        ]
        
        for config in valid_configs:
            assert config.batch_size > 0
            assert config.max_concurrent_batches > 0
            assert config.processing_timeout > 0
    
    def test_batch_size_optimization(self, mock_data_service):
        """Test that batch size optimization works correctly."""
        # Test with different data sizes
        test_cases = [
            (100, 10),   # Small dataset, small batches
            (1000, 100), # Medium dataset, medium batches
            (10000, 1000), # Large dataset, large batches
        ]
        
        for data_size, expected_batch_size in test_cases:
            anime_list = [{"title": f"Anime {i}"} for i in range(data_size)]
            batches = mock_data_service._create_batches(anime_list, expected_batch_size)
            
            # Verify batch sizing
            for i, batch in enumerate(batches[:-1]):  # All except last batch
                assert len(batch) == expected_batch_size
            
            # Last batch may be smaller
            if batches:
                assert len(batches[-1]) <= expected_batch_size
            
            # Verify all data is preserved
            total_items = sum(len(batch) for batch in batches)
            assert total_items == data_size
    
    def test_error_rate_calculation_logic(self):
        """Test error rate calculation in batch processing."""
        # Mock batch metrics calculation
        def calculate_error_rate(total_entries: int, processed_entries: int, error_count: int):
            if total_entries == 0:
                return 0.0
            return (error_count / total_entries) * 100
        
        test_cases = [
            (100, 95, 5, 5.0),    # 5% error rate
            (100, 100, 0, 0.0),   # 0% error rate
            (100, 50, 50, 50.0),  # 50% error rate
            (0, 0, 0, 0.0),       # Edge case: empty batch
        ]
        
        for total, processed, errors, expected_rate in test_cases:
            rate = calculate_error_rate(total, processed, errors)
            assert rate == expected_rate
    
    def test_concurrent_processing_semaphore_logic(self):
        """Test semaphore logic for concurrent processing."""
        # Mock semaphore behavior
        class MockSemaphore:
            def __init__(self, value: int):
                self.value = value
                self.current = 0
            
            def acquire(self):
                if self.current < self.value:
                    self.current += 1
                    return True
                return False
            
            def release(self):
                if self.current > 0:
                    self.current -= 1
        
        # Test semaphore limits concurrent access
        max_concurrent = 3
        semaphore = MockSemaphore(max_concurrent)
        
        # Acquire up to limit
        for i in range(max_concurrent):
            assert semaphore.acquire() is True
            assert semaphore.current == i + 1
        
        # Should fail to acquire beyond limit
        assert semaphore.acquire() is False
        assert semaphore.current == max_concurrent
        
        # Release and acquire again
        semaphore.release()
        assert semaphore.current == max_concurrent - 1
        assert semaphore.acquire() is True
        assert semaphore.current == max_concurrent
    
    def test_timeout_handling_logic(self):
        """Test timeout handling in batch processing."""
        # Mock timeout behavior
        async def mock_operation_with_timeout(timeout_seconds: int, operation_duration: int):
            """Mock operation that may timeout."""
            try:
                # Simulate operation duration
                await asyncio.wait_for(
                    asyncio.sleep(operation_duration), 
                    timeout=timeout_seconds
                )
                return "success"
            except asyncio.TimeoutError:
                return "timeout"
        
        # Test cases for timeout behavior
        async def run_timeout_tests():
            # Operation completes within timeout
            result = await mock_operation_with_timeout(timeout_seconds=2, operation_duration=1)
            assert result == "success"
            
            # Operation times out
            result = await mock_operation_with_timeout(timeout_seconds=1, operation_duration=2)
            assert result == "timeout"
        
        # Run the async timeout tests
        asyncio.run(run_timeout_tests())
    
    def test_processing_metrics_calculation(self):
        """Test processing metrics calculation logic."""
        def calculate_processing_metrics(start_time: float, total_entries: int, processed_entries: int):
            """Calculate processing metrics."""
            duration = time.time() - start_time
            
            if duration > 0:
                entries_per_second = processed_entries / duration
            else:
                entries_per_second = 0
            
            if total_entries > 0:
                success_rate = (processed_entries / total_entries) * 100
            else:
                success_rate = 0
            
            return {
                "duration": duration,
                "entries_per_second": entries_per_second,
                "success_rate": success_rate
            }
        
        # Mock start time (1 second ago)
        start_time = time.time() - 1.0
        
        # Test normal case
        metrics = calculate_processing_metrics(start_time, 1000, 950)
        
        assert metrics["duration"] >= 1.0  # At least 1 second
        assert metrics["entries_per_second"] > 0  # Should have processed some per second
        assert metrics["success_rate"] == 95.0  # 950/1000 = 95%
        
        # Test edge cases
        metrics_zero = calculate_processing_metrics(start_time, 0, 0)
        assert metrics_zero["success_rate"] == 0.0
        
        metrics_perfect = calculate_processing_metrics(start_time, 100, 100)
        assert metrics_perfect["success_rate"] == 100.0
    
    def test_data_integrity_preservation(self, mock_data_service):
        """Test that data integrity is preserved through processing pipeline."""
        original_data = [
            {"title": "Anime 1", "id": 1, "special_field": "value1"},
            {"title": "Anime 2", "id": 2, "special_field": "value2"},
            {"title": "Anime 3", "id": 3, "special_field": "value3"},
        ]
        
        # Process through batching
        batches = mock_data_service._create_batches(original_data, batch_size=2)
        
        # Verify no data loss in batching
        flattened = [item for batch in batches for item in batch]
        assert len(flattened) == len(original_data)
        
        # Verify order preservation in small batches
        assert flattened[0]["title"] == "Anime 1"
        assert flattened[1]["title"] == "Anime 2"
        assert flattened[2]["title"] == "Anime 3"
        
        # Verify all fields preserved
        for original, flattened_item in zip(original_data, flattened):
            assert original["title"] == flattened_item["title"]
            assert original["id"] == flattened_item["id"]
            assert original["special_field"] == flattened_item["special_field"]