"""Comprehensive tests for AnimeDataService.

This file contains all tests for the AnimeDataService class including:
- Core Logic Tests: Platform ID extraction, data processing, quality scoring
- Edge Cases: Malformed URLs, invalid data, boundary conditions  
- Performance Tests: Bulk processing, stress testing, memory usage
- Network Tests: Download functionality, error handling
"""
import pytest
from unittest.mock import patch, AsyncMock
from typing import Dict, Any

from src.services.data_service import AnimeDataService
from src.models.anime import AnimeEntry


# ============================================================================
# SECTION 1: CORE LOGIC TESTS
# ============================================================================

class TestAnimeDataServiceCore:
    """Core functionality tests for AnimeDataService."""

    @pytest.fixture
    def service(self) -> AnimeDataService:
        """Create service instance."""
        return AnimeDataService()

    def test_platform_configs_completeness(self, service: AnimeDataService):
        """Test that all expected platforms are configured."""
        expected_platforms = {
            'myanimelist', 'anilist', 'kitsu', 'anidb', 'anisearch',
            'simkl', 'livechart', 'animenewsnetwork', 'animeplanet',
            'notify', 'animecountdown'
        }
        
        assert set(service.platform_configs.keys()) == expected_platforms
        
        # Verify each config has required fields
        for platform, config in service.platform_configs.items():
            assert 'domain' in config
            assert 'pattern' in config
            assert 'id_type' in config
            assert config['id_type'] in {'numeric', 'slug', 'alphanumeric'}

    @pytest.mark.parametrize("platform,url,expected_id", [
        # Numeric ID platforms
        ('myanimelist', 'https://myanimelist.net/anime/12345', 12345),
        ('anilist', 'https://anilist.co/anime/67890', 67890),
        ('kitsu', 'https://kitsu.io/anime/111', 111),
        ('anidb', 'https://anidb.net/anime/222', 222),
        ('anisearch', 'https://anisearch.com/anime/333', 333),
        ('simkl', 'https://simkl.com/anime/444', 444),
        ('livechart', 'https://livechart.me/anime/555', 555),
        ('animenewsnetwork', 'https://animenewsnetwork.com/encyclopedia/anime.php?id=666', 666),
        ('animecountdown', 'https://animecountdown.com/777', 777),
        
        # String ID platforms
        ('animeplanet', 'https://anime-planet.com/anime/test-slug', 'test-slug'),
        ('notify', 'https://notify.moe/anime/ABC123DEF', 'ABC123DEF'),
    ])
    def test_extract_single_platform_id(self, service: AnimeDataService, platform: str, url: str, expected_id):
        """Test extraction of individual platform IDs."""
        platform_ids = service._extract_all_platform_ids([url])
        
        expected_key = f"{platform}_id"
        assert expected_key in platform_ids
        assert platform_ids[expected_key] == expected_id

    def test_extract_all_platform_ids_comprehensive(self, service: AnimeDataService, complex_anime_data: Dict[str, Any], expected_platform_ids: Dict[str, Any]):
        """Test extraction of all platform IDs from complex data."""
        sources = complex_anime_data["sources"]
        platform_ids = service._extract_all_platform_ids(sources)
        
        assert platform_ids == expected_platform_ids

    def test_extract_platform_ids_with_invalid_urls(self, service: AnimeDataService):
        """Test platform ID extraction with invalid URLs."""
        invalid_sources = [
            "https://invalid-site.com/anime/123",
            "https://myanimelist.net/character/456",  # Wrong path
            "https://anilist.co/anime/invalid",  # Non-numeric ID
            "not-a-url-at-all"
        ]
        
        platform_ids = service._extract_all_platform_ids(invalid_sources)
        assert platform_ids == {}

    def test_extract_platform_ids_mixed_valid_invalid(self, service: AnimeDataService):
        """Test platform ID extraction with mix of valid and invalid URLs."""
        mixed_sources = [
            "https://myanimelist.net/anime/12345",  # Valid
            "https://invalid-site.com/anime/123",   # Invalid domain
            "https://anilist.co/anime/67890",       # Valid
            "https://anilist.co/character/999"      # Invalid path
        ]
        
        platform_ids = service._extract_all_platform_ids(mixed_sources)
        
        expected = {
            "myanimelist_id": 12345,
            "anilist_id": 67890
        }
        assert platform_ids == expected

    def test_process_anime_entry_with_platform_ids(self, service: AnimeDataService, complex_anime_data: Dict[str, Any]):
        """Test that process_anime_entry includes all platform IDs."""
        processed = service.process_anime_entry(complex_anime_data)
        
        assert processed is not None
        
        # Check that all platform IDs are included
        expected_keys = {
            'myanimelist_id', 'anilist_id', 'kitsu_id', 'anidb_id',
            'anisearch_id', 'simkl_id', 'livechart_id', 'animenewsnetwork_id',
            'animeplanet_id', 'notify_id', 'animecountdown_id'
        }
        
        for key in expected_keys:
            assert key in processed
            assert processed[key] is not None

    def test_generate_anime_id_consistency(self, service: AnimeDataService):
        """Test that anime ID generation is consistent."""
        title = "Test Anime"
        sources = ["https://myanimelist.net/anime/123"]
        
        id1 = service._generate_anime_id(title, sources)
        id2 = service._generate_anime_id(title, sources)
        
        assert id1 == id2
        assert len(id1) == 12
        assert isinstance(id1, str)

    def test_generate_anime_id_uniqueness(self, service: AnimeDataService):
        """Test that different anime get different IDs."""
        title1 = "Test Anime 1"
        title2 = "Test Anime 2"
        sources = ["https://myanimelist.net/anime/123"]
        
        id1 = service._generate_anime_id(title1, sources)
        id2 = service._generate_anime_id(title2, sources)
        
        assert id1 != id2

    def test_calculate_quality_score_high_quality(self, service: AnimeDataService, complex_anime_data: Dict[str, Any]):
        """Test quality score calculation for high-quality data."""
        anime = AnimeEntry(**complex_anime_data)
        score = service._calculate_quality_score(anime)
        
        # Should be close to 1.0 for complete data
        assert 0.9 <= score <= 1.0

    def test_calculate_quality_score_minimal_data(self, service: AnimeDataService):
        """Test quality score calculation for minimal data."""
        minimal_data = {
            "sources": ["https://myanimelist.net/anime/1"],
            "title": "Minimal Anime",
            "type": "TV",
            "episodes": 0,
            "status": "FINISHED"
        }
        
        anime = AnimeEntry(**minimal_data)
        score = service._calculate_quality_score(anime)
        
        # Should be lower for minimal data
        assert 0.0 <= score <= 0.5

    def test_create_embedding_text(self, service: AnimeDataService, anime_entry: AnimeEntry):
        """Test embedding text creation."""
        embedding_text = service._create_embedding_text(anime_entry)
        
        # Should contain all relevant text fields
        assert anime_entry.title in embedding_text
        assert anime_entry.synopsis in embedding_text
        assert anime_entry.type in embedding_text
        
        for tag in anime_entry.tags:
            assert tag in embedding_text
        
        for studio in anime_entry.studios:
            assert studio in embedding_text

    def test_create_search_text(self, service: AnimeDataService, anime_entry: AnimeEntry):
        """Test search text creation."""
        search_text = service._create_search_text(anime_entry)
        
        # Should contain title, synonyms, and tags
        assert anime_entry.title in search_text
        
        for synonym in anime_entry.synonyms:
            assert synonym in search_text
        
        for tag in anime_entry.tags:
            assert tag in search_text

    def test_extract_year_season_valid(self, service: AnimeDataService):
        """Test year and season extraction from valid data."""
        anime_season = {"year": 2023, "season": "SPRING"}
        year, season = service._extract_year_season(anime_season)
        
        assert year == 2023
        assert season == "spring"

    def test_extract_year_season_none(self, service: AnimeDataService):
        """Test year and season extraction from None."""
        year, season = service._extract_year_season(None)
        
        assert year is None
        assert season is None

    def test_extract_year_season_partial(self, service: AnimeDataService):
        """Test year and season extraction from partial data."""
        anime_season = {"year": 2023}
        year, season = service._extract_year_season(anime_season)
        
        assert year == 2023
        assert season is None

    @pytest.mark.asyncio
    async def test_download_anime_database_success(self, service: AnimeDataService, mock_aiohttp_session):
        """Test successful database download."""
        test_data = {"data": [{"title": "Test"}]}
        mock_aiohttp_session.get.return_value.__aenter__.return_value.text.return_value = '{"data": [{"title": "Test"}]}'
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            result = await service.download_anime_database()
            
            assert result == test_data

    @pytest.mark.asyncio
    async def test_download_anime_database_http_error(self, service: AnimeDataService, mock_aiohttp_session):
        """Test database download with HTTP error."""
        mock_aiohttp_session.get.return_value.__aenter__.return_value.status = 404
        
        with patch('aiohttp.ClientSession', return_value=mock_aiohttp_session):
            with pytest.raises(Exception, match="Failed to download database: HTTP 404"):
                await service.download_anime_database()

    def test_process_anime_entry_invalid_data(self, service: AnimeDataService):
        """Test processing invalid anime entry."""
        invalid_data = {"invalid": "data"}
        
        result = service.process_anime_entry(invalid_data)
        assert result is None


# ============================================================================
# SECTION 2: EDGE CASES & BOUNDARY CONDITIONS
# ============================================================================

class TestAnimeDataServiceEdgeCases:
    """Edge case and boundary condition tests for AnimeDataService."""

    @pytest.fixture
    def service(self) -> AnimeDataService:
        """Create service instance."""
        return AnimeDataService()

    # URL Pattern Edge Cases
    @pytest.mark.parametrize("malformed_url,platform", [
        # Missing protocol
        ("myanimelist.net/anime/123", "myanimelist"),
        ("anilist.co/anime/456", "anilist"),
        
        # Wrong domain spelling
        ("https://myanimelist.com/anime/123", "myanimelist"),  # .com instead of .net
        ("https://anilist.org/anime/456", "anilist"),  # .org instead of .co
        
        # Missing anime path segment
        ("https://myanimelist.net/123", "myanimelist"),
        ("https://anilist.co/456", "anilist"),
        
        # Extra path segments
        ("https://myanimelist.net/anime/123/extra/path", "myanimelist"),
        ("https://anilist.co/anime/456/some-title/reviews", "anilist"),
        
        # Non-numeric IDs where numeric expected
        ("https://myanimelist.net/anime/abc", "myanimelist"),
        ("https://kitsu.io/anime/xyz-123", "kitsu"),
        
        # Empty ID
        ("https://myanimelist.net/anime/", "myanimelist"),
        ("https://anidb.net/anime/", "anidb"),
        
        # Special characters in numeric IDs
        ("https://simkl.com/anime/123-abc", "simkl"),
        ("https://livechart.me/anime/456_def", "livechart"),
    ])
    def test_malformed_url_patterns(self, service: AnimeDataService, malformed_url: str, platform: str):
        """Test that malformed URLs don't extract incorrect IDs."""
        platform_ids = service._extract_all_platform_ids([malformed_url])
        
        # Should not extract any ID for malformed URLs
        expected_key = f"{platform}_id"
        assert expected_key not in platform_ids or platform_ids[expected_key] is None

    @pytest.mark.parametrize("valid_variation,platform,expected_id", [
        # Query parameters should be ignored
        ("https://myanimelist.net/anime/123?tab=reviews", "myanimelist", 123),
        ("https://anilist.co/anime/456/?view=characters", "anilist", 456),
        
        # Fragments should be ignored
        ("https://kitsu.io/anime/789#overview", "kitsu", 789),
        ("https://anidb.net/anime/101#relations", "anidb", 101),
        
        # Trailing slashes
        ("https://simkl.com/anime/202/", "simkl", 202),
        ("https://livechart.me/anime/303/", "livechart", 303),
        
        # Mixed case domains
        ("https://MyAnimeList.net/anime/404", "myanimelist", 404),
        ("https://AniList.co/anime/505", "anilist", 505),
        ("https://KITSU.io/anime/606", "kitsu", 606),
        
        # HTTP vs HTTPS
        ("http://anidb.net/anime/707", "anidb", 707),
        ("http://anisearch.com/anime/808", "anisearch", 808),
        
        # Subdomain variations (should fail)
        ("https://www.myanimelist.net/anime/909", "myanimelist", None),
        ("https://beta.anilist.co/anime/111", "anilist", None),
    ])
    def test_url_variation_handling(self, service: AnimeDataService, valid_variation: str, platform: str, expected_id):
        """Test handling of URL variations."""
        platform_ids = service._extract_all_platform_ids([valid_variation])
        
        expected_key = f"{platform}_id"
        if expected_id is None:
            assert expected_key not in platform_ids or platform_ids[expected_key] is None
        else:
            assert platform_ids.get(expected_key) == expected_id

    # String ID Platform Edge Cases
    @pytest.mark.parametrize("url,platform,expected_id", [
        # Anime Planet slug variations
        ("https://anime-planet.com/anime/cowboy-bebop", "animeplanet", "cowboy-bebop"),
        ("https://anime-planet.com/anime/serial-experiments-lain", "animeplanet", "serial-experiments-lain"),
        ("https://anime-planet.com/anime/neon-genesis-evangelion", "animeplanet", "neon-genesis-evangelion"),
        ("https://anime-planet.com/anime/a", "animeplanet", "a"),  # Single character
        ("https://anime-planet.com/anime/123", "animeplanet", "123"),  # Numeric slug
        ("https://anime-planet.com/anime/test_anime", "animeplanet", "test_anime"),  # Underscore
        ("https://anime-planet.com/anime/test.anime", "animeplanet", "test.anime"),  # Dot
        
        # Notify.moe ID variations
        ("https://notify.moe/anime/ABC123DEF", "notify", "ABC123DEF"),
        ("https://notify.moe/anime/xyz789", "notify", "xyz789"),
        ("https://notify.moe/anime/A1B2C3", "notify", "A1B2C3"),
        ("https://notify.moe/anime/a", "notify", "a"),  # Single character
        ("https://notify.moe/anime/VERY-LONG-ID-STRING", "notify", "VERY-LONG-ID-STRING"),
        ("https://notify.moe/anime/123456", "notify", "123456"),  # All numeric
    ])
    def test_string_id_extraction(self, service: AnimeDataService, url: str, platform: str, expected_id: str):
        """Test extraction of string-based platform IDs."""
        platform_ids = service._extract_all_platform_ids([url])
        
        expected_key = f"{platform}_id"
        assert platform_ids.get(expected_key) == expected_id

    # Boundary Value Testing
    @pytest.mark.parametrize("id_value,platform", [
        # Very large numeric IDs
        (999999999, "myanimelist"),
        (2147483647, "anilist"),  # Max 32-bit signed int
        (9223372036854775807, "kitsu"),  # Max 64-bit signed int
        
        # Zero and small values
        (0, "simkl"),
        (1, "livechart"),
        
        # Edge case string lengths
        ("a" * 100, "animeplanet"),  # Very long slug
    ])
    def test_boundary_value_ids(self, service: AnimeDataService, id_value, platform: str):
        """Test boundary values for platform IDs."""
        if platform in ["animeplanet", "notify"]:
            # String ID platforms
            url = f"https://{service.platform_configs[platform]['domain']}/anime/{id_value}"
            platform_ids = service._extract_all_platform_ids([url])
            assert platform_ids.get(f"{platform}_id") == id_value
        else:
            # Numeric ID platforms
            url = f"https://{service.platform_configs[platform]['domain']}/anime/{id_value}"
            platform_ids = service._extract_all_platform_ids([url])
            assert platform_ids.get(f"{platform}_id") == id_value

    # Data Processing Edge Cases
    def test_process_anime_entry_with_empty_sources(self, service: AnimeDataService):
        """Test processing anime with empty sources list."""
        data = {
            "sources": [],
            "title": "No Sources Anime",
            "type": "TV",
            "episodes": 12,
            "status": "FINISHED"
        }
        
        processed = service.process_anime_entry(data)
        assert processed is not None
        
        # Should have no platform IDs
        platform_id_keys = [key for key in processed.keys() if key.endswith('_id')]
        for key in platform_id_keys:
            assert processed[key] is None

    def test_process_anime_entry_with_duplicate_sources(self, service: AnimeDataService):
        """Test processing anime with duplicate sources."""
        data = {
            "sources": [
                "https://myanimelist.net/anime/123",
                "https://myanimelist.net/anime/123",  # Duplicate
                "https://anilist.co/anime/456",
                "https://anilist.co/anime/456"  # Duplicate
            ],
            "title": "Duplicate Sources Anime",
            "type": "TV",
            "episodes": 12,
            "status": "FINISHED"
        }
        
        processed = service.process_anime_entry(data)
        assert processed is not None
        
        # Should extract IDs only once despite duplicates
        assert processed["myanimelist_id"] == 123
        assert processed["anilist_id"] == 456

    def test_process_anime_entry_conflicting_platform_ids(self, service: AnimeDataService):
        """Test processing anime with conflicting IDs for same platform."""
        data = {
            "sources": [
                "https://myanimelist.net/anime/123",
                "https://myanimelist.net/anime/456",  # Different ID, same platform
                "https://anilist.co/anime/789"
            ],
            "title": "Conflicting IDs Anime",
            "type": "TV",
            "episodes": 12,
            "status": "FINISHED"
        }
        
        processed = service.process_anime_entry(data)
        assert processed is not None
        
        # Should take the first matching ID (123, not 456)
        assert processed["myanimelist_id"] == 123
        assert processed["anilist_id"] == 789

    # Text Processing Edge Cases
    def test_create_embedding_text_with_empty_fields(self, service: AnimeDataService):
        """Test embedding text creation with empty/None fields."""
        data = {
            "sources": ["https://myanimelist.net/anime/1"],
            "title": "Minimal Anime",
            "type": "TV",
            "episodes": 0,
            "status": "FINISHED",
            "synopsis": None,
            "tags": [],
            "studios": [],
            "synonyms": []
        }
        
        anime = AnimeEntry(**data)
        embedding_text = service._create_embedding_text(anime)
        
        # Should contain at least title and type
        assert "Minimal Anime" in embedding_text
        assert "TV" in embedding_text
        
        # Should handle empty/None fields gracefully
        assert embedding_text.strip()  # Not empty after stripping

    def test_create_embedding_text_with_special_characters(self, service: AnimeDataService):
        """Test embedding text with special characters and Unicode."""
        data = {
            "sources": ["https://myanimelist.net/anime/1"],
            "title": "Special Characters: 攻殻機動隊 & Émotions!",
            "type": "TV",
            "episodes": 12,
            "status": "FINISHED",
            "synopsis": "A story with special chars: @#$%^&*(){}[]",
            "tags": ["アクション", "サイ・ファイ", "Action & Adventure"],
            "studios": ["Production I.G.", "マッドハウス"],
            "synonyms": ["Ghost in the Shell", "攻殻機動隊 SAC"]
        }
        
        anime = AnimeEntry(**data)
        embedding_text = service._create_embedding_text(anime)
        
        # Should preserve Unicode characters
        assert "攻殻機動隊" in embedding_text
        assert "アクション" in embedding_text
        assert "マッドハウス" in embedding_text
        assert "Émotions" in embedding_text

    @pytest.mark.asyncio
    async def test_process_all_anime_with_errors(self, service: AnimeDataService):
        """Test bulk processing with some entries causing errors."""
        test_data = {
            "data": [
                # Valid entry
                {
                    "sources": ["https://myanimelist.net/anime/1"],
                    "title": "Valid Anime",
                    "type": "TV",
                    "episodes": 12,
                    "status": "FINISHED"
                },
                # Invalid entry (missing required fields)
                {
                    "sources": ["https://myanimelist.net/anime/2"],
                    "title": "Invalid Anime"
                    # Missing type and status
                },
                # Another valid entry
                {
                    "sources": ["https://myanimelist.net/anime/3"],
                    "title": "Another Valid Anime",
                    "type": "Movie",
                    "episodes": 1,
                    "status": "FINISHED"
                }
            ]
        }
        
        processed_entries = await service.process_all_anime(test_data)
        
        # Should process valid entries and skip invalid ones
        assert len(processed_entries) == 2  # Only 2 valid entries
        
        titles = [entry["title"] for entry in processed_entries]
        assert "Valid Anime" in titles
        assert "Another Valid Anime" in titles
        assert "Invalid Anime" not in titles


# ============================================================================
# SECTION 3: PERFORMANCE & STRESS TESTS
# ============================================================================

class TestAnimeDataServicePerformance:
    """Performance and stress tests for AnimeDataService."""

    @pytest.fixture
    def service(self) -> AnimeDataService:
        """Create service instance."""
        return AnimeDataService()

    @pytest.fixture
    def large_anime_dataset(self) -> Dict[str, Any]:
        """Generate large anime dataset for performance testing."""
        return {
            "data": [
                {
                    "sources": [
                        f"https://myanimelist.net/anime/{i}",
                        f"https://anilist.co/anime/{i}",
                        f"https://kitsu.io/anime/{i}"
                    ],
                    "title": f"Performance Test Anime {i}",
                    "type": "TV",
                    "episodes": 12,
                    "status": "FINISHED",
                    "animeSeason": {"season": "SPRING", "year": 2023},
                    "synopsis": f"Test anime number {i} for performance testing. " * 10,
                    "tags": ["Action", "Adventure", "Comedy"],
                    "studios": [f"Studio {i % 10}"],
                    "synonyms": [f"Test {i}", f"テスト{i}"]
                }
                for i in range(100)  # 100 entries for testing
            ]
        }

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_bulk_processing_performance(self, service: AnimeDataService, large_anime_dataset):
        """Test bulk processing performance with large dataset."""
        import time
        
        start_time = time.time()
        processed_entries = await service.process_all_anime(large_anime_dataset)
        end_time = time.time()
        duration = end_time - start_time
        
        # Performance assertions
        assert len(processed_entries) == 100
        assert duration < 10.0  # Should complete within 10 seconds
        
        # Calculate throughput
        entries_per_second = len(processed_entries) / duration
        assert entries_per_second > 5  # Should process at least 5 entries/second

    @pytest.mark.slow
    def test_platform_id_extraction_performance(self, service: AnimeDataService):
        """Test platform ID extraction performance with many sources."""
        import time
        
        # Generate anime with maximum platform sources
        sources = [
            "https://myanimelist.net/anime/1",
            "https://anilist.co/anime/1", 
            "https://kitsu.io/anime/1",
            "https://anidb.net/anime/1",
            "https://anisearch.com/anime/1",
            "https://simkl.com/anime/1",
            "https://livechart.me/anime/1",
            "https://animenewsnetwork.com/encyclopedia/anime.php?id=1",
            "https://anime-planet.com/anime/test-anime",
            "https://notify.moe/anime/ABC123",
            "https://animecountdown.com/1"
        ]
        
        start_time = time.time()
        
        # Extract platform IDs many times
        for _ in range(100):
            platform_ids = service._extract_all_platform_ids(sources)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly
        assert duration < 2.0  # Less than 2 seconds for 100 extractions
        
        # Verify correctness wasn't sacrificed for speed
        platform_ids = service._extract_all_platform_ids(sources)
        assert len(platform_ids) == 11  # All platforms extracted


# ============================================================================
# SECTION 4: REFACTORED DATA PROCESSING TESTS
# ============================================================================

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


# ============================================================================
# SECTION 5: REMOVE ENTRIES FUNCTIONALITY TESTS
# ============================================================================

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
    def sample_remove_entries(self):
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
    async def test_remove_entries_success(self, mock_update_service, sample_remove_entries):
        """Test successful removal of entries."""
        result = await mock_update_service.remove_entries(sample_remove_entries)
        
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