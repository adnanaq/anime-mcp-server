"""End-to-end tests for platform cross-referencing functionality."""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from src.main import app
from src.services.data_service import AnimeDataService


class TestPlatformCrossReference:
    """End-to-end tests for the complete platform cross-referencing system."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def real_anime_sources(self):
        """Real-world anime sources for comprehensive testing."""
        return [
            "https://myanimelist.net/anime/1",
            "https://anilist.co/anime/1",
            "https://kitsu.io/anime/1",
            "https://anidb.net/anime/1",
            "https://anisearch.com/anime/1",
            "https://simkl.com/anime/1",
            "https://livechart.me/anime/1",
            "https://animenewsnetwork.com/encyclopedia/anime.php?id=1",
            "https://anime-planet.com/anime/cowboy-bebop",
            "https://notify.moe/anime/0KS4RHiyg",
            "https://animecountdown.com/1"
        ]

    @pytest.fixture
    def sample_anime_with_all_platforms(self, real_anime_sources):
        """Sample anime data with all platform sources."""
        return {
            "sources": real_anime_sources,
            "title": "Cowboy Bebop",
            "type": "TV",
            "episodes": 26,
            "status": "FINISHED",
            "animeSeason": {
                "season": "SPRING",
                "year": 1998
            },
            "picture": "https://example.com/bebop.jpg",
            "synonyms": ["カウボーイビバップ"],
            "tags": ["Action", "Drama", "Sci-Fi", "Space"],
            "studios": ["Sunrise"],
            "producers": ["Bandai Visual"],
            "synopsis": "The year 2071 A.D. That future is now..."
        }

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_complete_data_pipeline_with_crossref(self, sample_anime_with_all_platforms):
        """Test complete data processing pipeline with cross-referencing."""
        service = AnimeDataService()
        
        # Process the anime entry
        processed = service.process_anime_entry(sample_anime_with_all_platforms)
        
        assert processed is not None
        assert processed["title"] == "Cowboy Bebop"
        
        # Verify all platform IDs were extracted
        expected_platform_ids = {
            "myanimelist_id": 1,
            "anilist_id": 1,
            "kitsu_id": 1,
            "anidb_id": 1,
            "anisearch_id": 1,
            "simkl_id": 1,
            "livechart_id": 1,
            "animenewsnetwork_id": 1,
            "animeplanet_id": "cowboy-bebop",
            "notify_id": "0KS4RHiyg",
            "animecountdown_id": 1
        }
        
        for platform_id, expected_value in expected_platform_ids.items():
            assert platform_id in processed
            assert processed[platform_id] == expected_value
        
        # Verify embedding text contains all relevant information
        embedding_text = processed["embedding_text"]
        assert "Cowboy Bebop" in embedding_text
        assert "Action" in embedding_text
        assert "Sunrise" in embedding_text
        assert "2071" in embedding_text

    @pytest.mark.e2e
    def test_api_search_returns_platform_ids(self, client: TestClient):
        """Test that API search returns all platform IDs."""
        mock_results = [
            {
                "anime_id": "bebop123",
                "title": "Cowboy Bebop",
                "synopsis": "Space bounty hunters...",
                "type": "TV",
                "episodes": 26,
                "tags": ["Action", "Sci-Fi"],
                "studios": ["Sunrise"],
                "_score": 0.98,
                "year": 1998,
                "season": "spring",
                
                # All platform IDs
                "myanimelist_id": 1,
                "anilist_id": 1,
                "kitsu_id": 1,
                "anidb_id": 1,
                "anisearch_id": 1,
                "simkl_id": 1,
                "livechart_id": 1,
                "animenewsnetwork_id": 1,
                "animeplanet_id": "cowboy-bebop",
                "notify_id": "0KS4RHiyg",
                "animecountdown_id": 1
            }
        ]
        
        with patch('src.main.qdrant_client') as mock_client:
            mock_client.search.return_value = mock_results
            
            response = client.get("/api/search/?q=cowboy%20bebop&limit=10")
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data["results"]) == 1
            result = data["results"][0]
            
            # Verify all platform IDs are present and correct
            assert result["myanimelist_id"] == 1
            assert result["anilist_id"] == 1
            assert result["kitsu_id"] == 1
            assert result["anidb_id"] == 1
            assert result["anisearch_id"] == 1
            assert result["simkl_id"] == 1
            assert result["livechart_id"] == 1
            assert result["animenewsnetwork_id"] == 1
            assert result["animeplanet_id"] == "cowboy-bebop"
            assert result["notify_id"] == "0KS4RHiyg"
            assert result["animecountdown_id"] == 1

    @pytest.mark.e2e
    def test_partial_platform_coverage(self, client: TestClient):
        """Test behavior when anime has partial platform coverage."""
        mock_results = [
            {
                "anime_id": "partial123",
                "title": "Partial Coverage Anime",
                "synopsis": "Only on some platforms",
                "type": "TV",
                "episodes": 12,
                "tags": ["Action"],
                "studios": ["Small Studio"],
                "_score": 0.85,
                
                # Only some platform IDs available
                "myanimelist_id": 12345,
                "anilist_id": 67890,
                "animeplanet_id": "partial-anime",
                
                # Others should be None/null
                "kitsu_id": None,
                "anidb_id": None,
                "anisearch_id": None,
                "simkl_id": None,
                "livechart_id": None,
                "animenewsnetwork_id": None,
                "notify_id": None,
                "animecountdown_id": None
            }
        ]
        
        with patch('src.main.qdrant_client') as mock_client:
            mock_client.search.return_value = mock_results
            
            response = client.post(
                "/api/search/semantic",
                json={"query": "partial coverage", "limit": 10}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            result = data["results"][0]
            
            # Available platforms
            assert result["myanimelist_id"] == 12345
            assert result["anilist_id"] == 67890
            assert result["animeplanet_id"] == "partial-anime"
            
            # Unavailable platforms should be null
            assert result["kitsu_id"] is None
            assert result["anidb_id"] is None
            assert result["notify_id"] is None

    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_bulk_processing_with_crossref(self):
        """Test bulk processing maintains cross-referencing performance."""
        service = AnimeDataService()
        
        # Create test data with varying platform coverage
        test_data = {
            "data": [
                {
                    "sources": [
                        "https://myanimelist.net/anime/1",
                        "https://anilist.co/anime/1"
                    ],
                    "title": f"Test Anime {i}",
                    "type": "TV",
                    "episodes": 12,
                    "status": "FINISHED"
                }
                for i in range(100)  # Small batch for testing
            ]
        }
        
        # Process all entries
        processed_entries = await service.process_all_anime(test_data)
        
        assert len(processed_entries) == 100
        
        # Verify each entry has platform IDs extracted
        for entry in processed_entries:
            assert "myanimelist_id" in entry
            assert "anilist_id" in entry
            assert entry["myanimelist_id"] == 1
            assert entry["anilist_id"] == 1
            
            # Other platform IDs should be missing/None since not in sources
            assert entry.get("kitsu_id") is None
            assert entry.get("animeplanet_id") is None

    @pytest.mark.e2e
    def test_crossref_data_quality_impact(self):
        """Test that cross-referencing affects data quality scores."""
        service = AnimeDataService()
        
        # Anime with many platform sources (higher quality)
        high_quality_data = {
            "sources": [
                "https://myanimelist.net/anime/1",
                "https://anilist.co/anime/1",
                "https://kitsu.io/anime/1",
                "https://anidb.net/anime/1"
            ],
            "title": "High Quality Anime",
            "type": "TV",
            "episodes": 12,
            "status": "FINISHED",
            "synopsis": "A well-documented anime",
            "tags": ["Action", "Drama"],
            "studios": ["Famous Studio"]
        }
        
        # Anime with fewer platform sources (lower quality)
        low_quality_data = {
            "sources": ["https://myanimelist.net/anime/2"],
            "title": "Low Quality Anime",
            "type": "TV",
            "episodes": 0,
            "status": "FINISHED"
        }
        
        high_quality_processed = service.process_anime_entry(high_quality_data)
        low_quality_processed = service.process_anime_entry(low_quality_data)
        
        assert high_quality_processed["data_quality_score"] > low_quality_processed["data_quality_score"]
        
        # High quality should have multiple platform IDs
        platform_id_count_high = sum(1 for key in high_quality_processed.keys() 
                                   if key.endswith('_id') and high_quality_processed[key] is not None)
        
        platform_id_count_low = sum(1 for key in low_quality_processed.keys() 
                                  if key.endswith('_id') and low_quality_processed[key] is not None)
        
        assert platform_id_count_high > platform_id_count_low

    @pytest.mark.e2e
    def test_real_world_url_patterns(self):
        """Test against real-world URL patterns and edge cases."""
        service = AnimeDataService()
        
        real_world_sources = [
            # Standard patterns
            "https://myanimelist.net/anime/1/cowboy-bebop",
            "https://anilist.co/anime/1/Cowboy-Bebop/",
            "https://kitsu.io/anime/cowboy-bebop",
            
            # With query parameters
            "https://animenewsnetwork.com/encyclopedia/anime.php?id=13&page=anime",
            
            # With fragments
            "https://anime-planet.com/anime/cowboy-bebop#reviews",
            
            # Different protocols
            "http://anidb.net/anime/23",
            
            # Trailing slashes
            "https://simkl.com/anime/1234/",
            
            # Mixed case
            "https://LiveChart.me/anime/5678"
        ]
        
        test_data = {
            "sources": real_world_sources,
            "title": "Real World Test",
            "type": "TV",
            "episodes": 1,
            "status": "FINISHED"
        }
        
        processed = service.process_anime_entry(test_data)
        
        # Should extract IDs despite URL variations
        assert processed["myanimelist_id"] == 1
        assert processed["anilist_id"] == 1
        assert processed["animenewsnetwork_id"] == 13
        assert processed["anidb_id"] == 23
        assert processed["simkl_id"] == 1234
        assert processed["livechart_id"] == 5678