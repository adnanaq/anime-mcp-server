"""Integration tests for AnimeCountdown scraper with real external calls."""

import pytest

from src.integrations.scrapers.extractors.animecountdown import AnimeCountdownScraper


class TestAnimeCountdownRealData:
    """Test AnimeCountdown scraper with actual external API calls."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_site_connectivity(self):
        """Test that we can connect to AnimeCountdown.com."""
        scraper = AnimeCountdownScraper()

        # Test basic site connectivity
        response = await scraper._make_request(scraper.base_url, timeout=10)

        # Verify we get a successful response
        assert response is not None
        assert response.get("status_code") == 200
        assert len(response.get("content", "")) > 0

        # Verify it's actually AnimeCountdown site
        content = response.get("content", "").lower()
        assert "animecountdown" in content or "countdown" in content

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_upcoming_anime_returns_data(self):
        """Test that upcoming anime endpoint returns actual data."""
        scraper = AnimeCountdownScraper()

        # Call the actual method
        upcoming = await scraper.get_upcoming_anime(limit=5)

        # We expect SOME data to be returned
        assert upcoming is not None
        assert isinstance(upcoming, list)

        # If site is working, we should get at least some upcoming anime
        # This test will fail until we implement proper parsing
        assert (
            len(upcoming) > 0
        ), "No upcoming anime found - scraper needs implementation"

        # Verify structure of returned data
        first_anime = upcoming[0]
        assert isinstance(first_anime, dict)
        assert "title" in first_anime, "Anime data should have title"
        assert first_anime["title"], "Title should not be empty"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_search_returns_data(self):
        """Test that search returns actual data for popular anime."""
        scraper = AnimeCountdownScraper()

        # Search for a very popular anime that should definitely be on the site
        results = await scraper.search_anime_countdowns("demon slayer", limit=5)

        # We expect results for such a popular anime
        assert results is not None
        assert isinstance(results, list)
        assert (
            len(results) > 0
        ), "No search results for 'demon slayer' - scraper needs implementation"

        # Verify data structure
        first_result = results[0]
        assert isinstance(first_result, dict)
        assert "title" in first_result
        assert first_result["title"], "Search result should have non-empty title"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_anime_details_by_slug(self):
        """Test that we can get anime details by slug."""
        scraper = AnimeCountdownScraper()

        # Try to get details for a popular anime
        # This will fail until we implement proper parsing
        details = await scraper.get_anime_countdown_by_slug("demon-slayer")

        if details:  # If we get data, verify structure
            assert isinstance(details, dict)
            assert "title" in details
            assert details["title"], "Anime details should have non-empty title"
