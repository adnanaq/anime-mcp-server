"""Tests for LLM service Phase 6C AI-powered query understanding."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.services.llm_service import (
    LLMProvider,
    LLMService,
    SearchIntent,
    extract_search_intent,
    get_llm_service,
)


class TestSearchIntent:
    """Test SearchIntent model validation and functionality."""

    def test_search_intent_creation(self):
        """Test basic SearchIntent model creation."""
        intent = SearchIntent(
            query="mecha anime", limit=5, genres=["mecha", "action"], confidence=0.9
        )

        assert intent.query == "mecha anime"
        assert intent.limit == 5
        assert intent.genres == ["mecha", "action"]
        assert intent.confidence == 0.9
        assert intent.year_range is None
        assert len(intent.anime_types) == 0

    def test_search_intent_defaults(self):
        """Test SearchIntent default values."""
        intent = SearchIntent(query="test query")

        assert intent.query == "test query"
        assert intent.limit is None
        assert intent.genres == []
        assert intent.year_range is None
        assert intent.anime_types == []
        assert intent.studios == []
        assert intent.exclusions == []
        assert intent.mood_keywords == []
        assert intent.confidence == 0.0

    def test_search_intent_validation(self):
        """Test SearchIntent field validation."""
        # Valid limit range
        intent = SearchIntent(query="test", limit=25)
        assert intent.limit == 25

        # Year range list
        intent = SearchIntent(query="test", year_range=[2020, 2025])
        assert intent.year_range == [2020, 2025]


class TestLLMService:
    """Test LLM service functionality."""

    def test_service_initialization_openai(self):
        """Test LLM service initialization with OpenAI."""
        with patch("src.services.llm_service.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key = "test-key"

            with patch("openai.AsyncOpenAI") as mock_openai:
                service = LLMService(provider=LLMProvider.OPENAI)

                assert service.provider == LLMProvider.OPENAI
                mock_openai.assert_called_once_with(api_key="test-key")

    def test_service_initialization_anthropic(self):
        """Test LLM service initialization with Anthropic."""
        with patch("src.services.llm_service.get_settings") as mock_settings:
            mock_settings.return_value.anthropic_api_key = "test-key"

            with patch("anthropic.AsyncAnthropic") as mock_anthropic:
                service = LLMService(provider=LLMProvider.ANTHROPIC)

                assert service.provider == LLMProvider.ANTHROPIC
                mock_anthropic.assert_called_once_with(api_key="test-key")

    def test_service_no_api_key(self):
        """Test service initialization without API key."""
        with patch("src.services.llm_service.get_settings") as mock_settings:
            mock_settings.return_value.openai_api_key = None

            service = LLMService(provider=LLMProvider.OPENAI)
            assert service.client is None

    @pytest.mark.asyncio
    async def test_fallback_extraction_basic(self):
        """Test fallback extraction with basic query."""
        service = LLMService()
        service.client = None  # Force fallback

        result = await service.extract_search_intent("find action anime")

        assert isinstance(result, SearchIntent)
        assert result.query == "find action anime"
        assert "action" in result.genres
        assert result.limit is None
        assert result.confidence == 0.7

    @pytest.mark.asyncio
    async def test_fallback_extraction_with_limit(self):
        """Test fallback extraction with limit specified."""
        service = LLMService()
        service.client = None

        result = await service.extract_search_intent("show me 5 mecha anime")

        assert result.limit == 5
        assert "mecha" in result.genres
        assert "show me  mecha anime" in result.query or "mecha anime" in result.query

    @pytest.mark.asyncio
    async def test_fallback_extraction_year_range(self):
        """Test fallback extraction with year range."""
        service = LLMService()
        service.client = None

        result = await service.extract_search_intent("anime from 2020s")

        assert result.year_range == [2020, 2029]

    @pytest.mark.asyncio
    async def test_fallback_extraction_multiple_genres(self):
        """Test fallback extraction with multiple genres."""
        service = LLMService()
        service.client = None

        result = await service.extract_search_intent("action romance comedy anime")

        assert "action" in result.genres
        assert "romance" in result.genres
        assert "comedy" in result.genres

    @pytest.mark.asyncio
    async def test_fallback_extraction_complex_query(self):
        """Test fallback extraction with complex query."""
        service = LLMService()
        service.client = None

        result = await service.extract_search_intent(
            "find top 3 sci-fi anime from 2010s"
        )

        assert result.limit == 3
        assert "sci-fi" in result.genres
        assert result.year_range == [2010, 2019]

    @pytest.mark.asyncio
    async def test_openai_extraction_success(self):
        """Test successful OpenAI extraction."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = """
        {
            "query": "mecha anime",
            "limit": 5,
            "genres": ["mecha", "action"],
            "year_range": null,
            "anime_types": [],
            "studios": [],
            "exclusions": [],
            "mood_keywords": [],
            "confidence": 0.9
        }
        """
        mock_client.chat.completions.create.return_value = mock_response

        service = LLMService(provider=LLMProvider.OPENAI)
        service.client = mock_client

        result = await service.extract_search_intent("find 5 mecha anime")

        assert result.query == "mecha anime"
        assert result.limit == 5
        assert result.genres == ["mecha", "action"]
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_openai_extraction_failure_fallback(self):
        """Test OpenAI extraction failure falls back to regex."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        service = LLMService(provider=LLMProvider.OPENAI)
        service.client = mock_client

        result = await service.extract_search_intent("find 3 action anime")

        # Should fall back to regex extraction
        assert result.limit == 3
        assert "action" in result.genres
        assert result.confidence == 0.7  # Fallback confidence

    @pytest.mark.asyncio
    async def test_anthropic_extraction_success(self):
        """Test successful Anthropic extraction."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[
            0
        ].text = """
        {
            "query": "romance anime",
            "limit": 2,
            "genres": ["romance"],
            "year_range": null,
            "anime_types": [],
            "studios": [],
            "exclusions": [],
            "mood_keywords": ["light"],
            "confidence": 0.85
        }
        """
        mock_client.messages.create.return_value = mock_response

        service = LLMService(provider=LLMProvider.ANTHROPIC)
        service.client = mock_client

        result = await service.extract_search_intent("show 2 light romance anime")

        assert result.query == "romance anime"
        assert result.limit == 2
        assert result.genres == ["romance"]
        assert result.mood_keywords == ["light"]
        assert result.confidence == 0.85


class TestFallbackExtractionPatterns:
    """Test comprehensive fallback extraction patterns."""

    @pytest.mark.asyncio
    async def test_limit_patterns(self):
        """Test various limit extraction patterns."""
        service = LLMService()
        service.client = None

        test_cases = [
            ("limit to 5 anime", 5),
            ("show me 3 anime", 3),
            ("top 10 anime", 10),
            ("first 7 anime", 7),
            ("only 2 anime", 2),
            ("8 anime recommendations", 8),
            ("find 15 results", 15),
        ]

        for query, expected_limit in test_cases:
            result = await service.extract_search_intent(query)
            assert result.limit == expected_limit, f"Failed for query: {query}"

    @pytest.mark.asyncio
    async def test_genre_detection(self):
        """Test genre detection patterns."""
        service = LLMService()
        service.client = None

        test_cases = [
            ("action anime", ["action"]),
            ("romantic comedy", ["romance", "comedy"]),
            ("sci-fi space opera", ["sci-fi"]),
            ("mecha robot fighting", ["mecha", "action"]),
            ("horror thriller", ["horror"]),
            ("fantasy magical girl", ["fantasy"]),
            ("dramatic series", ["drama"]),
        ]

        for query, expected_genres in test_cases:
            result = await service.extract_search_intent(query)
            for genre in expected_genres:
                assert (
                    genre in result.genres
                ), f"Genre {genre} not found in {result.genres} for query: {query}"

    @pytest.mark.asyncio
    async def test_year_extraction(self):
        """Test year range extraction patterns."""
        service = LLMService()
        service.client = None

        test_cases = [
            ("anime from 1990s", [1990, 1999]),
            ("2000s series", [2000, 2009]),
            ("2020s anime", [2020, 2029]),
            ("anime from 2015", [2015, 2015]),
        ]

        for query, expected_range in test_cases:
            result = await service.extract_search_intent(query)
            assert result.year_range == expected_range, f"Failed for query: {query}"

    @pytest.mark.asyncio
    async def test_complex_queries(self):
        """Test complex multi-parameter queries."""
        service = LLMService()
        service.client = None

        # Complex query with limit, genre, and year
        result = await service.extract_search_intent(
            "find top 5 action anime from 2020s"
        )
        assert result.limit == 5
        assert "action" in result.genres
        assert result.year_range == [2020, 2029]

        # Query with multiple genres and limit
        result = await service.extract_search_intent("show me 3 sci-fi mecha anime")
        assert result.limit == 3
        assert "sci-fi" in result.genres
        assert "mecha" in result.genres


class TestGlobalServiceFunctions:
    """Test global service functions."""

    def test_get_llm_service_singleton(self):
        """Test get_llm_service returns singleton."""
        service1 = get_llm_service()
        service2 = get_llm_service()

        assert service1 is service2
        assert isinstance(service1, LLMService)

    @pytest.mark.asyncio
    async def test_extract_search_intent_function(self):
        """Test convenience extract_search_intent function."""
        with patch("src.services.llm_service.get_llm_service") as mock_get_service:
            mock_service = Mock()
            mock_service.extract_search_intent = AsyncMock(
                return_value=SearchIntent(query="test")
            )
            mock_get_service.return_value = mock_service

            result = await extract_search_intent("test query")

            assert isinstance(result, SearchIntent)
            assert result.query == "test"
            mock_service.extract_search_intent.assert_called_once_with("test query")


class TestLLMServiceIntegration:
    """Integration tests for LLM service with different scenarios."""

    @pytest.mark.asyncio
    async def test_service_with_realistic_queries(self):
        """Test service with realistic anime search queries."""
        service = LLMService()
        service.client = None  # Use fallback for consistent testing

        realistic_queries = [
            "find me some good action anime",
            "show me 5 Studio Ghibli movies",
            "top 10 anime from 2020s",
            "romantic comedy series",
            "mecha anime but not too violent",
            "sci-fi space anime from 90s",
            "limit results to 3 fantasy anime",
        ]

        for query in realistic_queries:
            result = await service.extract_search_intent(query)

            # All should return valid SearchIntent
            assert isinstance(result, SearchIntent)
            assert result.query is not None
            assert 0.0 <= result.confidence <= 1.0

            # If limit detected, should be reasonable
            if result.limit:
                assert 1 <= result.limit <= 50

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases and error conditions."""
        service = LLMService()
        service.client = None

        # Empty query
        result = await service.extract_search_intent("")
        assert result.query == ""

        # Very long query
        long_query = "anime " * 100
        result = await service.extract_search_intent(long_query)
        assert isinstance(result, SearchIntent)

        # Query with numbers but not limits
        result = await service.extract_search_intent("Attack on Titan season 4")
        assert result.limit is None  # Should not extract "4" as limit

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        """Test handling of invalid JSON from LLM."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json content"
        mock_client.chat.completions.create.return_value = mock_response

        service = LLMService(provider=LLMProvider.OPENAI)
        service.client = mock_client

        result = await service.extract_search_intent("test query")

        # Should fall back to regex extraction
        assert isinstance(result, SearchIntent)
        assert result.confidence == 0.7  # Fallback confidence
