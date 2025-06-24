"""LLM service for Phase 6C AI-powered query understanding."""

import logging
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from ..config import get_settings

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class SearchIntent(BaseModel):
    """Structured output schema for LLM query understanding."""

    query: str = Field(
        ..., description="Cleaned search query without limit/filter words"
    )
    limit: Optional[int] = Field(None, description="Number of results requested (1-50)")
    genres: List[str] = Field(default_factory=list, description="Detected anime genres")
    year_range: Optional[List[int]] = Field(
        None, description="Year range [start_year, end_year]"
    )
    anime_types: List[str] = Field(
        default_factory=list, description="Types: TV, Movie, OVA, Special, etc."
    )
    studios: List[str] = Field(
        default_factory=list, description="Specific animation studios"
    )
    exclusions: List[str] = Field(
        default_factory=list, description="Things to exclude/avoid"
    )
    mood_keywords: List[str] = Field(
        default_factory=list,
        description="Mood descriptors: dark, light, serious, funny",
    )
    confidence: float = Field(0.0, description="LLM confidence in extraction (0.0-1.0)")


class LLMService:
    """Service for AI-powered natural language query understanding."""

    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI):
        self.provider = provider
        self.settings = get_settings()
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client based on provider."""
        try:
            if self.provider == LLMProvider.OPENAI:
                from openai import AsyncOpenAI

                api_key = getattr(self.settings, "openai_api_key", None)
                if api_key:
                    self.client = AsyncOpenAI(api_key=api_key)
                    logger.info("Initialized OpenAI client")
                else:
                    logger.warning("OpenAI API key not found - LLM features disabled")

            elif self.provider == LLMProvider.ANTHROPIC:
                from anthropic import AsyncAnthropic

                api_key = getattr(self.settings, "anthropic_api_key", None)
                if api_key:
                    self.client = AsyncAnthropic(api_key=api_key)
                    logger.info("Initialized Anthropic client")
                else:
                    logger.warning(
                        "Anthropic API key not found - LLM features disabled"
                    )

        except ImportError as e:
            logger.warning(f"LLM provider {self.provider} not available: {e}")
            self.client = None

    async def extract_search_intent(self, user_query: str) -> SearchIntent:
        """Extract structured search parameters from natural language query."""
        if not self.client:
            # Fallback to basic extraction if LLM unavailable
            logger.info("LLM unavailable, using fallback extraction")
            return self._fallback_extraction(user_query)

        logger.info(f"Using LLM to extract intent from: {user_query}")

        try:
            if self.provider == LLMProvider.OPENAI:
                return await self._openai_extraction(user_query)
            elif self.provider == LLMProvider.ANTHROPIC:
                return await self._anthropic_extraction(user_query)
        except Exception as e:
            logger.error(
                f"LLM extraction failed: {e}, falling back to basic extraction"
            )
            return self._fallback_extraction(user_query)

    async def _openai_extraction(self, user_query: str) -> SearchIntent:
        """Extract using OpenAI structured output."""
        system_prompt = """You are an expert anime search assistant. Extract search parameters from user queries.

Rules:
- limit: Extract numbers like "5 anime", "top 3", "limit to 7", "show me 2"
- genres: Common anime genres (action, romance, comedy, drama, fantasy, sci-fi, mecha, shounen, etc.)
- year_range: "from 2020s" = (2020, 2029), "90s anime" = (1990, 1999), "recent" = (2020, 2024)
- anime_types: TV, Movie, OVA, Special, ONA
- studios: Specific animation studios like "Studio Ghibli", "Mappa", "Toei Animation"
- exclusions: "not horror", "but not romance", "except isekai"
- mood_keywords: "dark", "light-hearted", "serious", "funny", "emotional"
- query: Clean the main search intent, removing limit/filter words

Examples:
"find 5 mecha anime from 2020s but not too violent" →
  query: "mecha anime", limit: 5, genres: ["mecha"], year_range: (2020, 2029), exclusions: ["violent"]

"show me top 3 Studio Ghibli movies" →
  query: "Studio Ghibli movies", limit: 3, studios: ["Studio Ghibli"], anime_types: ["Movie"]
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Extract search parameters from: {user_query}",
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "search_intent",
                        "schema": SearchIntent.model_json_schema(),
                    },
                },
                temperature=0.1,
            )

            result_json = response.choices[0].message.content
            import json

            result_dict = json.loads(result_json)

            # Handle null values for list fields
            for field in [
                "genres",
                "anime_types",
                "studios",
                "exclusions",
                "mood_keywords",
            ]:
                if result_dict.get(field) is None:
                    result_dict[field] = []

            return SearchIntent(**result_dict)

        except Exception as e:
            logger.error(f"OpenAI extraction error: {e}")
            raise

    async def _anthropic_extraction(self, user_query: str) -> SearchIntent:
        """Extract using Anthropic Claude."""
        system_prompt = """You are an expert anime search assistant. Extract search parameters from user queries and respond with valid JSON only.

Extract these fields:
- query: Main search terms, cleaned of limit/filter words
- limit: Number of results (1-50) from phrases like "5 anime", "top 3", "limit to 7"  
- genres: Anime genres like action, romance, comedy, drama, fantasy, sci-fi, mecha
- year_range: [start_year, end_year] for "2020s" = [2020, 2029], "90s" = [1990, 1999]
- anime_types: ["TV", "Movie", "OVA", "Special", "ONA"]
- studios: ["Studio Name"] for specific animation studios
- exclusions: Things to avoid like "not horror", "except isekai"
- mood_keywords: ["dark", "light", "serious", "funny"]
- confidence: 0.0-1.0 confidence score

Respond with valid JSON only."""

        try:
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Extract search parameters: {user_query}",
                    }
                ],
                max_tokens=500,
                temperature=0.1,
            )

            result_json = response.content[0].text
            import json

            result_dict = json.loads(result_json)

            # Handle null values for list fields
            for field in [
                "genres",
                "anime_types",
                "studios",
                "exclusions",
                "mood_keywords",
            ]:
                if result_dict.get(field) is None:
                    result_dict[field] = []

            return SearchIntent(**result_dict)

        except Exception as e:
            logger.error(f"Anthropic extraction error: {e}")
            raise

    def _fallback_extraction(self, user_query: str) -> SearchIntent:
        """Basic regex-based extraction as fallback."""
        import re

        query_lower = user_query.lower()

        # Extract limit
        limit = None
        limit_patterns = [
            r"(?:limit.*?to|show.*?me|top|first|only)\s+(\d+)",
            r"(\d+)\s+(?:anime|results?)",
            r"find\s+(\d+)\s+",  # "find 3 action anime"
        ]
        for pattern in limit_patterns:
            match = re.search(pattern, query_lower)
            if match:
                try:
                    parsed_limit = int(match.group(1))
                    if 1 <= parsed_limit <= 50:
                        limit = parsed_limit
                        break
                except ValueError:
                    continue

        # Extract genres
        genre_keywords = {
            "action": ["action", "fighting", "battle"],
            "romance": ["romance", "romantic", "love"],
            "comedy": ["comedy", "funny", "humor"],
            "drama": ["drama", "dramatic"],
            "fantasy": ["fantasy", "magic", "magical"],
            "sci-fi": ["sci-fi", "science fiction", "futuristic", "space"],
            "mecha": ["mecha", "robot", "gundam"],
            "horror": ["horror", "scary", "dark"],
        }

        detected_genres = []
        for genre, keywords in genre_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_genres.append(genre)

        # Extract year range
        year_range = None
        year_match = re.search(r"\b(19|20)(\d{2})s?\b", user_query)
        if year_match:
            decade = int(year_match.group(1) + year_match.group(2))
            if year_match.group().endswith("s"):
                year_range = [decade, decade + 9]
            else:
                year_range = [decade, decade]

        # Clean query
        clean_query = user_query
        for pattern in limit_patterns:
            clean_query = re.sub(pattern, "", clean_query, flags=re.IGNORECASE)
        clean_query = clean_query.strip()

        return SearchIntent(
            query=clean_query,
            limit=limit,
            genres=detected_genres,
            year_range=year_range,
            confidence=0.7,  # Medium confidence for fallback
        )


# Global service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


async def extract_search_intent(user_query: str) -> SearchIntent:
    """Convenience function to extract search intent from user query."""
    service = get_llm_service()
    return await service.extract_search_intent(user_query)
