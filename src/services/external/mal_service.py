"""MAL/Jikan service integration following modular pattern."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...integrations.clients.mal_client import MALClient
from ...integrations.error_handling import ErrorContext
from .base_service import BaseExternalService

logger = logging.getLogger(__name__)


class MALService(BaseExternalService):
    """MAL/Jikan service wrapper for anime data operations."""

    def __init__(
        self, client_id: Optional[str] = None, client_secret: Optional[str] = None
    ):
        """Initialize MAL service with shared dependencies.

        Args:
            client_id: Optional MAL OAuth2 client ID
            client_secret: Optional MAL OAuth2 client secret
        """
        super().__init__(service_name="mal")

        # Initialize MAL/Jikan client
        self.client = MALClient(
            client_id=client_id,
            client_secret=client_secret,
            circuit_breaker=self.circuit_breaker,
            cache_manager=self.cache_manager,
            error_handler=ErrorContext(
                user_message="MAL service error",
                debug_info="MAL/Jikan API integration error",
            ),
        )

    async def search_anime(
        self,
        query: str = "",
        limit: int = 10,
        status: Optional[str] = None,
        genres: Optional[List[int]] = None,
        # New Jikan parameters - all 17 supported parameters
        anime_type: Optional[str] = None,
        score: Optional[float] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        rating: Optional[str] = None,
        sfw: Optional[bool] = None,
        genres_exclude: Optional[List[int]] = None,
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
        letter: Optional[str] = None,
        producers: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: Optional[int] = None,
        unapproved: Optional[bool] = None,
        correlation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for anime on MAL/Jikan with full parameter support.

        Args:
            query: Search query string (optional for letter-based search)
            limit: Maximum number of results (1-50)
            status: Anime status filter (airing, complete, upcoming)
            genres: List of genre IDs to include
            anime_type: Anime type filter (TV, Movie, OVA, ONA, Special)
            score: Exact score filter
            min_score: Minimum score filter (0.0-10.0)
            max_score: Maximum score filter (0.0-10.0)  
            rating: Content rating (G, PG, PG-13, R, R+, Rx)
            sfw: Filter out adult content (Safe For Work)
            genres_exclude: List of genre IDs to exclude
            order_by: Order results by field (title, score, rank, popularity, etc.)
            sort: Sort direction (asc, desc)
            letter: Return entries starting with letter
            producers: List of producer/studio IDs
            start_date: Start date filter (YYYY-MM-DD, YYYY-MM, YYYY)
            end_date: End date filter (YYYY-MM-DD, YYYY-MM, YYYY)
            page: Page number for pagination
            unapproved: Include unapproved entries
            correlation_id: Correlation ID for request tracing

        Returns:
            List of anime search results

        Raises:
            ValueError: If parameter validation fails
            Exception: If search fails
        """
        # Parameter validation
        self._validate_search_parameters(
            anime_type=anime_type,
            min_score=min_score,
            max_score=max_score,
            rating=rating,
            order_by=order_by,
            sort=sort,
            start_date=start_date,
            end_date=end_date,
        )

        try:
            logger.info(
                "MAL search: query='%s', limit=%d, status=%s, correlation_id=%s", 
                query, limit, status, correlation_id
            )
            
            return await self.client.search_anime(
                query=query,
                limit=limit,
                status=status,
                genres=genres,
                anime_type=anime_type,
                score=score,
                min_score=min_score,
                max_score=max_score,
                rating=rating,
                sfw=sfw,
                genres_exclude=genres_exclude,
                order_by=order_by,
                sort=sort,
                letter=letter,
                producers=producers,
                start_date=start_date,
                end_date=end_date,
                page=page,
                unapproved=unapproved,
                correlation_id=correlation_id,
            )
        except Exception as e:
            logger.error("MAL search failed: %s", e)
            raise

    async def get_anime_details(self, anime_id: int, correlation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get detailed anime information by ID.

        Args:
            anime_id: MAL anime ID
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Anime details or None if not found
        """
        try:
            logger.info("MAL anime details: anime_id=%d, correlation_id=%s", anime_id, correlation_id)
            return await self.client.get_anime_by_id(anime_id)
        except Exception as e:
            logger.error("MAL anime details failed: %s, correlation_id=%s", e, correlation_id)
            raise

    async def get_seasonal_anime(self, year: int, season: str, correlation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get seasonal anime from MAL/Jikan.

        Args:
            year: Year (e.g., 2024)
            season: Season (winter, spring, summer, fall)
            correlation_id: Optional correlation ID for request tracing

        Returns:
            List of seasonal anime

        Raises:
            ValueError: If season is invalid
        """
        valid_seasons = ["winter", "spring", "summer", "fall"]
        if season.lower() not in valid_seasons:
            raise ValueError(
                f"Invalid season '{season}'. Must be one of: {valid_seasons}"
            )

        try:
            logger.info("MAL seasonal: year=%d, season=%s, correlation_id=%s", year, season, correlation_id)
            return await self.client.get_seasonal_anime(year, season.lower())
        except Exception as e:
            logger.error("MAL seasonal failed: %s, correlation_id=%s", e, correlation_id)
            raise

    async def get_current_season(self, correlation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current season anime.

        Args:
            correlation_id: Optional correlation ID for request tracing

        Returns:
            List of current season anime
        """
        # Determine current season
        now = datetime.now()
        month = now.month
        year = now.year

        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        else:  # month in [9, 10, 11]
            season = "fall"

        try:
            logger.info("MAL current season: year=%d, season=%s, correlation_id=%s", year, season, correlation_id)
            return await self.client.get_seasonal_anime(year, season)
        except Exception as e:
            logger.error("MAL current season failed: %s, correlation_id=%s", e, correlation_id)
            raise

    async def get_anime_statistics(self, anime_id: int, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Get anime statistics (watching, completed, etc.).

        Args:
            anime_id: MAL anime ID
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Statistics data
        """
        try:
            logger.info("MAL anime statistics: anime_id=%d, correlation_id=%s", anime_id, correlation_id)
            return await self.client.get_anime_statistics(anime_id)
        except Exception as e:
            logger.error("MAL anime statistics failed: %s, correlation_id=%s", e, correlation_id)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check service health status.

        Returns:
            Health status information
        """
        try:
            # Simple health check - try to search with limit 1
            await self.client.search_anime(query="test", limit=1)

            return {
                "service": self.service_name,
                "status": "healthy",
                "circuit_breaker_open": self.circuit_breaker.is_open(),
            }

        except Exception as e:
            logger.warning("MAL health check failed: %s", e)
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_open": self.circuit_breaker.is_open(),
            }

    def _validate_search_parameters(
        self,
        anime_type: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        rating: Optional[str] = None,
        order_by: Optional[str] = None,
        sort: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """Validate search parameters according to Jikan API specification.
        
        Args:
            anime_type: Anime type to validate
            min_score: Minimum score to validate
            max_score: Maximum score to validate
            rating: Rating to validate
            order_by: Order by field to validate
            sort: Sort direction to validate
            start_date: Start date to validate
            end_date: End date to validate
            
        Raises:
            ValueError: If any parameter is invalid
        """
        # Valid anime types according to Jikan API
        valid_anime_types = {"TV", "Movie", "OVA", "ONA", "Special"}
        if anime_type and anime_type not in valid_anime_types:
            raise ValueError(
                f"Invalid anime type '{anime_type}'. Must be one of: {valid_anime_types}"
            )
        
        # Score range validation
        if min_score is not None and max_score is not None:
            if min_score > max_score:
                raise ValueError("min_score must be less than or equal to max_score")
        
        if min_score is not None and (min_score < 0.0 or min_score > 10.0):
            raise ValueError("min_score must be between 0.0 and 10.0")
            
        if max_score is not None and (max_score < 0.0 or max_score > 10.0):
            raise ValueError("max_score must be between 0.0 and 10.0")
        
        # Valid content ratings according to MAL
        valid_ratings = {"G", "PG", "PG-13", "R", "R+", "Rx"}
        if rating and rating not in valid_ratings:
            raise ValueError(
                f"Invalid rating '{rating}'. Must be one of: {valid_ratings}"
            )
        
        # Valid order_by fields according to Jikan API
        valid_order_by = {
            "title", "score", "rank", "popularity", "members", "favorites", 
            "start_date", "end_date", "episodes", "type"
        }
        if order_by and order_by not in valid_order_by:
            raise ValueError(
                f"Invalid order_by '{order_by}'. Must be one of: {valid_order_by}"
            )
        
        # Valid sort directions
        valid_sort = {"asc", "desc"}
        if sort and sort not in valid_sort:
            raise ValueError(
                f"Invalid sort '{sort}'. Must be one of: {valid_sort}"
            )
        
        # Date format validation (basic)
        import re
        date_pattern = re.compile(r'^\d{4}(-\d{2})?(-\d{2})?$')
        
        if start_date and not date_pattern.match(start_date):
            raise ValueError(
                f"Invalid start_date format '{start_date}'. Use YYYY, YYYY-MM, or YYYY-MM-DD"
            )
            
        if end_date and not date_pattern.match(end_date):
            raise ValueError(
                f"Invalid end_date format '{end_date}'. Use YYYY, YYYY-MM, or YYYY-MM-DD"
            )
