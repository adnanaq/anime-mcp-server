"""AniList GraphQL client implementation."""

from typing import Any, Dict, List, Optional

import aiohttp

from .base_client import BaseClient
from ..rate_limiting import AniListRateLimitAdapter
from ..rate_limiter import rate_limit_manager


class AniListClient(BaseClient):
    """AniList GraphQL API client."""

    def __init__(
        self,
        auth_token: Optional[str] = None,
        circuit_breaker=None,
        rate_limiter=None,
        cache_manager=None,
        error_handler=None,
    ):
        """Initialize AniList client.

        Args:
            auth_token: Optional OAuth2 bearer token for authenticated requests
            circuit_breaker: Circuit breaker instance
            rate_limiter: Rate limiter instance
            cache_manager: Cache manager instance
            error_handler: Error handler instance
        """
        super().__init__(service_name="anilist", circuit_breaker=circuit_breaker, cache_manager=cache_manager, error_handler=error_handler)
        self.base_url = "https://graphql.anilist.co"
        self.auth_token = auth_token
        
        # Register AniList rate limiting adapter with global rate limiter
        self._rate_limit_adapter = AniListRateLimitAdapter()
        rate_limit_manager.register_platform_adapter("anilist", self._rate_limit_adapter)

    async def _make_graphql_request(
        self, query: str, variables: Optional[Dict[str, Any]] = None, correlation_id: str = None
    ) -> Dict[str, Any]:
        """Make GraphQL request using BaseClient infrastructure with enhanced error handling."""
        # Prepare auth headers if token provided
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        # Use BaseClient's GraphQL support with correlation tracking
        return await self.make_graphql_request(
            url=self.base_url,
            query=query,
            variables=variables,
            correlation_id=correlation_id,
            headers=headers,
            endpoint="graphql"
        )

    async def get_anime_by_id(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get anime by AniList ID."""
        # Check cache first
        cache_key = f"anilist_anime_{anime_id}"
        if self.cache_manager:
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    return cached_result
            except:
                # Cache miss or error, continue to API call
                pass

        # Check circuit breaker before making API call
        if self.circuit_breaker and self.circuit_breaker.is_open():
            raise Exception("Circuit breaker is open")

        query = """
        query ($id: Int) {
            Media(id: $id, type: ANIME) {
                id
                title {
                    romaji
                    english
                    native
                }
                description
                episodes
                duration
                status
                seasonYear
                genres
                averageScore
                popularity
                coverImage {
                    large
                }
                studios {
                    nodes {
                        name
                        isAnimationStudio
                    }
                }
                characters {
                    edges {
                        node {
                            name {
                                full
                            }
                        }
                        role
                    }
                }
            }
        }
        """

        variables = {"id": anime_id}
        correlation_id = f"anilist-get-anime-{anime_id}"
        response = await self._make_graphql_request(query=query, variables=variables, correlation_id=correlation_id)

        if response and "data" in response and response["data"]["Media"]:
            return response["data"]["Media"]
        return None

    async def search_anime(
        self,
        # Text search parameters
        query: Optional[str] = None,
        
        # Content classification filters  
        genres: Optional[List[str]] = None,
        genres_exclude: Optional[List[str]] = None,
        status: Optional[str] = None,
        format: Optional[str] = None,
        
        # Temporal filters
        year: Optional[int] = None,
        season: Optional[str] = None,
        start_date_greater: Optional[str] = None,
        start_date_lesser: Optional[str] = None,
        end_date_greater: Optional[str] = None,
        end_date_lesser: Optional[str] = None,
        
        # Numeric range filters
        average_score_greater: Optional[int] = None,
        average_score_lesser: Optional[int] = None,
        popularity_greater: Optional[int] = None,
        popularity_lesser: Optional[int] = None,
        episodes_greater: Optional[int] = None,
        episodes_lesser: Optional[int] = None,
        duration_greater: Optional[int] = None,
        duration_lesser: Optional[int] = None,
        
        # Production filters
        studios: Optional[List[str]] = None,
        
        # Content filters
        tags: Optional[List[str]] = None,
        tags_exclude: Optional[List[str]] = None,
        
        # Result control
        limit: int = 10,
        sort: Optional[List[str]] = None,
        
        # Platform options
        is_adult: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Search anime by various criteria with comprehensive AniList parameter support."""
        search_query = """
        query (
            $search: String, 
            $genre_in: [String], 
            $genre_not_in: [String],
            $status: MediaStatus,
            $format: MediaFormat,
            $seasonYear: Int,
            $season: MediaSeason,
            $startDate_greater: FuzzyDateInt,
            $startDate_lesser: FuzzyDateInt,
            $endDate_greater: FuzzyDateInt,
            $endDate_lesser: FuzzyDateInt,
            $averageScore_greater: Int,
            $averageScore_lesser: Int,
            $popularity_greater: Int,
            $popularity_lesser: Int,
            $episodes_greater: Int,
            $episodes_lesser: Int,
            $duration_greater: Int,
            $duration_lesser: Int,
            $tag_in: [String],
            $tag_not_in: [String],
            $sort: [MediaSort],
            $isAdult: Boolean,
            $perPage: Int
        ) {
            Page(perPage: $perPage) {
                media(
                    search: $search,
                    genre_in: $genre_in,
                    genre_not_in: $genre_not_in,
                    status: $status,
                    format: $format,
                    seasonYear: $seasonYear,
                    season: $season,
                    startDate_greater: $startDate_greater,
                    startDate_lesser: $startDate_lesser,
                    endDate_greater: $endDate_greater,
                    endDate_lesser: $endDate_lesser,
                    averageScore_greater: $averageScore_greater,
                    averageScore_lesser: $averageScore_lesser,
                    popularity_greater: $popularity_greater,
                    popularity_lesser: $popularity_lesser,
                    episodes_greater: $episodes_greater,
                    episodes_lesser: $episodes_lesser,
                    duration_greater: $duration_greater,
                    duration_lesser: $duration_lesser,
                    tag_in: $tag_in,
                    tag_not_in: $tag_not_in,
                    sort: $sort,
                    isAdult: $isAdult,
                    type: ANIME
                ) {
                    id
                    title {
                        romaji
                        english
                        native
                    }
                    description
                    format
                    episodes
                    duration
                    status
                    seasonYear
                    season
                    startDate {
                        year
                        month
                        day
                    }
                    endDate {
                        year
                        month
                        day
                    }
                    averageScore
                    popularity
                    favourites
                    genres
                    tags {
                        name
                        category
                    }
                    coverImage {
                        large
                        medium
                    }
                    studios {
                        nodes {
                            name
                            isAnimationStudio
                        }
                    }
                    siteUrl
                    source
                    isAdult
                }
                pageInfo {
                    total
                    hasNextPage
                }
            }
        }
        """

        # Build variables dict with all supported parameters
        variables = {
            "search": query,
            "genre_in": genres,
            "genre_not_in": genres_exclude,
            "status": status,
            "format": format,
            "seasonYear": year,
            "season": season,
            "startDate_greater": self._parse_fuzzy_date(start_date_greater),
            "startDate_lesser": self._parse_fuzzy_date(start_date_lesser),
            "endDate_greater": self._parse_fuzzy_date(end_date_greater),
            "endDate_lesser": self._parse_fuzzy_date(end_date_lesser),
            "averageScore_greater": average_score_greater,
            "averageScore_lesser": average_score_lesser,
            "popularity_greater": popularity_greater,
            "popularity_lesser": popularity_lesser,
            "episodes_greater": episodes_greater,
            "episodes_lesser": episodes_lesser,
            "duration_greater": duration_greater,
            "duration_lesser": duration_lesser,
            "tag_in": tags,
            "tag_not_in": tags_exclude,
            "sort": sort,
            "isAdult": is_adult,
            "perPage": limit,
        }
        
        # Remove None values to avoid GraphQL errors
        variables = {k: v for k, v in variables.items() if v is not None}

        correlation_id = f"anilist-search-{hash(str(variables))}"
        response = await self._make_graphql_request(
            query=search_query, variables=variables, correlation_id=correlation_id
        )

        if response and "data" in response and "Page" in response["data"]:
            return response["data"]["Page"]["media"]
        return []
    
    def _parse_fuzzy_date(self, date_string: Optional[str]) -> Optional[int]:
        """Parse date string into AniList FuzzyDateInt format (YYYYMMDD).
        
        Args:
            date_string: Date in YYYY-MM-DD format
            
        Returns:
            FuzzyDateInt (YYYYMMDD) or None if invalid
        """
        if not date_string:
            return None
            
        try:
            # Handle YYYY-MM-DD format
            if '-' in date_string:
                parts = date_string.split('-')
                if len(parts) == 3:
                    year, month, day = parts
                    return int(f"{year:0>4}{month:0>2}{day:0>2}")
                elif len(parts) == 2:
                    year, month = parts
                    return int(f"{year:0>4}{month:0>2}01")
                elif len(parts) == 1:
                    year = parts[0]
                    return int(f"{year:0>4}0101")
            
            # Handle YYYY format
            elif len(date_string) == 4 and date_string.isdigit():
                return int(f"{date_string}0101")
                
            return None
        except (ValueError, IndexError):
            return None

    async def get_anime_characters(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime characters."""
        query = """
        query ($id: Int) {
            Media(id: $id, type: ANIME) {
                characters {
                    edges {
                        node {
                            id
                            name {
                                full
                            }
                            image {
                                large
                            }
                        }
                        role
                    }
                }
            }
        }
        """

        variables = {"id": anime_id}
        correlation_id = f"anilist-characters-{anime_id}"
        response = await self._make_graphql_request(query=query, variables=variables, correlation_id=correlation_id)

        if response and "data" in response and response["data"]["Media"]:
            edges = response["data"]["Media"]["characters"]["edges"]
            return [
                {
                    "name": edge["node"]["name"],
                    "role": edge["role"],
                    "id": edge["node"]["id"],
                    "image": edge["node"]["image"],
                }
                for edge in edges
            ]
        return []

    async def get_anime_staff(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime staff."""
        query = """
        query ($id: Int) {
            Media(id: $id, type: ANIME) {
                staff {
                    edges {
                        node {
                            id
                            name {
                                full
                            }
                        }
                        role
                    }
                }
            }
        }
        """

        variables = {"id": anime_id}
        correlation_id = f"anilist-staff-{anime_id}"
        response = await self._make_graphql_request(query=query, variables=variables, correlation_id=correlation_id)

        if response and "data" in response and response["data"]["Media"]:
            edges = response["data"]["Media"]["staff"]["edges"]
            return [
                {
                    "name": edge["node"]["name"],
                    "role": edge["role"],
                    "id": edge["node"]["id"],
                }
                for edge in edges
            ]
        return []
    
    # ANILIST-SPECIFIC RATE LIMITING OVERRIDES
    # Override BaseClient hooks with AniList-specific behavior
    
    async def handle_rate_limit_response(self, response):
        """Handle AniList-specific rate limiting using the registered strategy.
        
        This method is called by BaseClient when a 429 is encountered.
        It now delegates to the global rate limiter's AniList strategy.
        
        Args:
            response: HTTP response with 429 status
        """
        # Extract rate limit info using our adapter
        rate_info = self._rate_limit_adapter.extract_rate_limit_info(response)
        
        # Use the strategy to handle the response
        strategy = self._rate_limit_adapter.get_strategy()
        await strategy.handle_rate_limit_response(rate_info)
    
    async def monitor_rate_limits(self, response):
        """Monitor AniList-specific rate limit headers for proactive management.
        
        This method extracts AniList headers and logs warnings, but the actual
        rate limit data is now processed by the global rate limiter.
        
        Args:
            response: HTTP response object
        """
        # Extract rate limit info using our adapter
        rate_info = self._rate_limit_adapter.extract_rate_limit_info(response)
        
        if rate_info.remaining is not None and rate_info.limit:
            # Calculate usage percentage
            usage_percent = ((rate_info.limit - rate_info.remaining) / rate_info.limit) * 100
            
            # Log warnings at AniList-specific thresholds
            if usage_percent >= 90:
                self.logger.warning(
                    f"AniList rate limit critical: {rate_info.remaining}/{rate_info.limit} requests remaining ({usage_percent:.1f}% used)"
                )
            elif usage_percent >= 75:
                self.logger.warning(
                    f"AniList rate limit high: {rate_info.remaining}/{rate_info.limit} requests remaining ({usage_percent:.1f}% used)"
                )
            elif usage_percent >= 50:
                self.logger.info(
                    f"AniList rate limit moderate: {rate_info.remaining}/{rate_info.limit} requests remaining ({usage_percent:.1f}% used)"
                )
    
    async def calculate_backoff_delay(self, response, attempt: int = 0):
        """Calculate AniList-specific backoff delay using the registered strategy.
        
        Args:
            response: HTTP response object (may be None for retry scenarios)
            attempt: Current retry attempt number
            
        Returns:
            Delay in seconds (float)
        """
        # Extract rate limit info if response is available
        rate_info = None
        if response:
            rate_info = self._rate_limit_adapter.extract_rate_limit_info(response)
        
        # Use the strategy to calculate backoff
        strategy = self._rate_limit_adapter.get_strategy()
        return await strategy.calculate_backoff_delay(rate_info, attempt)
