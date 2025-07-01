"""AniList GraphQL client implementation."""

from typing import Any, Dict, List, Optional

import aiohttp

from .base_client import BaseClient


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

    async def _make_graphql_request(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make GraphQL request to AniList API."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        payload = {"query": query}

        if variables:
            payload["variables"] = variables

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url, json=payload, headers=headers
            ) as response:
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After", "60")
                    raise Exception(
                        f"Rate limit exceeded. Retry after {retry_after} seconds"
                    )

                result = await response.json()

                if "errors" in result:
                    raise Exception(f"GraphQL error: {result['errors'][0]['message']}")

                return result

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
        response = await self._make_graphql_request(query=query, variables=variables)

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

        response = await self._make_graphql_request(
            query=search_query, variables=variables
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
        response = await self._make_graphql_request(query=query, variables=variables)

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
        response = await self._make_graphql_request(query=query, variables=variables)

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
