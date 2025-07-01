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
        query: Optional[str] = None,
        genres: Optional[List[str]] = None,
        year: Optional[int] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search anime by various criteria."""
        search_query = """
        query ($search: String, $genre_in: [String], $seasonYear: Int, $perPage: Int) {
            Page(perPage: $perPage) {
                media(search: $search, genre_in: $genre_in, seasonYear: $seasonYear, type: ANIME) {
                    id
                    title {
                        romaji
                        english
                        native
                    }
                    averageScore
                }
            }
        }
        """

        variables = {
            "search": query,
            "genre_in": genres,
            "seasonYear": year,
            "perPage": limit,
        }

        response = await self._make_graphql_request(
            query=search_query, variables=variables
        )

        if response and "data" in response and "Page" in response["data"]:
            return response["data"]["Page"]["media"]
        return []

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
