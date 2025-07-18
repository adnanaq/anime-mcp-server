#!/usr/bin/env python3
"""
AniList Helper for AI Enrichment Integration

Test script to fetch and analyze AniList data for anime entries using GraphQL API.
"""

import argparse
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import aiohttp
import time

logger = logging.getLogger(__name__)


class AniListEnrichmentHelper:
    """Helper for AniList data fetching in AI enrichment pipeline."""
    
    def __init__(self):
        """Initialize AniList enrichment helper."""
        self.base_url = "https://graphql.anilist.co"
        self.session = None
        self.rate_limit_remaining = 90  # AniList allows 90 requests per minute
        self.rate_limit_reset = None
        
    async def _make_request(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GraphQL request to AniList API."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
                
            async with self.session.post(self.base_url, json=payload, headers=headers) as response:
                # Handle rate limiting according to AniList docs
                if 'X-RateLimit-Remaining' in response.headers:
                    self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
                    
                if response.status == 429:
                    # Rate limit exceeded - wait and retry
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    # Retry the request
                    return await self._make_request(query, variables)
                elif response.status == 200:
                    data = await response.json()
                    if "errors" in data:
                        logger.error(f"AniList GraphQL errors: {data['errors']}")
                        return {}
                    return data.get("data", {})
                else:
                    logger.warning(f"AniList API error: HTTP {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"AniList API request failed: {e}")
            return {}
    
    def _build_comprehensive_query(self) -> str:
        """Build comprehensive GraphQL query for anime data."""
        return """
        query ($idMal: Int) {
          Media(idMal: $idMal, type: ANIME) {
            id
            idMal
            title {
              romaji
              english
              native
              userPreferred
            }
            description(asHtml: false)
            source
            format
            episodes
            duration
            status
            season
            seasonYear
            countryOfOrigin
            isAdult
            hashtag
            coverImage {
              extraLarge
              large
              medium
              color
            }
            bannerImage
            trailer {
              id
              site
              thumbnail
            }
            averageScore
            meanScore
            popularity
            favourites
            trending
            genres
            synonyms
            tags {
              id
              name
              description
              category
              rank
              isGeneralSpoiler
              isMediaSpoiler
              isAdult
            }
            relations {
              edges {
                node {
                  id
                  title {
                    romaji
                    english
                  }
                  format
                  status
                }
                relationType
              }
            }
            studios {
              edges {
                node {
                  id
                  name
                  isAnimationStudio
                }
                isMain
              }
            }
            externalLinks {
              id
              url
              site
              type
              language
              color
              icon
            }
            streamingEpisodes {
              title
              thumbnail
              url
              site
            }
            nextAiringEpisode {
              episode
              airingAt
              timeUntilAiring
            }
            airingSchedule(perPage: 50) {
              edges {
                node {
                  id
                  episode
                  airingAt
                  timeUntilAiring
                  mediaId
                }
              }
            }
            rankings {
              id
              rank
              type
              format
              year
              season
              allTime
              context
            }
            stats {
              scoreDistribution {
                score
                amount
              }
              statusDistribution {
                status
                amount
              }
            }
            updatedAt
          }
        }
        """
    
    async def fetch_anime_by_mal_id(self, mal_id: int) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive anime data by MAL ID."""
        try:
            # Respect rate limit - wait if needed
            if self.rate_limit_remaining and self.rate_limit_remaining < 5:
                logger.info(f"Rate limit low ({self.rate_limit_remaining}), waiting 60 seconds...")
                await asyncio.sleep(60)
                self.rate_limit_remaining = 90  # Reset after wait
            
            query = self._build_comprehensive_query()
            variables = {"idMal": mal_id}
            
            response = await self._make_request(query, variables)
            return response.get("Media") if response else None
        except Exception as e:
            logger.error(f"Failed to fetch anime by MAL ID {mal_id}: {e}")
            return None
    
    async def fetch_all_characters(self, anilist_id: int) -> List[Dict[str, Any]]:
        """Fetch all characters for an anime with pagination."""
        all_characters = []
        page = 1
        has_next_page = True
        
        character_query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            characters(page: $page, perPage: 25, sort: ROLE) {
              pageInfo {
                hasNextPage
                currentPage
              }
              edges {
                node {
                  id
                  name {
                    full
                    native
                    alternative
                    alternativeSpoiler
                  }
                  image {
                    large
                    medium
                  }
                  description
                  gender
                  dateOfBirth {
                    year
                    month
                    day
                  }
                  age
                  bloodType
                  favourites
                }
                role
                voiceActors(language: JAPANESE, sort: RELEVANCE) {
                  id
                  name {
                    full
                    native
                    alternative
                  }
                  image {
                    large
                    medium
                  }
                  description
                  primaryOccupations
                  gender
                  dateOfBirth {
                    year
                    month
                    day
                  }
                  age
                  yearsActive
                  homeTown
                  bloodType
                  favourites
                }
                voiceActorRoles(language: JAPANESE, sort: RELEVANCE) {
                  voiceActor {
                    id
                    name {
                      full
                      native
                    }
                  }
                  roleNotes
                  dubGroup
                }
              }
            }
          }
        }
        """
        
        try:
            while has_next_page:
                # Respect rate limit
                if self.rate_limit_remaining and self.rate_limit_remaining < 5:
                    logger.info(f"Rate limit low ({self.rate_limit_remaining}), waiting 60 seconds...")
                    await asyncio.sleep(60)
                    self.rate_limit_remaining = 90
                
                variables = {"id": anilist_id, "page": page}
                response = await self._make_request(character_query, variables)
                
                if not response or not response.get("Media"):
                    break
                
                characters_data = response["Media"]["characters"]
                page_info = characters_data.get("pageInfo", {})
                edges = characters_data.get("edges", [])
                
                if edges:
                    all_characters.extend(edges)
                
                has_next_page = page_info.get("hasNextPage", False)
                page += 1
                
                # Small delay between requests to be respectful
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error fetching characters for AniList ID {anilist_id}: {e}")
        
        return all_characters

    async def fetch_all_staff(self, anilist_id: int) -> List[Dict[str, Any]]:
        """Fetch all staff for an anime with pagination."""
        all_staff = []
        page = 1
        has_next_page = True
        
        staff_query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            staff(page: $page, perPage: 25, sort: RELEVANCE) {
              pageInfo {
                hasNextPage
                currentPage
              }
              edges {
                node {
                  id
                  name {
                    full
                    native
                    alternative
                  }
                  image {
                    large
                    medium
                  }
                  description
                  primaryOccupations
                  gender
                  dateOfBirth {
                    year
                    month
                    day
                  }
                  age
                  yearsActive
                  homeTown
                  bloodType
                  favourites
                }
                role
              }
            }
          }
        }
        """
        
        try:
            while has_next_page:
                # Respect rate limit
                if self.rate_limit_remaining and self.rate_limit_remaining < 5:
                    logger.info(f"Rate limit low ({self.rate_limit_remaining}), waiting 60 seconds...")
                    await asyncio.sleep(60)
                    self.rate_limit_remaining = 90
                
                variables = {"id": anilist_id, "page": page}
                response = await self._make_request(staff_query, variables)
                
                if not response or not response.get("Media"):
                    break
                
                staff_data = response["Media"]["staff"]
                page_info = staff_data.get("pageInfo", {})
                edges = staff_data.get("edges", [])
                
                if edges:
                    all_staff.extend(edges)
                
                has_next_page = page_info.get("hasNextPage", False)
                page += 1
                
                # Small delay between requests to be respectful
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error fetching staff for AniList ID {anilist_id}: {e}")
        
        return all_staff

    async def fetch_all_episodes(self, anilist_id: int) -> List[Dict[str, Any]]:
        """Fetch all episodes/airing schedule for an anime with pagination."""
        all_episodes = []
        page = 1
        has_next_page = True
        
        episodes_query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            airingSchedule(page: $page, perPage: 50) {
              pageInfo {
                hasNextPage
                currentPage
              }
              edges {
                node {
                  id
                  episode
                  airingAt
                  timeUntilAiring
                  mediaId
                }
              }
            }
          }
        }
        """
        
        try:
            while has_next_page:
                # Respect rate limit
                if self.rate_limit_remaining and self.rate_limit_remaining < 5:
                    logger.info(f"Rate limit low ({self.rate_limit_remaining}), waiting 60 seconds...")
                    await asyncio.sleep(60)
                    self.rate_limit_remaining = 90
                
                variables = {"id": anilist_id, "page": page}
                response = await self._make_request(episodes_query, variables)
                
                if not response or not response.get("Media"):
                    break
                
                episodes_data = response["Media"]["airingSchedule"]
                page_info = episodes_data.get("pageInfo", {})
                edges = episodes_data.get("edges", [])
                
                if edges:
                    all_episodes.extend(edges)
                
                has_next_page = page_info.get("hasNextPage", False)
                page += 1
                
                # Small delay between requests to be respectful
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error fetching episodes for AniList ID {anilist_id}: {e}")
        
        return all_episodes

    async def fetch_all_data(self, mal_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch all AniList data for an anime by MAL ID.
        
        Args:
            mal_id: The MyAnimeList ID
            
        Returns:
            Dict containing comprehensive AniList data or None if not found
        """
        try:
            # First fetch basic anime data
            anime_data = await self.fetch_anime_by_mal_id(mal_id)
            if not anime_data:
                logger.warning(f"No AniList data found for MAL ID: {mal_id}")
                return None
            
            anilist_id = anime_data.get("id")
            if not anilist_id:
                logger.error("No AniList ID found in anime data")
                return anime_data
                
            # Fetch all characters with pagination
            logger.info(f"Fetching all characters for AniList ID: {anilist_id}")
            all_characters = await self.fetch_all_characters(anilist_id)
            if all_characters:
                anime_data["characters"] = {"edges": all_characters}
                logger.info(f"Total characters fetched: {len(all_characters)}")
            
            # Fetch all staff with pagination
            logger.info(f"Fetching all staff for AniList ID: {anilist_id}")
            all_staff = await self.fetch_all_staff(anilist_id)
            if all_staff:
                anime_data["staff"] = {"edges": all_staff}
                logger.info(f"Total staff fetched: {len(all_staff)}")
            
            # Fetch all episodes/airing schedule with pagination
            logger.info(f"Fetching all episodes for AniList ID: {anilist_id}")
            all_episodes = await self.fetch_all_episodes(anilist_id)
            if all_episodes:
                anime_data["airingSchedule"] = {"edges": all_episodes}
                logger.info(f"Total episodes fetched: {len(all_episodes)}")
                
            logger.info(f"Successfully fetched comprehensive AniList data for MAL ID: {mal_id}")
            return anime_data
            
        except Exception as e:
            logger.error(f"Error in fetch_all_data for MAL ID {mal_id}: {e}")
            return None
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()


async def main():
    """Main function for testing AniList data fetching."""
    parser = argparse.ArgumentParser(description="Test AniList data fetching")
    parser.add_argument("--mal-id", type=int, required=True, help="MyAnimeList ID to fetch")
    parser.add_argument("--output", type=str, default="test_anilist_output.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    helper = AniListEnrichmentHelper()
    
    try:
        # Fetch data
        anime_data = await helper.fetch_all_data(args.mal_id)
        
        if anime_data:
            # Save to file
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(anime_data, f, indent=2, ensure_ascii=False)
        else:
            logger.error(f"No data found for MAL ID: {args.mal_id}")
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
    finally:
        await helper.close()


if __name__ == "__main__":
    asyncio.run(main())