#!/usr/bin/env python3
"""Demonstrate anime data access using free APIs instead of scraping.

This shows how to get anime data without complex scraping setup.
"""

import asyncio
import json
from typing import Dict, List, Optional
from urllib.parse import quote

import aiohttp


class AnimeAPIClient:
    """Client for accessing anime data through free APIs."""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_jikan(self, query: str, limit: int = 5) -> List[Dict]:
        """Search anime using Jikan API (unofficial MAL API).
        
        Jikan provides free REST API access to MyAnimeList data.
        No authentication required, but has rate limits.
        """
        # Jikan v4 API endpoint
        search_url = f"https://api.jikan.moe/v4/anime?q={quote(query)}&limit={limit}"
        
        try:
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    for anime in data.get('data', []):
                        results.append({
                            'source': 'myanimelist',
                            'mal_id': anime.get('mal_id'),
                            'title': anime.get('title'),
                            'title_english': anime.get('title_english'),
                            'synopsis': anime.get('synopsis'),
                            'type': anime.get('type'),
                            'episodes': anime.get('episodes'),
                            'score': anime.get('score'),
                            'scored_by': anime.get('scored_by'),
                            'status': anime.get('status'),
                            'aired': anime.get('aired', {}).get('string'),
                            'genres': [g['name'] for g in anime.get('genres', [])],
                            'themes': [t['name'] for t in anime.get('themes', [])],
                            'demographics': [d['name'] for d in anime.get('demographics', [])],
                            'studios': [s['name'] for s in anime.get('studios', [])],
                            'producers': [p['name'] for p in anime.get('producers', [])],
                            'year': anime.get('year'),
                            'season': anime.get('season'),
                            'rating': anime.get('rating'),
                            'image': anime.get('images', {}).get('jpg', {}).get('large_image_url'),
                            'url': anime.get('url')
                        })
                    
                    return results
                elif response.status == 429:
                    print("⚠️  Jikan API rate limit hit. Wait 1-2 seconds between requests.")
                    return []
                else:
                    print(f"Jikan API error: {response.status}")
                    return []
                    
        except Exception as e:
            print(f"Error calling Jikan API: {e}")
            return []
    
    async def search_anilist(self, query: str, limit: int = 5) -> List[Dict]:
        """Search anime using AniList's GraphQL API.
        
        AniList provides a public GraphQL API that doesn't require authentication
        for basic queries. More reliable than web scraping.
        """
        url = 'https://graphql.anilist.co'
        
        # GraphQL query for anime search with rich data
        graphql_query = '''
        query ($search: String, $perPage: Int) {
            Page(page: 1, perPage: $perPage) {
                pageInfo {
                    total
                    hasNextPage
                }
                media(search: $search, type: ANIME, sort: SEARCH_MATCH) {
                    id
                    idMal
                    title {
                        romaji
                        english
                        native
                    }
                    description
                    type
                    format
                    status
                    episodes
                    duration
                    chapters
                    volumes
                    season
                    seasonYear
                    averageScore
                    meanScore
                    popularity
                    favourites
                    genres
                    tags {
                        name
                        rank
                    }
                    studios {
                        nodes {
                            name
                            isAnimationStudio
                        }
                    }
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
                    coverImage {
                        extraLarge
                        large
                        medium
                    }
                    bannerImage
                    trailer {
                        id
                        site
                    }
                    synonyms
                    isAdult
                    countryOfOrigin
                    source
                    siteUrl
                }
            }
        }
        '''
        
        variables = {
            'search': query,
            'perPage': limit
        }
        
        try:
            async with self.session.post(
                url,
                json={'query': graphql_query, 'variables': variables},
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    for anime in data.get('data', {}).get('Page', {}).get('media', []):
                        # Clean description HTML
                        description = anime.get('description', '')
                        if description:
                            # Simple HTML tag removal
                            description = description.replace('<br>', '\n')
                            description = description.replace('<br/>', '\n')
                            description = description.replace('<i>', '')
                            description = description.replace('</i>', '')
                            description = description.replace('<b>', '')
                            description = description.replace('</b>', '')
                        
                        results.append({
                            'source': 'anilist',
                            'anilist_id': anime.get('id'),
                            'mal_id': anime.get('idMal'),
                            'title': anime.get('title', {}).get('romaji'),
                            'title_english': anime.get('title', {}).get('english'),
                            'title_native': anime.get('title', {}).get('native'),
                            'description': description,
                            'type': anime.get('type'),
                            'format': anime.get('format'),
                            'status': anime.get('status'),
                            'episodes': anime.get('episodes'),
                            'duration': anime.get('duration'),
                            'season': anime.get('season'),
                            'year': anime.get('seasonYear'),
                            'score': anime.get('averageScore'),
                            'popularity': anime.get('popularity'),
                            'favourites': anime.get('favourites'),
                            'genres': anime.get('genres', []),
                            'tags': [tag['name'] for tag in anime.get('tags', [])[:5]],  # Top 5 tags
                            'studios': [s['name'] for s in anime.get('studios', {}).get('nodes', []) if s.get('isAnimationStudio')],
                            'synonyms': anime.get('synonyms', []),
                            'adult': anime.get('isAdult', False),
                            'source_material': anime.get('source'),
                            'country': anime.get('countryOfOrigin'),
                            'image': anime.get('coverImage', {}).get('large'),
                            'banner': anime.get('bannerImage'),
                            'trailer': anime.get('trailer'),
                            'url': anime.get('siteUrl')
                        })
                    
                    return results
                else:
                    print(f"AniList API error: {response.status}")
                    return []
                    
        except Exception as e:
            print(f"Error calling AniList API: {e}")
            return []
    
    async def get_anime_details_jikan(self, mal_id: int) -> Optional[Dict]:
        """Get detailed anime information from Jikan API."""
        url = f"https://api.jikan.moe/v4/anime/{mal_id}/full"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    anime = data.get('data', {})
                    
                    return {
                        'source': 'myanimelist',
                        'mal_id': anime.get('mal_id'),
                        'title': anime.get('title'),
                        'title_japanese': anime.get('title_japanese'),
                        'title_synonyms': anime.get('title_synonyms', []),
                        'synopsis': anime.get('synopsis'),
                        'background': anime.get('background'),
                        'type': anime.get('type'),
                        'episodes': anime.get('episodes'),
                        'status': anime.get('status'),
                        'aired': anime.get('aired'),
                        'premiered': anime.get('premiered'),
                        'broadcast': anime.get('broadcast'),
                        'producers': [p['name'] for p in anime.get('producers', [])],
                        'licensors': [l['name'] for l in anime.get('licensors', [])],
                        'studios': [s['name'] for s in anime.get('studios', [])],
                        'source': anime.get('source'),
                        'genres': [g['name'] for g in anime.get('genres', [])],
                        'themes': [t['name'] for t in anime.get('themes', [])],
                        'demographics': [d['name'] for d in anime.get('demographics', [])],
                        'duration': anime.get('duration'),
                        'rating': anime.get('rating'),
                        'score': anime.get('score'),
                        'scored_by': anime.get('scored_by'),
                        'rank': anime.get('rank'),
                        'popularity': anime.get('popularity'),
                        'members': anime.get('members'),
                        'favorites': anime.get('favorites'),
                        'relations': anime.get('relations', []),
                        'theme': anime.get('theme', {}),
                        'external': anime.get('external', []),
                        'streaming': anime.get('streaming', [])
                    }
                else:
                    return None
        except Exception as e:
            print(f"Error getting anime details: {e}")
            return None
    
    async def get_seasonal_anime(self, year: int, season: str) -> List[Dict]:
        """Get anime from a specific season using Jikan API."""
        # Season must be: winter, spring, summer, or fall
        url = f"https://api.jikan.moe/v4/seasons/{year}/{season.lower()}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    for anime in data.get('data', []):
                        results.append({
                            'mal_id': anime.get('mal_id'),
                            'title': anime.get('title'),
                            'type': anime.get('type'),
                            'episodes': anime.get('episodes'),
                            'status': anime.get('status'),
                            'aired': anime.get('aired', {}).get('string'),
                            'score': anime.get('score'),
                            'genres': [g['name'] for g in anime.get('genres', [])],
                            'synopsis': anime.get('synopsis'),
                            'image': anime.get('images', {}).get('jpg', {}).get('image_url')
                        })
                    
                    return results
                else:
                    return []
        except Exception as e:
            print(f"Error getting seasonal anime: {e}")
            return []
    
    async def demonstrate_apis(self):
        """Demonstrate various API capabilities."""
        print("🚀 Anime Data Access via Free APIs")
        print("=" * 60)
        
        # Test search
        search_query = "Steins Gate"
        
        # 1. Search with Jikan (MAL)
        print(f"\n1. Searching '{search_query}' via Jikan API...")
        mal_results = await self.search_jikan(search_query, limit=3)
        if mal_results:
            print(f"   ✅ Found {len(mal_results)} results from MyAnimeList")
            for anime in mal_results[:1]:  # Show first result
                print(f"\n   Title: {anime['title']}")
                print(f"   Score: {anime['score']}/10 (by {anime['scored_by']:,} users)")
                print(f"   Episodes: {anime['episodes']} ({anime['status']})")
                print(f"   Genres: {', '.join(anime['genres'])}")
                print(f"   Studios: {', '.join(anime['studios'])}")
                if anime['synopsis']:
                    print(f"   Synopsis: {anime['synopsis'][:150]}...")
        
        # Add delay to respect rate limits
        await asyncio.sleep(1)
        
        # 2. Search with AniList
        print(f"\n2. Searching '{search_query}' via AniList GraphQL API...")
        anilist_results = await self.search_anilist(search_query, limit=3)
        if anilist_results:
            print(f"   ✅ Found {len(anilist_results)} results from AniList")
            for anime in anilist_results[:1]:  # Show first result
                print(f"\n   Title: {anime['title']} ({anime['title_native']})")
                print(f"   Score: {anime['score']}/100")
                print(f"   Episodes: {anime['episodes']} ({anime['format']})")
                print(f"   Genres: {', '.join(anime['genres'])}")
                print(f"   Tags: {', '.join(anime['tags'])}")
                print(f"   Studios: {', '.join(anime['studios'])}")
        
        await asyncio.sleep(1)
        
        # 3. Get detailed info
        print("\n3. Getting detailed info for Steins;Gate (MAL ID: 9253)...")
        details = await self.get_anime_details_jikan(9253)
        if details:
            print("   ✅ Retrieved full details")
            print(f"   Aired: {details['aired']['string']}")
            print(f"   Duration: {details['duration']}")
            print(f"   Rating: {details['rating']}")
            print(f"   Themes: {', '.join(details['themes'])}")
            print(f"   Producers: {', '.join(details['producers'][:3])}...")
            print(f"   External Links: {len(details['external'])} sites")
            print(f"   Streaming: {len(details['streaming'])} platforms")
        
        await asyncio.sleep(1)
        
        # 4. Get seasonal anime
        print("\n4. Getting current season anime (Winter 2025)...")
        seasonal = await self.get_seasonal_anime(2025, "winter")
        if seasonal:
            print(f"   ✅ Found {len(seasonal)} anime this season")
            # Show top 3 by score
            top_rated = sorted(seasonal, key=lambda x: x.get('score') or 0, reverse=True)[:3]
            for i, anime in enumerate(top_rated, 1):
                print(f"\n   #{i}: {anime['title']}")
                print(f"       Score: {anime['score'] or 'N/A'}")
                print(f"       Type: {anime['type']} - {anime['episodes'] or '?'} episodes")
        
        print("\n" + "=" * 60)
        print("📊 API Comparison:")
        print("\nJikan (MAL) API:")
        print("  ✅ Comprehensive MAL data")
        print("  ✅ No authentication needed") 
        print("  ⚠️  Rate limited (60 requests/minute)")
        print("  📖 Docs: https://docs.api.jikan.moe/")
        
        print("\nAniList GraphQL API:")
        print("  ✅ Rich, flexible queries")
        print("  ✅ Higher rate limits")
        print("  ✅ Real-time data")
        print("  📖 Docs: https://anilist.github.io/ApiV2-GraphQL-Docs/")
        
        print("\n💡 Recommendation:")
        print("Use these free APIs instead of scraping when possible!")
        print("Only scrape sites that don't have APIs (Anime-Planet, etc.)")


async def main():
    """Run the API demonstration."""
    async with AnimeAPIClient() as client:
        await client.demonstrate_apis()


if __name__ == "__main__":
    asyncio.run(main())