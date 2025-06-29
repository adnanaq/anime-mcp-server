#!/usr/bin/env python3
"""Proof of concept for anime data scraping using existing libraries.

This demonstrates practical scraping approaches without requiring additional dependencies.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional
from urllib.parse import quote, urljoin

import aiohttp
from bs4 import BeautifulSoup


class SimpleAnimeScraper:
    """Simple scraper using only aiohttp and BeautifulSoup."""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def extract_json_ld(self, html: str) -> Optional[Dict]:
        """Extract JSON-LD structured data from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all JSON-LD scripts
        scripts = soup.find_all('script', type='application/ld+json')
        
        for script in scripts:
            try:
                data = json.loads(script.string)
                
                # Look for anime-related types
                if isinstance(data, dict):
                    schema_type = data.get('@type', '')
                    if schema_type in ['TVSeries', 'Movie', 'CreativeWork', 'VideoObject']:
                        return data
                elif isinstance(data, list):
                    # Sometimes there are multiple schemas
                    for item in data:
                        if isinstance(item, dict) and item.get('@type') in ['TVSeries', 'Movie', 'CreativeWork']:
                            return item
            except json.JSONDecodeError:
                continue
        
        return None
    
    async def extract_opengraph(self, html: str) -> Dict[str, str]:
        """Extract OpenGraph metadata."""
        soup = BeautifulSoup(html, 'html.parser')
        og_data = {}
        
        # Find all OpenGraph meta tags
        og_tags = soup.find_all('meta', property=re.compile(r'^og:'))
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            content = tag.get('content', '')
            if property_name and content:
                og_data[property_name] = content
        
        return og_data
    
    async def scrape_myanimelist_api(self, anime_id: int) -> Optional[Dict]:
        """Scrape MyAnimeList using their unofficial API endpoints.
        
        Note: MAL has some endpoints that return JSON data without authentication.
        """
        # MAL has some public endpoints that return JSON
        url = f"https://myanimelist.net/anime/{anime_id}"
        
        try:
            async with self.session.get(url, headers=self.headers) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Extract data from various sources
                    data = {
                        'source': 'myanimelist',
                        'id': anime_id,
                        'url': url
                    }
                    
                    # Try JSON-LD first
                    json_ld = await self.extract_json_ld(html)
                    if json_ld:
                        data['structured_data'] = json_ld
                    
                    # Extract OpenGraph data
                    og_data = await self.extract_opengraph(html)
                    if og_data:
                        data['title'] = og_data.get('title', '')
                        data['description'] = og_data.get('description', '')
                        data['image'] = og_data.get('image', '')
                    
                    # Parse HTML for additional data
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Title
                    title_elem = soup.find('h1', class_='title-name')
                    if title_elem:
                        data['title'] = title_elem.text.strip()
                    
                    # Score
                    score_elem = soup.find('div', class_='score-label')
                    if score_elem:
                        data['score'] = score_elem.text.strip()
                    
                    # Synopsis
                    synopsis_elem = soup.find('p', itemprop='description')
                    if synopsis_elem:
                        data['synopsis'] = synopsis_elem.text.strip()
                    
                    # Information sidebar
                    info_dict = {}
                    info_section = soup.find('td', class_='borderClass')
                    if info_section:
                        for elem in info_section.find_all('div'):
                            text = elem.text.strip()
                            if ':' in text:
                                key, value = text.split(':', 1)
                                info_dict[key.strip()] = value.strip()
                    
                    data['info'] = info_dict
                    
                    return data
                    
        except Exception as e:
            print(f"Error scraping MAL: {e}")
            return None
    
    async def scrape_jikan_api(self, query: str) -> List[Dict]:
        """Use Jikan API (unofficial MAL API) for searching.
        
        Jikan provides free REST API access to MyAnimeList data.
        """
        # Jikan v4 API endpoint
        search_url = f"https://api.jikan.moe/v4/anime?q={quote(query)}&limit=5"
        
        try:
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    for anime in data.get('data', []):
                        results.append({
                            'mal_id': anime.get('mal_id'),
                            'title': anime.get('title'),
                            'title_english': anime.get('title_english'),
                            'synopsis': anime.get('synopsis'),
                            'type': anime.get('type'),
                            'episodes': anime.get('episodes'),
                            'score': anime.get('score'),
                            'status': anime.get('status'),
                            'genres': [g['name'] for g in anime.get('genres', [])],
                            'studios': [s['name'] for s in anime.get('studios', [])],
                            'year': anime.get('year'),
                            'season': anime.get('season'),
                            'image': anime.get('images', {}).get('jpg', {}).get('image_url'),
                            'url': anime.get('url')
                        })
                    
                    return results
                else:
                    print(f"Jikan API error: {response.status}")
                    return []
                    
        except Exception as e:
            print(f"Error calling Jikan API: {e}")
            return []
    
    async def scrape_anilist_graphql(self, query: str) -> List[Dict]:
        """Use AniList's public GraphQL API.
        
        AniList provides a public GraphQL API that doesn't require authentication
        for basic queries.
        """
        url = 'https://graphql.anilist.co'
        
        # GraphQL query for anime search
        graphql_query = '''
        query ($search: String) {
            Page(page: 1, perPage: 5) {
                media(search: $search, type: ANIME) {
                    id
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
                    averageScore
                    genres
                    studios {
                        nodes {
                            name
                        }
                    }
                    startDate {
                        year
                        month
                        day
                    }
                    coverImage {
                        large
                        medium
                    }
                    siteUrl
                }
            }
        }
        '''
        
        variables = {'search': query}
        
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
                        results.append({
                            'anilist_id': anime.get('id'),
                            'title': anime.get('title', {}).get('romaji'),
                            'title_english': anime.get('title', {}).get('english'),
                            'title_native': anime.get('title', {}).get('native'),
                            'description': anime.get('description', '').replace('<br>', '\n').replace('<i>', '').replace('</i>', ''),
                            'type': anime.get('type'),
                            'format': anime.get('format'),
                            'status': anime.get('status'),
                            'episodes': anime.get('episodes'),
                            'duration': anime.get('duration'),
                            'score': anime.get('averageScore'),
                            'genres': anime.get('genres', []),
                            'studios': [s['name'] for s in anime.get('studios', {}).get('nodes', [])],
                            'year': anime.get('startDate', {}).get('year'),
                            'image': anime.get('coverImage', {}).get('large'),
                            'url': anime.get('siteUrl')
                        })
                    
                    return results
                else:
                    print(f"AniList API error: {response.status}")
                    return []
                    
        except Exception as e:
            print(f"Error calling AniList API: {e}")
            return []
    
    async def demonstrate_scraping(self):
        """Demonstrate various scraping approaches."""
        print("🔍 Anime Scraping Proof of Concept")
        print("=" * 60)
        
        # Test search query
        search_query = "Cowboy Bebop"
        
        # 1. Try Jikan API (MAL)
        print(f"\n1. Searching via Jikan API (MyAnimeList)...")
        mal_results = await self.scrape_jikan_api(search_query)
        if mal_results:
            print(f"   ✅ Found {len(mal_results)} results")
            for i, anime in enumerate(mal_results[:2], 1):
                print(f"\n   Result {i}:")
                print(f"   - Title: {anime['title']}")
                print(f"   - Score: {anime['score']}")
                print(f"   - Episodes: {anime['episodes']}")
                print(f"   - Genres: {', '.join(anime['genres'])}")
        else:
            print("   ❌ No results or API error")
        
        # 2. Try AniList GraphQL API
        print(f"\n2. Searching via AniList GraphQL API...")
        anilist_results = await self.scrape_anilist_graphql(search_query)
        if anilist_results:
            print(f"   ✅ Found {len(anilist_results)} results")
            for i, anime in enumerate(anilist_results[:2], 1):
                print(f"\n   Result {i}:")
                print(f"   - Title: {anime['title']}")
                print(f"   - Score: {anime['score']}/100")
                print(f"   - Episodes: {anime['episodes']}")
                print(f"   - Genres: {', '.join(anime['genres'])}")
        else:
            print("   ❌ No results or API error")
        
        # 3. Direct HTML scraping example
        print(f"\n3. Direct HTML scraping from MyAnimeList...")
        # Cowboy Bebop MAL ID is 1
        mal_data = await self.scrape_myanimelist_api(1)
        if mal_data:
            print("   ✅ Successfully scraped page")
            print(f"   - Title: {mal_data.get('title', 'N/A')}")
            print(f"   - Has structured data: {'structured_data' in mal_data}")
            print(f"   - Has OpenGraph data: {bool(mal_data.get('image'))}")
        else:
            print("   ❌ Failed to scrape")
        
        print("\n" + "=" * 60)
        print("📊 Summary:")
        print("- Jikan API: Free, reliable, no auth needed")
        print("- AniList GraphQL: Free, powerful, no auth for public data")
        print("- Direct scraping: Works but requires careful parsing")
        print("\n💡 Recommendation: Use APIs when available, scrape as fallback")


async def main():
    """Run the scraping demonstration."""
    async with SimpleAnimeScraper() as scraper:
        await scraper.demonstrate_scraping()


if __name__ == "__main__":
    asyncio.run(main())