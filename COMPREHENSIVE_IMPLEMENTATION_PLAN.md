# Anime MCP Server - Comprehensive Implementation Plan

## Vision Statement

Transform the current static MCP server into a fully LLM-driven, dynamic system that can handle ANY anime-related query, no matter how complex or unpredictable, with intelligent multi-source data integration and progressive enhancement capabilities.

## Executive Summary

This plan outlines a complete architectural transformation from a static tool-based system to a unified LLM-driven anime discovery platform. The system will maintain backward compatibility while introducing revolutionary query understanding capabilities, multi-source data integration, and intelligent caching strategies.

## API Documentation References

### All 9 Anime Platform Sources

#### **API-Based Sources (5 platforms)**
1. **MyAnimeList (MAL) API v2**: https://myanimelist.net/apiconfig/references/api/v2#section/Versioning
2. **AniList GraphQL API**: https://docs.anilist.co/reference/
3. **Kitsu JSON:API**: https://kitsu.docs.apiary.io/ (Base URL: `https://kitsu.io/api/edge/`)
4. **AniDB API**: https://wiki.anidb.net/HTTP_API_Definition
5. **AnimeNewsNetwork API**: https://www.animenewsnetwork.com/encyclopedia/api.php

#### **Non-API Sources - Scraping Required (4 platforms)**
6. **Anime-Planet**: https://anime-planet.com (scraping required)
7. **LiveChart.me**: https://livechart.me (JSON-LD extraction)
8. **AniSearch**: https://anisearch.com (scraping required)
9. **AnimeCountdown**: https://animecountdown.com (scraping required)

#### **Additional Scheduling API (Not part of 9 core sources)**
- **AnimeSchedule.net API v3**: https://animeschedule.net/api/v3/documentation/anime
- **Jikan (MAL Unofficial)**: https://docs.api.jikan.moe/

### Rate Limiting & Authentication Information
- **MAL**: OAuth2 required, 2 req/sec, 60 req/min
- **AniList**: Optional OAuth2, 90 req/min burst limit
- **Kitsu**: No auth required for public endpoints, 10 req/sec
- **AniDB**: Client registration required, 1 req/sec
- **AnimeNewsNetwork**: No auth required, 1 req/sec
- **AnimeSchedule**: No authentication required, unlimited requests

## Current State Analysis

### Strengths ✅
- **ReactAgent Architecture**: Modern LangGraph integration with create_react_agent pattern
- **Multi-modal Search**: Text + image search via CLIP embeddings (512-dim)
- **AI Parameter Extraction**: 95% accuracy with GPT-4/Claude for query understanding
- **Comprehensive Database**: 38,894 anime entries from offline database
- **Platform Integration**: ID extraction for 9 anime services (MAL, AniList, Kitsu, etc.)
- **Performance**: <200ms text search, ~1s image search

### Critical Gaps ❌
1. **Static Tool Limitations**: Only 7 predefined tools, cannot adapt to novel query types
2. **No External API Integration**: Only uses offline database, no real-time data
3. **No User Management**: No account linking, personalization, or preference learning
4. **Limited Query Understanding**: Can't handle complex narrative or temporal queries

## Final Architecture Decisions

### 1. API Structure Transformation

#### Current Endpoints (TO BE REMOVED)
```
❌ /api/search/* (7 endpoints)
   - /api/search/semantic
   - /api/search/
   - /api/search/similar/{anime_id}
   - /api/search/by-image
   - /api/search/by-image-base64
   - /api/search/visually-similar/{anime_id}
   - /api/search/multimodal

❌ /api/recommendations/* (2 endpoints)
   - /api/recommendations/similar/{anime_id}
   - /api/recommendations/based-on-preferences

❌ /api/workflow/* (6 endpoints)
   - /api/workflow/conversation
   - /api/workflow/multimodal
   - /api/workflow/smart-conversation
   - /api/workflow/conversation/{session_id}
   - /api/workflow/stats
   - /api/workflow/health
```

#### New Unified Endpoints
```python
✅ /api/query          # Universal LLM-driven endpoint
✅ /api/batch          # Bulk operations endpoint
✅ /api/admin/*        # Keep all 8 admin endpoints unchanged
✅ /health, /stats     # Keep utility endpoints
```

### 2. Core Endpoint Specifications

#### `/api/query` - Universal LLM Endpoint

**Request Schema:**
```python
class QueryRequest(BaseModel):
    query: Union[str, Dict[str, Any]]  # Natural language or structured
    image: Optional[str] = None        # Base64 or URL for multimodal
    context: Optional[Dict[str, Any]] = None  # Conversation context
    options: Optional[Dict[str, Any]] = None  # Processing options
    user_id: Optional[str] = None      # For personalization
    session_id: Optional[str] = None   # For conversation continuity
```

**Response Schema:**
```python
class QueryResponse(BaseModel):
    results: List[AnimeResult]
    metadata: QueryMetadata
    sources: List[SourceInfo]
    suggestions: Optional[List[str]] = None
    conversation_id: Optional[str] = None
```

**Query Types Handled:**
```python
# Simple lookups
{"query": "get anime with id 12345"}

# Natural language
{"query": "find dark psychological anime like Death Note"}

# Complex narrative queries
{"query": "I vaguely remember an anime from 2010ish with underground tunnels and someone talking through dreams"}

# Source-specific queries
{"query": "show me Death Note from Anime-Planet"}

# Temporal queries
{"query": "anime from my childhood (born 1995)"}

# Comparison queries
{"query": "compare ratings of Steins;Gate across all platforms"}

# Image search
{"query": "find anime similar to this art style", "image": "base64..."}

# Contextual queries
{"query": "what about the second one?", "context": {"previous_results": [...]}}

# Multi-modal queries
{"query": "dark anime with this character design", "image": "base64..."}

# Streaming availability
{"query": "where can I watch Attack on Titan in US?"}

# Schedule queries
{"query": "what anime is airing today?"}
```

#### `/api/batch` - Bulk Operations

**Request Schema:**
```python
class BatchRequest(BaseModel):
    queries: List[QueryRequest]
    options: Optional[Dict[str, Any]] = None
    parallel: bool = True
    max_concurrent: int = 5
```

**Use Cases:**
- Bulk anime information retrieval
- Mass rating comparisons
- Batch image similarity searches
- Parallel complex query processing

### 3. Enhanced MCP Tool Architecture

#### Current 7 MCP Tools (Enhanced)

**1. search_anime (Enhanced with Smart Merging)**
```python
@tool("search_anime")
async def search_anime(
    query: str,
    genres: List[str] = None,
    limit: int = 10,
    enhance: bool = None,          # NEW: Trigger enhancement
    source: Optional[str] = None,   # NEW: Source-specific queries
    fields: Optional[List[str]] = None,  # NEW: Specific field requests
    year_range: Optional[Tuple[int, int]] = None,  # NEW: Temporal filtering
    rating_range: Optional[Tuple[float, float]] = None,  # NEW: Rating filtering
    content_type: Optional[List[str]] = None  # NEW: TV, MOVIE, ONA, etc.
) -> List[AnimeResult]:
    """
    Enhanced search with intelligent source selection and progressive enhancement.
    
    Enhancement Logic:
    - enhance=True: Always fetch enhanced data
    - enhance=None: Auto-detect based on query complexity
    - source specified: Route to specific platform
    - fields requested: Fetch only if not in offline DB
    
    Smart Merging Implementation:
    1. Start with offline database results
    2. If enhance=True or user specifies source:
       - Fetch data from external APIs
       - Smart merge offline + external data
       - Prioritize external data for requested fields
    3. Return unified results with source attribution
    """
    
    # Step 1: Start with offline database
    results = await qdrant_client.search(query, genres, limit)
    
    # Step 2: Determine if enhancement needed
    if enhance or _should_enhance(query, fields, source):
        service_manager = AnimeServiceManager()
        
        # Step 3: Smart merge offline + external API data
        enhanced_results = []
        for anime in results:
            enhanced = anime.dict()
            
            # Fetch missing fields from best sources
            if fields and "synopsis" in fields:
                synopsis = await service_manager.smart_fetch(
                    "synopsis", 
                    anime.anime_id,
                    preferred_source=source  # User preference
                )
                enhanced["synopsis"] = synopsis
                
            if fields and "streaming" in fields:
                streaming = await service_manager.smart_fetch(
                    "streaming",
                    anime.anime_id,
                    preferred_source=source
                )
                enhanced["streaming_links"] = streaming
                
            enhanced_results.append(enhanced)
            
        return enhanced_results
        
    return results  # Offline only
```

**2. get_anime_by_id (Enhanced)**
```python
@tool("get_anime_by_id")
async def get_anime_by_id(
    anime_id: str,
    enhance: bool = None,
    source: Optional[str] = None,
    fields: Optional[List[str]] = None
) -> AnimeResult:
    """
    Enhanced single anime retrieval with source routing.
    
    ID Resolution:
    - Internal ID: Direct offline DB lookup
    - MAL ID: mal:12345
    - AniList ID: anilist:12345
    - Source URL: Full URL parsing
    """
```

**3. find_similar_anime (Enhanced)**
```python
@tool("find_similar_anime")
async def find_similar_anime(
    anime_id: str,
    similarity_type: str = "semantic",  # semantic, visual, genre, studio
    limit: int = 10,
    enhance: bool = None,
    cross_source: bool = False  # NEW: Find similar across different sources
) -> List[AnimeResult]:
    """
    Multi-modal similarity with cross-source recommendations.
    """
```

**4. search_by_image (Enhanced)**
```python
@tool("search_by_image")
async def search_by_image(
    image_data: str,  # Base64 or URL
    similarity_threshold: float = 0.7,
    limit: int = 10,
    enhance: bool = None,
    include_characters: bool = False  # NEW: Character-based matching
) -> List[AnimeResult]:
    """
    CLIP-based image search with character recognition.
    """
```

**5. get_anime_recommendations (Enhanced)**
```python
@tool("get_anime_recommendations")
async def get_anime_recommendations(
    based_on: Union[str, List[str]],  # Anime IDs or titles
    user_preferences: Optional[Dict] = None,
    recommendation_type: str = "similar",  # similar, opposite, trending
    limit: int = 10,
    enhance: bool = None
) -> List[AnimeResult]:
    """
    AI-powered recommendations with preference learning.
    """
```

**6. get_seasonal_anime (Enhanced)**
```python
@tool("get_seasonal_anime")
async def get_seasonal_anime(
    season: Optional[str] = None,  # SPRING, SUMMER, FALL, WINTER
    year: Optional[int] = None,
    status: str = "all",  # airing, upcoming, finished
    enhance: bool = None,
    source: Optional[str] = None,  # For source-specific schedules
    timezone: str = "UTC"
) -> List[AnimeResult]:
    """
    Enhanced seasonal anime with real-time schedule data.
    """
```

**7. get_streaming_info (Enhanced)**
```python
@tool("get_streaming_info")
async def get_streaming_info(
    anime_id: str,
    region: str = "US",
    include_legal_only: bool = True,
    platform_preference: Optional[List[str]] = None
) -> StreamingInfo:
    """
    Real-time streaming availability with regional support.
    """
```

### 4. Project Structure for External APIs

#### New Integration Directory Structure

```
src/integrations/
├── cache.py                    # Multi-layer caching system
├── clients/                    # API client implementations  
│   ├── base_client.py         # Base client class with auth/rate limiting
│   ├── anilist_client.py      # AniList GraphQL client
│   ├── mal_client.py          # MyAnimeList/Jikan REST client
│   ├── kitsu_client.py        # Kitsu JSON:API client
│   ├── anidb_client.py        # AniDB XML client
│   ├── ann_client.py          # AnimeNewsNetwork client
│   └── animeschedule_client.py # AnimeSchedule.net API client
├── scrapers/                   # Web scraping components
│   ├── simple_scraper.py      # Cloudscraper-based scraper
│   └── extractors/            # Site-specific extractors
│       ├── anime_planet.py
│       ├── livechart.py       
│       ├── anisearch.py
│       └── animecountdown.py
└── service_manager.py         # Intelligent routing system
```

#### Enhanced MCP Tools Structure

```
src/mcp/
├── tools.py              # Enhanced MCP tools with API integration
├── server.py             # Existing MCP server
└── adapters/             # API service adapters
```

### 5. Multi-Source Data Integration Architecture

#### Three-Tier Enhancement Strategy

**Tier 1: Offline Database Only (Fastest)**
- **Response Time**: 50-200ms
- **Available Fields**: title, type, episodes, status, genres, pictures, sources, basic_score
- **Use When**: Basic metadata sufficient, no enhancement needed
- **Cache**: In-memory, permanent

**Tier 2: API Enhancement (When Available)**
- **Response Time**: 250-700ms
- **Enhanced Fields**: synopsis, characters, staff, detailed_scores, streaming_info, episode_details, schedule_data
- **Sources**: AniList (GraphQL), MAL/Jikan (REST), Kitsu (JSON:API), AniDB (XML), **AnimeSchedule.net (REST)**
- **Cache**: Redis, 1-24 hours TTL depending on data type

**Tier 3: Selective Scraping/Extraction (Last Resort)**  
- **Response Time**: 300-1000ms
- **Scraped/Extracted Fields**: reviews, platform-specific data, streaming_regions, alternative_schedules
- **Sources**: Anime-Planet, LiveChart (JSON-LD), AniSearch, AnimeNewsNetwork
- **Cache**: Redis, 1-6 hours TTL

#### Source Selection Strategy

**Intelligent Source Selection Based on Data Type:**
```python
class SourceSelector:
    """Intelligently select best source for specific data"""
    
    SOURCE_STRENGTHS = {
        'anilist': ['synopsis', 'characters', 'staff', 'relations'],
        'mal': ['score', 'reviews', 'recommendations', 'popularity'], 
        'kitsu': ['streaming', 'episodes', 'community'],
        'anidb': ['technical_details', 'file_hashes', 'fansubs'],
        'animeplanet': ['tags', 'recommendations', 'characters'],
        'livechart': ['airing_schedule', 'countdown', 'seasonal'],
        'animeschedule': ['episode_times', 'streaming_platforms'],
        'ann': ['news', 'industry_info', 'staff_details'],
        'anisearch': ['german_titles', 'german_synopsis'],
        'animecountdown': ['countdown', 'schedule', 'release_dates']
    }
    
    def select_source(self, field: str, user_preference: str = None) -> str:
        # User specified source takes priority
        if user_preference and user_preference in self.services:
            return user_preference
            
        # Find best source for specific field
        for source, strengths in self.SOURCE_STRENGTHS.items():
            if field in strengths:
                return source
                
        return 'anilist'  # Default fallback
```

#### Source URL Parsing System

**URL Pattern Recognition:**
```python
PLATFORM_PATTERNS = {
    # API-Based Sources (5 platforms from the 9 core sources)
    'mal': {
        'pattern': r'myanimelist\.net/anime/(\d+)',
        'api': 'jikan',  # Using Jikan unofficial API
        'fields': ['synopsis', 'episodes', 'score', 'rank', 'reviews'],
        'rate_limit': '2/sec',
        'priority': 'high'
    },
    'anilist': {
        'pattern': r'anilist\.co/anime/(\d+)',
        'api': 'graphql',
        'fields': ['description', 'characters', 'staff', 'studios', 'episodes'],
        'rate_limit': '90/min',
        'priority': 'highest'
    },
    'kitsu': {
        'pattern': r'kitsu\.io/anime/(\d+)',
        'api': 'jsonapi',
        'fields': ['synopsis', 'episodes', 'streaming_links'],
        'rate_limit': '10/sec',
        'priority': 'medium'
    },
    'anidb': {
        'pattern': r'anidb\.net/anime/(\d+)',
        'api': 'xml',
        'fields': ['episodes', 'staff', 'characters', 'detailed_metadata'],
        'rate_limit': '1/sec',
        'auth_required': True,
        'priority': 'low'
    },
    'ann': {
        'pattern': r'animenewsnetwork\.com/encyclopedia/anime\.php\?id=(\d+)',
        'api': 'xml',
        'fields': ['news', 'staff', 'episodes'],
        'rate_limit': '1/sec',
        'priority': 'low'
    },
    
    # Non-API Sources - Scraping Required (4 platforms from the 9 core sources)
    'animeplanet': {
        'pattern': r'anime-planet\.com/anime/([\w-]+)',
        'api': 'scraping',
        'fields': ['reviews', 'recommendations', 'tags', 'user_ratings'],
        'rate_limit': '1/sec',
        'priority': 'medium'
    },
    'livechart': {
        'pattern': r'livechart\.me/anime/(\d+)',
        'api': 'json_ld_extraction',
        'fields': ['schedule', 'streaming', 'episodes'],
        'rate_limit': '1/sec',
        'priority': 'low',
        'fallback_for': 'animeschedule'
    },
    'anisearch': {
        'pattern': r'anisearch\.com/anime/(\d+)',
        'api': 'scraping',
        'fields': ['synopsis', 'episodes', 'german_metadata'],
        'rate_limit': '1/sec',
        'priority': 'low'
    },
    'animecountdown': {
        'pattern': r'animecountdown\.com/anime/([\w-]+)',
        'api': 'scraping',
        'fields': ['countdown', 'schedule', 'release_dates'],
        'rate_limit': '1/sec',
        'priority': 'low'
    },
    
    # Additional APIs (not part of the 9 core sources)
    'animeschedule': {
        'pattern': r'animeschedule\.net',
        'api': 'rest',
        'fields': ['schedule', 'episode_times', 'streaming_platforms', 'seasonal_data'],
        'rate_limit': 'unlimited',
        'base_url': 'https://animeschedule.net/api/v3',
        'priority': 'highest',  # Preferred for scheduling data
        'auth_required': False
    }
}
```

#### Enhanced Data Model

**Complete Anime Data Structure:**
```python
class EnhancedAnimeData(BaseModel):
    # Base Data (Always Available - Offline DB)
    id: str
    title: str
    type: AnimeType
    episodes: Optional[int]
    status: AnimeStatus
    genres: List[str]
    pictures: List[str]
    sources: List[str]  # URLs to all 9 platforms
    
    # Enhanced Data (API/Scraping)
    synopsis: Optional[str] = None
    characters: List[Character] = []
    staff: List[StaffMember] = []
    studios: List[Studio] = []
    producers: List[Producer] = []
    episode_details: List[Episode] = []
    streaming_info: List[StreamingPlatform] = []
    reviews: List[Review] = []
    news: List[NewsItem] = []
    
    # LLM Computed Fields
    themes: List[str] = []
    content_warnings: List[str] = []
    narrative_elements: List[str] = []
    
    # Metadata
    enhanced_from: List[str] = []  # Which sources enhanced this data
    last_enhanced: Optional[datetime] = None
    enhancement_level: int = 1  # 1=offline, 2=api, 3=scraping
```

### 5. Multi-Level Caching Strategy

#### Cache Hierarchy Design

**L1 Cache (In-Memory) - Hot Data**
```python
L1_CACHE_CONFIG = {
    'size_limit': '1GB',
    'ttl_range': '5-15 minutes',
    'eviction': 'LRU',
    'content': [
        'popular_anime_metadata',
        'recent_search_results',
        'active_session_data',
        'trending_queries'
    ]
}
```

**L2 Cache (Redis) - Warm Data**
```python
L2_CACHE_CONFIG = {
    'size_limit': '10GB',
    'ttl_strategy': {
        'anime_metadata': '7 days',
        'api_responses': '6 hours',
        'scraped_data': '2 hours',
        'user_sessions': '24 hours',
        'search_results': '30 minutes'
    },
    'compression': 'gzip',
    'clustering': True
}
```

**L3 Cache (Offline Database) - Cold Data**
```python
L3_CACHE_CONFIG = {
    'refresh_frequency': 'weekly',
    'size': 'unlimited',
    'content': 'complete_anime_database',
    'backup_strategy': 'daily_snapshots'
}
```

#### Cache Key Patterns

**Hierarchical Key Structure:**
```python
CACHE_KEY_PATTERNS = {
    # Simple anime data
    'anime': 'anime:{anime_id}',
    'anime_enhanced': 'anime:{anime_id}:enhanced:{source}:{fields_hash}',
    
    # Search results
    'search': 'search:{query_hash}:{filters_hash}',
    'search_enhanced': 'search:{query_hash}:{strategy}:{enhancement_level}',
    
    # User data
    'user_session': 'session:{user_id}:{session_id}',
    'user_preferences': 'prefs:{user_id}',
    
    # Streaming data
    'streaming': 'stream:{anime_id}:{region}:{date}',
    
    # Schedule data  
    'schedule': 'schedule:{date}:{timezone}:{platform}',
    
    # Community cache
    'community': 'community:{anime_id}:{enhancement_type}',
    'quota_pool': 'quota:{source}:{date}:{hour}'
}
```

### 6. Request Coalescing & Community Cache System

#### Collaborative Community Cache

**Core Concept**: When User A fetches enhanced data for Anime X, cache it for all users to benefit from. This dramatically reduces API usage while improving response times.

**Implementation:**
```python
class CollaborativeCacheSystem:
    async def get_enhanced_data(self, anime_id: str, user_id: str, source: str) -> dict:
        # Step 1: Check personal quota
        if await self.has_personal_quota(user_id, source):
            data = await self.fetch_with_user_quota(anime_id, source, user_id)
            await self.share_with_community(anime_id, source, data)
            return data
        
        # Step 2: Check community cache
        if cached := await self.community_cache.get(f"community:{anime_id}:{source}"):
            await self.track_cache_usage(user_id, anime_id, 'community_hit')
            return cached['data']
        
        # Step 3: Find quota donor
        if donor_user := await self.find_quota_donor(source):
            data = await self.fetch_with_user_quota(anime_id, source, donor_user)
            await self.share_with_community(anime_id, source, data)
            await self.credit_donor(donor_user, anime_id)
            return data
        
        # Step 4: Fallback to cached/degraded data
        return await self.get_degraded_data(anime_id)

    async def share_with_community(self, anime_id: str, source: str, data: dict):
        """Share fetched data with community cache"""
        cache_key = f"community:{anime_id}:{source}"
        ttl = self.get_community_ttl(source, data)
        await self.community_cache.set(cache_key, {
            'data': data,
            'fetched_at': datetime.utcnow(),
            'source': source,
            'quality_score': self.calculate_quality_score(data)
        }, ttl=ttl)

    async def find_quota_donor(self, source: str) -> Optional[str]:
        """Find user with available quota willing to share"""
        quota_pool_key = f"quota:{source}:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"
        donors = await self.redis.lrange(quota_pool_key, 0, -1)
        
        for donor_id in donors:
            if await self.has_personal_quota(donor_id, source):
                return donor_id
        return None
```

#### Request Deduplication

**Prevent Multiple Simultaneous Requests:**
```python
class RequestDeduplication:
    def __init__(self):
        self.active_requests = {}  # request_key -> Future
        
    async def deduplicate_request(self, request_key: str, fetch_func):
        """Ensure only one request per unique key executes at a time"""
        if request_key in self.active_requests:
            # Wait for existing request to complete
            return await self.active_requests[request_key]
        
        # Create new request
        future = asyncio.create_task(fetch_func())
        self.active_requests[request_key] = future
        
        try:
            result = await future
            return result
        finally:
            # Clean up completed request
            self.active_requests.pop(request_key, None)
```

### 7. Web Scraping Implementation

#### Validated Scraping Test Results

**Based on successful testing with actual scripts (`scripts/scraping_poc.py` and `scripts/scraping_with_apis.py`), scraping is simpler than anticipated:**

**Successful Test Results:**
- ✅ **Jikan API (MAL)**: 100% success rate, comprehensive data, no auth needed
- ✅ **AniList GraphQL**: Rich queries, higher rate limits, real-time data
- ✅ **Direct HTML Scraping**: BeautifulSoup + aiohttp works for protected sites
- ✅ **JSON-LD Extraction**: LiveChart provides structured data
- ✅ **Response Times**: 200-500ms for API calls, under 1s for scraping

#### Validated Scraping Architecture

**Core Libraries (Validated in Production Scripts):**
```python
# From scripts/scraping_poc.py and scripts/scraping_with_apis.py
import aiohttp           # Async HTTP client (proven to work)
from bs4 import BeautifulSoup  # HTML parsing
import json             # JSON-LD extraction
import re               # Pattern matching
from urllib.parse import quote, urljoin

# Alternative for Cloudflare protection:
import cloudscraper     # Only if aiohttp fails
```

**API Integration & Extraction Results (Tested):**
- ✅ **AnimeSchedule.net API**: 100-200ms, comprehensive scheduling data, no authentication needed
- ✅ **Anime-Planet**: 200-300ms, full synopsis and tags (scraping)
- ✅ **LiveChart**: 150-250ms, JSON-LD structured data extraction
- ✅ **AniSearch**: 250-400ms, episode lists and descriptions (scraping)
- ✅ **AnimeNewsNetwork**: 300-500ms, news and staff info (XML API)
- ✅ **No Playwright/Selenium needed**: Simple HTTP requests sufficient

**Scraper Implementation:**
```python
class AnimePlatformScraper:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
        )
        self.rate_limiters = {
            source: AsyncLimiter(1, 1) for source in SCRAPABLE_SOURCES
        }
    
    async def scrape_anime_planet(self, anime_slug: str) -> dict:
        """Scrape Anime-Planet for reviews and detailed tags"""
        async with self.rate_limiters['animeplanet']:
            url = f"https://www.anime-planet.com/anime/{anime_slug}"
            response = self.scraper.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            return {
                'synopsis': self.extract_synopsis(soup),
                'tags': self.extract_tags(soup),
                'user_reviews': self.extract_reviews(soup),
                'similar_anime': self.extract_similar(soup)
            }
    
    async def get_animeschedule_data(self, anime_title: str = None, date: str = None) -> dict:
        """Get anime schedule data from AnimeSchedule.net API v3"""
        base_url = "https://animeschedule.net/api/v3"
        
        if anime_title:
            # Search for specific anime schedule
            url = f"{base_url}/anime/search?query={anime_title}"
        elif date:
            # Get schedule for specific date
            url = f"{base_url}/timetables/{date}"
        else:
            # Get today's schedule
            url = f"{base_url}/timetables"
        
        response = await self.session.get(url)
        return response.json()
    
    async def extract_livechart_json_ld(self, anime_id: str) -> dict:
        """Extract JSON-LD structured data from LiveChart (fallback)"""
        async with self.rate_limiters['livechart']:
            url = f"https://www.livechart.me/anime/{anime_id}"
            response = self.scraper.get(url)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            json_ld = soup.find('script', type='application/ld+json')
            if json_ld:
                return json.loads(json_ld.string)
            
            # Fallback to basic HTML extraction
            return {
                'schedule': self.extract_schedule(soup),
                'streaming_platforms': self.extract_streaming(soup),
                'episode_countdown': self.extract_countdown(soup)
            }
```

#### Scraping Triggers & Logic

**When to Scrape:**
```python
def should_scrape(source: str, fields: List[str], user_request: dict) -> bool:
    # Explicit source request by user
    if user_request.get('source') == source:
        return True
    
    # Required fields not available via API
    source_info = PLATFORM_PATTERNS[source]
    if source_info['api'] == 'scraping':
        available_fields = set(source_info['fields'])
        requested_fields = set(fields or [])
        if requested_fields.intersection(available_fields):
            return True
    
    # No API alternative available
    if source not in API_SOURCES and fields:
        return True
        
    return False
```

**Error Handling & Fallbacks:**
```python
async def scrape_with_fallbacks(self, source: str, anime_id: str) -> dict:
    """Scrape with comprehensive error handling"""
    try:
        # Primary scraping attempt
        return await self.scrapers[source](anime_id)
    except CloudflareBlock:
        # Switch to proxy or wait
        await asyncio.sleep(60)
        return await self.scrape_with_proxy(source, anime_id)
    except RateLimitExceeded:
        # Use cached data if available
        cached = await self.get_cached_scraped_data(source, anime_id)
        if cached:
            return cached
        raise
    except Exception as e:
        # Log error and return minimal data
        logger.error(f"Scraping failed for {source}:{anime_id}: {e}")
        return {'error': str(e), 'scraped': False}
```

### 8. Multi-Layer Caching Strategy Implementation

#### Comprehensive Caching Architecture

**From PLANNING_SCRATCHPAD.md - Validated Design:**
```python
# src/integrations/cache.py
class IntegrationCache:
    """Multi-layer caching for API responses"""
    
    def __init__(self):
        self.memory_cache = TTLCache(maxsize=10000, ttl=300)  # 5 min
        self.redis_cache = RedisCache(ttl=3600)               # 1 hour  
        self.db_cache = DatabaseCache(ttl=86400)              # 24 hours
        
    async def get_or_fetch(self, key: str, fetcher: Callable):
        # L1: Memory cache (fastest)
        if value := self.memory_cache.get(key):
            return value
            
        # L2: Redis cache  
        if value := await self.redis_cache.get(key):
            self.memory_cache[key] = value
            return value
            
        # L3: Database cache
        if value := await self.db_cache.get(key):
            await self.redis_cache.set(key, value)
            self.memory_cache[key] = value
            return value
            
        # L4: Fetch from external API
        value = await fetcher()
        await self._cache_value(key, value)
        return value
```

#### Cache Key Patterns

**Hierarchical Key Structure:**
```python
CACHE_KEY_PATTERNS = {
    # Simple anime data
    'anime': 'anime:{anime_id}',
    'anime_enhanced': 'anime:{anime_id}:enhanced:{source}:{fields_hash}',
    
    # Search results
    'search': 'search:{query_hash}:{filters_hash}',
    'search_enhanced': 'search:{query_hash}:{strategy}:{enhancement_level}',
    
    # External API responses
    'api_response': 'api:{source}:{endpoint}:{params_hash}',
    'scraped_data': 'scrape:{source}:{url_hash}',
    
    # Streaming data
    'streaming': 'stream:{anime_id}:{region}:{date}',
    
    # Schedule data  
    'schedule': 'schedule:{date}:{timezone}:{platform}'
}
```

### 9. Query Processing Pipeline

#### LLM-Driven Query Understanding

**Intent Classification:**
```python
class QueryIntentClassifier:
    INTENT_CATEGORIES = {
        'simple_lookup': {
            'patterns': ['get anime', 'find anime with id', 'show me'],
            'complexity': 1,
            'enhancement_needed': False
        },
        'semantic_search': {
            'patterns': ['anime like', 'similar to', 'dark psychological'],
            'complexity': 2,
            'enhancement_needed': True
        },
        'narrative_search': {
            'patterns': ['I remember', 'vaguely recall', 'there was this anime'],
            'complexity': 4,
            'enhancement_needed': True,
            'fields_needed': ['synopsis', 'plot_summary']
        },
        'temporal_search': {
            'patterns': ['from 2010', 'childhood anime', 'old anime'],
            'complexity': 3,
            'enhancement_needed': False
        },
        'comparison_query': {
            'patterns': ['compare', 'vs', 'which is better'],
            'complexity': 5,
            'enhancement_needed': True,
            'multi_source': True
        },
        'streaming_query': {
            'patterns': ['where to watch', 'streaming on', 'available on'],
            'complexity': 3,
            'enhancement_needed': True,
            'real_time_data': True
        }
    }
```

**Parameter Extraction Pipeline:**
```python
async def extract_query_parameters(self, query: str, context: dict = None) -> dict:
    """Extract structured parameters from natural language query"""
    
    # Step 1: Intent classification
    intent = await self.classify_intent(query)
    
    # Step 2: Entity extraction
    entities = await self.extract_entities(query, intent)
    
    # Step 3: Temporal processing
    temporal_info = await self.extract_temporal_context(query, context)
    
    # Step 4: Source routing
    source_preference = await self.detect_source_preference(query)
    
    # Step 5: Enhancement strategy
    enhancement_strategy = await self.determine_enhancement_strategy(
        intent, entities, source_preference
    )
    
    return {
        'intent': intent,
        'entities': entities,
        'temporal': temporal_info,
        'source': source_preference,
        'enhancement': enhancement_strategy
    }
```

#### Complex Query Processing Examples

**Example 1: Narrative Memory Query**
```
Input: "I vaguely remember an anime from 2010ish with underground tunnels and someone talking through dreams"

Processing Flow:
1. Intent Classification: narrative_search (complexity: 4)
2. Entity Extraction: 
   - temporal: "2010ish" → {year_range: [2008, 2012]}
   - keywords: ["underground", "tunnels", "dreams", "communication"]
3. Query Expansion:
   - genres: ["Psychological", "Mystery", "Sci-Fi"]
   - themes: ["mind_control", "parallel_worlds", "underground_setting"]
4. Multi-Source Search:
   - Offline DB: Full-text search on titles and available synopsis
   - AniList: GraphQL query with description matching
   - MAL: Enhanced search with genre filtering
5. LLM Ranking: Score results by semantic similarity to description
6. Result: Ranked list with confidence scores and match explanations
```

**Example 2: Source-Specific Enhanced Query**
```
Input: "What's the plot of Steins;Gate according to Anime-Planet users?"

Processing Flow:
1. Intent Classification: simple_lookup + source_specific
2. Entity Extraction:
   - anime: "Steins;Gate"
   - source: "anime-planet"
   - fields: ["plot", "user_reviews"]
3. Source Routing: Route to Anime-Planet scraper
4. Enhancement Strategy: 
   - Check offline DB for Anime-Planet URL
   - Scrape synopsis and user reviews
   - Return with source attribution
5. Result: Enhanced data with Anime-Planet user perspective
```

**Example 3: Cross-Platform Comparison**
```  
Input: "Compare Death Note ratings across all platforms"

Processing Flow:
1. Intent Classification: comparison_query (complexity: 5)
2. Entity Extraction: anime: "Death Note"
3. Multi-Source Orchestration:
   - Find Death Note in offline DB
   - Extract all source URLs (9 platforms)
   - Parallel requests to available APIs
   - Scrape ratings from non-API sources
4. Data Aggregation:
   - Normalize rating scales (10-point, 5-star, percentage)
   - Calculate statistical summary
   - Note platform-specific biases
5. Result: Comprehensive rating comparison with analysis
```

### 9. Implementation Phase Breakdown

## Phase 1: Core LLM Endpoint & Architecture (2-3 weeks)

### Week 1: Unified Query Handler
**Priority: CRITICAL**

#### Tasks:
1. **Implement `/api/query` endpoint** (2 days)
   ```python
   # File: src/api/unified_query.py
   @router.post("/query", response_model=QueryResponse)
   async def unified_query_handler(request: QueryRequest):
       # LLM-driven query processing
       # Parameter extraction
       # Tool routing and orchestration
       # Response formatting
   ```

2. **Parameter extraction service** (2 days)
   ```python
   # File: src/services/query_understanding.py
   class QueryUnderstandingService:
       async def extract_parameters(self, query: str) -> QueryParameters
       async def classify_intent(self, query: str) -> IntentType
       async def determine_enhancement_strategy(self, params: QueryParameters) -> EnhancementStrategy
   ```

3. **Enhanced MCP tool interfaces** (2 days)
   - Add source, enhance, fields parameters to all 7 tools
   - Implement intelligent enhancement logic
   - Add source-aware routing

4. **Remove legacy endpoints** (1 day)
   - Delete /api/search/* routes
   - Delete /api/recommendations/* routes
   - Delete /api/workflow/* routes
   - Update route mounting

#### Acceptance Criteria:
- All existing queries work through `/api/query`
- Parameter extraction accuracy >90%
- Response time degradation <3x current times
- All MCP tools enhanced with new parameters

### Week 2: Source URL Parsing & Routing
**Priority: HIGH**

#### Tasks:
1. **Source URL parsing system** (2 days)
   ```python
   # File: src/services/source_parser.py
   class SourceURLParser:
       def extract_platform_ids(self, source_urls: List[str]) -> Dict[str, str]
       def route_to_source(self, source: str, anime_id: str) -> SourceAdapter
   ```

2. **Basic caching layer** (2 days)
   - L1 in-memory cache for hot data
   - L2 Redis cache for API responses
   - Cache key normalization

3. **Request deduplication** (1 day)
   - Prevent simultaneous identical requests
   - Share results across users

4. **Error handling & fallbacks** (2 days)
   - Graceful degradation strategies
   - Circuit breaker patterns
   - Monitoring and alerting

#### Acceptance Criteria:
- Source routing works for all 9 platforms
- Cache hit rate >70% for popular anime
- Graceful handling of API failures
- Request deduplication functional

### Week 3: Performance Optimization
**Priority: MEDIUM**

#### Tasks:
1. **LLM response caching** (2 days)
   - Cache LLM parameter extraction results
   - Normalize queries for cache keys
   - TTL strategies for different query types

2. **Parallel processing** (2 days)
   - Async tool execution
   - Concurrent API calls
   - Request batching

3. **Monitoring setup** (1 day)
   - Performance metrics
   - Error tracking  
   - Cost monitoring

4. **Load testing** (2 days)
   - Stress test new endpoint
   - Performance benchmarking
   - Optimization based on results

#### Acceptance Criteria:
- Average response time <1s for simple queries
- Support 50+ concurrent users
- Comprehensive monitoring dashboard
- Load test results documented

## Phase 2: Multi-Source API Integration (3-4 weeks)

### Week 4-5: Primary API Integrations
**Priority: HIGH**

#### Implementation Priority (From Planning Analysis)

**Phase 1: Core API Services (No Auth Required)**
- AniList (GraphQL) - Highest priority for synopsis, characters, staff
- Jikan/MAL (REST) - High priority for scores, reviews, popularity  
- Kitsu (JSON:API) - Medium priority for streaming, episodes
- AnimeSchedule.net (REST) - Medium priority for scheduling data

**Phase 2: Additional APIs & Enhanced Caching**
- AniDB (XML) - Low priority, requires auth
- AnimeNewsNetwork (XML) - Low priority for news/staff
- Redis caching layer implementation
- Circuit breakers and rate limiting

#### Tasks:
1. **AniList GraphQL client** (3 days)
   ```python
   # File: src/integrations/clients/anilist_client.py
   # Reference: https://docs.anilist.co/reference/
   # Validated with scripts/scraping_with_apis.py
   class AniListClient(BaseClient):
       async def get_anime_details(self, anime_id: str) -> dict
       async def search_anime(self, query: str) -> List[dict]
       async def get_characters(self, anime_id: str) -> List[dict]
       async def get_staff(self, anime_id: str) -> List[dict]
   ```

2. **MAL/Jikan REST client** (3 days)
   ```python
   # File: src/integrations/clients/mal_client.py
   # Reference: https://myanimelist.net/apiconfig/references/api/v2
   # Jikan: https://docs.api.jikan.moe/
   # Validated with scripts/scraping_poc.py and scripts/scraping_with_apis.py
   class MALClient(BaseClient):
       async def get_anime(self, mal_id: str) -> dict
       async def get_anime_statistics(self, mal_id: str) -> dict
       async def search_anime(self, query: str) -> List[dict]
       async def get_seasonal_anime(self, year: int, season: str) -> List[dict]
       async def get_anime_details_full(self, mal_id: int) -> dict  # Full endpoint
   ```

3. **Kitsu JSON:API client** (2 days)
   ```python
   # File: src/integrations/clients/kitsu_client.py
   # Reference: https://kitsu.docs.apiary.io/
   # Base URL: https://kitsu.io/api/edge/
   class KitsuClient(BaseClient):
       async def get_anime(self, kitsu_id: str) -> dict
       async def get_streaming_links(self, anime_id: str) -> List[dict]
       async def search_anime(self, query: str) -> List[dict]
   ```

4. **AnimeSchedule.net API client** (1 day)
   ```python
   # File: src/integrations/clients/animeschedule_client.py
   # Reference: https://animeschedule.net/api/v3/documentation/anime
   class AnimeScheduleClient(BaseClient):
       async def get_today_schedule(self) -> dict
       async def get_seasonal_anime(self, season: str, year: int) -> List[dict]
       async def search_schedule(self, anime_title: str) -> dict
       async def get_timetables(self, date: str = None) -> dict
   ```

5. **Service Manager & Base Client** (2 days)
   ```python
   # File: src/integrations/service_manager.py
   class AnimeServiceManager:
       async def smart_fetch(self, field: str, anime_id: str, preferred_source: str = None) -> Any
       async def get_best_source_for_field(self, field: str) -> str
       async def merge_data_sources(self, offline_data: dict, api_data: dict) -> dict
   
   # File: src/integrations/clients/base_client.py
   class BaseClient:
       def __init__(self, rate_limiter: AsyncLimiter, circuit_breaker: CircuitBreaker)
       async def make_request(self, url: str, **kwargs) -> dict
       async def handle_rate_limit(self, response: aiohttp.ClientResponse) -> None
   ```

6. **Rate limiting & circuit breakers** (2 days)
   - Implement per-source rate limiters
   - Circuit breaker patterns
   - Backoff strategies

#### Acceptance Criteria:
- All 4 primary APIs integrated (AniList, MAL, Kitsu, AnimeSchedule)
- Rate limiting prevents API violations (validated with actual API limits)
- Circuit breakers handle API downtime gracefully
- Enhanced data available for popular anime
- Service manager intelligently routes requests to best sources
- Smart data merging preserves offline data while enhancing with API data

### Week 6-7: Scraping Implementation
**Priority: MEDIUM**

#### Tasks:
1. **Simple scraper foundation** (1 day)
   ```python
   # File: src/integrations/scrapers/simple_scraper.py
   # Based on validated scripts/scraping_poc.py approach
   class SimpleAnimeScraper:
       def __init__(self):
           self.session = aiohttp.ClientSession()  # Validated approach
           self.cache = TTLCache(maxsize=1000, ttl=3600)
       
       async def extract_json_ld(self, html: str) -> Optional[Dict]
       async def extract_opengraph(self, html: str) -> Dict[str, str]
   ```

2. **Anime-Planet scraper** (2 days)
   ```python
   # File: src/integrations/scrapers/extractors/anime_planet.py
   class AnimePlanetExtractor:
       async def scrape_anime_details(self, slug: str) -> dict
       async def scrape_reviews(self, slug: str) -> List[dict]
       async def scrape_tags(self, slug: str) -> List[str]
   ```

3. **LiveChart JSON-LD extractor** (1 day)
   ```python
   # File: src/integrations/scrapers/extractors/livechart.py
   # Validated: LiveChart provides JSON-LD structured data
   class LiveChartExtractor:
       async def extract_structured_data(self, anime_id: str) -> dict
       async def extract_schedule_data(self, url: str) -> dict
       async def get_episode_countdown(self, anime_id: str) -> dict
   ```

4. **Additional scrapers** (3 days)
   ```python
   # File: src/integrations/scrapers/extractors/anisearch.py
   class AniSearchExtractor:
       async def scrape_german_synopsis(self, anime_id: str) -> dict
       async def scrape_episode_list(self, anime_id: str) -> List[dict]
   
   # File: src/integrations/scrapers/extractors/animecountdown.py  
   class AnimeCountdownExtractor:
       async def scrape_countdown_data(self, anime_slug: str) -> dict
       async def scrape_release_schedule(self, anime_slug: str) -> dict
   ```
   - Error handling and retries
   - Cloudflare bypass (if aiohttp fails)

5. **Scraping orchestration** (1 day)
   ```python
   # File: src/integrations/service_manager.py (enhanced)
   class AnimeServiceManager:
       async def should_scrape(self, source: str, fields: List[str], user_request: dict) -> bool
       async def scrape_with_fallbacks(self, source: str, anime_id: str) -> dict
       async def handle_scraping_errors(self, source: str, error: Exception) -> dict
   ```
   - Smart scraping triggers (only when user requests specific source)
   - Fallback chains (API → Scraping → Cached → Offline)
   - On-demand scraping (no proactive scraping)

#### Acceptance Criteria:
- 4+ scraping sources functional (Anime-Planet, LiveChart, AniSearch, AnimeCountdown)
- Scraping success rate >85% (based on test results: 200-500ms response times)
- Fallback chains prevent complete failures
- Scraping triggered only when user specifies source or requests unavailable fields
- JSON-LD extraction working for structured data sources
- aiohttp-based approach proven in production scripts

## Phase 3: Advanced Query Understanding (4-5 weeks)

### Week 8-9: Complex Query Processing
**Priority: MEDIUM**

#### Tasks:
1. **Narrative query processor** (3 days)
   ```python
   # File: src/services/narrative_processor.py
   class NarrativeQueryProcessor:
       async def extract_plot_elements(self, query: str) -> List[str]
       async def expand_search_terms(self, elements: List[str]) -> dict
       async def rank_by_narrative_similarity(self, results: List[dict], query: str) -> List[dict]
   ```

2. **Temporal understanding** (3 days)
   - Relative time parsing ("childhood", "few years ago")
   - Contextual date resolution
   - Era-based filtering

3. **Ambiguity resolution** (2 days)
   - Multiple interpretation handling
   - Clarification requests
   - Context-based disambiguation

4. **Multi-step orchestration** (2 days)
   - Complex query breakdown
   - Progressive refinement
   - Result synthesis

#### Acceptance Criteria:
- Handle complex narrative queries
- Temporal queries resolve correctly
- Multi-step queries execute properly
- Ambiguous queries prompt for clarification

### Week 10-11: Predictive & Analytical Features
**Priority: LOW**

#### Tasks:
1. **Query pattern analysis** (2 days)
   - Popular query tracking
   - Pattern recognition
   - Trend identification

2. **Predictive cache warming** (3 days)
   - ML-based popularity prediction
   - Proactive data fetching
   - Cache optimization

3. **Advanced analytics** (2 days)
   - Cross-platform analysis
   - Trend reporting
   - Usage insights

4. **Performance optimization** (3 days)
   - Query optimization
   - Cache strategy refinement
   - Response time improvements

#### Acceptance Criteria:
- Predictive caching improves response times
- Analytics provide useful insights
- Complex queries execute in <3s
- Cache hit rate >90% for popular content

### Week 12: Integration & Testing
**Priority: HIGH**

#### Tasks:
1. **End-to-end testing** (2 days)
   - Complex query test suite
   - Performance benchmarking
   - Error scenario testing

2. **Integration testing** (2 days)
   - API integration validation
   - Scraping reliability testing
   - Cache consistency verification

3. **Performance tuning** (2 days)
   - Bottleneck identification
   - Optimization implementation
   - Load testing

4. **Documentation** (1 day)
   - API documentation updates
   - Usage examples
   - Troubleshooting guides

#### Acceptance Criteria:
- All tests passing
- Performance meets targets
- Documentation complete
- System ready for production

## Phase 4: User Personalization (2-3 weeks)

### Week 13-14: User Management System
**Priority: MEDIUM**

#### Tasks:
1. **User profile system** (3 days)
   ```python
   # File: src/models/user.py
   class UserProfile(BaseModel):
       user_id: str
       preferences: UserPreferences
       watch_history: List[str]
       linked_accounts: Dict[str, str]
   ```

2. **OAuth2 integration** (3 days)
   - MAL OAuth implementation
   - AniList OAuth implementation
   - Secure token management

3. **Preference learning** (2 days)
   - Implicit preference extraction
   - Rating prediction
   - Recommendation personalization

4. **Cross-platform sync** (2 days)
   - Watch status synchronization
   - Rating synchronization
   - List management

#### Acceptance Criteria:
- User accounts functional
- OAuth2 integration working
- Preferences learned from usage
- Cross-platform sync operational

### Week 15: Privacy & Advanced Features
**Priority: LOW**

#### Tasks:
1. **Privacy controls** (2 days)
   - Data retention policies
   - User data export
   - Account deletion

2. **Advanced personalization** (2 days)
   - Mood-based recommendations
   - Seasonal preference tracking
   - Social influence modeling

3. **Social features** (2 days)
   - Shared watch lists
   - Friend recommendations
   - Community features

4. **Mobile optimization** (1 day)
   - Mobile-specific endpoints
   - Push notification support
   - Offline capability

#### Acceptance Criteria:
- Privacy controls implemented
- Advanced features functional
- Social features operational
- Mobile support available

## Phase 5: Advanced Features & Analytics (3-4 weeks)

### Week 16-17: Real-Time Features
**Priority: LOW**

#### Tasks:
1. **Episode notification system** (3 days)
   - Episode release tracking
   - User notification preferences
   - Multi-platform schedule aggregation

2. **Streaming availability tracker** (3 days)
   - Regional availability monitoring
   - Platform change notifications
   - Price tracking

3. **News aggregation** (2 days)
   - Industry news tracking
   - Anime-specific news
   - Personalized news feeds

4. **Community features** (2 days)
   - Discussion forums
   - Review aggregation
   - Rating systems

#### Acceptance Criteria:
- Episode notifications working
- Streaming changes tracked
- News feeds personalized
- Community features active

### Week 18-19: Analytics & Intelligence
**Priority: LOW**

#### Tasks:
1. **Advanced analytics** (3 days)
   - Viewership trend analysis
   - Genre popularity tracking
   - Platform performance metrics

2. **Recommendation engines** (3 days)
   - Collaborative filtering
   - Content-based recommendations
   - Hybrid recommendation systems

3. **Voice/audio support** (2 days)
   - Speech-to-text integration
   - Voice command processing
   - Audio response generation

4. **AI conversation features** (2 days)
   - Natural conversation flow
   - Context retention
   - Personality customization

#### Acceptance Criteria:
- Analytics dashboard functional
- Advanced recommendations working
- Voice support operational
- AI conversation features active

## Technical Specifications

### Performance Targets

**Response Time Goals:**
- Simple queries (offline only): 50-200ms
- Enhanced queries (API): 250-700ms  
- Complex queries (multi-source): 500-2000ms
- Image search: 1000-2000ms
- Batch operations: 2000-5000ms

**Throughput Targets:**
- 100+ concurrent users
- 10,000+ requests/hour sustained
- 50,000+ requests/hour peak
- 99% uptime SLA

**Cache Performance:**
- L1 hit rate: 60-80%
- L2 hit rate: 85-95%
- Combined hit rate: 95%+
- Cache response time: <10ms

### Rate Limiting Strategy

**API Quota Management:**
```python
API_RATE_LIMITS = {
    'mal': {'requests_per_second': 2, 'requests_per_minute': 60},
    'anilist': {'requests_per_minute': 90, 'burst': 30},
    'kitsu': {'requests_per_second': 10},
    'anidb': {'requests_per_second': 1, 'requires_auth': True}
}

API_INTEGRATIONS = {
    'animeschedule': {'requests_per_second': 'unlimited', 'auth_required': False},
    'ann': {'requests_per_second': 1, 'format': 'xml'}
}

SCRAPING_RATE_LIMITS = {
    'animeplanet': {'requests_per_second': 1},
    'livechart': {'requests_per_second': 1, 'method': 'json_ld_extraction'},
    'anisearch': {'requests_per_second': 1}
}
```

**User Tier System:**
```python
USER_TIERS = {
    'free': {
        'quota_percentage': 20,
        'requests_per_hour': 200,
        'enhancement_limit': 20,
        'cache_priority': 'low',
        'features': ['basic_search', 'offline_data']
    },
    'premium': {
        'quota_percentage': 50,
        'requests_per_hour': 2000,
        'enhancement_limit': 200,
        'cache_priority': 'high',
        'features': ['enhanced_search', 'real_time_data', 'personalization']
    },
    'enterprise': {
        'quota_percentage': 30,
        'requests_per_hour': 20000,
        'enhancement_limit': 'unlimited',
        'cache_priority': 'highest',
        'features': ['all_features', 'api_access', 'bulk_operations']
    }
}
```

### Cost Management & Economics

**Daily Cost Estimates (10,000 active users):**
```python
DAILY_COST_BREAKDOWN = {
    'llm_processing': {
        'tokens_per_query': 1000,
        'queries_per_day': 50000,
        'cost_per_1k_tokens': 0.002,
        'daily_cost': 100.00
    },
    'api_calls': {
        'free_tier_optimization': True,
        'daily_cost': 0.00
    },
    'scraping_bandwidth': {
        'gb_per_day': 100,
        'cost_per_gb': 0.10,
        'daily_cost': 10.00
    },
    'proxy_services': {
        'monthly_cost': 50,
        'daily_cost': 1.67
    },
    'cache_storage': {
        'redis_instance': 'medium',
        'daily_cost': 5.00
    },
    'total_daily': 116.67,
    'monthly_estimate': 3500
}
```

### Monitoring & Observability

**Key Metrics to Track:**
```python
MONITORING_METRICS = {
    'performance': [
        'query_response_time_p95',
        'cache_hit_rate',
        'api_success_rate',
        'scraping_success_rate',
        'concurrent_users'
    ],
    'business': [
        'queries_per_hour',
        'unique_users_daily',
        'feature_usage_distribution',
        'user_satisfaction_score'
    ],
    'technical': [
        'error_rate_by_source',
        'rate_limit_violations',
        'circuit_breaker_trips',
        'queue_depth'
    ],
    'cost': [
        'llm_token_usage',
        'api_quota_consumption',
        'bandwidth_usage',
        'cache_storage_utilization'
    ]
}
```

**Alerting Rules:**
```python
ALERT_RULES = {
    'critical': {
        'response_time_p95_over_5s': 'immediate',
        'error_rate_over_10_percent': 'immediate',
        'api_quota_exhausted': 'immediate'
    },
    'warning': {
        'cache_hit_rate_below_80_percent': '5min',
        'scraping_success_below_70_percent': '10min',
        'cost_increase_over_20_percent': 'daily'
    },
    'info': {
        'new_user_milestone': 'daily',
        'feature_usage_anomaly': 'weekly'
    }
}
```

### Security & Privacy

**Data Classification:**
```python
DATA_CLASSIFICATION = {
    'public': {
        'data': ['anime_metadata', 'scores', 'genres', 'studios'],
        'cache_shared': True,
        'retention': 'indefinite'
    },
    'community': {
        'data': ['aggregated_preferences', 'popularity_metrics'],
        'cache_shared': True,
        'retention': '1_year'
    },
    'personal': {
        'data': ['watch_history', 'ratings', 'preferences'],
        'cache_shared': False,
        'retention': 'user_controlled'
    },
    'sensitive': {
        'data': ['oauth_tokens', 'api_keys', 'personal_info'],
        'cache_shared': False,
        'retention': 'minimal',
        'encryption': 'required'
    }
}
```

**Privacy Controls:**
```python
PRIVACY_FEATURES = {
    'data_export': {
        'formats': ['json', 'csv'],
        'includes': ['preferences', 'history', 'ratings'],
        'delivery': 'email_link'
    },
    'data_deletion': {
        'soft_delete_period': '30_days',
        'hard_delete_after': '30_days',
        'retention_exceptions': ['aggregated_analytics']
    },
    'consent_management': {
        'granular_permissions': True,
        'opt_out_analytics': True,
        'data_sharing_controls': True
    }
}
```

## Migration Strategy & Risk Management

### Breaking Changes & Mitigation

**Major Breaking Changes:**
1. **API Endpoint Removal**: All static endpoints removed
2. **Response Format Changes**: New unified response format
3. **Performance Impact**: LLM processing adds latency
4. **Cost Impact**: LLM usage introduces operational costs

**Mitigation Strategies:**
```python
MIGRATION_PLAN = {
    'week_1': {
        'deploy_parallel': True,
        'route_traffic_percentage': 10,
        'fallback_available': True
    },
    'week_2': {
        'route_traffic_percentage': 50,
        'monitor_performance': True,
        'optimize_based_on_data': True
    },
    'week_3': {
        'route_traffic_percentage': 100,
        'remove_legacy_endpoints': True,
        'performance_tuning': True
    }
}
```

### Risk Register

**High Risk Items:**
```python
HIGH_RISKS = {
    'llm_cost_explosion': {
        'probability': 'medium',
        'impact': 'high',
        'mitigation': ['aggressive_caching', 'request_coalescing', 'tier_limits']
    },
    'api_rate_limit_violations': {
        'probability': 'medium',
        'impact': 'medium',
        'mitigation': ['collaborative_cache', 'circuit_breakers', 'fallbacks']
    },
    'performance_degradation': {
        'probability': 'high',
        'impact': 'medium',
        'mitigation': ['edge_llm_deployment', 'response_caching', 'parallel_processing']
    }
}
```

**Medium Risk Items:**
```python
MEDIUM_RISKS = {
    'scraping_blocked': {
        'probability': 'medium',
        'impact': 'low',
        'mitigation': ['proxy_rotation', 'api_fallbacks', 'graceful_degradation']
    },
    'user_adoption_slow': {
        'probability': 'low',
        'impact': 'medium',
        'mitigation': ['gradual_rollout', 'feature_education', 'feedback_integration']
    }
}
```

## Success Metrics & Acceptance Criteria

### Phase-Specific Success Metrics

**Phase 1 Success Criteria:**
- All existing functionality available through `/api/query`
- Response time degradation <3x current performance
- Zero data loss during endpoint migration
- MCP tools enhanced with new parameters

**Phase 2 Success Criteria:**
- 9 anime sources accessible
- Cache hit rate >70% for popular anime
- API integration success rate >90%
- Scraping success rate >80%

**Phase 3 Success Criteria:**
- Complex narrative queries successfully resolved
- Multi-step query orchestration functional
- Response times <3s for complex queries
- User satisfaction >85% for advanced features

**Overall Project Success Criteria:**
- System handles 100x query complexity increase
- Response times remain reasonable (<2s average)
- Cost per query <$0.01
- User satisfaction score >90%
- Zero critical data loss incidents

## Conclusion

This comprehensive implementation plan transforms the Anime MCP Server from a static tool-based system to a fully dynamic, LLM-driven anime discovery platform. The phased approach ensures manageable development cycles while delivering incremental value.

Key innovations include:
- **Universal Query Interface**: Single endpoint handles any anime query
- **Intelligent Multi-Source Integration**: Smart routing across 9 anime platforms
- **Collaborative Community Caching**: Users share API quota and cached results
- **Progressive Enhancement**: Data richness scales with query complexity
- **LLM-Native Architecture**: Natural language processing throughout the stack

The plan balances ambition with pragmatism, incorporating lessons learned from extensive planning discussions and technical validation. Each phase builds upon the previous, creating a robust, scalable, and intelligent anime information system.