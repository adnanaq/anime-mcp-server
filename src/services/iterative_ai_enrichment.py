#!/usr/bin/env python3
"""
Iterative AI Enrichment Agent - Simple and Clean Approach

This file will:
1. Import the full anime schema we want to achieve
2. Accept anime data from offline database (for vector indexing)
3. Use AI to enrich the data step by step
4. Build up functionality iteratively
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional

# AI clients with lazy imports
AI_CLIENTS = {
    "openai": {
        "module": "openai",
        "class": "AsyncOpenAI", 
        "key": "OPENAI_API_KEY",
        "default_model": "gpt-4o"
    },
    "anthropic": {
        "module": "anthropic", 
        "class": "AsyncAnthropic",
        "key": "ANTHROPIC_API_KEY", 
        "default_model": "claude-3-haiku-20240307"
    }
}

# Import the full schema we're targeting
try:
    from ..models.anime import (
        AnimeEntry,
        CharacterEntry,
        StaffEntry,
        StatisticsEntry,
        EnrichmentMetadata
    )
except ImportError:
    from src.models.anime import (
        AnimeEntry,
        CharacterEntry,
        StaffEntry,
        StatisticsEntry,
        EnrichmentMetadata
    )

logger = logging.getLogger(__name__)


class IterativeAIEnrichmentAgent:
    """
    Simple AI enrichment agent that works iteratively.
    
    Takes anime data from offline database and enriches it using AI,
    following the full schema structure.
    """
    
    def __init__(self, ai_provider: Optional[str] = None):
        """Initialize with AI provider (auto-detects if None)"""
        logger.info("Initializing Iterative AI Enrichment Agent")
        
        self.ai_provider = ai_provider or self._detect_provider()
        self.ai_client = self._create_client(self.ai_provider) if self.ai_provider else None
        
        if not self.ai_client:
            logger.warning("No AI provider configured")
    
    def _detect_provider(self) -> Optional[str]:
        """Auto-detect available provider"""
        for name, config in AI_CLIENTS.items():
            if os.getenv(config["key"]) and self._has_module(config["module"]):
                return name
        return None
    
    def _has_module(self, module_name: str) -> bool:
        """Check if module is available"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def _create_client(self, provider: str):
        """Create AI client using factory pattern"""
        if provider not in AI_CLIENTS:
            raise ValueError(f"Unknown provider: {provider}")
        
        config = AI_CLIENTS[provider]
        api_key = os.getenv(config["key"])
        
        if not api_key:
            raise ValueError(f"{config['key']} not found in environment")
        
        try:
            module = __import__(config["module"])
            client_class = getattr(module, config["class"])
            client = client_class(api_key=api_key)
            logger.info(f"{provider} client initialized")
            return client
        except ImportError:
            raise ImportError(f"{config['module']} package not installed")
        except AttributeError:
            raise AttributeError(f"Class {config['class']} not found in {config['module']}")
    
    async def enrich_anime_from_offline_data(self, offline_anime_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: Enrich anime data from offline database.
        
        Args:
            offline_anime_data: Raw anime data from anime-offline-database
            
        Returns:
            Enriched anime data following our full schema
        """
        logger.info(f"Starting enrichment for: {offline_anime_data.get('title', 'Unknown')}")
        
        # Start with the offline data as base
        enriched_data = {
            **offline_anime_data,  # Keep all original data
        }
        
        # AI enrichment
        if self.ai_client:
            logger.info(f"AI enriching anime data...")
            ai_enriched = await self._ai_enrich_data(offline_anime_data)
            if ai_enriched:
                logger.info(f"AI enrichment completed for: {enriched_data.get('title')}")
                return ai_enriched  # Return AI's complete enriched data
            else:
                logger.warning(f"AI enrichment failed for: {enriched_data.get('title')}")
        else:
            logger.warning(f"No AI client configured, skipping enrichment")
        
        logger.info(f"Base data prepared for: {enriched_data.get('title')}")
        
        return enriched_data  # Return original data if AI enrichment failed
    
    async def _call_ai(self, prompt: str, model: Optional[str] = None) -> str:
        """Call AI provider with unified interface"""
        if not self.ai_client:
            raise ValueError("No AI client configured")
        
        model = model or AI_CLIENTS[self.ai_provider]["default_model"]
        
        # Unified call method based on provider
        if self.ai_provider == "openai":
            response = await self.ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
            
        elif self.ai_provider == "anthropic":
            response = await self.ai_client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.content[0].text
            
        else:
            raise ValueError(f"Unknown provider: {self.ai_provider}")
    
    async def _ai_enrich_data(self, anime_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        AI enriches anime data by processing real API response data.
        
        Args:
            anime_data: Anime data from offline database
            
        Returns:
            Enriched data or None if failed
        """
        # Step 1: Extract MAL ID and fetch real API data
        sources = anime_data.get('sources', [])
        mal_id = None
        
        for url in sources:
            if 'myanimelist.net/anime/' in url:
                mal_id = url.split('/anime/')[-1].split('/')[0]
                break
        
        if not mal_id:
            logger.info("No MAL URL found, returning original data")
            return anime_data
        
        # Step 2: Fetch real Jikan API data
        logger.info(f"Fetching Jikan data for MAL ID: {mal_id}")
        jikan_data = await self._fetch_jikan_anime_full(mal_id)
        
        if "error" in jikan_data:
            logger.error(f"Failed to fetch Jikan data: {jikan_data['error']}")
            return anime_data
        
        # Step 3: Have AI process the real API response with comprehensive schema
        prompt = f"""
Process this anime data with the real Jikan API response and map it to the comprehensive schema:

ORIGINAL ANIME DATA:
{anime_data}

JIKAN API RESPONSE:
{jikan_data}

ID EXTRACTION TASKS:
1. Extract MAL ID from MAL URLs (myanimelist.net/anime/{{id}})
2. Extract AniList ID from AniList URLs (anilist.co/anime/{{id}})  

MANDATORY RELATEDANIME PROCESSING:
1. ALWAYS include the "relatedAnime" field in your output
2. PROCESS EVERY SINGLE original relatedAnime URL from the input data  
3. CREATE ONE SEPARATE ENTRY for each original URL - do not group or merge
4. PRESERVE EXACT URLS: Copy each original URL exactly as provided
5. EXTRACT CREATIVE TITLES: Use URL path analysis and site content to create meaningful titles
6. TITLE EXTRACTION (be creative and intelligent):
   - Extract from URL path with intelligent formatting (convert dashes to spaces, proper capitalization)
   - Analyze site content and context to infer proper titles
   - Use fuzzy matching to connect URLs to meaningful content
   - For unclear paths, create descriptive titles from available URL components
7. NEVER USE "Unknown Title" - always extract something meaningful from the URL
8. VERIFICATION: Your output count must match input count exactly

API FUNCTION CALLS (targeted based on source assignments):
- fetch_jikan_anime_details (for demographics, content_warnings, aired_dates, staff, themes, episodes, relations, awards, statistics)

INTELLIGENT DATA STANDARDIZATION:
- For statistics, intelligently map platform-specific fields to uniform schema:
   - Convert rating scales (e.g., AniList 82 → score 8.2)
   - Map field names (e.g., "averageScore" → "score", "popularity" → "members")
   - Handle missing fields gracefully (set to null)
   
Map the API data to this comprehensive JSON schema (preserve all original fields + add enriched fields):

{{
    "sources": "Preserve from original data",
    "title": "Preserve from original data", 
    "type": "Preserve from original data",
    "episodes": "Preserve from original data",
    "status": "Preserve from original data",
    "animeSeason": "Preserve from original data",
    "picture": "Preserve from original data",
    "thumbnail": "Preserve from original data", 
    "duration": "Preserve from original data",
    "score": "Preserve from original data",
    "synonyms": "Preserve from original data",
    "studios": "Preserve from original data",
    "producers": "Preserve from original data",
    "tags": "Preserve from original data",
    "relatedAnime": [{{\"anime_id\": \"from_URL\", \"relation_type\": \"infer_type\", \"title\": \"extract_from_URL\", \"urls\": {{\"platform\": \"exact_original_URL\"}}}}] (one entry per original URL - do not group)
    
    "synopsis": "Extract from jikan_data.data.synopsis",
    "genres": ["Extract genre names from jikan_data.data.genres array"],
    "demographics": ["Shounen"],
    "themes": [{{"name": "Theme Name", "description": "Theme description"}}],
    "source_material": "manga",
    "rating": "PG-13",
    "content_warnings": ["Violence"],
    "aired_dates": {{
        "from": "2019-04-06T16:25:00Z",
        "to": "2019-09-28T16:25:00Z",
        "string": "Apr 6, 2019 to Sep 28, 2019"
    }},
    "broadcast": {{
        "day": "Saturday",
        "time": "23:30",
        "timezone": "JST"
    }},
    "title_japanese": "Extract from jikan_data.data.title_japanese",
    "title_english": "Extract from jikan_data.data.title_english",    
    "streaming_info": [{{
        "name": "Platform name from jikan_data.data.streaming",
        "url": "Platform URL"
    }}],
    "licensors": ["Extract licensor names from jikan_data.data.licensors"],
    "streaming_licenses": ["Extract from streaming data"],
    "staff": [{{
        "name": "Director Name",
        "role": "Director",
        "positions": ["Director", "Episode Director"]
    }}],
    "opening_themes": [{{
        "title": "Theme Title",
        "artist": "Artist Name",
        "episodes": "1-26"
    }}],
    "ending_themes": [{{
        "title": "Theme Title", 
        "artist": "Artist Name",
        "episodes": "1-26"
    }}],
    
    "statistics": {{
        "mal": {{
            "score": "Rating score (normalize to 0-10 scale if needed)",
            "scored_by": "Number of users who rated", 
            "rank": "Overall ranking position",
            "popularity_rank": "Popularity ranking position",
            "members": "Total members/users tracking",
            "favorites": "Number of users who favorited"
        }}
    }},
    
    "trailer": {{
        "url": "http//...",
        "thumbnail": "Extract max size image"
    }},
    
    "external_links": {{
        "official_website": "Extract from jikan_data.data.external if available"
    }},
    
    "images": {{
        "covers": [{{
            "url": "https://...", choose the largest one
            "source": "jikan",
            "type": "cover"
        }}]
    }},
    "episode_details": [],
    
    "relations": [{{ CRITICAL: ONLY for NON-ANIME relationships (manga, light novels, etc.). DO NOT put anime entries here - all anime go in relatedAnime field above!
        "anime_id": "manga-id",
        "relation_type": "Adaptation", 
        "title": "Manga Title",
        "urls": {{
            "mal": "https://myanimelist.net/manga/123"
        }}
    }}],
    
    "awards": [{{
        "name": "Award Name",
        "organization": "Organization",
        "year": 2020
    }}],
    
    "jikan_response": "Include the complete jikan_data object"
}}

CRITICAL INSTRUCTIONS:
1. Use REAL values from the jikan_data API response - do not make up data
2. If a field is not available in API response, use null or empty array/object
3. Preserve ALL original anime fields exactly as they are
4. Extract actual values from the nested JSON structure
5. For relatedAnime: PROCESS ALL ORIGINAL URLs from input data
6. BE INCLUSIVE: Process every URL unless clearly manga/novels
7. INTELLIGENT TITLE EXTRACTION: Use fuzzy matching, URL analysis, and site content to create meaningful titles
8. CREATIVE FORMATTING: Convert URL slugs to proper titles (dashes to spaces, proper capitalization)
9. URL PARSING: Extract meaningful identifiers from various URL structures (slugs, IDs, path segments)
10. FUZZY MATCHING: Use pattern recognition to infer relationships and titles from URL structure
11. CRITICAL SEPARATION: relatedAnime = ALL anime relationships, relations = ONLY non-anime (manga/novels)
12. URL PRESERVATION: Copy every original URL exactly - no modifications, substitutions, or changes
13. OUTPUT COUNT: Must match the exact number of original relatedAnime URLs
14. VALID RELATIONSHIP TYPES: Sequel, Prequel, Side Story, Alternative Version, Spin-off, Parent Story, Summary, Alternative Setting, Character, Other
15. Return valid JSON only, no explanations

Map the real API response data to this comprehensive schema structure."""
        
        try:
            logger.info(f"Processing real API data with AI...")
            response = await self._call_ai(prompt)
            response = response.strip()
            
            # Clean up response if wrapped in markdown
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            logger.info(f"AI response received")
            
            # Parse AI response
            try:
                import json
                enriched_anime = json.loads(response)
                logger.info(f"Successfully parsed enriched anime data")
                return enriched_anime
            except json.JSONDecodeError as e:
                logger.error(f"AI returned invalid JSON: {e}")
                logger.error(f"Response was: {response}")
                return None
                
        except Exception as e:
            logger.error(f"AI enrichment failed: {e}")
            return None
    
    
    async def _fetch_jikan_anime_full(self, mal_id: str) -> Dict[str, Any]:
        """Fetch complete anime data from Jikan API"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.jikan.moe/v4/anime/{mal_id}/full"
                logger.info(f"Fetching Jikan data from: {url}")
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Successfully fetched Jikan data for MAL ID: {mal_id}")
                        return data
                    else:
                        logger.error(f"Jikan API error: {response.status}")
                        return {"error": f"HTTP {response.status}"}
                        
        except Exception as e:
            logger.error(f"Failed to fetch Jikan data: {e}")
            return {"error": str(e)}
    


# Convenience function for vector indexing integration
async def enrich_anime_for_vector_indexing(
    offline_anime_data: Dict[str, Any], 
    ai_provider: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for vector indexing pipeline.
    
    Args:
        offline_anime_data: Anime data from offline database
        ai_provider: "openai" or "anthropic" (auto-detects if None)
        
    Returns:
        Enriched anime data ready for vector indexing
    """
    agent = IterativeAIEnrichmentAgent(ai_provider=ai_provider)
    return await agent.enrich_anime_from_offline_data(offline_anime_data)