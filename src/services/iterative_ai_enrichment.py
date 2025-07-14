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
import json
import logging
import os
from typing import Dict, Any, Optional, List

# Import the new prompt template manager and multi-stage implementation
try:
    from .prompts.prompt_template_manager import PromptTemplateManager
    from .iterative_ai_enrichment_v2 import MultiStageEnrichmentMixin
except ImportError:
    from src.services.prompts.prompt_template_manager import PromptTemplateManager
    from src.services.iterative_ai_enrichment_v2 import MultiStageEnrichmentMixin


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


class IterativeAIEnrichmentAgent(MultiStageEnrichmentMixin):
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
        
        # Initialize prompt template manager for modular prompts
        self.prompt_manager = PromptTemplateManager()
        
        if not self.ai_client:
            logger.warning("No AI provider configured")
        else:
            logger.info(f"Initialized with AI provider: {self.ai_provider}")
            
        # Validate prompt templates
        template_validation = self.prompt_manager.validate_templates()
        failed_templates = [t for t, valid in template_validation.items() if not valid]
        if failed_templates:
            logger.warning(f"Failed to load templates: {failed_templates}")
        else:
            logger.info("All prompt templates validated successfully")
    
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
        """Call AI provider with unified interface and function calling support"""
        if not self.ai_client:
            raise ValueError("No AI client configured")
        
        model = model or AI_CLIENTS[self.ai_provider]["default_model"]
        
        # No function calling for now - use pure AI analysis
        
        # Unified call method based on provider
        if self.ai_provider == "openai":
            response = await self.ai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=3600.0  # 1 hour timeout for long-running anime processing
            )
            return response.choices[0].message.content
            
        elif self.ai_provider == "anthropic":
            # Anthropic doesn't support function calling in the same way
            # We'll need to handle this differently for Anthropic
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
        jikan_data = await self._fetch_jikan_anime_full(mal_id)
        
        if "error" in jikan_data:
            logger.error(f"Failed to fetch Jikan data: {jikan_data['error']}")
            return anime_data
        
        # Step 3: Fetch episode details
        episodes_data = await self._fetch_jikan_episodes(mal_id)
        
        if "error" in episodes_data:
            logger.warning(f"Failed to fetch episode data: {episodes_data['error']}")
            episodes_data = {"data": []}  # Continue without episodes
        
        
        # Step 5: Pre-process episode data to avoid massive AI prompts
        processed_episodes = []
        for episode in episodes_data.get("data", []):
            # Extract only the fields we need, excluding mal_id and forum_url as requested
            processed_episode = {
                "url": episode.get("url"),
                "title": episode.get("title"),
                "title_japanese": episode.get("title_japanese"),
                "title_romanji": episode.get("title_romanji"),
                "aired": episode.get("aired"),
                "score": episode.get("score"),
                "filler": episode.get("filler", False),
                "recap": episode.get("recap", False),
                "duration": episode.get("duration"),
                "synopsis": episode.get("synopsis")
            }
            processed_episodes.append(processed_episode)
        
        logger.info(f"Pre-processed {len(processed_episodes)} episodes for AI")
        
        
        # Safety check for None values
        if processed_episodes is None:
            processed_episodes = []
        
        # Step 6: Have AI process the real API response with comprehensive schema
        prompt = f"""
# ROLE & CONTEXT
You are an expert anime database enrichment specialist. Your task is to process anime data from an offline database and enrich it with real-time Jikan API data, creating a comprehensive unified schema.

# OBJECTIVE
Transform the provided anime data into a structured, enriched JSON object that preserves all original data while adding detailed metadata from the Jikan API response.

# INPUT DATA
## ORIGINAL ANIME DATA:
{anime_data}

## JIKAN API RESPONSE:
{jikan_data}

## EPISODE DATA (Pre-processed):
{processed_episodes}


# PROCESSING INSTRUCTIONS

## 1. DATA PRESERVATION REQUIREMENTS
- Preserve ALL original anime fields exactly as provided unless specified otherwise below

## 2. RELATEDANIME PROCESSING (CRITICAL TASK)
This is the most important part of your task. Follow these steps exactly:

### STEP 1: IDENTIFY ORIGINAL URLS
- Review all URLs in the original data's "relatedAnime" field
- Process EVERY URL provided - do not filter or exclude any URLs

For each URL, perform intelligent title extraction and relationship inference:

### STEP 2: TITLE EXTRACTION RULES
- Convert dashes/underscores to spaces
- Apply proper capitalization
- Extract meaningful identifiers from URL paths
- For unclear paths, use descriptive titles from URL components 
- NEVER use "Unknown Title" - always extract something meaningful

**TITLE EXTRACTION EXAMPLES:**

**EXAMPLE 1:**
Input: "https://anime-planet.com/anime/dandadan-2nd-season"
→ Extract: "Dandadan 2nd Season"

**EXAMPLE 2:**
Input: "https://anime-planet.com/anime/creepy-nuts-otonoke"
→ Extract: "Creepy Nuts Otonoke"

**EXAMPLE 3:**
Input: "https://myanimelist.net/anime/60543"
→ Extract: "Dandadan 2nd Season" (or similar meaningful title)

**EXAMPLE 4:**
Input: "https://anilist.co/anime/185586"
→ Extract: "Dandadan 2nd Season" (or similar meaningful title)

### STEP 3: RELATIONSHIP TYPE INFERENCE
Based on title analysis, determine:
- "Sequel": For season/part continuations (e.g., "2nd Season", "Part 2")
- "Prequel": For earlier entries
- "Side Story": For spin-offs
- "Other": For theme songs, specials, unclear relationships


## 3. DATA EXTRACTION FROM JIKAN API
- Extract real values from jikan_data.data structure
- Convert rating scales appropriately (maintain 0-10 scale)
- Format dates as ISO strings
- Handle missing fields gracefully (use null/empty arrays)

## 4. SCHEMA SEPARATION RULE
- **relatedAnime**: ALL anime-related entries (from original URLs)
- **relations**: ONLY non-anime relationships (manga, novels), smartly observe the url for hints.


# OUTPUT SCHEMA
Map the data to this comprehensive JSON structure (preserve ALL original fields + add these enriched fields):

{{
    // All original fields preserved as-is
    ...original_anime_data,
    
    // Enhanced relatedAnime processing
    "relatedAnime": [{{\"relation_type\": \"infer_type\", \"title\": \"extract_from_URL\", \"url\": \"exact_original_URL\"}}] (PROCESS ALL original URLs - one entry per URL, no filtering)
    
    // New enriched fields
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
    "background": "Extract from jikan_data.data.background",
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
    
    "trailers": [{{
        "youtube_url": "http//...",
        "title": "Trailer title or use default",
        "thumbnail_url": "Extract max size image"
    }}],
    
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
    "episode_details": processed_episodes (CRITICAL: Use ALL pre-processed episodes exactly as provided - DO NOT skip any episodes),
    
    
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
    }}]
}}

# RESPONSE REQUIREMENTS

## FORMAT CONSTRAINTS
- Return ONLY valid JSON - no explanations, no markdown formatting
- No text before or after the JSON object
- Ensure proper JSON structure with correct brackets and commas


## QUALITY STANDARDS
- Extract meaningful titles from URL analysis
- Use intelligent relationship type inference
- PROCESS ALL ORIGINAL URLs - do not filter or exclude any URLs
- Output count must match input count (all 16 URLs should be processed)
- Focus on intelligent title extraction and relationship categorization
- Maintain data integrity throughout processing
- Handle edge cases gracefully

Now process the data and return the enriched JSON following this schema structure."""
        
        try:
            logger.info(f"Processing anime data with AI...")
            response = await self._call_ai(prompt)
            response = response.strip()
            
            # Clean up response if wrapped in markdown
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse AI response
            try:
                import json
                enriched_anime = json.loads(response)
                logger.info(f"Successfully enriched anime data")
                return enriched_anime
            except json.JSONDecodeError as e:
                logger.error(f"AI returned invalid JSON: {e}")
                logger.error(f"Response excerpt: {response[:500]}...")  # Show first 500 chars
                return anime_data  # Return original data instead of None
                
        except Exception as e:
            logger.error(f"AI enrichment failed: {e}")
            return anime_data  # Return original data instead of None
    
    
    async def _fetch_jikan_anime_full(self, mal_id: str) -> Dict[str, Any]:
        """Fetch complete anime data from Jikan API"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.jikan.moe/v4/anime/{mal_id}/full"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.error(f"Jikan API error: {response.status}")
                        return {"error": f"HTTP {response.status}"}
                        
        except Exception as e:
            logger.error(f"Failed to fetch Jikan data: {e}")
            return {"error": str(e)}
    
    async def _fetch_jikan_episodes(self, mal_id: str) -> Dict[str, Any]:
        """Fetch detailed episode data from Jikan API with optimized rate limiting and batching"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                # First get the episodes list to know how many episodes there are
                episodes_list_url = f"https://api.jikan.moe/v4/anime/{mal_id}/episodes"
                
                async with session.get(episodes_list_url) as response:
                    if response.status != 200:
                        logger.error(f"Episodes list API error: {response.status}")
                        return {"error": f"HTTP {response.status}"}
                    
                    episodes_list = await response.json()
                    episode_count = len(episodes_list.get("data", []))
                
                if episode_count == 0:
                    return {"data": []}
                
                # Optimized batch fetching with controlled concurrency
                detailed_episodes = []
                batch_size = 2  # Conservative batch size to avoid rate limits
                rate_limit_delay = 0.8  # 800ms between batches (more conservative)
                
                for batch_start in range(1, episode_count + 1, batch_size):
                    batch_end = min(batch_start + batch_size - 1, episode_count)
                    batch_tasks = []
                    
                    # Create batch of concurrent requests
                    for episode_num in range(batch_start, batch_end + 1):
                        task = self._fetch_single_episode(session, mal_id, episode_num)
                        batch_tasks.append(task)
                    
                    # Execute batch concurrently
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process batch results
                    for episode_num, result in zip(range(batch_start, batch_end + 1), batch_results):
                        if isinstance(result, Exception):
                            logger.warning(f"Failed to fetch episode {episode_num}: {result}")
                            continue
                        elif result is None:
                            continue
                        else:
                            detailed_episodes.append(result)
                    
                    # Rate limiting delay between batches (except for last batch)
                    if batch_end < episode_count:
                        await asyncio.sleep(rate_limit_delay)
                
                logger.info(f"Successfully fetched detailed data for {len(detailed_episodes)}/{episode_count} episodes")
                return {"data": detailed_episodes}
                        
        except Exception as e:
            logger.error(f"Failed to fetch episode data: {e}")
            return {"error": str(e)}
    
    async def _fetch_single_episode(self, session: 'aiohttp.ClientSession', mal_id: str, episode_num: int) -> Optional[Dict[str, Any]]:
        """Fetch a single episode with error handling"""
        episode_url = f"https://api.jikan.moe/v4/anime/{mal_id}/episodes/{episode_num}"
        
        try:
            async with session.get(episode_url) as response:
                if response.status == 200:
                    episode_data = await response.json()
                    return episode_data["data"]
                elif response.status == 429:
                    logger.warning(f"Rate limited on episode {episode_num}, applying backoff")
                    # Progressive backoff for rate limits
                    await asyncio.sleep(2.0)  # Wait 2 seconds before retry
                    # Retry once with longer delay
                    async with session.get(episode_url) as retry_response:
                        if retry_response.status == 200:
                            retry_data = await retry_response.json()
                            return retry_data["data"]
                        elif retry_response.status == 429:
                            logger.warning(f"Still rate limited on episode {episode_num}, skipping")
                            return None
                        else:
                            logger.error(f"Retry failed for episode {episode_num}: HTTP {retry_response.status}")
                            return None
                else:
                    logger.warning(f"Failed to fetch episode {episode_num}: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching episode {episode_num}: {e}")
            return None
    


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