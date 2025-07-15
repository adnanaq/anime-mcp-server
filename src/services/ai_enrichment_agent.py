# """
# AI-Powered Enrichment Agent

# This agent autonomously handles the entire anime enrichment process:
# - Makes API calls to MAL and AniList
# - Fetches character data and synopsis
# - Intelligently matches and merges data
# - Creates comprehensive enriched content

# The AI agent has access to API documentation and can make targeted calls
# for exactly the data it needs.
# """

# import asyncio
# import json
# import logging
# import os
# import time
# from typing import Dict, List, Optional, Any

# from pydantic import BaseModel
# try:
#     from ..integrations.rate_limiting.core import rate_limit_manager
#     from .source_mapping import (
#         SINGLE_SOURCE_MAPPING, 
#         MULTI_SOURCE_MAPPING,
#         get_all_required_sources,
#         get_fields_for_source
#     )
#     from .schema_validator import validate_ai_enrichment_result
# except ImportError:
#     from src.integrations.rate_limiting.core import rate_limit_manager
#     from src.services.source_mapping import (
#         SINGLE_SOURCE_MAPPING,
#         MULTI_SOURCE_MAPPING, 
#         get_all_required_sources,
#         get_fields_for_source
#     )
#     from src.services.schema_validator import validate_ai_enrichment_result

# logger = logging.getLogger(__name__)




# class EnrichedAnimeResult(BaseModel):
#     """Comprehensive result from AI enrichment agent matching AnimeEntry schema"""
    
#     # Basic anime identification
#     title: str
#     titles: Dict[str, str] = {}  # romaji, english, native
    
#     # Enriched synopsis (AI-merged from multiple sources)
#     enriched_synopsis: Optional[str] = None
#     synopsis_sources: List[str] = []  # Which APIs contributed
    
#     # Character data (AI-matched and merged from all sources)
#     characters: List[Dict[str, Any]] = []
    
#     # Media content
#     trailers: List[Dict[str, Any]] = []  # Official trailers and PVs
    
#     # NEW COMPREHENSIVE FIELDS matching AnimeEntry schema
#     genres: List[str] = []
#     demographics: List[str] = []
#     themes: List[Dict[str, Any]] = []  # {name, description}
#     source_material: Optional[str] = None
#     rating: Optional[str] = None
#     content_warnings: List[str] = []
    
#     # Detailed timing information
#     aired_dates: Optional[Dict[str, Any]] = None
#     broadcast: Optional[Dict[str, Any]] = None
    
#     # Streaming and availability
#     streaming_info: List[Dict[str, Any]] = []
#     licensors: List[str] = []
#     streaming_licenses: List[str] = []
    
#     # Staff and music
#     staff: List[Dict[str, Any]] = []
#     opening_themes: List[Dict[str, Any]] = []
#     ending_themes: List[Dict[str, Any]] = []
    
#     # Statistics from multiple platforms
#     statistics: Dict[str, Dict[str, Any]] = {}
    
#     # External links
#     external_links: Dict[str, str] = {}
    
#     # Enhanced images with source attribution
#     images: Dict[str, List[Dict[str, Any]]] = {}
    
#     # Episode details
#     episode_details: List[Dict[str, Any]] = []
    
#     # Relations with multi-platform URLs
#     relations: List[Dict[str, Any]] = []
    
#     # Awards and recognition
#     awards: List[Dict[str, Any]] = []
    
#     # Popularity trends
#     popularity_trends: Optional[Dict[str, Any]] = None
    
#     # Enhanced metadata
#     enhanced_metadata: Dict[str, Any] = {
#         "enhancement_version": "2.0",
#         "enhancement_sources": [],
#         "data_quality_score": 0.0,
#         "completeness_score": 0.0,
#         "processing_time": 0.0,
#         "apis_used": [],
#         "character_matches": 0,
#         "processing_notes": ""
#     }


# class AIEnrichmentAgent:
#     """AI agent that autonomously enriches anime data"""
    
#     def __init__(self, timeout: int = 30):
#         """
#         Initialize AI enrichment agent.
        
#         Args:
#             timeout: Timeout for API calls in seconds
#         """
#         self.timeout = timeout
        
#     async def enrich_anime(self, anime_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Autonomously enrich anime data using AI agent.
#         AI handles everything: URL parsing, ID extraction, API calls, and data merging.
        
#         Args:
#             anime_data: Raw anime info with title and source URLs (from anime-offline-database)
            
#         Returns:
#             Original anime data PLUS comprehensive enriched data
#         """
#         start_time = asyncio.get_event_loop().time()
        
#         # Check if we have source URLs to work with
#         sources = anime_data.get('sources', [])
#         if not sources:
#             logger.warning("No source URLs provided for enrichment")
#             return {
#                 **anime_data,  # Preserve all original properties
#                 "enrichment_metadata": {
#                     "error": "No source URLs provided",
#                     "processing_time": asyncio.get_event_loop().time() - start_time
#                 }
#             }
        
#         # Create AI prompt - AI handles everything autonomously
#         # Call new structured AI enrichment workflow
#         logger.info(f"AI agent autonomously enriching: {anime_data.get('title')}")
#         print(f"🔄 Starting structured enrichment workflow...")
#         enriched_result = await self._ai_enrich(anime_data)
        
#         # Calculate processing time and update enhanced metadata
#         processing_time = asyncio.get_event_loop().time() - start_time
#         enriched_result.enhanced_metadata["processing_time"] = processing_time
#         enriched_result.enhanced_metadata["last_enhanced"] = asyncio.get_event_loop().time()
        
#         # MERGE original anime data with enriched data - preserve all original fields
#         enriched_dict = enriched_result.dict()
        
#         # Map enriched fields to match AnimeEntry schema naming
#         final_result = {
#             **anime_data,  # Preserve ALL original properties (sources, type, episodes, status, season, etc.)
#             "synopsis": enriched_dict.get("enriched_synopsis"),  # Map to AnimeEntry field name
#             "characters": enriched_dict.get("characters", []),
#             "trailers": enriched_dict.get("trailers", []),
            
#             # Add all new comprehensive fields
#             "genres": enriched_dict.get("genres", []),
#             "demographics": enriched_dict.get("demographics", []),
#             "themes": enriched_dict.get("themes", []),
#             "source_material": enriched_dict.get("source_material"),
#             "rating": enriched_dict.get("rating"),
#             "content_warnings": enriched_dict.get("content_warnings", []),
#             "aired_dates": enriched_dict.get("aired_dates"),
#             "broadcast": enriched_dict.get("broadcast"),
#             "streaming_info": enriched_dict.get("streaming_info", []),
#             "licensors": enriched_dict.get("licensors", []),
#             "streaming_licenses": enriched_dict.get("streaming_licenses", []),
#             "staff": enriched_dict.get("staff", []),
#             "opening_themes": enriched_dict.get("opening_themes", []),
#             "ending_themes": enriched_dict.get("ending_themes", []),
#             "statistics": enriched_dict.get("statistics", {}),
#             "external_links": enriched_dict.get("external_links", {}),
#             "images": enriched_dict.get("images", {}),
#             "episode_details": enriched_dict.get("episode_details", []),
#             "relations": enriched_dict.get("relations", []),
#             "awards": enriched_dict.get("awards", []),
#             "popularity_trends": enriched_dict.get("popularity_trends"),
#             "enhanced_metadata": enriched_dict.get("enhanced_metadata", {})
#         }
        
#         # SCHEMA VALIDATION - ensure output matches enhanced_anime_schema_example.json
#         logger.info("🔍 Validating enriched data against schema...")
#         is_valid, validation_report = validate_ai_enrichment_result(final_result)
        
#         if is_valid:
#             logger.info("✅ Schema validation passed")
#         else:
#             logger.warning("⚠️ Schema validation failed")
#             logger.warning(validation_report)
            
#         # Add validation metadata
#         final_result["enhanced_metadata"]["schema_validation"] = {
#             "valid": is_valid,
#             "validated_at": asyncio.get_event_loop().time(),
#             "validation_summary": "Schema compliance verified" if is_valid else "Schema validation failed"
#         }
        
#         return final_result
    
#     def _create_autonomous_enrichment_prompt(self, anime_data: Dict[str, Any]) -> str:
#         """Create comprehensive AI prompt for enrichment using source mapping configuration"""
        
#         # Generate source assignments dynamically from configuration
#         single_source_assignments = []
#         for field, config in SINGLE_SOURCE_MAPPING.items():
#             single_source_assignments.append(f"- {field}: {config['primary']} (primary), {config['fallback']} (fallback)")
        
#         multi_source_assignments = []
#         for field, sources in MULTI_SOURCE_MAPPING.items():
#             multi_source_assignments.append(f"- {field}: {', '.join(sources)} (fetch from ALL)")
        
#         prompt = f"""Extract IDs from anime sources and use the provided functions to fetch targeted data based on source assignments:

# {json.dumps(anime_data, indent=2)}

# SINGLE-SOURCE FIELDS (use primary source, fallback if primary fails):
# {chr(10).join(single_source_assignments)}

# MULTI-SOURCE FIELDS (fetch from ALL specified sources):
# {chr(10).join(multi_source_assignments)}

# CRITICAL INSTRUCTIONS:
# 1. For single-source fields: Only call the primary source. If it fails, try fallback.
# 2. For multi-source fields: Call ALL specified sources. Leave empty if source lacks data.
# 3. For statistics: Create standardized schema with these exact properties:
#    - score: Rating score (normalize to 0-10 scale if needed)
#    - scored_by: Number of users who rated
#    - rank: Overall ranking position  
#    - popularity_rank: Popularity ranking position
#    - members: Total members/users tracking
#    - favorites: Number of users who favorited

# EXTRACTION TASKS:
# 1. Extract MAL ID from MAL URLs (myanimelist.net/anime/{{id}})
# 2. Extract AniList ID from AniList URLs (anilist.co/anime/{{id}})  
# 3. Extract Kitsu slug from Kitsu URLs (kitsu.app/anime/{{slug}} or kitsu.io/anime/{{slug}})
# 4. Generate correct AnimeSchedule slug from anime title

# API FUNCTION CALLS (targeted based on source assignments):
# - fetch_jikan_characters (for characters - multi-source)
# - fetch_jikan_anime_details (for demographics, content_warnings, aired_dates, staff, themes, episodes, relations, awards, statistics)
# - fetch_anilist_characters (for characters - multi-source)  
# - fetch_anilist_anime_details (for genres, themes, statistics)
# - fetch_kitsu_anime_details (for rating, streaming_info, statistics)
# - fetch_animeschedule_anime_details (for broadcast, streaming_info, external_links, licensors, streaming_licenses, statistics)
#   IMPORTANT: Both Kitsu and AnimeSchedule use slug-based search:
#   - Extract slug from URL: "kitsu.io/anime/kimetsu-no-yaiba" → "kimetsu-no-yaiba"
#   - Or convert from title: "Kimetsu no Yaiba" → "kimetsu-no-yaiba"
#   - Kitsu will search by slug and find the matching anime automatically

# INTELLIGENT DATA STANDARDIZATION:
# 4. For statistics, intelligently map platform-specific fields to uniform schema:
#    - Convert rating scales (e.g., AniList 82 → score 8.2)
#    - Map field names (e.g., "averageScore" → "score", "popularity" → "members")
#    - Handle missing fields gracefully (set to null)
# 5. For characters, smart deduplication across sources:
#    - Match characters by name variations (fuzzy matching)
#    - Combine character IDs from different platforms
#    - Collect images from all available sources
#    - Merge voice actor information

# Return comprehensive JSON matching AnimeEntry schema:
# {{
#     "title": "Anime Title",
#     "enriched_synopsis": "Combined synopsis from multiple sources...",
#     "synopsis_sources": ["jikan", "anilist"],
#     "characters": [{{
#         "name": "Character Name",
#         "role": "Main", 
#         "name_variations": ["Alt Name 1"],
#         "name_kanji": "漢字名",
#         "name_native": "Native Name",
#         "character_ids": {{"mal": ID, "anilist": ID}},
#         "images": {{"mal": "url", "anilist": "url"}},
#         "description": "Character description",
#         "age": "15",
#         "gender": "Male",
#         "voice_actors": [{{"name": "VA Name", "language": "Japanese"}}]
#     }}],
#     "genres": ["Action", "Adventure"],
#     "demographics": ["Shounen"],
#     "themes": [{{
#         "name": "Theme Name",
#         "description": "Theme description"
#     }}],
#     "source_material": "manga",
#     "rating": "PG-13",
#     "content_warnings": ["Violence"],
#     "aired_dates": {{
#         "from": "2019-04-06T16:25:00Z",
#         "to": "2019-09-28T16:25:00Z",
#         "string": "Apr 6, 2019 to Sep 28, 2019"
#     }},
#     "broadcast": {{
#         "day": "Saturday",
#         "time": "23:30",
#         "timezone": "JST"
#     }},
#     "streaming_info": [{{
#         "platform": "Crunchyroll",
#         "url": "https://crunchyroll.com/...",
#         "region": "Global",
#         "free": true,
#         "premium_required": false,
#         "subtitle_languages": ["English"]
#     }}],
#     "staff": [{{
#         "name": "Director Name",
#         "role": "Director",
#         "positions": ["Director", "Episode Director"]
#     }}],
#     "opening_themes": [{{
#         "title": "Theme Title",
#         "artist": "Artist Name",
#         "episodes": "1-26"
#     }}],
#     "ending_themes": [{{
#         "title": "Theme Title", 
#         "artist": "Artist Name",
#         "episodes": "1-26"
#     }}],
#     "statistics": {{
#         "mal": {{"score": 8.43, "scored_by": 1000000, "rank": 68, "popularity_rank": 12, "members": 3500000, "favorites": 89234}},
#         "anilist": {{"score": 8.2, "scored_by": null, "rank": null, "popularity_rank": null, "members": 147329, "favorites": 53821}},
#         "kitsu": {{"score": 8.21, "scored_by": 45123, "rank": null, "popularity_rank": null, "members": null, "favorites": 12456}},
#         "animeschedule": {{"score": 9.07, "scored_by": 338, "rank": null, "popularity_rank": null, "members": 2867, "favorites": null}}
#     }},
#     "external_links": {{
#         "official_website": "https://...",
#         "twitter": "https://..."
#     }},
#     "licensors": ["Funimation"],
#     "streaming_licenses": ["Crunchyroll"],
#     "images": {{
#         "covers": [{{
#             "url": "https://...",
#             "source": "jikan",
#             "type": "cover"
#         }}]
#     }},
#     "episode_details": [{{
#         "number": 1,
#         "title": "Episode Title",
#         "aired": "2019-04-06T16:25:00Z",
#         "synopsis": "Episode synopsis"
#     }}],
#     "relations": [{{
#         "anime_id": "sequel-id",
#         "relation_type": "sequel",
#         "title": "Sequel Title",
#         "urls": {{
#             "mal": "https://myanimelist.net/anime/123",
#             "anilist": "https://anilist.co/anime/123"
#         }}
#     }}],
#     "awards": [{{
#         "name": "Award Name",
#         "organization": "Organization",
#         "year": 2020
#     }}],
#     "enhanced_metadata": {{
#         "enhancement_version": "2.0",
#         "last_enhanced": "2025-07-12T13:45:00Z",
#         "enhancement_sources": ["jikan", "anilist", "kitsu", "animeschedule"],
#         "data_quality_score": 0.96,
#         "completeness_score": 0.94
#     }}
# }}

# Use ALL 6 functions to get REAL data. Follow the data source priority exactly. Merge intelligently with comprehensive coverage."""

#         return prompt
    
#     async def _ai_enrich(self, anime_data: Dict[str, Any]) -> EnrichedAnimeResult:
#         """Two-step enrichment: 1) Call APIs separately, 2) AI merges all data"""
#         logger.info("🔍 Starting two-step enrichment process...")
        
#         # STEP 1: Call all APIs separately to get raw data
#         logger.info("📡 Step 1: Fetching raw data from all APIs...")
#         raw_api_data = await self._fetch_all_apis_separately(anime_data)
        
#         # STEP 2: Use AI to intelligently merge all raw data  
#         logger.info("🧠 Step 2: AI merging and schema building...")
#         merged_result = await self._call_ai_for_data_merging(anime_data, raw_api_data)
        
#         return merged_result
    
#     async def _fetch_all_apis_separately(self, anime_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Simple direct API calls to get raw data"""
#         logger.info("🔍 Extracting IDs from source URLs...")
        
#         # Extract IDs from URLs (simple approach)
#         sources = anime_data.get('sources', [])
#         mal_id = None
#         anilist_id = None
#         kitsu_slug = None
        
#         for url in sources:
#             if 'myanimelist.net/anime/' in url:
#                 mal_id = url.split('/anime/')[-1].split('/')[0]
#             elif 'anilist.co/anime/' in url:
#                 anilist_id = url.split('/anime/')[-1].split('/')[0]
#             elif 'kitsu.app/anime/' in url or 'kitsu.io/anime/' in url:
#                 kitsu_slug = url.split('/anime/')[-1].split('/')[0]
        
#         logger.info(f"📊 Extracted IDs: MAL={mal_id}, AniList={anilist_id}, Kitsu={kitsu_slug}")
        
#         # Make direct API calls
#         api_responses = {}
        
#         # Call Jikan (MAL) API
#         if mal_id:
#             try:
#                 try:
#                     from ..integrations.clients.jikan_client import JikanClient
#                 except ImportError:
#                     from src.integrations.clients.jikan_client import JikanClient
#                 jikan_client = JikanClient()
                
#                 logger.info(f"🎯 Fetching Jikan anime details for ID: {mal_id}")
#                 anime_details = await jikan_client.get_anime_by_id(int(mal_id))
                
#                 logger.info(f"🎭 Fetching Jikan characters for ID: {mal_id}")
#                 characters = await jikan_client.get_anime_characters(int(mal_id))
                
#                 api_responses['jikan'] = {
#                     'anime_details': anime_details,
#                     'characters': characters
#                 }
#                 # Handle different response formats
#                 if isinstance(characters, dict):
#                     char_count = len(characters.get('data', []))
#                 elif isinstance(characters, list):
#                     char_count = len(characters)
#                 else:
#                     char_count = 0
#                 logger.info(f"✅ Jikan: Got {char_count} characters")
                
#             except Exception as e:
#                 logger.warning(f"⚠️ Jikan API failed: {e}")
#                 api_responses['jikan'] = {'error': str(e)}
        
#         # Call AniList API  
#         if anilist_id:
#             try:
#                 try:
#                     from ..integrations.clients.anilist_client import AniListClient
#                 except ImportError:
#                     from src.integrations.clients.anilist_client import AniListClient
#                 anilist_client = AniListClient()
                
#                 logger.info(f"🎯 Fetching AniList details for ID: {anilist_id}")
#                 anime_details = await anilist_client.get_anime_by_id(int(anilist_id))
                
#                 logger.info(f"🎭 Fetching AniList characters for ID: {anilist_id}")
#                 characters = await anilist_client.get_anime_characters(int(anilist_id))
                
#                 api_responses['anilist'] = {
#                     'anime_details': anime_details,
#                     'characters': characters
#                 }
#                 logger.info(f"✅ AniList: Got character data")
                
#             except Exception as e:
#                 logger.warning(f"⚠️ AniList API failed: {e}")
#                 api_responses['anilist'] = {'error': str(e)}
        
#         # Call Kitsu API
#         if kitsu_slug:
#             try:
#                 try:
#                     from ..integrations.clients.kitsu_client import KitsuClient
#                 except ImportError:
#                     from src.integrations.clients.kitsu_client import KitsuClient
#                 kitsu_client = KitsuClient()
                
#                 logger.info(f"🎯 Fetching Kitsu details for slug: {kitsu_slug}")
#                 anime_details = await kitsu_client.get_anime_by_id(int(kitsu_slug))
                
#                 api_responses['kitsu'] = {
#                     'anime_details': anime_details
#                 }
#                 logger.info(f"✅ Kitsu: Got anime details")
                
#             except Exception as e:
#                 logger.warning(f"⚠️ Kitsu API failed: {e}")
#                 api_responses['kitsu'] = {'error': str(e)}
        
#         return api_responses
    
#     async def _call_ai_for_data_merging(self, anime_data: Dict[str, Any], raw_api_data: Dict[str, Any]) -> EnrichedAnimeResult:
#         """Use AI to merge API responses with chunked character processing for all characters"""
        
#         # First, get basic anime metadata (no characters)
#         basic_data = await self._process_basic_anime_data(anime_data, raw_api_data)
        
#         # Then, process characters in chunks to ensure we get all of them
#         all_characters = await self._process_characters_in_chunks(raw_api_data)
        
#         # Combine results
#         final_result = EnrichedAnimeResult(
#             title=anime_data.get('title', 'Unknown'),
#             synopsis=basic_data.get('synopsis'),
#             genres=basic_data.get('genres', []),
#             demographics=basic_data.get('demographics', []), 
#             rating=basic_data.get('rating'),
#             characters=all_characters,
#             statistics=basic_data.get('statistics', {}),
#             staff=basic_data.get('staff', [])
#         )
        
#         logger.info(f"✅ Final merge: {len(all_characters)} characters, {len(basic_data.get('genres', []))} genres")
        
#         return final_result
    
#     async def _process_basic_anime_data(self, anime_data: Dict[str, Any], raw_api_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Process basic anime metadata (synopsis, genres, stats) without characters"""
        
#         basic_prompt = f"""Extract basic anime metadata from this API response (NO CHARACTERS):

# ANIME: {anime_data.get('title')}
# API DATA: {json.dumps(raw_api_data, indent=2)}

# Extract and return ONLY this JSON structure:
# {{
#     "synopsis": "Extract synopsis from jikan anime_details.data.synopsis", 
#     "genres": ["Extract genres from jikan data"],
#     "demographics": ["Extract demographics"],
#     "rating": "Extract content rating",
#     "statistics": {{"mal": {{"score": "jikan_score", "members": "jikan_members"}}}},
#     "staff": [
#         {{"name": "Staff Name", "role": "Director/Producer"}}
#     ]
# }}

# Focus only on basic metadata. Do NOT include characters."""
        
#         try:
#             if os.getenv("OPENAI_API_KEY"):
#                 response = await self._call_openai_for_merging(basic_prompt)
#             else:
#                 response = await self._call_anthropic_for_merging(basic_prompt)
            
#             # Parse response
#             if "```json" in response:
#                 start = response.find("```json") + 7
#                 end = response.find("```", start)
#                 if end != -1:
#                     response = response[start:end].strip()
            
#             return json.loads(response)
            
#         except Exception as e:
#             logger.error(f"❌ Basic data processing failed: {e}")
#             return {}
    
#     async def _process_characters_in_chunks(self, raw_api_data: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Process all characters in chunks to ensure we get every single one"""
        
#         # Get characters from API data
#         characters_data = raw_api_data.get('jikan', {}).get('characters', [])
#         if not characters_data:
#             logger.warning("⚠️ No character data found in API response")
#             return []
        
#         logger.info(f"🎭 Processing {len(characters_data)} characters in chunks of 10...")
        
#         all_processed_characters = []
#         chunk_size = 10
        
#         for i in range(0, len(characters_data), chunk_size):
#             chunk = characters_data[i:i+chunk_size]
#             chunk_start = i + 1
#             chunk_end = min(i + chunk_size, len(characters_data))
            
#             logger.info(f"🔄 Processing characters {chunk_start}-{chunk_end} ({len(chunk)} characters)")
            
#             chunk_prompt = f"""Convert these {len(chunk)} characters to structured format:

# {json.dumps(chunk, indent=2)}

# Convert EACH character exactly like this:
# {{
#     "name": "character.character.name",
#     "role": "character.role",
#     "name_variations": [],
#     "name_kanji": "",
#     "character_ids": {{"mal": "character.character.mal_id"}},
#     "images": {{"mal": "character.character.images.jpg.image_url"}},
#     "description": "",
#     "age": "",
#     "gender": "",
#     "voice_actors": [
#         // Convert voice_actors: {{"name": "voice_actor.person.name", "language": "voice_actor.language"}}
#     ]
# }}

# Return JSON array with exactly {len(chunk)} characters."""
            
#             try:
#                 if os.getenv("OPENAI_API_KEY"):
#                     response = await self._call_openai_for_merging(chunk_prompt)
#                 else:
#                     response = await self._call_anthropic_for_merging(chunk_prompt)
                
#                 # Parse response
#                 if "```json" in response:
#                     start = response.find("```json") + 7
#                     end = response.find("```", start)
#                     if end != -1:
#                         response = response[start:end].strip()
                
#                 chunk_characters = json.loads(response)
#                 all_processed_characters.extend(chunk_characters)
                
#                 logger.info(f"✅ Processed chunk {chunk_start}-{chunk_end}: got {len(chunk_characters)} characters")
                
#             except Exception as e:
#                 logger.error(f"❌ Failed to process chunk {chunk_start}-{chunk_end}: {e}")
#                 # Add placeholder characters for failed chunk
#                 for char in chunk:
#                     placeholder = {
#                         "name": char.get('character', {}).get('name', 'Unknown'),
#                         "role": char.get('role', 'Supporting'),
#                         "name_variations": [],
#                         "name_kanji": "",
#                         "character_ids": {"mal": char.get('character', {}).get('mal_id')},
#                         "images": {"mal": char.get('character', {}).get('images', {}).get('jpg', {}).get('image_url', '')},
#                         "description": "",
#                         "age": "",
#                         "gender": "",
#                         "voice_actors": []
#                     }
#                     all_processed_characters.append(placeholder)
        
#         logger.info(f"🎯 Chunked processing complete: {len(all_processed_characters)} total characters")
#         return all_processed_characters
    
#     async def _call_openai_for_merging(self, prompt: str) -> str:
#         """Call OpenAI to merge API data"""
#         try:
#             import openai
#             client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
#             response = await client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.1
#             )
            
#             return response.choices[0].message.content
            
#         except Exception as e:
#             logger.error(f"❌ OpenAI merging failed: {e}")
#             raise
    
#     async def _call_anthropic_for_merging(self, prompt: str) -> str:
#         """Call Anthropic to merge API data"""
#         try:
#             import anthropic
#             client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
#             response = await client.messages.create(
#                 model="claude-3-haiku-20240307",
#                 max_tokens=4000,
#                 messages=[{"role": "user", "content": prompt}]
#             )
            
#             return response.content[0].text
            
#         except Exception as e:
#             logger.error(f"❌ Anthropic merging failed: {e}")
#             raise
