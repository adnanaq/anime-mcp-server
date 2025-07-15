#!/usr/bin/env python3
"""
Multi-Stage AI Enrichment Implementation

This is the refactored _ai_enrich_data method with 4-stage pipeline
to replace the monolithic 200+ line prompt approach.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import the prompt template manager
try:
    from .prompts.prompt_template_manager import PromptTemplateManager
    from .animeschedule_helper import AnimScheduleEnrichmentHelper
except ImportError:
    from src.services.prompts.prompt_template_manager import PromptTemplateManager
    from src.services.animeschedule_helper import AnimScheduleEnrichmentHelper

logger = logging.getLogger(__name__)

# AI clients configuration
AI_CLIENTS = {
    "openai": {
        "module": "openai",
        "class": "AsyncOpenAI", 
        "key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-2024-11-20"
    },
    "anthropic": {
        "module": "anthropic", 
        "class": "AsyncAnthropic",
        "key": "ANTHROPIC_API_KEY", 
        "default_model": "claude-sonnet-4-20250514"
    }
}


class MultiStageEnrichmentMixin:
    """Mixin providing 4-stage enrichment pipeline methods"""
    
    async def _ai_enrich_data_v2(self, anime_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Multi-stage AI enrichment using modular prompts for optimal performance.
        
        Replaces the monolithic 200+ line prompt with 5 specialized stages:
        1. Metadata Extraction (synopsis, genres, basic info)
        2. Episode Processing (episode details, timing)
        3. Relationship Analysis (relatedAnime URL processing)
        4. Statistics & Media (stats, trailers, images, staff)
        5. Character Processing (characters, voice actors)
        
        Args:
            anime_data: Anime data from offline database
            
        Returns:
            Enriched data or None if failed
        """
        try:
            # Step 1: Extract MAL ID
            sources = anime_data.get('sources', [])
            mal_id = None
            for url in sources:
                if 'myanimelist.net/anime/' in url:
                    mal_id = url.split('/anime/')[-1].split('/')[0]
                    break
            
            if not mal_id:
                logger.info("No MAL URL found, returning original data")
                return anime_data

            # Step 2: Concurrently fetch ALL external API data
            import asyncio
            logger.info("Fetching ALL external API data concurrently...")
            
            api_tasks = {
                "jikan_anime": self._fetch_jikan_anime_full(mal_id),
                "jikan_episodes": self._fetch_jikan_episodes(mal_id),
                "jikan_characters": self._fetch_jikan_characters(mal_id),
                "animeschedule": self.animeschedule_helper.find_anime_match(anime_data)
            }

            api_results = await asyncio.gather(*api_tasks.values(), return_exceptions=True)
            jikan_data, episodes_data, characters_data, animeschedule_data = api_results

            # Handle API fetch results
            if isinstance(jikan_data, Exception) or "error" in jikan_data:
                logger.error(f"Failed to fetch Jikan data: {jikan_data}")
                return anime_data

            if isinstance(episodes_data, Exception) or "error" in episodes_data:
                logger.warning(f"Failed to fetch episode data: {episodes_data}")
                episodes_data = {"data": []}

            if isinstance(characters_data, Exception) or "error" in characters_data:
                logger.warning(f"Failed to fetch character data: {characters_data}")
                characters_data = {"data": []}

            if isinstance(animeschedule_data, Exception):
                logger.warning(f"Failed to fetch AnimSchedule data: {animeschedule_data}")
                animeschedule_data = None

            # Step 3: Pre-process episode data
            processed_episodes = self._preprocess_episodes(episodes_data.get("data", []))
            logger.info(f"Pre-processed {len(processed_episodes)} episodes for multi-stage AI processing")
            
            # Step 4: Execute 5-stage enrichment pipeline with all API data available
            enriched_data = await self._execute_enrichment_pipeline(
                anime_data, jikan_data, processed_episodes, characters_data, animeschedule_data
            )
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Multi-stage AI enrichment failed: {e}")
            return anime_data  # Return original data instead of None
    
    def _preprocess_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pre-process episode data to extract only needed fields"""
        processed_episodes = []
        for episode in episodes:
            # Extract fields from individual episode endpoint (which includes detailed info)
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
        return processed_episodes
    
    async def _execute_enrichment_pipeline(
        self,
        anime_data: Dict[str, Any],
        jikan_data: Dict[str, Any],
        processed_episodes: List[Dict[str, Any]],
        characters_data: Dict[str, Any],
        animeschedule_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute the 5-stage enrichment pipeline concurrently with field-specific extraction.
        
        Each stage extracts ONLY its specific fields, then programmatically merged.
        Massive token reduction and performance improvement vs old merge approach.
        """
        import asyncio
        pipeline_start = datetime.now()
        
        try:
            # Create tasks for all enrichment stages with staggered delays to avoid rate limits
            async def delayed_stage(delay_seconds, stage_func, *args):
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
                return await stage_func(*args)
            
            tasks = {
                "metadata": delayed_stage(0, self._execute_stage_with_retry, 1, self._execute_stage_1, anime_data, jikan_data, animeschedule_data),
                "episodes": delayed_stage(2, self._execute_stage_with_retry, 2, self._execute_stage_2, processed_episodes, jikan_data),
                "relationships": delayed_stage(4, self._execute_stage_with_retry, 3, self._execute_stage_3, anime_data, jikan_data),
                "stats_media": delayed_stage(6, self._execute_stage_with_retry, 4, self._execute_stage_4, jikan_data),
                "characters": delayed_stage(8, self._execute_stage_with_retry, 5, self._execute_stage_5, characters_data)
            }

            # Run tasks concurrently and gather results
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # Map results back to stage names
            stage_results = dict(zip(tasks.keys(), results))

            # Handle exceptions and log errors
            for stage, result in stage_results.items():
                if isinstance(result, Exception):
                    logger.error(f"Stage '{stage}' failed: {result}")
                    stage_results[stage] = {} # Set empty dict for failed stage

            # Stage 6: Programmatic Assembly (eliminate AI dependency)
            logger.info("Stage 6: Programmatic assembly...")
            final_result = self._programmatic_merge(anime_data, stage_results)
            
            pipeline_time = (datetime.now() - pipeline_start).total_seconds()
            logger.info(f"✅ 5-stage enrichment pipeline completed in {pipeline_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Enrichment pipeline failed: {e}")
            # Attempt partial recovery using completed stages
            return self._attempt_partial_recovery_v2({}, anime_data, e)
    
    async def _execute_stage_with_retry(self, stage_num: int, stage_func, *args, max_retries: int = 2):
        """Execute a stage with retry logic and error recovery"""
        for attempt in range(max_retries + 1):
            try:
                result = await stage_func(*args)
                if attempt > 0:
                    logger.info(f"Stage {stage_num} succeeded on retry {attempt}")
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Stage {stage_num} JSON decode error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    raise Exception(f"Stage {stage_num} failed after {max_retries + 1} attempts: JSON parsing failed")
                    
            except Exception as e:
                logger.warning(f"Stage {stage_num} error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    raise Exception(f"Stage {stage_num} failed after {max_retries + 1} attempts: {e}")
    
    def _attempt_partial_recovery_v2(self, stage_results: Dict[str, Any], original_data: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """Attempt to recover with partial data when pipeline fails using programmatic merge"""
        logger.warning(f"Attempting partial recovery from pipeline failure: {error}")
        
        if not stage_results:
            logger.warning("No stages completed successfully, returning original data")
            return original_data
        
        # Use programmatic merge with whatever stages completed
        completed_stages = list(stage_results.keys())
        logger.info(f"Recovering with completed stages: {completed_stages}")
        
        # Merge completed stages programmatically
        partial_result = self._programmatic_merge(original_data, stage_results)
        
        # Update metadata to reflect partial recovery
        partial_result["enrichment_metadata"] = {
            "source": "iterative_ai_enrichment_partial",
            "enriched_at": datetime.now().isoformat(),
            "success": False,
            "stages_completed": completed_stages,
            "pipeline_version": "field_extraction_v1",
            "partial_recovery": True,
            "error_message": str(error),
            "optimization": "programmatic_merge"
        }
        
        return partial_result

    async def _process_in_batches(self, items: List[Any], batch_processor, batch_size: int = 10, **kwargs) -> List[Any]:
        """
        Generic function to process a list of items in concurrent batches.
        
        Args:
            items: The list of items to process.
            batch_processor: An async function that takes a batch (list) and kwargs.
            batch_size: The number of items in each batch.
            **kwargs: Additional arguments to pass to the batch_processor.
            
        Returns:
            A list of aggregated results from all batches.
        """
        import asyncio
        
        tasks = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            task = batch_processor(batch, **kwargs)
            tasks.append(task)
            
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"A batch failed during processing: {result}")
            elif result:
                final_results.extend(result)
        
        return final_results

    async def _process_episode_batch(self, batch: List[Dict[str, Any]], jikan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Processes a batch of episodes and returns the list of episode details."""
        prompt = self.prompt_manager.build_stage_prompt(
            2,
            episodes_data=json.dumps(batch, indent=2)
        )
        response = await self._call_ai_with_cleanup(prompt)
        result = json.loads(response)
        return result.get("episode_details", [])

    async def _process_character_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a batch of characters and returns the list of characters."""
        batch_data = {"data": batch}
        prompt = self.prompt_manager.build_stage_prompt(
            5,
            characters_data=json.dumps(batch_data, indent=2)
        )
        response = await self._call_ai_with_cleanup(prompt)
        result = json.loads(response)
        return result.get("characters", [])

    async def _execute_stage_1(self, anime_data: Dict[str, Any], jikan_data: Dict[str, Any], animeschedule_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Stage 1: Metadata extraction - ONLY extract specific metadata fields"""
        # Extract only core fields from jikan_data to reduce token count
        jikan_core = {
            "data": {
                "synopsis": jikan_data.get("data", {}).get("synopsis"),
                "genres": jikan_data.get("data", {}).get("genres", []),
                "demographics": jikan_data.get("data", {}).get("demographics", []),
                "themes": jikan_data.get("data", {}).get("themes", []),
                "source": jikan_data.get("data", {}).get("source"),
                "rating": jikan_data.get("data", {}).get("rating"),
                "title_japanese": jikan_data.get("data", {}).get("title_japanese"),
                "title_english": jikan_data.get("data", {}).get("title_english"),
                "background": jikan_data.get("data", {}).get("background"),
                "aired": jikan_data.get("data", {}).get("aired"),
                "broadcast": jikan_data.get("data", {}).get("broadcast")
            }
        }
        
        # AnimSchedule data is now passed as parameter (already fetched concurrently)
        if animeschedule_data:
            logger.info(f"Processing AnimSchedule data for '{anime_data.get('title')}'")
        else:
            logger.info(f"No AnimSchedule data available for '{anime_data.get('title')}'")
        
        prompt = self.prompt_manager.build_stage_prompt(
            1,
            jikan_core_data=json.dumps(jikan_core, indent=2),
            animeschedule_data=json.dumps(animeschedule_data, indent=2) if animeschedule_data else "{}"
        )
        
        response = await self._call_ai_with_cleanup(prompt)
        result = json.loads(response)
        
        # Add AnimSchedule data directly to result for programmatic merge
        if animeschedule_data:
            result["_animeschedule_data"] = animeschedule_data
        
        return result
    
    async def _execute_stage_2(self, episodes: List[Dict[str, Any]], jikan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Episode processing - process all episodes in batches."""
        total_episodes = len(episodes)
        logger.info(f"Processing {total_episodes} episodes in batches")

        processed_episode_details = await self._process_in_batches(
            episodes, self._process_episode_batch, batch_size=10, jikan_data=jikan_data
        )
        
        logger.info(f"Successfully processed {len(processed_episode_details)}/{total_episodes} episodes")

        return {
            "episode_details": processed_episode_details
        }
    
    async def _execute_stage_3(self, anime_data: Dict[str, Any], jikan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Relationship analysis - ONLY extract relationship fields"""
        related_urls = anime_data.get("relatedAnime", [])
        jikan_relations = {
            "data": {
                "relations": jikan_data.get("data", {}).get("relations", [])
            }
        }
        
        prompt = self.prompt_manager.build_stage_prompt(
            3,
            related_anime_urls=json.dumps(related_urls, indent=2),
            jikan_relations_data=json.dumps(jikan_relations, indent=2)
        )
        
        response = await self._call_ai_with_cleanup(prompt)
        return json.loads(response)
    
    async def _execute_stage_4(self, jikan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 4: Statistics and media - ONLY extract stats and media fields"""
        jikan_stats = {
            "data": {
                "score": jikan_data.get("data", {}).get("score"),
                "scored_by": jikan_data.get("data", {}).get("scored_by"),
                "rank": jikan_data.get("data", {}).get("rank"),
                "popularity": jikan_data.get("data", {}).get("popularity"),
                "members": jikan_data.get("data", {}).get("members"),
                "favorites": jikan_data.get("data", {}).get("favorites")
            }
        }
        
        jikan_media = {
            "data": {
                "trailer": jikan_data.get("data", {}).get("trailer"),
                "images": jikan_data.get("data", {}).get("images"),
                "staff": jikan_data.get("data", {}).get("staff", []),
                "theme": jikan_data.get("data", {}).get("theme", {}),
                "streaming": jikan_data.get("data", {}).get("streaming", []),
                "licensors": jikan_data.get("data", {}).get("licensors", []),
                "external": jikan_data.get("data", {}).get("external", [])
            }
        }
        
        prompt = self.prompt_manager.build_stage_prompt(
            4,
            jikan_statistics_data=json.dumps(jikan_stats, indent=2),
            jikan_media_data=json.dumps(jikan_media, indent=2)
        )
        
        response = await self._call_ai_with_cleanup(prompt)
        return json.loads(response)
    
    def _programmatic_merge(self, original_data: Dict[str, Any], stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Programmatically merge all stage outputs with original data.
        
        This replaces AI-based final assembly with deterministic code merge.
        Much faster and more reliable than AI JSON generation.
        """
        # Start with original anime data
        final_result = original_data.copy()
        
        # Merge Stage 1: Metadata fields (including timing data and AnimSchedule data)
        if "metadata" in stage_results:
            metadata = stage_results["metadata"]
            for field in ["synopsis", "genres", "demographics", "themes", "source_material", 
                         "rating", "content_warnings", "title_japanese", "title_english", "background",
                         "aired_dates", "broadcast", "external_links", "statistics", "images", "month"]:
                if field in metadata:
                    final_result[field] = metadata[field]
        
        # Merge Stage 2: Episode fields only
        if "episodes" in stage_results:
            episodes = stage_results["episodes"]
            for field in ["episode_details"]:
                if field in episodes:
                    final_result[field] = episodes[field]
        
        # Merge Stage 3: Relationship fields
        if "relationships" in stage_results:
            relationships = stage_results["relationships"]
            for field in ["relatedAnime", "relations"]:
                if field in relationships:
                    final_result[field] = relationships[field]
        
        # Merge Stage 4: Statistics and media fields
        if "stats_media" in stage_results:
            stats_media = stage_results["stats_media"]
            for field in ["trailers", "staff", "opening_themes", 
                         "ending_themes", "streaming_info", "licensors", "streaming_licenses", "awards"]:
                if field in stats_media:
                    final_result[field] = stats_media[field]
            
            # Special handling for statistics: merge instead of overwrite
            if "statistics" in stats_media:
                if "statistics" not in final_result:
                    final_result["statistics"] = {}
                # Merge Stage 4 statistics with existing statistics from Stage 1
                final_result["statistics"].update(stats_media["statistics"])
            
            # Special handling for external_links: merge instead of overwrite
            if "external_links" in stats_media:
                if "external_links" not in final_result:
                    final_result["external_links"] = {}
                # Merge Stage 4 external_links with existing external_links from Stage 1
                final_result["external_links"].update(stats_media["external_links"])
            
            # Special handling for images: merge covers arrays instead of overwrite
            if "images" in stats_media and "covers" in stats_media["images"]:
                if "images" not in final_result:
                    final_result["images"] = {"covers": []}
                if "covers" not in final_result["images"]:
                    final_result["images"]["covers"] = []
                # Merge Stage 4 covers with existing covers from Stage 1
                final_result["images"]["covers"].extend(stats_media["images"]["covers"])
        
        # Merge Stage 5: Character fields
        if "characters" in stage_results:
            characters = stage_results["characters"]
            if "characters" in characters:
                final_result["characters"] = characters["characters"]
        
        # Add enrichment metadata
        final_result["enrichment_metadata"] = {
            "source": "iterative_ai_enrichment_optimized",
            "enriched_at": datetime.now().isoformat(),
            "success": True,
            "stages_completed": list(stage_results.keys()),
            "pipeline_version": "field_extraction_v1",
            "optimization": "programmatic_merge"
        }
        
        return final_result
    
    async def _call_ai_with_cleanup(self, prompt: str) -> str:
        """Call AI with response cleanup for JSON parsing"""
        response = await self._call_ai(prompt)
        
        # Find the start and end of the JSON object
        start_index = response.find('{')
        end_index = response.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return response[start_index:end_index+1].strip()
        else:
            # Fallback for basic markdown cleanup if JSON object not found
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            return response.strip()
    
    async def _fetch_jikan_characters(self, mal_id: str) -> Dict[str, Any]:
        """Fetch character data from Jikan API"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.jikan.moe/v4/anime/{mal_id}/characters"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.error(f"Jikan characters API error: {response.status}")
                        return {"error": f"HTTP {response.status}"}
                        
        except Exception as e:
            logger.error(f"Failed to fetch Jikan characters: {e}")
            return {"error": str(e)}
    
    async def _execute_stage_5(self, characters_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 5: Character processing - process all characters in batches."""
        characters_core = []
        for char_entry in characters_data.get("data", []):
            character = char_entry.get("character", {})
            voice_actors = char_entry.get("voice_actors", [])
            char_core = {
                "character": {
                    "mal_id": character.get("mal_id"),
                    "name": character.get("name"),
                    "name_kanji": character.get("name_kanji"),
                    "images": {"jpg": character.get("images", {}).get("jpg", {}).get("image_url")},
                    "about": character.get("about", "")[:200] if character.get("about") else ""
                },
                "role": char_entry.get("role"),
                "voice_actors": [
                    {
                        "person": {
                            "name": va.get("person", {}).get("name"),
                            "images": {"jpg": va.get("person", {}).get("images", {}).get("jpg", {}).get("image_url")}
                        },
                        "language": va.get("language")
                    }
                    for va in voice_actors
                ]
            }
            characters_core.append(char_core)

        total_characters = len(characters_core)
        logger.info(f"Processing {total_characters} characters in batches")

        processed_characters = await self._process_in_batches(
            characters_core, self._process_character_batch, batch_size=10
        )

        logger.info(f"Successfully processed {len(processed_characters)}/{total_characters} characters")
        return {"characters": processed_characters}
    
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
        import asyncio
        
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
        import asyncio
        
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


class IterativeAIEnrichmentAgent(MultiStageEnrichmentMixin):
    """
    Complete AI enrichment agent using the v2 multi-stage pipeline.
    """
    
    def __init__(self, ai_provider: Optional[str] = None):
        """Initialize with AI provider (auto-detects if None)"""
        logger.info("Initializing Iterative AI Enrichment Agent")
        
        self.ai_provider = ai_provider or self._detect_provider()
        self.ai_client = self._create_client(self.ai_provider) if self.ai_provider else None
        
        # Initialize prompt template manager for modular prompts
        self.prompt_manager = PromptTemplateManager()
        
        # Initialize AnimSchedule helper for enrichment
        self.animeschedule_helper = AnimScheduleEnrichmentHelper()
        
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
                temperature=0.1,
                timeout=3600.0  # 1 hour timeout for long-running processing
            )
            return response.choices[0].message.content
            
        elif self.ai_provider == "anthropic":
            response = await self.ai_client.messages.create(
                model=model,
                max_tokens=8192,  # Increased for complex multi-agent responses
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=3600.0  # 1 hour timeout for long-running processing
            )
            return response.content[0].text
            
        else:
            raise ValueError(f"Unknown provider: {self.ai_provider}")
    
    async def enrich_anime_from_offline_data(self, offline_anime_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: Enrich anime data from offline database using v2 pipeline.
        
        Args:
            offline_anime_data: Raw anime data from anime-offline-database
            
        Returns:
            Enriched anime data following our full schema
        """
        logger.info(f"Starting v2 enrichment for: {offline_anime_data.get('title', 'Unknown')}")
        
        # Use the v2 multi-stage pipeline
        if self.ai_client:
            logger.info(f"AI enriching anime data with v2 pipeline...")
            ai_enriched = await self._ai_enrich_data_v2(offline_anime_data)
            if ai_enriched:
                logger.info(f"v2 AI enrichment completed for: {ai_enriched.get('title')}")
                return ai_enriched
            else:
                logger.warning(f"v2 AI enrichment failed for: {offline_anime_data.get('title')}")
        else:
            logger.warning(f"No AI client configured, skipping enrichment")
        
        return offline_anime_data  # Return original data if AI enrichment failed


# Convenience function for vector indexing integration
async def enrich_anime_for_vector_indexing(
    offline_anime_data: Dict[str, Any], 
    ai_provider: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for vector indexing pipeline using v2 system.
    
    Args:
        offline_anime_data: Anime data from offline database
        ai_provider: "openai" or "anthropic" (auto-detects if None)
        
    Returns:
        Enriched anime data ready for vector indexing
    """
    agent = IterativeAIEnrichmentAgent(ai_provider=ai_provider)
    return await agent.enrich_anime_from_offline_data(offline_anime_data)