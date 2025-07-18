# Gemini Self-Instruction Manual for Anime Data Enrichment

This document outlines the PRODUCTION-LEVEL process for enriching anime data based on the logic in `src/services/iterative_ai_enrichment.py`.

## 1. Objective

The primary goal is to take a raw anime data object (from an offline database) and enrich it with additional information from external APIs (Jikan, AnimSchedule, Kitsu, Anime-Planet) and AI processing. This is a PRODUCTION implementation of the `enrich_anime_from_offline_data` method.

**CRITICAL: This is NOT a simulation. This is a production-level enrichment process that must use REAL API calls and REAL data.**

## 2. Inputs

- `offline_anime_data`: A JSON object representing a single anime. This will be provided at the start of the process.

## 3. The Process

**CRITICAL RULE: I must not proceed to the next step unless the current step has been completed successfully.**

### Step 1: Extract Platform IDs

1.  Check the `sources` field in the `offline_anime_data`.
2.  **MAL ID Extraction:**
    - Find the URL containing "myanimelist.net/anime/".
    - Extract the numerical ID from this URL.
    - If no MAL ID is found, the process stops, and the original `offline_anime_data` is returned.
3.  **Kitsu ID Extraction:**
    - Find the URL containing "kitsu.io/anime/" or "kitsu.app/anime/".
    - Extract the numerical ID from this URL.
    - If no Kitsu ID is found, Kitsu data fetching will be skipped.
4.  **Anime-Planet Slug Extraction:**
    - Find the URL containing "anime-planet.com/anime/".
    - Extract the slug from this URL (e.g., "dandadan" from "https://www.anime-planet.com/anime/dandadan").
    - If no Anime-Planet URL is found, Anime-Planet data fetching will be skipped.

### Step 2: Concurrent External API Data Fetching

1.  **Jikan Anime Full Data:** Fetch full anime data from Jikan API using the MAL ID. Save in temporary file in temp/jikan.json to be used later
    - URL: `https://api.jikan.moe/v4/anime/{mal_id}/full`
2.  **Jikan Episodes Data:** Fetch episode data from Jikan API. Use episodes property from the `offline_anime_data` (from base_anime_sample.json). Do not skip any episode. Save in temporary file in temp/episodes.json to be used later
    - **FIRST:** Create temp/episodes.json with the episode count from offline_anime_data: `{"episodes": <episode_count>}`
    - **IMPORTANT:** For anime with >100 episodes, use the reusable script:
      `python src/batch_enrichment/jikan_helper.py episodes {mal_id} temp/episodes.json temp/episodes_detailed.json`
    - **CRITICAL:**
      - NEVER give up on fetching all episodes regardless of time taken. Wait for the reusable script to complete fully before proceeding. ALL episodes MUST be fetched - no exceptions.
      - The reusable script will read the episode count from temp/episodes.json and fetch detailed data for each episode from the Jikan API endpoints
    - URL: `https://api.jikan.moe/v4/anime/{mal_id}/episodes/{episode_num}`
3.  **Jikan Characters Data:** Fetch character data from Jikan API. Do not skip any character. Save in temporary file in temp/characters.json to be used later
    - **IMPORTANT:** For anime with >50 characters, use the reusable script: `python src/batch_enrichment/jikan_helper.py characters {mal_id} temp/characters.json temp/characters_detailed.json`
    - **CRITICAL:** NEVER give up on fetching all characters regardless of time taken. Wait for the reusable script to complete fully before proceeding. ALL characters MUST be fetched - no exceptions.
    - URL: `https://api.jikan.moe/v4/anime/{mal_id}/characters`
4.  **AnimSchedule Data:** Find a matching anime on AnimSchedule using REAL API calls. Save in temporary file in temp/as.json to be used later
    - URL: `https://animeschedule.net/api/v3/anime?q={search_term}`
    - This involves a smart search using title, synonyms, and other metadata from `offline_anime_data`. Follow the logic in `animeschedule_helper.py` to implement proper search strategy.
    - **NEVER mock this data** - Always make real API calls to AnimSchedule to get accurate, up-to-date information including statistics, images, and external links.
5.  **Kitsu Data:** Fetch comprehensive Kitsu data using the extracted Kitsu ID. Save in temporary file in temp/kitsu.json to be used later
    - **ONLY if Kitsu ID was found in Step 1** - otherwise skip this step entirely
    - Use the `KitsuEnrichmentHelper.fetch_all_data(kitsu_id)` method from `src/batch_enrichment/kitsu_helper.py`
    - **NEVER mock this data** - Always make real API calls to Kitsu to get accurate information including categories, statistics, images, and NSFW flags.
6.  **Anime-Planet Data:** Fetch comprehensive Anime-Planet data using web scraping. Save in temporary file in temp/animeplanet.json to be used later
    - **ONLY if Anime-Planet URL was found in Step 1** - otherwise skip this step entirely
    - Use the `AnimePlanetEnrichmentHelper.fetch_all_data(offline_anime_data)` method from `src/batch_enrichment/animeplanet_helper.py`
    - **NEVER mock this data** - Always make real web scraping calls to Anime-Planet to get accurate information including ratings, images, rankings, and genre data.

### Step 3: Pre-process Episode Data

1.  From the fetched Jikan episodes data, create a simplified list of episodes, extracting only the following fields for each episode: `url`, `title`, `title_japanese`, `title_romanji`, `aired`, `score`, `filler`, `recap`, `duration`, `synopsis`.

### Step 4: Execute 5-Stage Enrichment Pipeline

This is the core of the process, where AI is used to process the collected data. Act as expert data scientist who is collecting, sanitizing and organizing anime data, and gnerate the expected JSON output for each stage based on the provided data and the logic in the corresponding prompt templates. Follow eahc stage systematically, DO NOT load all stage prompts at once. And when creating script, DO NOT use ChatGPT or ANthropic API.

**IMPORTANT: Strictly follow AnimeEtry schema from `src/models/anime.py`at each stage**

1.  **Stage 1: Metadata Extraction** PROMPT: src/services/prompts/stages/01_metadata_extraction_v2.txt
    - **Inputs:** `offline_anime_data`, core Jikan data, AnimSchedule data, Kitsu data, Anime-Planet data.
    - **Action:** Generate a JSON object containing `synopsis`, `genres`, `demographics`, `themes`, `source_material`, `rating`, `content_warnings`, `nsfw`, `title_japanese`, `title_english`, `background`, `aired_dates`, `broadcast`, `broadcast_schedule`, `premiere_dates`, `delay_information`, `episode_overrides`, `external_links`, `statistics`, `images`, `month`.
2.  **Stage 2: Episode Processing** PROMPT: src/services/prompts/stages/02_episode_processing_multi_agent.txt
    - **Inputs:** The pre-processed episode list.
    - **Action:** Process episodes in batches. For each batch, generate a list of `episode_details`. DO NOT skip any episode
3.  **Stage 3: Relationship Analysis** PROMPT: src/services/prompts/stages/03_relationship_analysis.txt
    - **Inputs:** `relatedAnime` URLs from `offline_anime_data`, and `relations` from Jikan data.
    - **Action:** Generate a JSON object with `relatedAnime` and `relations` fields.
    - **CRITICAL RULES:**
      - Process EVERY URL. The number of output `relatedAnime` entries must exactly match the number of input URLs.
      - Use "Intelligent Title Extraction":
        - Scan all URLs to find explicit titles (e.g., from anime-planet).
        - Visit each site to find the approprioate title and relation
        - Do not use numeric ID from url as the title.
      - **FORBIDDEN PATTERNS:** Do not use generic titles like "Anime [ID]", "Unknown Title", or "Anime 19060".
4.  **Stage 4: Statistics and Media** PROMPT: src/services/prompts/stages/04_statistics_media.txt
    - **Inputs:** Jikan statistics and media data.
    - **Action:** Generate a JSON object with `trailers`, `staff`, `opening_themes`, `ending_themes`, `streaming_info`, `licensors`, `streaming_licenses`, `awards`, `statistics`, `external_links`, and `images`.
    - **CRITICAL RULES:**
      - The `statistics` field must be a nested object with source as a key, like `mal`, `animeschedule`, `kitsu`, `animeplanet` key (e.g., `{"statistics": {"mal": {...}, "kitsu": {...}, "animeplanet": {...}}}`). There could be multiple sources.
5.  **Stage 5: Character Processing** PROMPT: src/services/prompts/stages/05_character_processing_multi_agent.txt
    - **Inputs:** Jikan characters data.
    - **Action:** Process characters in batches. For each batch, generate a list of `characters`. DO NOT skip any charatcer

### Step 5: Programmatic Assembly

1.  Merge the results from all five stages into a single JSON object.
2.  Start with the original `offline_anime_data`, and append animeschedule url for the relevent anime in the sources proeprty.
3.  Update the fields with the data from each stage's output following AnimeEtry schema from `src/models/anime.py`
4.  Add an `enrichment_metadata` object.
5.  **CRITICAL: Unicode Character Handling** - When saving the final JSON output, always use `ensure_ascii=False` and `encoding='utf-8'` to properly display international characters (Greek, Cyrillic, Japanese, etc.) instead of Unicode escape sequences.

## 4. Output Schema

The final output of this process must be a single JSON object that validates against the `AnimeEntry` Pydantic model defined in `src/models/anime.py`. I will ensure that the final, merged object adheres strictly to this schema, including all specified fields and their data types.

## 5. My Role

You will act as the Data Enrichment Expert. You will go through each step of the process, making REAL requests to the external APIs and then generating the expected JSON output for each of the five AI-driven stages. I will then perform the final programmatic merge to produce the final, enriched anime data object.

**PRODUCTION REQUIREMENTS:**

- Always make real API calls to Jikan, AnimSchedule, Kitsu (when IDs are available), and Anime-Planet (when URLs are available)
- Never mock or simulate data - this is production-level enrichment
- Handle API rate limits and errors gracefully
- Save all API responses to temporary files for reproducibility
- Use proper error handling for failed API calls
- Follow the exact API endpoints and data structures

I am now ready to begin. Please provide the `offline_anime_data` JSON object.
