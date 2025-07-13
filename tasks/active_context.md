# Active Context

# Anime MCP Server - Current Implementation Session

## Current Work Focus

**Project Status**: AI-Powered Data Enrichment Enhancement

- **Current Priority**: Enhance AI enrichment agent for comprehensive anime data standardization
- **System State**: Basic enrichment agent exists but needs intelligent multi-source data processing
- **Immediate Goal**: Implement AI-driven standardization and smart merging for multi-platform anime data

## Active Decisions and Considerations

**AI-Driven Data Enrichment Strategy**:
- **Multi-Source Integration**: Jikan, AniList, Kitsu, AnimeSchedule APIs based on `_sources` schema mapping
- **Intelligent Standardization**: AI normalizes different API schemas into uniform properties
- **Smart Character Merging**: AI deduplicates and enhances characters across multiple sources
- **Schema Compliance**: Output matches enhanced_anime_schema_example.json structure exactly

**Why AI-Powered Approach**:
- **Problem**: Different APIs have inconsistent field names, formats, and data structures
- **Solution**: AI intelligently maps, converts, and standardizes data from multiple sources
- **Advantage**: Comprehensive coverage with intelligent merging eliminates manual field mapping

**AI Enrichment Technical Implementation**:

**Source Mapping Strategy** (based on enhanced_anime_schema_example.json `_sources`):
```python
# Single-source fields (primary + fallback)
SINGLE_SOURCE_MAPPING = {
    "genres": {"primary": "anilist", "fallback": "jikan"},
    "demographics": {"primary": "jikan", "fallback": "anilist"},
    "themes": {"primary": "anilist", "fallback": "jikan"},
    "broadcast": {"primary": "animeschedule", "fallback": "jikan"},
    "rating": {"primary": "kitsu", "fallback": "jikan"}
}

# Multi-source fields (fetch from all)
MULTI_SOURCE_MAPPING = {
    "statistics": ["jikan", "anilist", "kitsu", "animeschedule"],
    "images": ["jikan", "anilist", "kitsu", "animeschedule"],
    "characters": ["jikan", "anilist"],  # Character merging
    "streaming_info": ["animeschedule", "kitsu"]
}
```

**AI Standardization Examples**:
```python
# AI-standardized statistics across platforms
"statistics": {
    "mal": {
        "score": 8.43,           # AI maps from raw "score"
        "scored_by": 2251158,    # AI maps from "scored_by"
        "rank": 68,              # AI maps from "rank"
        "popularity_rank": 12,   # AI maps from "popularity"
        "members": 3500000,      # AI maps from "members"
        "favorites": 89234       # AI maps from "favorites"
    },
    "anilist": {
        "score": 8.2,            # AI converts averageScore 82 → 8.2
        "scored_by": null,       # Not available
        "rank": null,            # Not available
        "popularity_rank": null, # Not available
        "members": 147329,       # AI maps from "popularity"
        "favorites": 53821       # AI maps from "favourites"
    },
    "kitsu": {
        "score": 8.21,           # AI converts averageRating 82.1 → 8.21
        "scored_by": 45123,      # AI maps from "ratingCount"
        "rank": null,            # Not available
        "popularity_rank": null, # Not available
        "members": null,         # Not available
        "favorites": 12456       # AI maps from "favoritesCount"
    }
}

# AI-merged character data
"characters": [
    {
        "name": "Kamado, Tanjirou",
        "name_variations": ["Tanjiro Kamado", "炭治郎"],  # AI merges variations
        "character_ids": {"mal": 146156, "anilist": 127212},  # AI combines IDs
        "images": {
            "mal": "https://cdn.myanimelist.net/images/characters/1/364490.jpg",
            "anilist": "https://s4.anilist.co/file/anilistcdn/character/large/b127212-AqNr8yCAAhQI.png"
        },  # AI collects from all sources
        "voice_actors": [...]  # AI deduplicates across sources
    }
]
```

**AI Enhancement Key Principles**:

**Multi-Source Data Handling**:
- **Single-Source Fields**: Use primary source, fallback to alternative if primary fails
- **Multi-Source Fields**: Fetch from ALL relevant APIs, leave empty if source lacks data
- **No Cross-Substitution**: For multi-source fields, don't substitute missing data with alternative sources

**AI Standardization Approach**:
- **Uniform Statistics Schema**: AI maps different field names to consistent properties
- **Unit Conversion**: AI handles scale conversions (82/100 → 8.2/10)
- **Character Deduplication**: AI identifies same characters across sources using fuzzy matching
- **Image Collection**: AI gathers character images from all available sources

**Performance Considerations**:
- **Existing API Infrastructure**: Use current rate limiting and pagination logic
- **No Additional Calls**: Work within existing fetch mechanisms
- **Character Chunking**: Maintain existing large dataset processing for characters
- **Schema Compliance**: Output must match enhanced_anime_schema_example.json exactly

**Critical Data Management Considerations**:

**Weekly Database Update Strategy**:
- **Current Update Process**: Need to investigate how weekly anime-offline-database updates work
- **Enrichment Preservation**: Enriched data (synopsis, characters, trailers) should NOT be overwritten during updates
- **Update Logic Required**: 
  ```python
  # Proposed update strategy
  if existing_anime.has_enrichment_data():
      # Only update core fields (title, episodes, status, etc.)
      # Preserve enriched fields (synopsis, characters, trailers)
      merge_core_fields_only(existing_anime, new_anime_data)
  else:
      # Full update for non-enriched entries
      full_update(existing_anime, new_anime_data)
  ```

**Data Staleness and Change Frequency**:
- **Static Data** (rarely changes):
  - Synopsis: Fixed once anime completes production
  - Character core info: Names, roles, basic descriptions
  - Trailers: PVs are typically static
- **Dynamic Data** (potential changes):
  - Character count: New characters might be added during long-running series
  - Character images: CDN links may break or be updated
  - Trailer links: YouTube videos may be removed/replaced
- **Refresh Strategy**: Need periodic re-enrichment for data validation

**CDN and External Link Reliability**:
- **Image CDN Issues**: 
  - MAL/AniList CDN links may break over time
  - Need image download and local caching strategy
  - Implement image validation and re-fetching mechanisms
- **YouTube Trailer Links**:
  - Trailer videos may be removed/made private
  - Need trailer validation and fallback mechanisms
- **API Availability**: Jikan/AniList APIs may have downtime

**Incremental Enrichment Strategy**:
- **Skip Already Enriched**: Check for existing synopsis/characters before API calls
- **Selective Re-enrichment**: Only re-fetch data older than X months
- **Validation Checks**: Periodically validate image URLs and trailer links
- **Metadata Tracking**: Track enrichment timestamps and sources for each anime

## Recent Changes

**API Testing Completed (Latest Session)**:
- **Validated data sources**: Confirmed synopsis, character, and trailer availability across platforms
- **Identified optimal strategy**: Jikan primary + AniList secondary approach
- **Fixed BaseClient issue**: Resolved deprecated `correlation_logger` attribute error
- **Discovered pagination**: AniList actually has 86 characters (not 25) when using pagination
- **Coverage analysis**: 74.6% MAL coverage (29,007 anime) available for immediate enrichment
- **📋 Detailed Results**: See [API_TESTING_RESULTS.md](API_TESTING_RESULTS.md) for comprehensive findings, data quality comparisons, and implementation recommendations

## Next Steps

**Immediate (Next Session)**:

- **PRIORITY 1**: Data Enrichment Implementation ⏳ **READY TO START**
  - **Phase 1a**: URL Parser Service (`src/services/url_parser.py`)
    - Extract MAL/AniList IDs from anime-offline-database source URLs
    - Handle edge cases (malformed URLs, missing IDs)
    - Validate extracted IDs before API calls
  - **Phase 1b**: Jikan Enrichment Service (`src/services/jikan_enrichment.py`)
    - Use existing Jikan client to fetch synopsis, characters, trailers
    - Implement batch processing with rate limiting (existing infrastructure)
    - Process 29,007 anime with MAL URLs (74.6% coverage)
  - **Phase 1c**: Database Schema Enhancement (`src/models/anime.py`)
    - Extend AnimeEntry model with synopsis, characters, trailers fields
    - Maintain backward compatibility with existing data
    - Add enrichment metadata (sources, timestamps, confidence, last_validated)
    - Implement enrichment flags to prevent overwriting during weekly updates
  - **Phase 1d**: Vector Generation Enhancement (`src/vector/qdrant_client.py`)
    - Include synopsis in text embedding generation
    - Store character data for fine-tuning dataset preparation
    - Update processing pipeline to handle enriched data

- **PRIORITY 2**: Enhanced Fine-Tuning Dataset Preparation
  - **Phase 2a**: Character Recognition Dataset (`src/vector/anime_dataset.py`)
    - Use enriched character images and names for training data
    - Create character-anime association datasets
    - Generate character similarity datasets across anime
  - **Phase 2b**: Enhanced Text Embeddings (`src/vector/text_processor.py`)
    - Include synopsis content in embedding generation: `title + synopsis + tags + studios`
    - Improve semantic search quality with richer content
    - Generate genre classification datasets from synopsis content
  - **Phase 2c**: Art Style Classification (`src/vector/art_style_classifier.py`)
    - Use character images as additional training data
    - Cross-reference with studio/year metadata for style classification
    - Create visual similarity datasets using character designs

- **PRIORITY 3**: Cross-Platform Data Validation
  - **Phase 3a**: AniList Pagination Implementation
    - Fix existing `get_anime_characters` to fetch all pages (86 vs 25 characters)
    - Implement proper GraphQL pagination with page tracking
    - Test with Attack on Titan to validate full character retrieval
  - **Phase 3b**: Character ID Mapping (`src/services/character_mapping.py`)
    - Match characters between Jikan (MAL IDs) and AniList (AniList IDs)
    - Use name similarity + role matching for cross-platform validation
    - Create unified character database with both platform IDs
  - **Phase 3c**: Data Quality Assurance
    - Validate synopsis length and quality across platforms
    - Cross-check character counts and image availability
    - Implement fallback mechanisms for missing or low-quality data
    - Create update strategy to preserve enriched data during weekly database updates
    - Implement periodic validation for CDN links and external resources
    - Design incremental re-enrichment for stale or broken data