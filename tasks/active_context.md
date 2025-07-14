# Active Context

# Anime MCP Server - Current Implementation Session

## Current Work Focus

**Project Status**: Schema Compliance Crisis and Implementation Gaps

- **Current Priority**: Fix critical schema violations in iterative AI enrichment system
- **System State**: Proof-of-concept AI enrichment working but 60%+ schema non-compliant
- **Immediate Goal**: Align implementation with AnimeEntry Pydantic models and fix type system violations
- **Critical Issue**: AI returns unvalidated dictionaries instead of proper Pydantic models

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

**Iterative AI Enrichment Implementation (Latest Session)**:
- **✅ Proof of Concept Complete**: `src/services/iterative_ai_enrichment.py` (593 lines) working
- **✅ Episode Processing Optimized**: Batch fetching (2 per batch, 800ms delays) for performance
- **✅ Real Test Results**: Successfully enriches Dandadan from 15 → 40+ fields
- **✅ API Integration**: Full Jikan API integration with anime details and episodes
- **❌ Character Processing Removed**: Eliminated per user performance feedback
- **❌ Schema Compliance Crisis**: 60%+ of AnimeEntry schema fields missing or incorrectly typed
- **❌ Type System Violations**: AI returns dictionaries instead of Pydantic models
- **⚠️ Critical Gap**: Implementation doesn't validate against `src/models/anime.py` AnimeEntry schema

## Next Steps

**Immediate (Next Session)**:

- **PRIORITY 1**: Modular Prompt Chunking Strategy ⚡ **HIGH PRIORITY**
  - **Critical Issue**: 200+ line monolithic prompt causing performance delays and timeouts
  - **Phase 1a**: Extract mega-prompt into specialized template files (metadata, episodes, relationships)
    - Create `src/services/prompts/` directory structure with base, specialized, and schema folders
    - Break current prompt into focused, domain-specific prompts
    - Implement token management and context window optimization
  - **Phase 1b**: Multi-Stage Processing Pipeline Implementation
    - Refactor `_ai_enrich_data()` into 5-stage pipeline (metadata → episodes → relationships → stats → assembly)
    - Add `PromptTemplateManager` class for file-based prompt loading and substitution
    - Implement stage-specific error handling and retry logic
  - **Expected**: 60-80% performance improvement, 90%+ success rate, timeout elimination

- **PRIORITY 2**: Schema Compliance Crisis Resolution 🚨 **CRITICAL**
  - **Phase 1a**: Schema Validation Implementation (`src/services/iterative_ai_enrichment.py`)
    - Add Pydantic model validation wrapper around AI output processing
    - Ensure AI dictionaries convert to proper ThemeEntry, StaffEntry, StreamingEntry models
    - Implement AnimeEntry validation before return
    - Add comprehensive error handling for schema violations
  - **Phase 1b**: Enrichment Metadata Tracking
    - Implement `EnrichmentMetadata` model creation during AI processing
    - Add success/failure tracking and timestamp logging
    - Enable update age checking and re-enrichment logic
    - Track data quality scores and completeness metrics
  - **Phase 1c**: Character Processing Decision Resolution
    - Decide: Remove `characters: List[CharacterEntry]` from schema OR re-implement character processing
    - Update schema documentation to match implementation reality
    - Ensure schema and implementation are aligned
  - **Phase 1d**: Type Safety and Model Integration
    - Convert all AI dictionary outputs to proper Pydantic models
    - Add runtime validation and error reporting
    - Implement comprehensive testing with AnimeEntry model validation

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