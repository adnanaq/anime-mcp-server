# Active Context

# Anime MCP Server - Current Implementation Session

## Current Work Focus

**Project Status**: 5-Stage Modular Enrichment Pipeline Complete - Ready for Multi-Source Integration

- **Current Priority**: Add multi-source data integration (AniList, Kitsu, AnimeSchedule, AniDB)
- **System State**: 5-stage modular prompt system working optimally with character processing
- **Recent Achievement**: Successfully implemented Stage 5 character processing with 100% character coverage
- **Immediate Goal**: Extend current Jikan-only system to multi-source data aggregation
- **Next Phase**: Multi-source integration → Large anime testing → Schema validation

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

**5-Stage Modular Enrichment Pipeline (Latest Session)**:
- **✅ Modular Prompt System Complete**: `src/services/iterative_ai_enrichment_v2.py` (556 lines) with 5 specialized stages
- **✅ Stage 5 Character Processing**: Successfully processes ALL characters (38 for Dandadan) with multi-language voice actors
- **✅ Token Optimization**: 90%+ token reduction per stage, eliminated timeout issues
- **✅ Performance Achieved**: 4-minute total pipeline (vs 4+ minute single stage previously)
- **✅ API Independence**: Complete Jikan API integration (15 calls: 3 base + 12 episodes + 1 characters)
- **✅ Programmatic Assembly**: Field-specific extraction with deterministic merge (no AI assembly)
- **✅ Character Schema Compliance**: Full character data with proper voice actor and image mapping
- **⚠️ Single-Source Limitation**: Currently Jikan-only, needs multi-source integration
- **⚠️ Large Anime Untested**: Not tested on 100+ episode series like One Piece
- **⚠️ Schema Validation Missing**: No runtime validation against enhanced_anime_schema_example.json

## Next Steps

**Immediate (Next Session)**:

- **PRIORITY 1**: Multi-Source Data Integration 🚀 **HIGH PRIORITY**
  - **Current State**: 5-stage pipeline works perfectly with Jikan, needs multi-source extension
  - **Target Sources**: AniList, Kitsu, AnimeSchedule, AniDB (based on enhanced_anime_schema_example.json)
  - **Phase 1a**: Stage 6 - Multi-Source Data Fetching
    - Add parallel API calls to all 4 additional sources alongside existing Jikan
    - Implement source-specific data extraction and normalization
    - Add intelligent fallback mechanisms when sources are unavailable
  - **Phase 1b**: Stage 7 - Multi-Source Data Merge
    - Implement AI-powered multi-source data merging based on `_sources` mapping
    - Handle single-source fields (primary + fallback) vs multi-source fields (aggregate all)
    - Add conflict resolution for contradictory data across sources
  - **Expected**: Complete multi-source coverage matching enhanced_anime_schema_example.json

- **PRIORITY 2**: Large Anime Testing & Optimization 📏 **CRITICAL**
  - **Current Limitation**: Only tested on 12-episode anime (Dandadan)
  - **Phase 2a**: One Piece Testing Protocol
    - Test 5-stage pipeline on One Piece (1000+ episodes)
    - Monitor API call scaling (3 base + 1000+ episodes + 1 characters = 1000+ calls)
    - Implement episode chunking and rate limiting optimization
  - **Phase 2b**: Performance Optimization for Large Series
    - Add episode count limits and sampling strategies
    - Implement intelligent episode selection (key episodes, season finales, etc.)
    - Add configurable episode processing limits for different anime sizes
  - **Expected**: Handle anime with 100+ episodes without timeout or rate limiting issues

- **PRIORITY 3**: Schema Validation Implementation 🔍 **MEDIUM**
  - **Current Gap**: No runtime validation against enhanced_anime_schema_example.json structure
  - **Phase 3a**: Pydantic Model Validation
    - Add comprehensive validation against enhanced AnimeEntry schema
    - Implement proper type conversion from AI dictionaries to Pydantic models
    - Add validation error reporting and recovery mechanisms
  - **Phase 3b**: Data Quality Assurance
    - Implement enhanced_metadata tracking with data quality scores
    - Add cross-platform data correlation validation
    - Enable periodic re-validation and data freshness checks
  - **Expected**: 100% schema compliance with production-ready validation

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