# Active Context

# Anime MCP Server - Current Implementation Session

## Current Work Focus

**Project Status**: Data Enrichment Implementation for Enhanced Fine-Tuning

- **Current Priority**: Implement anime content enrichment pipeline (Task #119)
- **System State**: Fine-tuning infrastructure complete, but limited by missing synopsis/character data
- **Immediate Goal**: Enrich 29,007+ anime with synopsis, characters, and trailers to unlock fine-tuning potential

## Active Decisions and Considerations

**Data Enrichment Strategy**:
- **Primary Source**: Jikan API (comprehensive, no auth required, 77 characters per anime)
- **Secondary Source**: AniList API (validation, 86 characters with pagination)
- **Coverage**: 74.6% immediate coverage (29,007 anime) using existing MAL URLs
- **Approach**: Direct ID extraction from anime-offline-database source URLs (no fuzzy matching)

**Why This Approach**:
- **Problem**: Task #118 fine-tuning limited by missing synopsis and character data in anime-offline-database
- **Solution**: Enrich existing database with comprehensive content data to unlock fine-tuning potential
- **Advantage**: Existing source URLs eliminate need for fuzzy matching, ensuring 100% accuracy

**Technical Implementation Details**:

**URL Parsing Strategy**:
```python
# Extract IDs from existing sources like:
# "https://myanimelist.net/anime/16498" → mal_id: 16498
# "https://anilist.co/anime/16498" → anilist_id: 16498
def extract_platform_ids(sources):
    platform_ids = {}
    for source in sources:
        if 'myanimelist.net/anime/' in source:
            platform_ids['mal'] = int(source.split('/anime/')[1])
        elif 'anilist.co/anime/' in source:
            platform_ids['anilist'] = int(source.split('/anime/')[1])
    return platform_ids
```

**Data Structure Plan**:
```python
# Enhanced anime entry structure
{
    "title": "Attack on Titan",
    "sources": [...],  # Existing
    "synopsis": "Centuries ago, mankind was slaughtered...",  # NEW from Jikan
    "characters": [  # NEW from Jikan + AniList
        {
            "name": "Eren Yeager",
            "role": "Main",
            "character_ids": {
                "mal": 40882,      # From Jikan
                "anilist": 40882   # From AniList cross-reference
            },
            "images": {
                "jikan": "https://cdn.myanimelist.net/images/characters/10/216895.jpg",
                "anilist": "https://s4.anilist.co/file/anilistcdn/character/large/b40882-dsj2Ibw8VlpA.jpg"
            }
        }
    ],
    "trailers": [  # NEW from Jikan/AniList
        {
            "youtube_id": "LHtdKWJdif4",
            "source": "jikan",
            "thumbnail": "https://i.ytimg.com/vi/LHtdKWJdif4/hqdefault.jpg"
        }
    ]
}
```

**API Integration Specifics**:
- **Jikan Client**: Already implemented (`src/integrations/clients/jikan_client.py`)
- **AniList Client**: Already implemented with pagination support discovered
- **Character Endpoints**: 
  - Jikan: `/anime/{id}/characters` (77 characters for Attack on Titan)
  - AniList: GraphQL with pagination (86 characters for Attack on Titan)
- **Rate Limiting**: Both clients have existing rate limiting infrastructure

**Key Technical Considerations**:
- **Character Data**: AniList pagination discovered (86 vs 25 characters) - need to implement full pagination
- **Synopsis Quality**: Jikan/MAL provides longer descriptions (1157 vs 837 chars) - use Jikan as primary
- **Image Quality**: Both platforms provide high-quality character images - collect from both for redundancy
- **Cross-Platform IDs**: Need mapping between AniList and MAL character IDs for comprehensive coverage
- **Storage Strategy**: Extend existing AnimeEntry model vs create separate enrichment tables
- **Vector Integration**: Include synopsis in text embeddings, character data for fine-tuning datasets

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