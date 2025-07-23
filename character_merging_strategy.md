# Multi-Source Character Merging Strategy

## Overview

Intelligent character merging from 4 sources (Jikan, AniList, AniDB, Kitsu) using AI-powered matching and hierarchical data integration.

## Stage 5 Single-Agent Architecture

### Core Responsibilities

**MULTI-SOURCE CHARACTER PROCESSOR**: Single AI agent responsible for:

1. **Character Matching**: Intelligent cross-platform character identification
2. **Data Integration**: Hierarchical merging with conflict resolution
3. **Quality Validation**: Completeness and consistency checks
4. **Output Generation**: Unified character schema

## Input Data Structure

```yaml
Input Sources:
  jikan_characters: 126 characters with detailed biographies
  anilist_characters: 69 characters with structured metadata
  anidb_characters: 94 characters with types and ratings
  kitsu_characters: 10 characters (relationships only)
```

## Character Matching Algorithm

### Phase 1: Primary Name Matching

```yaml
Matching Strategy:
  1. Exact Name Match:
    - Jikan.name ↔ AniList.name.full
    - Jikan.name_kanji ↔ AniList.name.native

  2. Alternative Name Validation:
    - Check AniList.name.alternative array
    - Cross-reference with Jikan nicknames

  3. Fuzzy Name Matching:
    - Handle romanization differences
    - Account for spelling variations
```

### Phase 2: Role Consistency Check

```yaml
Role Validation:
  - Jikan "Main" ↔ AniList "MAIN" ↔ AniDB "main character"
  - Jikan "Supporting" ↔ AniList "SUPPORTING" ↔ AniDB "secondary character"
  - Flag mismatches for manual review
```

### Phase 3: Content Similarity Validation

```yaml
Similarity Checks:
  - Description content overlap (>30% similarity)
  - Voice actor name matching (Japanese VAs)
  - Physical characteristics consistency
```

## Data Integration Hierarchy

### Single-Source Fields (Use Best Source)

| Field            | Priority                | Reasoning                            |
| ---------------- | ----------------------- | ------------------------------------ |
| **Primary Name** | Jikan → AniList → AniDB | Jikan most standardized              |
| **Description**  | Jikan → AniList → AniDB | Jikan has richest biographies        |
| **URL**          | Jikan → AniList → AniDB | Jikan provides direct character page |
| **Birth Date**   | AniList → Jikan → AniDB | AniList structured format            |
| **Age**          | AniList → Jikan → AniDB | AniList most reliable                |
| **Gender**       | AniList → AniDB → Jikan | AniList explicit field               |
| **Blood Type**   | AniList → AniDB         | AniList priority                     |

### Multi-Source Fields (Collect All)

| Field               | Sources     | Merge Strategy     |
| ------------------- | ----------- | ------------------ |
| **Name Variations** | All sources | Deduplicated array |
| **Nicknames**       | Jikan only  | Direct from Jikan API |
| **Images**          | All sources | Source attribution |
| **Voice Actors**    | All sources | Language-grouped   |
| **Character IDs**   | All sources | Platform mapping   |

### Calculated Fields

| Field                | Calculation Method                           |
| -------------------- | -------------------------------------------- |
| **Popularity Score** | Jikan.favorites + AniList.favourites         |
| **Rating**           | AniDB.rating (unique to AniDB)               |
| **Character Type**   | AniDB.character_type (unique classification) |

## Enhanced Schema Output

```json
{
  "characters": [
    {
      "name": "Spike Spiegel",
      "role": "Main",
      "name_variations": [
        "スパイク・スピーゲル",
        "Spike Spike",
        "Swimming Bird"
      ],
      "name_kanji": "スパイク・スピーゲル",
      "name_native": "スパイク・スピーゲル",
      "nicknames": [],
      "character_ids": {
        "mal": 1,
        "anilist": 1,
        "anidb": 118,
        "kitsu": null
      },
      "images": {
        "jikan": "https://cdn.myanimelist.net/images/characters/11/516853.jpg",
        "anilist": "https://s4.anilist.co/file/anilistcdn/character/large/b1-ChxaldmieFlQ.png",
        "anidb": "https://cdn.anidb.net/images/main/14555.jpg"
      },
      "description": "Birthdate: June 26, 2044...", // From Jikan (richest)
      "url": "https://myanimelist.net/character/1/Spike_Spiegel", // From Jikan (direct link)
      "age": "27", // From AniList (structured)
      "gender": "Male", // From AniList (explicit)
      "dateOfBirth": {
        "year": 2044,
        "month": 6,
        "day": 26
      }, // From AniList (structured)
      "bloodType": "O", // From AniDB if available
      "character_type": "main character", // From AniDB (unique)
      "popularity": {
        "mal_favorites": 48162,
        "anilist_favourites": 12577,
        "combined_score": 60739
      },
      "rating": {
        "anidb_rating": 8.5,
        "anidb_votes": 150
      },
      "voice_actors": [
        {
          "name": "Yamadera, Kouichi",
          "language": "Japanese"
        },
        {
          "name": "Blum, Steven",
          "language": "English"
        }
      ]
    }
  ]
}
```

## Matching Confidence Levels

### High Confidence (Auto-Merge)

- Exact name + role match
- Voice actor confirmation
- Description similarity >70%

### Medium Confidence (Review)

- Name variations match
- Role consistency
- Some content overlap

### Low Confidence (Manual)

- Fuzzy name match only
- Role conflicts
- No supporting evidence

## Quality Validation Rules

### Completeness Checks

- ✅ Every Jikan character processed (primary source)
- ✅ AniList main characters matched
- ✅ AniDB characters integrated where possible
- ✅ No duplicate character entries

### Consistency Validation

- ✅ Name variations are logical
- ✅ Roles align across sources
- ✅ Voice actors match for Japanese VAs
- ✅ Age/birth date consistency

### Error Handling

- 🚨 Flag unmatched high-importance characters
- 🚨 Report role conflicts
- 🚨 Log character count discrepancies
- 🚨 Validate image URL accessibility

## Implementation Notes

### Performance Considerations

- Single-agent approach (15x faster than multi-agent)
- Batch processing with intelligent caching
- Progressive matching (exact → fuzzy → manual)

### Data Preservation

- Maintain all source attributions
- Preserve original IDs for cross-referencing
- Include matching confidence scores
- Store merge metadata for debugging

This strategy ensures comprehensive character data while maintaining accuracy through intelligent matching and hierarchical integration.
