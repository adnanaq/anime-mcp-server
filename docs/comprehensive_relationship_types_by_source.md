# Comprehensive Relationship Types by Source

## Overview

This document provides the complete catalog of relationship types available across all anime database sources. This information is critical for Stage 3 relationship processing to ensure proper standardization and mapping between different platforms.

## **Why This Documentation is Critical**

Our initial analysis using Cowboy Bebop (1998) only revealed limited relationship types:
- `Adaptation`, `Side Story`, `Summary` (Jikan)
- `sideStories`, `other` (AnimSchedule)  
- `same_franchise` (AnimePlanet)

However, modern anime franchises use **much more comprehensive relationship vocabularies**. This document provides the complete reference for all possible types across sources.

---

## **1. Jikan (MyAnimeList) - Most Comprehensive**

### **Core Relationship Types**
```yaml
Primary Types:
  "Adaptation"           # Source material → Anime adaptation
  "Sequel"              # Direct continuation of story
  "Prequel"             # Events occurring before original
  "Side Story"          # Related but separate storyline
  "Summary"             # Compilation/recap episodes
  "Alternative Setting" # Alternative universe/timeline
  "Alternative Version" # Different adaptation of same source
  "Character"           # Character-focused spin-off
  "Full Story"          # Complete/extended version
  "Parent Story"        # Main/original story
  "Spin-off"            # New series in same universe
  "Other"               # General/unspecified relationship
```

### **Extended Types** (Less Common)
```yaml
"Alternative"         # Generic alternative version
"Compilation"         # Collection/anthology
"Music"               # Music videos/concerts
"Other"               # Catchall category
```

### **Data Structure**
```json
{
  "relations": [
    {
      "relation": "Sequel",
      "entry": [
        {
          "mal_id": 123,
          "type": "anime",
          "name": "Series Name Season 2",
          "url": "https://myanimelist.net/anime/123/Series_Name_Season_2"
        }
      ]
    }
  ]
}
```

---

## **2. AnimePlanet - Dual System**

### **Primary Classification**
```yaml
relation_type: "same_franchise"  # Always this value
```

### **Detailed Subtypes**
```yaml
relation_subtype:
  "Sequel"              # Direct sequel series
  "Prequel"             # Prequel series
  "Alternative Version" # Different version/adaptation
  "Side Story"          # Side story content
  "Spin-off"            # Spin-off series
  "Movie"               # Related movies
  "OVA"                 # Original Video Animation
  "Special"             # Special episodes
  "Music Video"         # Related music videos
  "Live Action"         # Live-action adaptations
```

### **Data Structure**
```json
{
  "related_anime": [
    {
      "title": "Series Name Season 2 2024-01-01 - 2024-03-31 TV: 12 ep",
      "slug": "series-name-season-2",
      "url": "https://www.anime-planet.com/anime/series-name-season-2",
      "relation_type": "same_franchise",
      "relation_subtype": "Sequel",
      "type": "TV",
      "year": 2024,
      "start_date": "2024-01-01"
    }
  ]
}
```

---

## **3. AnimSchedule - Category-Based**

### **Relationship Categories**
```yaml
"sequels"      # Sequel series
"prequels"     # Prequel series  
"sideStories"  # Side stories and spin-offs
"movies"       # Related movies
"specials"     # Special episodes/OVAs
"adaptations"  # Source material adaptations
"other"        # Miscellaneous relationships
"music"        # Music videos/concerts
"liveAction"   # Live-action versions
```

### **Data Structure**
```json
{
  "relations": {
    "sequels": [
      "series-name-season-2",
      "series-name-season-3"
    ],
    "movies": [
      "series-name-movie-1"  
    ],
    "sideStories": [
      "series-name-side-story"
    ]
  }
}
```

### **Route Conversion**
- Base URL: `https://animeschedule.net/anime/{route}`
- Title extraction from slug: `series-name-season-2` → "Series Name Season 2"

---

## **4. AniList - GraphQL Schema Types**

### **MediaRelation Enum Values**
```yaml
"ADAPTATION"       # Adaptation of source material
"SEQUEL"           # Sequel
"PREQUEL"          # Prequel
"ALTERNATIVE"      # Alternative version
"SPIN_OFF"         # Spin-off series
"SIDE_STORY"       # Side story
"CHARACTER"        # Character-focused content
"SUMMARY"          # Summary/compilation
"FULL_STORY"       # Extended full story
"PARENT"           # Parent/main story
"OTHER"            # Other relationships
"COMPILATION"      # Compilation work
```

### **Data Structure** (GraphQL)
```graphql
{
  relations {
    edges {
      node {
        id
        title {
          romaji
          english
          native
        }
        type
        format
      }
      relationType
    }
  }
}
```

---

## **5. Kitsu - JSON:API Format**

### **Relationship Types**
```yaml
"sequel"               # Sequel
"prequel"             # Prequel
"adaptation"          # Source adaptation
"side_story"          # Side story
"parent_story"        # Parent story
"alternative_setting" # Alternative universe
"alternative_version" # Alternative version
"full_story"          # Complete story
"summary"             # Summary/compilation
"character"           # Character spin-off
"other"               # Other relationship
```

### **Data Structure** (JSON:API)
```json
{
  "relationships": {
    "mediaRelationships": {
      "links": {
        "self": "https://kitsu.io/api/edge/anime/1/relationships/media-relationships",
        "related": "https://kitsu.io/api/edge/anime/1/media-relationships"
      }
    }
  }
}
```

**Note**: Kitsu requires additional API calls to get actual relationship data.

---

## **6. AniDB - Relationship Types**

### **Relation Categories**
```yaml
"sequel"          # Sequel
"prequel"         # Prequel  
"same setting"    # Same universe/setting
"alternative setting" # Alternative universe
"alternative version" # Different version
"character"       # Character-based relation
"side story"      # Side story
"parent story"    # Main story
"summary"         # Summary/compilation
"full story"      # Extended version
"other"           # Other relationships
```

### **Data Access**
- Requires web scraping or dedicated API integration
- Data typically embedded in HTML pages
- No public JSON API available

---

## **7. Offline Database**

### **Relationship Data**
```yaml
Format: Raw URL arrays
Types: None provided - URLs only
Processing: Requires title extraction and type inference
```

### **Data Structure**
```json
{
  "relatedAnime": [
    "https://myanimelist.net/anime/123",
    "https://anilist.co/anime/456", 
    "https://anime-planet.com/anime/series-name-season-2"
  ]
}
```

---

## **Cross-Source Relationship Type Mapping**

### **Standardized Mapping Table**

| Standard Type | Jikan | AnimePlanet | AnimSchedule | AniList | Kitsu |
|---------------|--------|-------------|--------------|---------|--------|
| **Sequel** | `Sequel` | `Sequel` | `sequels` | `SEQUEL` | `sequel` |
| **Prequel** | `Prequel` | `Prequel` | `prequels` | `PREQUEL` | `prequel` |
| **Side Story** | `Side Story` | `Side Story` | `sideStories` | `SIDE_STORY` | `side_story` |
| **Spin-off** | `Spin-off` | `Spin-off` | `sideStories` | `SPIN_OFF` | `other` |
| **Alternative Version** | `Alternative Version` | `Alternative Version` | `other` | `ALTERNATIVE` | `alternative_version` |
| **Adaptation** | `Adaptation` | *(manga only)* | `adaptations` | `ADAPTATION` | `adaptation` |
| **Summary** | `Summary` | *(not specific)* | `specials` | `SUMMARY` | `summary` |
| **Character** | `Character` | *(not specific)* | `other` | `CHARACTER` | `character` |
| **Parent Story** | `Parent Story` | *(not specific)* | `other` | `PARENT` | `parent_story` |
| **Full Story** | `Full Story` | *(not specific)* | `other` | `FULL_STORY` | `full_story` |
| **Other** | `Other` | *(same_franchise)* | `other` | `OTHER` | `other` |

### **Priority Hierarchy for Conflicts**
1. **Jikan** - Most standardized and comprehensive
2. **AniList** - Well-structured GraphQL types  
3. **AnimePlanet** - Good detail with subtypes
4. **AnimSchedule** - Category-based, good coverage
5. **Kitsu** - Limited API access
6. **AniDB** - Requires scraping

---

## **Implementation Guidelines for Stage 3**

### **Enhanced Relationship Type Vocabulary** (Preserve Granularity)

```yaml
# Specific Types (High Quality - Preserve from AnimePlanet)
Specific_Content_Types:
  - "Movie"              # Preserve AnimePlanet distinction
  - "OVA"                # Preserve AnimePlanet distinction  
  - "Special"            # Preserve AnimePlanet distinction
  - "Music Video"        # Preserve AnimePlanet distinction
  - "Live Action"        # Preserve AnimePlanet distinction

# Story Relationship Types  
Story_Relationship_Types:
  - "Sequel"
  - "Prequel" 
  - "Spin-off"
  - "Side Story"
  - "Alternative Version"
  - "Character"          # Character-focused content
  
# Source Relationship Types
Source_Relationship_Types:
  - "Adaptation"         # From source material
  - "Parent Story"       # Main story
  - "Full Story"         # Extended version
  - "Summary"            # Compilation/recap

# Fallback Only
Generic_Types:
  - "Other"              # Use only when no specific type available
```

### **Revised Source Processing Strategy** (Jikan + AnimePlanet Base)

**Phase 1: Co-Primary Sources**
1. **Jikan Processing**: Use comprehensive types as-is (12+ types, manga support)
2. **AnimePlanet Processing**: Preserve granular subtypes (Movie, OVA, Special, etc.)

**Phase 2: Supplementary Sources**
3. **AnimSchedule Processing**: Use for validation and gap-filling only
4. **Offline URLs**: Comprehensive expansion with intelligent type inference

**Key Principle**: Build on Jikan + AnimePlanet foundation, supplement with others

### **Enhanced Fallback Strategy** (Preserve Granularity)

```yaml
Intelligent_Type_Inference (For Offline URLs Only):
  1. Pattern-based inference (preserve specificity):
     - "Season 2/3/etc" → "Sequel"
     - "Movie/Film" → "Movie"           # Keep specific
     - "OVA" → "OVA"                   # Keep specific
     - "Special" → "Special"           # Keep specific
     - "Music Video" → "Music Video"   # Keep specific
     - "Live Action" → "Live Action"   # Keep specific
     
  2. Use source anime context for relationship inference
  3. **Last resort only**: Default to "Other"

Priority_Rules:
  - Always prefer specific types over generic
  - Never downgrade AnimePlanet granularity 
  - Only use "Other" when absolutely no type can be determined
```

### **Quality Assurance**

```yaml
Validation_Rules:
  1. Ensure all relationship types are in standard vocabulary
  2. Validate logical consistency (no anime being its own sequel)
  3. Check for circular references in relationships
  4. Verify URL format correctness
  5. Ensure title meaningfulness (no "Anime ID 12345" patterns)
```

---

## **Key Insights for Stage 3 Processing**

### **1. Vocabulary Complexity**
- **Jikan**: 12+ distinct relationship types
- **AnimePlanet**: 10+ subtypes within "same_franchise"  
- **AnimSchedule**: 8+ category types
- Total unique concepts: 15+ standardized types needed

### **2. Processing Challenges**
- **Type Inconsistency**: Same relationship expressed differently across sources
- **Granularity Differences**: Some sources more specific than others
- **Missing Data**: Not all sources provide relationship types

### **3. Standardization Benefits**
- **Consistent Output**: Unified vocabulary regardless of source
- **Better User Experience**: Predictable relationship categorization
- **Improved Search**: Standardized filtering capabilities
- **Cross-Platform Compatibility**: Same types across all anime entries

This comprehensive reference ensures Stage 3 relationship processing can handle the full spectrum of anime relationships across all database sources, not just the limited examples we saw with older anime like Cowboy Bebop.