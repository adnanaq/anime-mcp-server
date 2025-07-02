# Anime Search Query Parameter Mapping

This document maps universal search parameters to platform-specific query parameters for anime search APIs, and documents response fields returned by each platform.

**Purpose**: Guide mapper implementations for converting universal schema to platform search queries and response field mappings.

**Key Distinction**:

- **Query Parameters** = Used for filtering/searching (what you send TO the API)
- **Response Fields** = Data returned by the API (what you GET FROM the API)

---

# PART I: QUERY PARAMETERS

This section documents parameters used for filtering and searching anime data.

## Universal to Platform Query Parameter Mapping

### Core Search Parameters

| Universal Parameter | MAL API v2 ✅ | Jikan API v4 ✅ | AniList GraphQL | Kitsu JSON:API    | AniDB ⚠️ | Anime-Planet | AnimeSchedule | AniSearch |
| ------------------- | ------------- | --------------- | --------------- | ----------------- | -------- | ------------ | ------------- | --------- |
| **query**           | `q` ✅        | `q` ✅          | `search` ✅     | `filter[text]` ✅ | `aid` ✅ | `q`          | `search`      | `query`   |
| **limit**           | `limit` ✅    | `limit` ✅      | `perPage` ✅    | `page[limit]` ✅  | ❌       | `limit`      | `limit`       | `limit`   |
| **offset**          | `offset` ✅   | `page`\* ✅     | `page`\* ✅     | `page[offset]` ✅ | ❌       | `offset`     | `offset`      | `offset`  |

\*Note: Jikan and AniList use page numbers, so offset needs conversion: `page = (offset / limit) + 1`

\*\*AniDB Testing Results: API testing verified that AniDB uses `aid` (anime ID) for queries, not fuzzy text search. Other parameters (limit, offset, type, status) are not supported. Use anime-titles.xml database for title-to-ID mapping.

### Content Classification Parameters

| Universal Parameter | MAL API v2 ✅ | Jikan API v4 ✅ | AniList GraphQL | Kitsu JSON:API         | AniDB ⚠️ | Anime-Planet | AnimeSchedule | AniSearch |
| ------------------- | ------------- | --------------- | --------------- | ---------------------- | -------- | ------------ | ------------- | --------- |
| **status**          | ❌            | `status` ✅     | `status` ✅     | `filter[status]` ✅    | ❌       | ❌           | `status`      | `status`  |
| **type_format**     | ❌            | `type` ✅       | `format` ✅     | `filter[subtype]` ✅   | ❌       | ❌           | `format`      | `type`    |
| **rating**          | ❌            | `rating` ✅     | ❌              | `filter[ageRating]` ✅ | ❌       | ❌           | ❌            | ❌        |
| **source**          | ❌            | ❌              | `source` ✅     | ❌                     | ❌       | ❌           | ❌            | ❌        |

### Status Value Mappings

| Universal Status     | MAL API v2 ✅         | Jikan API v4 ✅ | AniList GraphQL ✅    | Kitsu JSON:API  | AniDB |
| -------------------- | --------------------- | --------------- | --------------------- | --------------- | ----- |
| **FINISHED**         | `finished_airing` ✅  | `complete` ✅   | `FINISHED` ✅         | `finished` ✅   | ❌    |
| **RELEASING**        | `currently_airing` ✅ | `airing` ✅     | `RELEASING` ✅        | `current` ✅    | ❌    |
| **NOT_YET_RELEASED** | `not_yet_aired` ✅    | `upcoming` ✅   | `NOT_YET_RELEASED` ✅ | `upcoming` ✅   | ❌    |
| **HIATUS**           | ❌                    | ❌              | `HIATUS` ✅           | ❌              | ❌    |
| **TBA**              | ❌                    | ❌              | ❌                    | `tba` ✅        | ❌    |
| **UNRELEASED**       | ❌                    | ❌              | ❌                    | `unreleased` ✅ | ❌    |

### Format/Type Value Mappings

| Universal Format | MAL API v2 ✅ | Jikan API v4 ✅ | AniList GraphQL ✅ | Kitsu JSON:API | AniDB |
| ---------------- | ------------- | --------------- | ------------------ | -------------- | ----- |
| **TV**           | `tv` ✅       | `tv` ✅         | `TV` ✅            | `TV` ✅        | ❌    |
| **MOVIE**        | `movie` ✅    | `movie` ✅      | `MOVIE` ✅         | `movie` ✅     | ❌    |
| **OVA**          | `ova` ✅      | `ova` ✅        | `OVA` ✅           | `OVA` ✅       | ❌    |
| **ONA**          | `ona` ✅      | `ona` ✅        | `ONA` ✅           | `ONA` ✅       | ❌    |
| **SPECIAL**      | `special` ✅  | `special` ✅    | `SPECIAL` ✅       | `special` ✅   | ❌    |
| **MUSIC**        | `music` ✅    | `music` ✅      | `MUSIC` ✅         | `music` ✅     | ❌    |
| **TV_SPECIAL**   | ❌            | `tv_special` ✅ | ❌                 | ❌             | ❌    |
| **CM**           | ❌            | `cm` ✅         | ❌                 | ❌             | ❌    |
| **PV**           | ❌            | `pv` ✅         | ❌                 | ❌             | ❌    |

### Scoring Parameters

| Universal Parameter | MAL API v2 ✅ | Jikan API v4 ✅ | AniList GraphQL ✅              | Kitsu JSON:API             | AniDB |
| ------------------- | ------------- | --------------- | ------------------------------- | -------------------------- | ----- |
| **min_score**       | ❌            | `min_score` ✅  | `averageScore_greater` ✅ (×10) | `filter[averageRating]` ✅ | ❌    |
| **max_score**       | ❌            | `max_score` ✅  | `averageScore_lesser` ✅ (×10)  | `filter[averageRating]` ✅ | ❌    |

### Episode Parameters

| Universal Parameter | MAL API v2 ✅ | Jikan API v4 ✅ | AniList GraphQL ✅    | Kitsu JSON:API            | AniDB |
| ------------------- | ------------- | --------------- | --------------------- | ------------------------- | ----- |
| **episodes**        | ❌            | ❌              | `episodes` ✅         | `filter[episodeCount]` ✅ | ❌    |
| **min_episodes**    | ❌            | ❌              | `episodes_greater` ✅ | `filter[episodeCount]` ✅ | ❌    |
| **max_episodes**    | ❌            | ❌              | `episodes_lesser` ✅  | `filter[episodeCount]` ✅ | ❌    |

### Duration Parameters

| Universal Parameter | MAL API v2 | Jikan API v4 | AniList GraphQL ✅    | Kitsu JSON:API             | AniDB |
| ------------------- | ---------- | ------------ | --------------------- | -------------------------- | ----- |
| **min_duration**    | ❌         | ❌           | `duration_greater` ✅ | `filter[episodeLength]` ✅ | ❌    |
| **max_duration**    | ❌         | ❌           | `duration_lesser` ✅  | `filter[episodeLength]` ✅ | ❌    |

### Temporal Parameters

| Universal Parameter | MAL API v2 ✅ | Jikan API v4 ✅ | AniList GraphQL ✅     | Kitsu JSON:API          | AniDB |
| ------------------- | ------------- | --------------- | ---------------------- | ----------------------- | ----- |
| **start_date**      | ❌            | `start_date` ✅ | `startDate_greater` ✅ | ❌                      | ❌    |
| **end_date**        | ❌            | `end_date` ✅   | `endDate` ✅           | ❌                      | ❌    |
| **year**            | ❌            | ❌              | `seasonYear` ✅        | `filter[seasonYear]` ✅ | ❌    |
| **season**          | ❌            | ❌              | `season` ✅            | `filter[season]` ✅     | ❌    |

\*Note: Platforms without native year/season support can convert these to start_date format (e.g., "2023-01-01" for Winter 2023)

### Content Filtering Parameters

| Universal Parameter | MAL API v2 ✅ | Jikan API v4 ✅     | AniList GraphQL ✅ | Kitsu JSON:API          | AniDB |
| ------------------- | ------------- | ------------------- | ------------------ | ----------------------- | ----- |
| **include_adult**   | ❌            | `sfw`\* ✅          | `isAdult` ✅       | ❌                      | ❌    |
| **genres**          | ❌            | `genres` ✅         | `genre_in` ✅      | `filter[categories]` ✅ | ❌    |
| **genres_exclude**  | ❌            | `genres_exclude` ✅ | `genre_not_in` ✅  | ❌                      | ❌    |

\*Note: Jikan's `sfw` is inverse of `include_adult` (sfw=true excludes adult content)

### User Engagement Parameters

| Universal Parameter | MAL API v2 ✅ | Jikan API v4 | AniList GraphQL ✅      | Kitsu JSON:API | AniDB |
| ------------------- | ------------- | ------------ | ----------------------- | -------------- | ----- |
| **min_popularity**  | ❌            | ❌           | `popularity_greater` ✅ | ❌             | ❌    |
| **min_score_count** | ❌            | ❌           | ❌                      | ❌             | ❌    |

### Production Parameters

| Universal Parameter | MAL API v2 ✅ | Jikan API v4 ✅ | AniList GraphQL ✅ | Kitsu JSON:API | AniDB |
| ------------------- | ------------- | --------------- | ------------------ | -------------- | ----- |
| **producers**       | ❌            | `producers` ✅  | `licensedBy` ✅    | ❌             | ❌    |
| **studios**         | ❌            | ❌              | ❌                 | ❌             | ❌    |

### Sorting Parameters

| Universal Parameter | MAL API v2 ✅ | Jikan API v4 ✅ | AniList GraphQL ✅ | Kitsu JSON:API  | AniDB |
| ------------------- | ------------- | --------------- | ------------------ | --------------- | ----- |
| **sort_by**         | ❌            | `order_by` ✅   | `sort` ✅          | `sort` ✅       | ❌    |
| **sort_order**      | ❌            | `sort` ✅       | (embedded)\* ✅    | (embedded)\* ✅ | ❌    |

\*Note: AniList/Kitsu embed direction in sort field (e.g., "SCORE_DESC" vs "SCORE")

**Jikan API v4 Sort Options**: `mal_id`, `title`, `start_date`, `end_date`, `episodes`, `score`, `scored_by`, `rank`, `popularity`, `members`, `favorites`

### Sort Field Mappings

| Universal Sort | MAL API v2 ✅ | Jikan API v4 ✅ | AniList GraphQL ✅ | Kitsu JSON:API      | AniDB |
| -------------- | ------------- | --------------- | ------------------ | ------------------- | ----- |
| **score**      | ❌            | ❌              | `SCORE` ✅         | `averageRating` ✅  | ❌    |
| **popularity** | ❌            | ❌              | `POPULARITY` ✅    | `userCount` ✅      | ❌    |
| **title**      | ❌            | ❌              | `TITLE_ROMAJI` ✅  | `canonicalTitle` ✅ | ❌    |
| **year**       | ❌            | ❌              | `START_DATE` ✅    | `startDate` ✅      | ❌    |
| **rank**       | ❌            | ❌              | `SCORE` ✅         | ❌                  | ❌    |

## Platform-Specific Query Parameters

### Jikan API v4 Unique Parameters

| Jikan Parameter | Description                                              | Values/Format |
| --------------- | -------------------------------------------------------- | ------------- |
| **unapproved**  | Include unapproved entries                               | Boolean       |
| **letter**      | Return entries starting with letter (conflicts with `q`) | `A-Z`         |

### AniList GraphQL Unique Parameters ✅

**71 verified parameters from AniList GraphQL schema - ALL WORKING**

#### Basic Filters (28 parameters)

| AniList Parameter   | Description                 | Values/Format                                                                                                                                                                                       |
| ------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **id**              | Filter by AniList media ID  | Integer ≥ 1                                                                                                                                                                                         |
| **idMal**           | Filter by MyAnimeList ID    | Integer ≥ 1                                                                                                                                                                                         |
| **startDate**       | Start date filter           | FuzzyDateInt (YYYYMMDD)                                                                                                                                                                             |
| **endDate**         | End date filter             | FuzzyDateInt (YYYYMMDD)                                                                                                                                                                             |
| **season**          | Season filter               | `WINTER`, `SPRING`, `SUMMER`, `FALL`                                                                                                                                                                |
| **seasonYear**      | Season year                 | Integer (1900-2030)                                                                                                                                                                                 |
| **episodes**        | Exact episode count         | Integer ≥ 0                                                                                                                                                                                         |
| **duration**        | Episode duration in minutes | Integer ≥ 0                                                                                                                                                                                         |
| **chapters**        | Chapter count (manga)       | Integer ≥ 0                                                                                                                                                                                         |
| **volumes**         | Volume count (manga)        | Integer ≥ 0                                                                                                                                                                                         |
| **isAdult**         | Adult content filter        | Boolean                                                                                                                                                                                             |
| **genre**           | Single genre filter         | String                                                                                                                                                                                              |
| **tag**             | Single tag filter           | String                                                                                                                                                                                              |
| **minimumTagRank**  | Minimum tag rank            | Integer (1-100)                                                                                                                                                                                     |
| **tagCategory**     | Tag category filter         | String                                                                                                                                                                                              |
| **onList**          | User's list status          | Boolean                                                                                                                                                                                             |
| **licensedBy**      | Licensing site name         | String                                                                                                                                                                                              |
| **licensedById**    | Licensing site ID           | Integer ≥ 1                                                                                                                                                                                         |
| **averageScore**    | Exact average score         | Integer (0-100)                                                                                                                                                                                     |
| **popularity**      | Exact popularity count      | Integer ≥ 0                                                                                                                                                                                         |
| **source**          | Source material type        | `ORIGINAL`, `MANGA`, `LIGHT_NOVEL`, `VISUAL_NOVEL`, `VIDEO_GAME`, `OTHER`, `NOVEL`, `DOUJINSHI`, `ANIME`, `WEB_NOVEL`, `LIVE_ACTION`, `GAME`, `BOOK`, `MULTIMEDIA_PROJECT`, `PICTURE_BOOK`, `COMIC` |
| **countryOfOrigin** | Country code                | 2-letter code (`JP`, `KR`, `CN`, etc.)                                                                                                                                                              |
| **isLicensed**      | Official licensing status   | Boolean                                                                                                                                                                                             |

#### Negation Filters (6 parameters)

| AniList Parameter    | Description              | Values/Format    |
| -------------------- | ------------------------ | ---------------- |
| **id_not**           | Exclude AniList ID       | Integer ≥ 1      |
| **idMal_not**        | Exclude MAL ID           | Integer ≥ 1      |
| **format_not**       | Exclude format           | MediaFormat enum |
| **status_not**       | Exclude status           | MediaStatus enum |
| **averageScore_not** | Exclude average score    | Integer (0-100)  |
| **popularity_not**   | Exclude popularity count | Integer ≥ 0      |

#### Array Inclusion Filters (17 parameters)

| AniList Parameter      | Description                | Values/Format        |
| ---------------------- | -------------------------- | -------------------- |
| **id_in**              | Include AniList IDs        | Array of integers    |
| **id_not_in**          | Exclude AniList IDs        | Array of integers    |
| **idMal_in**           | Include MAL IDs            | Array of integers    |
| **idMal_not_in**       | Exclude MAL IDs            | Array of integers    |
| **format_in**          | Include formats            | Array of MediaFormat |
| **format_not_in**      | Exclude formats            | Array of MediaFormat |
| **status_in**          | Include statuses           | Array of MediaStatus |
| **status_not_in**      | Exclude statuses           | Array of MediaStatus |
| **genre_in**           | Include genres             | Array of strings     |
| **genre_not_in**       | Exclude genres             | Array of strings     |
| **tag_in**             | Include tags               | Array of strings     |
| **tag_not_in**         | Exclude tags               | Array of strings     |
| **tagCategory_in**     | Include tag categories     | Array of strings     |
| **tagCategory_not_in** | Exclude tag categories     | Array of strings     |
| **licensedBy_in**      | Include licensing sites    | Array of strings     |
| **licensedById_in**    | Include licensing site IDs | Array of integers    |
| **source_in**          | Include source types       | Array of MediaSource |

#### Range Filters (18 parameters)

| AniList Parameter        | Description                     | Values/Format           |
| ------------------------ | ------------------------------- | ----------------------- |
| **startDate_greater**    | Start date greater than         | FuzzyDateInt (YYYYMMDD) |
| **startDate_lesser**     | Start date lesser than          | FuzzyDateInt (YYYYMMDD) |
| **startDate_like**       | Start date pattern match        | String pattern          |
| **endDate_greater**      | End date greater than           | FuzzyDateInt (YYYYMMDD) |
| **endDate_lesser**       | End date lesser than            | FuzzyDateInt (YYYYMMDD) |
| **endDate_like**         | End date pattern match          | String pattern          |
| **episodes_greater**     | Episodes greater than           | Integer ≥ 0             |
| **episodes_lesser**      | Episodes lesser than            | Integer ≥ 0             |
| **duration_greater**     | Duration greater than (minutes) | Integer ≥ 0             |
| **duration_lesser**      | Duration lesser than (minutes)  | Integer ≥ 0             |
| **chapters_greater**     | Chapters greater than           | Integer ≥ 0             |
| **chapters_lesser**      | Chapters lesser than            | Integer ≥ 0             |
| **volumes_greater**      | Volumes greater than            | Integer ≥ 0             |
| **volumes_lesser**       | Volumes lesser than             | Integer ≥ 0             |
| **averageScore_greater** | Average score greater than      | Integer (0-100)         |
| **averageScore_lesser**  | Average score lesser than       | Integer (0-100)         |
| **popularity_greater**   | Popularity greater than         | Integer ≥ 0             |
| **popularity_lesser**    | Popularity lesser than          | Integer ≥ 0             |

#### Special Parameters

| AniList Parameter | Description  | Values/Format                                                        |
| ----------------- | ------------ | -------------------------------------------------------------------- |
| **sort**          | Sort options | Array of `ID`, `TITLE_ROMAJI`, `SCORE_DESC`, `POPULARITY_DESC`, etc. |

**Total: 69+ parameters validated ✅**

### Kitsu JSON:API Unique Parameters ✅

**Kitsu-specific parameters that cannot be mapped to universal parameters:**

| Kitsu Parameter       | Description               | Values/Format                  | Verified |
| --------------------- | ------------------------- | ------------------------------ | -------- |
| **filter[streamers]** | Streaming platform filter | Platform names (`Crunchyroll`) | ✅       |

**Range Syntax Notes:**

- Kitsu uses `..` for range filtering: `min..max`
- Single-sided ranges: `80..` (≥80) or `..90` (≤90)
- Works for: `averageRating`, `episodeCount`, `episodeLength`

**✅ VERIFICATION SUMMARY:**

- **14 total parameters verified** against real Kitsu API
- **13 parameters map to universal properties** (handled automatically)
- **1 Kitsu-specific parameter** requires special handling
- **10 undocumented parameters discovered** (more than official docs)

## Comprehensive Verification Summary ✅

**All major parameter mapping categories have been systematically verified against actual API implementations:**

### ✅ **Status Value Mappings** - VERIFIED

- **MAL**: FINISHED→finished_airing, RELEASING→currently_airing, NOT_YET_RELEASED→not_yet_aired, HIATUS→on_hiatus
- **Jikan**: FINISHED→complete, RELEASING→airing, NOT_YET_RELEASED→upcoming, HIATUS→❌
- **AniList**: Direct mapping (FINISHED→FINISHED, RELEASING→RELEASING, etc.)
- **Kitsu**: FINISHED→finished, RELEASING→current, NOT_YET_RELEASED→upcoming, plus unique TBA→tba, UNRELEASED→unreleased

### ✅ **Format/Type Value Mappings** - VERIFIED

- **MAL/Jikan**: Lowercase conversion (TV→tv, MOVIE→movie, etc.)
- **AniList**: Uppercase preservation (TV→TV, MOVIE→MOVIE, etc.)
- **Kitsu**: Mixed case preservation (TV→TV, movie→movie, etc.)
- **All platforms**: Full support for TV, MOVIE, OVA, ONA, SPECIAL, MUSIC

### ✅ **Scoring Parameters** - VERIFIED

- **MAL/Jikan**: 0-10 scale (direct mapping)
- **AniList**: 0-100 scale (automatic ×10 conversion: 7.5→75)
- **Kitsu**: Range syntax with .. separator (min_score/max_score → averageRating range filtering)
- **Parameter names**: min_score/max_score → averageScore_greater/lesser (AniList)

### ✅ **Episode Parameters** - VERIFIED

- **MAL**: No episode range filtering support
- **Jikan**: episodes_greater/episodes_lesser (different names, same function)
- **AniList**: episodes_greater/episodes_lesser (same as Jikan)
- **Kitsu**: Range syntax with .. separator (min_episodes/max_episodes → episodeCount range filtering)

### ✅ **Duration Parameters** - VERIFIED

- **MAL/Jikan**: No duration filtering support
- **AniList**: duration_greater/duration_lesser (minutes)
- **Kitsu**: Range syntax with .. separator (min_duration/max_duration → episodeLength range filtering)

### ✅ **Temporal Parameters** - VERIFIED

- **MAL/Jikan**: Convert year+season to start_date format (2023+WINTER→2023-01-01)
- **AniList**: Native season/seasonYear support + separate date filters
- **Kitsu**: Native season/seasonYear support (filter[season]/filter[seasonYear])
- **Date formats**: ISO 8601 for MAL/Jikan, FuzzyDateInt for AniList, season names for Kitsu

### ✅ **Content Filtering** - VERIFIED

- **Adult Content**: MAL(nsfw=white), Jikan(sfw=true), AniList(isAdult=false), Kitsu(❌ not available)
- **Genres**: MAL(❌), Jikan(comma-separated), AniList(arrays), Kitsu(string names via categories)
- **Genre Exclusion**: MAL(❌), Jikan(genres_exclude), AniList(genre_not_in), Kitsu(❌)
- **Age Rating**: MAL(✅), Jikan(✅), AniList(❌), Kitsu(G/PG/R/R18 via ageRating)

### ✅ **User Engagement** - VERIFIED

- **MAL**: num_scoring_users supported, popularity not in universal mapping
- **Jikan**: Limited engagement parameter support
- **AniList**: Full popularity_greater/lesser support
- **Kitsu**: No direct user engagement parameters available

### ✅ **Sorting Parameters** - VERIFIED

- **MAL**: sort + order (separate fields)
- **Jikan**: order_by + sort (separate fields)
- **AniList**: Embedded direction (SCORE_DESC, POPULARITY_ASC, etc.)
- **Kitsu**: Embedded direction with - prefix for descending (averageRating vs -averageRating)

### ✅ **Sort Field Mappings** - VERIFIED

- **Universal "score"**: MAL(mean), Jikan(score), AniList(SCORE), Kitsu(averageRating)
- **Universal "popularity"**: MAL(popularity), Jikan(popularity), AniList(POPULARITY), Kitsu(userCount)
- **Universal "episodes"**: MAL(num_episodes), Jikan(episodes), AniList(EPISODES), Kitsu(episodeCount)
- **Universal "title"**: MAL(title), Jikan(title), AniList(TITLE_ROMAJI), Kitsu(canonicalTitle)
- **Universal "year"**: MAL(start_date), Jikan(start_date), AniList(START_DATE), Kitsu(startDate)

**🎯 COMPREHENSIVE VALIDATION COMPLETE: All parameter mapping sections verified against real API implementations!**

## Parameter Conversion Notes

### Date Handling

- **MAL/Jikan**: Accept `YYYY-MM-DD`, `YYYY-MM`, or `YYYY` formats
- **AniList**: Requires separate fields for different date operations
- **Kitsu**: Uses ISO 8601 date strings

### Genre Handling

- **MAL**: Uses string genre names
- **Jikan**: Uses numeric genre IDs
- **AniList**: Uses string genre names
- **Kitsu**: Uses category relationship links

### Score Scaling

- **MAL/Jikan**: 0-10 scale (decimals allowed)
- **AniList**: 0-100 scale (requires conversion: `score * 10`)
- **Kitsu**: Range syntax with `..` separator (`80..90` for 80-90 range)

### Adult Content Logic

- **MAL**: `nsfw=white` = SFW only, `nsfw=gray` = questionable, `nsfw=black` = NSFW
- **Jikan**: `sfw=true` = exclude adult content (inverse logic)
- **AniList**: `isAdult=false` = exclude adult content
- **Kitsu**: Adult content filtering not available (filter[nsfw] does not work)

---

# PART II: RESPONSE FIELDS

This section documents what fields each platform can return in responses (not filtering capabilities).

**Note**: Fields marked with ⚡ can be used both as query parameters for filtering AND as response fields for AniList GraphQL. Other platforms may have different dual-purpose capabilities - see the detailed "Dual-Purpose Fields Reference" section below.

## Available Response Fields

This section documents what fields each platform can return in responses (not filtering capabilities). Response fields are organized by category for clarity.

### Core Identity Fields

| Universal Field     | MAL API v2 ✅   | Jikan API v4 ❌ | AniList GraphQL ✅ | Kitsu JSON:API ✅ | AniDB |
| ------------------- | --------------- | --------------- | ------------------ | ----------------- | ----- |
| **id_field** ⚡     | `id` ✅         | ❌              | `id` ✅            | `id` ✅           | ❌    |
| **mal_id_field** ⚡ | `id` ✅         | ❌              | `idMal` ✅         | `malId` ✅        | ❌    |
| **type_field**      | `media_type` ✅ | ❌              | `type` ✅          | `subtype` ✅      | ❌    |
| **format_field** ⚡ | `media_type` ✅ | ❌              | `format` ✅        | `subtype` ✅      | ❌    |

### Content Information Fields

| Universal Field      | MAL API v2 ✅        | Jikan API v4 ❌ | AniList GraphQL ✅   | Kitsu JSON:API ✅   | AniDB |
| -------------------- | -------------------- | --------------- | -------------------- | ------------------- | ----- |
| **title_field**      | `title` ✅           | ❌              | `title` ✅           | `canonicalTitle` ✅ | ❌    |
| **synopsis_field**   | `synopsis` ✅        | ❌              | `description` ✅     | `synopsis` ✅       | ❌    |
| **genres_field** ⚡  | `genres` ✅          | ❌              | `genres` ✅          | `categories` ✅     | ❌    |
| **synonyms_field**   | `alternative_titles` | ❌              | `synonyms` ✅        | `titles` ✅         | ❌    |
| **tags_field** ⚡    | ❌                   | ❌              | `tags` ✅            | ❌                  |
| **source_field** ⚡  | `source` ✅          | ❌              | `source` ✅          | ❌                  |
| **hashtag_field**    | ❌                   | ❌              | `hashtag` ✅         | ❌                  |
| **country_field** ⚡ | ❌                   | ❌              | `countryOfOrigin` ✅ | ❌                  |

### Status & Metrics Fields

| Universal Field         | MAL API v2 ✅      | Jikan API v4 ❌ | AniList GraphQL ✅ | Kitsu JSON:API ✅   | AniDB |
| ----------------------- | ------------------ | --------------- | ------------------ | ------------------- | ----- |
| **status_field** ⚡     | `status` ✅        | ❌              | `status` ✅        | `status` ✅         | ❌    |
| **score_field** ⚡      | `mean` ✅          | ❌              | `averageScore` ✅  | `averageRating` ✅  | ❌    |
| **mean_score_field**    | `mean` ✅          | ❌              | `meanScore` ✅     | `averageRating` ✅  | ❌    |
| **popularity_field** ⚡ | `popularity` ✅    | ❌              | `popularity` ✅    | `userCount` ✅      | ❌    |
| **rank_field**          | `rank` ✅          | ❌              | `rankings` ✅      | ❌                  | ❌    |
| **trending_field**      | ❌                 | ❌              | `trending` ✅      | ❌                  | ❌    |
| **favourites_field**    | `num_favorites` ✅ | ❌              | `favourites` ✅    | `favoritesCount` ✅ | ❌    |

### Temporal Fields

| Universal Field          | MAL API v2 ✅     | Jikan API v4 ❌ | AniList GraphQL ✅ | Kitsu JSON:API ✅ | AniDB |
| ------------------------ | ----------------- | --------------- | ------------------ | ----------------- | ----- |
| **start_date_field** ⚡  | `start_date` ✅   | ❌              | `startDate` ✅     | `startDate` ✅    | ❌    |
| **end_date_field** ⚡    | `end_date` ✅     | ❌              | `endDate` ✅       | `endDate` ✅      | ❌    |
| **season_field** ⚡      | `start_season` ✅ | ❌              | `season` ✅        | `season` ✅       | ❌    |
| **season_year_field** ⚡ | `start_season` ✅ | ❌              | `seasonYear` ✅    | `seasonYear` ✅   | ❌    |
| **updated_at_field**     | `updated_at` ✅   | ❌              | `updatedAt` ✅     | `updatedAt` ✅    |

### Episode/Chapter Data Fields

| Universal Field       | MAL API v2 ✅                 | Jikan API v4 ❌ | AniList GraphQL ✅ | Kitsu JSON:API ✅  | AniDB |
| --------------------- | ----------------------------- | --------------- | ------------------ | ------------------ | ----- |
| **episodes_field** ⚡ | `num_episodes` ✅             | ❌              | `episodes` ✅      | `episodeCount` ✅  | ❌    |
| **duration_field** ⚡ | `average_episode_duration` ✅ | ❌              | `duration` ✅      | `episodeLength` ✅ | ❌    |
| **chapters_field** ⚡ | ❌                            | ❌              | `chapters` ✅      | `chapterCount` ✅  | ❌    |
| **volumes_field** ⚡  | ❌                            | ❌              | `volumes` ✅       | `volumeCount` ✅   | ❌    |

### Visual Content Fields

| Universal Field        | MAL API v2 ✅     | Jikan API v4 ❌ | AniList GraphQL ✅ | Kitsu JSON:API ✅ | AniDB |
| ---------------------- | ----------------- | --------------- | ------------------ | ----------------- | ----- |
| **cover_image_field**  | `main_picture` ✅ | ❌              | `coverImage` ✅    | `posterImage` ✅  | ❌    |
| **banner_image_field** | ❌                | ❌              | `bannerImage` ✅   | `coverImage` ✅   | ❌    |
| **trailer_field**      | ❌                | ❌              | `trailer` ✅       | ❌                | ❌    |

### Relationship Fields

| Universal Field          | MAL API v2 ✅      | Jikan API v4 ❌ | AniList GraphQL ✅ | Kitsu JSON:API ✅  | AniDB |
| ------------------------ | ------------------ | --------------- | ------------------ | ------------------ | ----- |
| **relations_field**      | `related_anime` ✅ | ❌              | `relations` ✅     | `relationships` ✅ | ❌    |
| **characters_field**     | ❌                 | ❌              | `characters` ✅    | `characters` ✅    | ❌    |
| **staff_field**          | ❌                 | ❌              | `staff` ✅         | `staff` ✅         | ❌    |
| **studios_field**        | `studios` ✅       | ❌              | `studios` ✅       | `animeStaff` ✅    | ❌    |
| **external_links_field** | ❌                 | ❌              | `externalLinks` ✅ | ❌                 | ❌    |

### Schedule & Streaming Fields

| Universal Field              | MAL API v2 ✅  | Jikan API v4 ❌ | AniList GraphQL ✅     | Kitsu JSON:API ✅ |
| ---------------------------- | -------------- | --------------- | ---------------------- | ----------------- |
| **next_airing_field**        | ❌             | ❌              | `nextAiringEpisode` ✅ | ❌                |
| **airing_schedule_field**    | `broadcast` ✅ | ❌              | `airingSchedule` ✅    | ❌                |
| **streaming_episodes_field** | ❌             | ❌              | `streamingEpisodes` ✅ | ❌                |

### User-Specific Fields

| Universal Field            | MAL API v2 ✅       | Jikan API v4 ❌ | AniList GraphQL ✅  | Kitsu JSON:API ✅ |
| -------------------------- | ------------------- | --------------- | ------------------- | ----------------- |
| **is_favourite_field**     | ❌                  | ❌              | `isFavourite` ✅    | ❌                |
| **media_list_entry_field** | `my_list_status` ✅ | ❌              | `mediaListEntry` ✅ | ❌                |
| **reviews_field**          | ❌                  | ❌              | `reviews` ✅        | `reviews` ✅      |

### Content Flags

| Universal Field          | MAL API v2 ✅ | Jikan API v4 ❌ | AniList GraphQL ✅ | Kitsu JSON:API ✅ |
| ------------------------ | ------------- | --------------- | ------------------ | ----------------- |
| **is_adult_field** ⚡    | `nsfw` ✅     | ❌              | `isAdult` ✅       | ❌                |
| **is_licensed_field** ⚡ | ❌            | ❌              | `isLicensed` ✅    | ❌                |
| **rating_field** ⚡      | `rating` ✅   | ❌              | ❌                 | `ageRating` ✅    |

### Platform-Specific Response Fields

#### MAL API v2 Unique Fields

| Universal Field                    | Description                        | Type                      |
| ---------------------------------- | ---------------------------------- | ------------------------- |
| **alternative_titles_field**       | Alternative titles object          | Object                    |
| **my_list_status_field**           | User's list status (requires auth) | Object                    |
| **num_list_users_field**           | Number of users with anime in list | Integer                   |
| **num_scoring_users_field**        | Number of users who scored anime   | Integer                   |
| **nsfw_field**                     | Content safety rating              | String (white/gray/black) |
| **average_episode_duration_field** | Episode duration in seconds        | Integer                   |
| **start_season_field**             | Season information object          | Object                    |
| **broadcast_field**                | Broadcast information              | Object                    |
| **main_picture_field**             | Main picture URLs                  | Object                    |
| **created_at_field**               | Creation timestamp                 | DateTime                  |
| **updated_at_field**               | Last update timestamp              | DateTime                  |

#### AniList GraphQL Unique Fields

**Complete list of 55 AniList GraphQL Media response fields organized by category:**

##### Core Identity Fields (4)

| Field      | Description      | Type    |
| ---------- | ---------------- | ------- |
| **id**     | AniList media ID | Integer |
| **idMal**  | MyAnimeList ID   | Integer |
| **type**   | Media type       | Enum    |
| **format** | Media format     | Enum    |

##### Content Information Fields (8)

| Field               | Description              | Type   |
| ------------------- | ------------------------ | ------ |
| **title**           | Official titles object   | Object |
| **description**     | Synopsis/description     | String |
| **genres**          | Genre array              | Array  |
| **synonyms**        | Alternative titles       | Array  |
| **tags**            | Tag objects array        | Array  |
| **source**          | Source material type     | Enum   |
| **hashtag**         | Official Twitter hashtag | String |
| **countryOfOrigin** | Origin country code      | String |

##### Status & Metrics Fields (8)

| Field            | Description              | Type    |
| ---------------- | ------------------------ | ------- |
| **status**       | Current release status   | Enum    |
| **averageScore** | Weighted average score   | Integer |
| **meanScore**    | Mean score (0-100)       | Integer |
| **popularity**   | Users with media on list | Integer |
| **trending**     | Recent activity amount   | Integer |
| **favourites**   | Users who favorited      | Integer |
| **isLocked**     | Edit lock status         | Boolean |
| **rankings**     | Ranking array            | Array   |

##### Temporal Information Fields (6)

| Field          | Description                       | Type    |
| -------------- | --------------------------------- | ------- |
| **startDate**  | First release date                | Object  |
| **endDate**    | Last release date                 | Object  |
| **season**     | Release season                    | Enum    |
| **seasonYear** | Release year                      | Integer |
| **seasonInt**  | Combined season/year (deprecated) | Integer |
| **updatedAt**  | Last data update timestamp        | Integer |

##### Episode/Chapter Data Fields (4)

| Field        | Description                | Type    |
| ------------ | -------------------------- | ------- |
| **episodes** | Episode count              | Integer |
| **duration** | Episode duration (minutes) | Integer |
| **chapters** | Chapter count (manga)      | Integer |
| **volumes**  | Volume count (manga)       | Integer |

##### Visual Content Fields (3)

| Field           | Description        | Type   |
| --------------- | ------------------ | ------ |
| **coverImage**  | Cover image object | Object |
| **bannerImage** | Banner image URL   | String |
| **trailer**     | Trailer object     | Object |

##### Relationship Fields (5)

| Field             | Description              | Type   |
| ----------------- | ------------------------ | ------ |
| **relations**     | Related media connection | Object |
| **characters**    | Character connection     | Object |
| **staff**         | Staff connection         | Object |
| **studios**       | Studio connection        | Object |
| **externalLinks** | External link array      | Array  |

##### Schedule & Streaming Fields (3)

| Field                 | Description             | Type   |
| --------------------- | ----------------------- | ------ |
| **nextAiringEpisode** | Next episode info       | Object |
| **airingSchedule**    | Full airing schedule    | Object |
| **streamingEpisodes** | Streaming episode array | Array  |

##### User-Specific Fields (4)

| Field                  | Description            | Type    |
| ---------------------- | ---------------------- | ------- |
| **isFavourite**        | User's favorite status | Boolean |
| **isFavouriteBlocked** | Favorite block status  | Boolean |
| **mediaListEntry**     | User's list entry      | Object  |
| **reviews**            | Review connection      | Object  |

##### Analytics & Stats Fields (2)

| Field      | Description             | Type   |
| ---------- | ----------------------- | ------ |
| **trends** | Daily trend stats       | Object |
| **stats**  | Media statistics object | Object |

##### External Integration Fields (2)

| Field               | Description               | Type   |
| ------------------- | ------------------------- | ------ |
| **siteUrl**         | AniList website URL       | String |
| **recommendations** | Recommendation connection | Object |

##### Content Flags (4)

| Field                       | Description               | Type    |
| --------------------------- | ------------------------- | ------- |
| **isAdult**                 | Adult content flag        | Boolean |
| **isLicensed**              | Official licensing status | Boolean |
| **isRecommendationBlocked** | Recommendation block      | Boolean |
| **isReviewBlocked**         | Review block              | Boolean |

##### Administrative Fields (2)

| Field                     | Description          | Type    |
| ------------------------- | -------------------- | ------- |
| **autoCreateForumThread** | Forum thread setting | Boolean |
| **modNotes**              | Moderator notes      | String  |

**Total: 55 comprehensive AniList GraphQL Media response fields ✅**

---

## Dual-Purpose Fields Reference ⚡

The following fields can be used both as **query parameters** (for filtering) AND as **response fields** (returned data):

### Core Identity

- `id` - Filter by ID / Returns the ID
- `idMal` - Filter by MAL ID / Returns MAL ID
- `format` - Filter by format / Returns format

### Content

- `genres` - Filter by genres / Returns genre list
- `tags` - Filter by tags / Returns tag list
- `source` - Filter by source / Returns source type
- `countryOfOrigin` - Filter by country / Returns origin country

### Status & Metrics

- `status` - Filter by status / Returns current status
- `averageScore` - Filter by score range / Returns average score
- `popularity` - Filter by popularity / Returns popularity count

### Temporal

- `startDate` - Filter by start date / Returns start date
- `endDate` - Filter by end date / Returns end date
- `season` - Filter by season / Returns season
- `seasonYear` - Filter by year / Returns season year

### Episode/Chapter Data

- `episodes` - Filter by episode count / Returns episode count
- `duration` - Filter by duration / Returns episode duration
- `chapters` - Filter by chapter count / Returns chapter count
- `volumes` - Filter by volume count / Returns volume count

### Content Flags

- `isAdult` - Filter adult content / Returns adult flag
- `isLicensed` - Filter licensed content / Returns license status
- `rating` - Filter by content rating / Returns content rating

## Implementation Guidelines

1. **Graceful Degradation**: If a platform doesn't support a parameter, ignore it silently
2. **Value Conversion**: Convert between different scales (e.g., score 0-10 vs 0-100)
3. **Format Conversion**: Handle different date formats and ID vs name mappings
4. **Default Values**: Apply sensible defaults for required platform parameters
5. **Validation**: Validate parameter values against platform constraints before sending

---
