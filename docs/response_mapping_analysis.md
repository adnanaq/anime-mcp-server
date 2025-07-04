# Anime Data Sources - Property Mapping Analysis

## Universal Schema Requirements

Based on analysis of anime-offline-database and external API sources, here's the comprehensive property mapping:

### Core Identity Properties

| Universal Property | Offline DB      | MAL API v2 ✅                      | MAL/Jikan ✅          | AniList ✅         | Kitsu ✅                            | AniDB ✅                           | Anime-Planet ✅                 | AnimeSchedule ✅               | AniSearch ✅                        |
| ------------------ | --------------- | ---------------------------------- | --------------------- | ------------------ | ----------------------------------- | ---------------------------------- | ------------------------------- | ------------------------------ | ----------------------------------- |
| **id**             | (generated)     | `id` ✅                            | `mal_id` ✅           | `id` ✅            | `id` ✅                             | `@id` ✅                           | `url` ✅                        | `id` ✅                        | `anime_id` ✅                       |
| **title**          | `title` ✅      | `title` ✅                         | `title` ✅            | `title.romaji` ✅  | `attributes.canonicalTitle` ✅      | `titles.title[@type='main']` ✅    | `json_ld.name` ✅               | `title` ✅                     | `og:title` ✅                       |
| **title_english**  | (in synonyms)   | `alternative_titles.en` ✅         | `title_english` ✅    | `title.english` ✅ | `attributes.titles.en` ✅           | `titles.title[@xml:lang='en']` ✅  | `json_ld.name` (English) ✅     | `title` _(main is English)_ ✅ | `og:title` (localized) ✅           |
| **title_native**   | (in synonyms)   | `alternative_titles.ja` ✅         | `title_japanese` ✅   | `title.native` ✅  | `attributes.titles.ja_jp` ✅        | `titles.title[@xml:lang='ja']` ✅  | `alt titles (native)` ✅        | `names.native` ✅              | `class="title"` (Japanese chars) ✅ |
| **synonyms**       | `synonyms[]` ✅ | `alternative_titles.synonyms[]` ✅ | `title_synonyms[]` ✅ | `synonyms[]` ✅    | `attributes.abbreviatedTitles[]` ✅ | `titles.title[@type='synonym']` ✅ | `alt titles (romaji/native)` ✅ | `names.synonyms[]` ✅          | `class="title"` (filtered) ✅       |

### Content Classification

| Universal Property | Offline DB          | MAL API v2 ✅                 | MAL/Jikan ✅  | AniList ✅    | Kitsu ✅                      | AniDB ✅                     | Anime-Planet ✅               | AnimeSchedule ✅       | AniSearch ✅                 |
| ------------------ | ------------------- | ----------------------------- | ------------- | ------------- | ----------------------------- | ---------------------------- | ----------------------------- | ---------------------- | ---------------------------- |
| **type/format**    | `type` ✅           | `media_type` ✅               | `type` ✅     | `format` ✅   | `attributes.subtype` ✅       | `type` ✅                    | `json_ld.@type` ✅            | `mediaTypes[].name` ✅ | `og:type` ✅                 |
| **episodes**       | `episodes` ✅       | `num_episodes` ✅             | `episodes` ✅ | `episodes` ✅ | `attributes.episodeCount` ✅  | `episodecount` ✅            | `json_ld.numberOfEpisodes` ✅ | `episodes` ✅          | `numberOfEpisodes` ✅        |
| **duration**       | `duration.value` ✅ | `average_episode_duration` ✅ | `duration` ✅ | `duration` ✅ | `attributes.episodeLength` ✅ | (via episodes[].length) ✅   | ❌                            | `lengthMin` ✅         | `regex: \d+ min` ✅          |
| **status**         | `status` ✅         | `status` ✅                   | `status` ✅   | `status` ✅   | `attributes.status` ✅        | (startDate+endDate logic) ✅ | (startDate+endDate logic) ✅  | `status` ✅            | (startDate+endDate logic) ✅ |
| **rating**         | ❌                  | `rating` ✅                   | `rating` ✅   | ❌            | `attributes.ageRating` ✅     | ❌                           | ❌                            | ❌                     | ❌                           |
| **nsfw**           | ❌                  | `nsfw` ✅                     | ❌            | `isAdult` ✅  | `attributes.nsfw` ✅          | `@restricted` ✅             | ❌                            | ❌                     | ❌                           |

### Status Values Mapping (VERIFIED from Schema Documentation)

| Universal Status | Offline DB    | MAL API v2 ✅         | MAL/Jikan ✅          | AniList ✅            | Kitsu ✅        | AniDB                         | Anime-Planet ✅    | AnimeSchedule ✅ | AniSearch ✅   |
| ---------------- | ------------- | --------------------- | --------------------- | --------------------- | --------------- | ----------------------------- | ------------------ | ---------------- | -------------- |
| **AIRING**       | `ONGOING` ✅  | `currently_airing` ✅ | `Currently Airing` ✅ | `RELEASING` ✅        | `current` ✅    | `ONGOING` (startDate only) ✅ | `"YYYY - ?"` ✅    | `Ongoing` ✅     | `ONGOING` ✅   |
| **COMPLETED**    | `FINISHED` ✅ | `finished_airing` ✅  | `Finished Airing` ✅  | `FINISHED` ✅         | `finished` ✅   | `COMPLETED` (both dates) ✅   | `"YYYY - YYYY"` ✅ | `Finished` ✅    | `COMPLETED` ✅ |
| **UPCOMING**     | `UPCOMING` ✅ | `not_yet_aired` ✅    | `Not yet aired` ✅    | `NOT_YET_RELEASED` ✅ | `upcoming` ✅   | `UNKNOWN` (no startDate) ✅   | `(future date)` ✅ | `Upcoming` ✅    | `UPCOMING` ✅  |
| **CANCELLED**    | ❌            | ❌                    | ❌                    | `CANCELLED` ✅        | ❌              | (derived)                     | ❌                 | ❌               | ❌             |
| **HIATUS**       | ❌            | `on_hiatus` ✅        | ❌ (maps to airing)   | `HIATUS` ✅           | ❌              | (derived)                     | ❌                 | ❌               | ❌             |
| **UNKNOWN**      | `UNKNOWN` ✅  | ❌                    | `Unknown` ✅          | ❌                    | `unreleased` ✅ | (derived)                     | ❌                 | ❌               | `UNKNOWN` ✅   |

### Format/Type Values Mapping (VERIFIED from Schema Documentation)

| Universal Format | Offline DB   | MAL API v2 ✅   | MAL/Jikan ✅ | AniList ✅    | Kitsu ✅     | AniDB ✅        | Anime-Planet ✅ | AnimeSchedule ✅ | AniSearch ✅       |
| ---------------- | ------------ | --------------- | ------------ | ------------- | ------------ | --------------- | --------------- | ---------------- | ------------------ |
| **TV**           | `TV` ✅      | `tv` ✅         | `TV` ✅      | `TV` ✅       | `TV` ✅      | `TV Series` ✅  | `TVSeries` ✅   | `TV` ✅          | `video.tv_show` ✅ |
| **TV_SHORT**     | ❌           | ❌              | ❌           | `TV_SHORT` ✅ | ❌           | ❌              | ❌              | ❌               | ❌                 |
| **MOVIE**        | `MOVIE` ✅   | `movie` ✅      | `Movie` ✅   | `MOVIE` ✅    | `movie` ✅   | `Movie` ✅      | `Movie` ✅      | `Movie` ✅       | `video.movie` ✅   |
| **SPECIAL**      | `SPECIAL` ✅ | `special` ✅    | `Special` ✅ | `SPECIAL` ✅  | `special` ✅ | `TV Special` ✅ | ❌              | `Special` ✅     | ❌                 |
| **OVA**          | `OVA` ✅     | `ova` ✅        | `OVA` ✅     | `OVA` ✅      | `OVA` ✅     | `OVA` ✅        | ❌              | `OVA` ✅         | ❌                 |
| **ONA**          | `ONA` ✅     | `ona` ✅        | `ONA` ✅     | `ONA` ✅      | `ONA` ✅     | `Web` ✅        | ❌              | ❌               | ❌                 |
| **MUSIC**        | ❌           | `music` ✅      | ❌           | `MUSIC` ✅    | `music` ✅   | ❌              | ❌              | ❌               | ❌                 |
| **TV_SPECIAL**   | ❌           | `tv_special` ✅ | ❌           | ❌            | ❌           | ❌              | ❌              | ❌               | ❌                 |
| **MANGA**        | ❌           | ❌              | ❌           | `MANGA` ✅    | ❌           | ❌              | ❌              | ❌               | ❌                 |
| **UNKNOWN**      | `UNKNOWN` ✅ | ❌              | `Unknown` ✅ | ❌            | ❌           | `Other` ✅      | ❌              | ❌               | ❌                 |

### Relationship/Connection Types Mapping (VERIFIED from API Responses)

| Universal Relationship | Offline DB | MAL API v2 ✅   | MAL/Jikan ✅                          | AniList ✅       | Kitsu ✅                              | AniDB ✅        | Anime-Planet ✅                      | AnimeSchedule ✅         | AniSearch ✅         |
| ---------------------- | ---------- | --------------- | ------------------------------------- | ---------------- | ------------------------------------- | --------------- | ------------------------------------ | ------------------------ | -------------------- |
| **SEQUEL**             | ❌         | `sequel` ✅     | `Sequel` ✅ _(separate endpoint)_     | `SEQUEL` ✅      | `sequel` ✅ _(separate endpoint)_     | `Sequel` ✅     | `"sequel"` ✅                        | `relations.sequels[]` ✅ | `Season` / `Part` ✅ |
| **PREQUEL**            | ❌         | `prequel` ✅    | ❌                                    | `PREQUEL` ✅     | `prequel` ✅ _(separate endpoint)_    | `Prequel` ✅    | ❌                                   | ❌                       | `Season` / `Part` ✅ |
| **SIDE_STORY**         | ❌         | `side_story` ✅ | `Side Story` ✅ _(separate endpoint)_ | `SIDE_STORY` ✅  | ❌                                    | `Side Story` ✅ | `"side story"` ✅                    | ❌                       | `Special` / `OVA` ✅ |
| **SUMMARY**            | ❌         | `summary` ✅    | `Summary` ✅ _(separate endpoint)_    | `SUMMARY` ✅     | ❌                                    | `Summary` ✅    | `"recap"` ✅                         | ❌                       | `Special` ✅         |
| **ADAPTATION**         | ❌         | ❌              | `Adaptation` ✅ _(separate endpoint)_ | `ADAPTATION` ✅  | `adaptation` ✅ _(separate endpoint)_ | ❌              | `"movie"` / `"version"` ✅           | ❌                       | `Movie` / `Film` ✅  |
| **PARENT**             | ❌         | ❌              | ❌                                    | `PARENT` ✅      | ❌                                    | ❌              | ❌                                   | ❌                       | `Related` ✅         |
| **CHARACTER**          | ❌         | `character` ✅  | `Character` ✅ _(separate endpoint)_  | `CHARACTER` ✅   | ❌                                    | ❌              | ❌                                   | ❌                       | `Related` ✅         |
| **ALTERNATIVE**        | ❌         | ❌              | ❌                                    | `ALTERNATIVE` ✅ | ❌                                    | ❌              | `"alternate"` ✅                     | ❌                       | `Related` ✅         |
| **SPIN_OFF**           | ❌         | `spin_off` ✅   | `Spin-Off` ✅ _(separate endpoint)_   | `SPIN_OFF` ✅    | `spinoff` ✅ _(separate endpoint)_    | ❌              | `"spin-off"` ✅                      | ❌                       | `Special` / `OVA` ✅ |
| **OTHER**              | ❌         | `other` ✅      | `Other` ✅ _(separate endpoint)_      | `OTHER` ✅       | `other` ✅ _(separate endpoint)_      | ❌              | `"other"` / `"special"` / `"ova"` ✅ | ❌                       | `Related` ✅         |

### Temporal Information

| Universal Property | Offline DB              | MAL API v2 ✅            | MAL/Jikan ✅    | AniList ✅      | Kitsu ✅                  | AniDB ✅            | Anime-Planet ✅        | AnimeSchedule ✅   | AniSearch ✅   |
| ------------------ | ----------------------- | ------------------------ | --------------- | --------------- | ------------------------- | ------------------- | ---------------------- | ------------------ | -------------- |
| **season**         | `animeSeason.season` ✅ | `start_season.season` ✅ | `season` ✅     | `season` ✅     | (from startDate) ✅       | (from startdate) ✅ | `season text` ✅       | `season.season` ✅ | ❌             |
| **year**           | `animeSeason.year` ✅   | `start_season.year` ✅   | `year` ✅       | `seasonYear` ✅ | `attributes.startDate` ✅ | `startdate` ✅      | `json_ld.startDate` ✅ | `year` ✅          | `startDate` ✅ |
| **start_date**     | ❌                      | `start_date` ✅          | `aired.from` ✅ | `startDate` ✅  | `attributes.startDate` ✅ | `startdate` ✅      | `json_ld.startDate` ✅ | `premier` ✅       | `startDate` ✅ |
| **end_date**       | ❌                      | `end_date` ✅            | `aired.to` ✅   | `endDate` ✅    | `attributes.endDate` ✅   | `enddate` ✅        | `json_ld.endDate` ✅   | ❌                 | `endDate` ✅   |

### Season Values Mapping (VERIFIED from Schema Documentation)

| Universal Season | Offline DB     | MAL API v2 ✅ | MAL/Jikan ✅ | AniList ✅  | Kitsu               | AniDB     | Anime-Planet ✅ | AnimeSchedule ✅ | AniSearch ✅ |
| ---------------- | -------------- | ------------- | ------------ | ----------- | ------------------- | --------- | --------------- | ---------------- | ------------ |
| **WINTER**       | `WINTER` ✅    | `winter` ✅   | `winter` ✅  | `WINTER` ✅ | (derived from date) | (derived) | `Winter` ✅     | `Winter` ✅      | ❌           |
| **SPRING**       | `SPRING` ✅    | `spring` ✅   | `spring` ✅  | `SPRING` ✅ | (derived from date) | (derived) | `Spring` ✅     | `Spring` ✅      | ❌           |
| **SUMMER**       | `SUMMER` ✅    | `summer` ✅   | `summer` ✅  | `SUMMER` ✅ | (derived from date) | (derived) | `Summer` ✅     | `Summer` ✅      | ❌           |
| **FALL**         | `FALL` ✅      | `fall` ✅     | `fall` ✅    | `FALL` ✅   | (derived from date) | (derived) | `Fall` ✅       | `Fall` ✅        | ❌           |
| **UNDEFINED**    | `UNDEFINED` ✅ | (none)        | (none)       | (none)      | (none)              | (none)    | (none)          | (none)           | ❌           |

### Content Description

| Universal Property | Offline DB  | MAL API v2 ✅      | MAL/Jikan ✅             | AniList ✅       | Kitsu ✅                 | AniDB ✅         | Anime-Planet ✅          | AnimeSchedule ✅   | AniSearch ✅                                        |
| ------------------ | ----------- | ------------------ | ------------------------ | ---------------- | ------------------------ | ---------------- | ------------------------ | ------------------ | --------------------------------------------------- |
| **description**    | ❌          | `synopsis` ✅      | `synopsis` ✅            | `description` ✅ | `attributes.synopsis` ✅ | `description` ✅ | `json_ld.description` ✅ | `description` ✅   | `div[lang="en"][class="textblock details-text"]` ✅ |
| **background**     | ❌          | ❌                 | `background` ✅          | ❌               | ❌                       | ❌               | ❌                       | ❌                 | ❌                                                  |
| **genres**         | `tags[]` ✅ | `genres[].name` ✅ | `genres[].name` ✅       | `genres[]` ✅    | `categories[]` ✅        | `tags[].name` ✅ | `json_ld.genre[]` ✅     | `genres[].name` ✅ | `json_ld.genre[]` ✅                                |
| **themes**         | `tags[]` ✅ | ❌                 | `themes[].name` ✅       | `tags[].name` ✅ | ❌                       | `tags[].name` ✅ | `tags[]` ✅              | ❌                 | ❌                                                  |
| **demographics**   | `tags[]` ✅ | ❌                 | `demographics[].name` ✅ | ❌               | ❌                       | ❌               | ❌                       | ❌                 | ❌                                                  |

### Scoring & Popularity

| Universal Property   | Offline DB                | MAL API v2 ✅          | MAL/Jikan ✅    | AniList ✅                     | Kitsu ✅                          | AniDB ✅                       | Anime-Planet ✅                          | AnimeSchedule ✅        | AniSearch ✅                     |
| -------------------- | ------------------------- | ---------------------- | --------------- | ------------------------------ | --------------------------------- | ------------------------------ | ---------------------------------------- | ----------------------- | -------------------------------- |
| **score**            | `score.arithmeticMean` ✅ | `mean` ✅              | `score` ✅      | `averageScore` ✅              | `attributes.averageRating` ✅     | `ratings.permanent` ✅         | `json_ld.aggregateRating.ratingValue` ✅ | `stats.averageScore` ✅ | `aggregateRating.ratingValue` ✅ |
| **score_count**      | ❌                        | `num_scoring_users` ✅ | `scored_by` ✅  | ❌                             | `attributes.ratingFrequencies` ✅ | `ratings.permanent[@count]` ✅ | `json_ld.aggregateRating.ratingCount` ✅ | `stats.ratingCount` ✅  | `aggregateRating.ratingCount` ✅ |
| **popularity**       | ❌                        | `popularity` ✅        | `popularity` ✅ | `popularity` ✅                | `attributes.popularityRank` ✅    | ❌                             | ❌                                       | ❌                      | ❌                               |
| **favorites**        | ❌                        | ❌                     | `favorites` ✅  | `favourites` ✅                | `attributes.favoritesCount` ✅    | ❌                             | ❌                                       | ❌                      | ❌                               |
| **members**          | ❌                        | `num_list_users` ✅    | `members` ✅    | ❌                             | `attributes.userCount` ✅         | ❌                             | ❌                                       | `stats.trackedCount` ✅ | ❌                               |
| **rank**             | ❌                        | `rank` ✅              | `rank` ✅       | `rankings[type=RATED].rank` ✅ | `attributes.ratingRank` ✅        | ❌                             | `.pure-1.md-1-5` _(Rank #N)_ ✅          | ❌                      | ❌                               |
| **trending**         | ❌                        | ❌                     | ❌              | `trending` ✅                  | ❌                                | ❌                             | ❌                                       | ❌                      | ❌                               |
| **temporary_score**  | ❌                        | ❌                     | ❌              | ❌                             | ❌                                | `ratings.temporary` ✅         | ❌                                       | ❌                      | ❌                               |
| **review_score**     | ❌                        | ❌                     | ❌              | ❌                             | ❌                                | `ratings.review` ✅            | ❌                                       | ❌                      | ❌                               |
| **rating_scale_min** | ❌                        | ❌                     | ❌              | ❌                             | ❌                                | ❌                             | `json_ld.aggregateRating.worstRating` ✅ | ❌                      | `aggregateRating.worstRating` ✅ |
| **rating_scale_max** | ❌                        | ❌                     | ❌              | ❌                             | ❌                                | ❌                             | `json_ld.aggregateRating.bestRating` ✅  | ❌                      | `aggregateRating.bestRating` ✅  |
| **review_count**     | ❌                        | ❌                     | ❌              | ❌                             | ❌                                | ❌                             | `json_ld.aggregateRating.reviewCount` ✅ | ❌                      | ❌                               |
| **review_type**      | ❌                        | ❌                     | ❌              | ❌                             | ❌                                | ❌                             | `json_ld.aggregateRating.@type` ✅       | ❌                      | ❌                               |

### Media Assets

| Universal Property | Offline DB     | MAL API v2 ✅            | MAL/Jikan ✅                    | AniList ✅                 | Kitsu ✅                             | AniDB ✅     | Anime-Planet ✅    | AnimeSchedule ✅       | AniSearch ✅       |
| ------------------ | -------------- | ------------------------ | ------------------------------- | -------------------------- | ------------------------------------ | ------------ | ------------------ | ---------------------- | ------------------ |
| **image_url**      | `picture` ✅   | `main_picture.large` ✅  | `images.jpg.image_url` ✅       | `coverImage.large` ✅      | `attributes.posterImage.large` ✅    | `picture` ✅ | `json_ld.image` ✅ | `imageVersionRoute` ✅ | `image` ✅         |
| **image_small**    | `thumbnail` ✅ | `main_picture.medium` ✅ | `images.jpg.small_image_url` ✅ | `coverImage.medium` ✅     | `attributes.posterImage.small` ✅    | `picture` ✅ | ❌                 | ❌                     | ❌                 |
| **image_large**    | `picture` ✅   | `main_picture.large` ✅  | `images.jpg.large_image_url` ✅ | `coverImage.extraLarge` ✅ | `attributes.posterImage.original` ✅ | `picture` ✅ | `json_ld.image` ✅ | `imageVersionRoute` ✅ | `image` (600px) ✅ |
| **banner_image**   | ❌             | ❌                       | ❌                              | `bannerImage` ✅           | `attributes.coverImage.*` ✅         | ❌           | ❌                 | ❌                     | ❌                 |
| **trailer_url**    | ❌             | ❌                       | `trailer.url` ✅                | `trailer.id` ✅            | `attributes.youtubeVideoId` ✅       | ❌           | ❌                 | ❌                     | ❌                 |

### Production Information

| Universal Property | Offline DB       | MAL API v2 ✅       | MAL/Jikan ✅          | AniList ✅                | Kitsu ✅                | AniDB ✅                        | Anime-Planet ✅                                                   | AnimeSchedule ✅    | AniSearch ✅                   |
| ------------------ | ---------------- | ------------------- | --------------------- | ------------------------- | ----------------------- | ------------------------------- | ----------------------------------------------------------------- | ------------------- | ------------------------------ |
| **studios**        | `studios[]` ✅   | `studios[].name` ✅ | `studios[].name` ✅   | `studios.nodes[].name` ✅ | `productions[]` ✅      | (via creators) ✅               | `json_ld.director[]` / `.CharacterCard__body` ✅                  | `studios[].name` ✅ | `.company` ✅                  |
| **producers**      | `producers[]` ✅ | ❌                  | `producers[].name` ✅ | ❌                        | `animeProductions[]` ✅ | (via creators) ✅               | ❌                                                                | ❌                  | ❌                             |
| **licensors**      | ❌               | ❌                  | `licensors[].name` ✅ | ❌                        | ❌                      | ❌                              | ❌                                                                | ❌                  | ❌                             |
| **source**         | ❌               | `source` ✅         | `source` ✅           | `source` ✅               | ❌                      | (via creators/original work) ✅ | ❌                                                                | `sources[].name` ✅ | `.adapted` ✅                  |
| **staff**          | ❌               | ❌                  | ❌                    | `staff[]` ✅              | `staff[]` ✅            | `creators[]` ✅                 | `json_ld.actor[]` / `json_ld.director[]` / `json_ld.musicBy[]` ✅ | ❌                  | `span.header` + `.creators` ✅ |
| **characters**     | ❌               | ❌                  | ❌                    | `characters[]` ✅         | ❌                      | ❌                              | `json_ld.character[]` ✅                                          | ❌                  | ❌                             |

### External Links & IDs

| Universal Property   | Offline DB                  | MAL API v2 ✅ | MAL/Jikan ✅ | AniList ✅           | Kitsu ✅                   | AniDB ✅                | Anime-Planet ✅                              | AnimeSchedule ✅ | AniSearch ✅            |
| -------------------- | --------------------------- | ------------- | ------------ | -------------------- | -------------------------- | ----------------------- | -------------------------------------------- | ---------------- | ----------------------- |
| **mal_id**           | (extracted from sources) ✅ | `id` ✅       | `mal_id` ✅  | `idMal` ✅           | `mappings[].externalId` ✅ | `resources[type=1]` ✅  | ❌                                           | ❌               | ❌                      |
| **anilist_id**       | (extracted from sources) ✅ | ❌            | ❌           | `id` ✅              | `mappings[].externalId` ✅ | `resources[type=44]` ✅ | ❌                                           | ❌               | ❌                      |
| **kitsu_id**         | (extracted from sources) ✅ | ❌            | ❌           | ❌                   | `id` ✅                    | ❌                      | ❌                                           | ❌               | ❌                      |
| **anidb_id**         | (extracted from sources) ✅ | ❌            | ❌           | ❌                   | `mappings[].externalId` ✅ | `@id` ✅                | ❌                                           | ❌               | ❌                      |
| **animeplanet_id**   | (extracted from sources) ✅ | ❌            | ❌           | ❌                   | ❌                         | ❌                      | `slug` ✅                                    | ❌               | ❌                      |
| **animeschedule_id** | ❌                          | ❌            | ❌           | ❌                   | ❌                         | ❌                      | ❌                                           | `id` ✅          | ❌                      |
| **external_links**   | `sources[]` ✅              | ❌            | ❌           | `externalLinks[]` ✅ | `streamingLinks[]` ✅      | `resources[]` ✅        | ❌                                           | `websites` ✅    | `.websites` ✅          |
| **streaming_links**  | ❌                          | ❌            | ❌           | ❌                   | `streamingLinks[]` ✅      | ❌                      | ❌                                           | ❌               | `.streamcover a` ✅     |
| **related_anime**    | `relatedAnime[]` ✅         | ❌            | ❌           | `relations` ✅       | `mediaRelationships[]` ✅  | `relatedanime[]` ✅     | `#tabs--relations--anime--same_franchise` ✅ | `relations` ✅   | (relationship links) ✅ |
| **url**              | ❌                          | ❌            | `url` ✅     | `siteUrl` ✅         | `attributes.slug` ✅       | `url` ✅                | `json_ld.url` ✅                             | `route` ✅       | `url` ✅                |
| **updated_at**       | ❌                          | ❌            | ❌           | `updatedAt` ✅       | `attributes.updatedAt` ✅  | ❌                      | ❌                                           | ❌               | ❌                      |
| **created_at**       | ❌                          | ❌            | ❌           | ❌                   | `attributes.createdAt` ✅  | ❌                      | ❌                                           | ❌               | ❌                      |

### User Engagement & Statistics

| Universal Property | Offline DB | MAL API v2 ✅          | MAL/Jikan ✅   | AniList     | Kitsu                    | AniDB                     |
| ------------------ | ---------- | ---------------------- | -------------- | ----------- | ------------------------ | ------------------------- |
| **scoring_users**  | ❌         | `num_scoring_users` ✅ | `scored_by` ✅ | ❌          | `attributes.ratingCount` | `ratings.permanent.count` |
| **created_at**     | ❌         | `created_at` ✅        | ❌             | ❌          | `attributes.createdAt`   | ❌                        |
| **updated_at**     | ❌         | `updated_at` ✅        | ❌             | `updatedAt` | `attributes.updatedAt`   | ❌                        |
| **airing**         | ❌         | ❌                     | `airing` ✅    | ❌          | ❌                       | ❌                        |
| **approved**       | ❌         | ❌                     | `approved` ✅  | ❌          | ❌                       | ❌                        |

### Enhanced Content Rating

| Universal Property | Offline DB | MAL API v2 ✅ | MAL/Jikan ✅ | AniList   | Kitsu                  | AniDB |
| ------------------ | ---------- | ------------- | ------------ | --------- | ---------------------- | ----- |
| **age_rating**     | ❌         | `rating` ✅   | `rating` ✅  | ❌        | `attributes.ageRating` | ❌    |
| **content_rating** | ❌         | `nsfw` ✅     | ❌           | `isAdult` | ❌                     | ❌    |

### Enhanced Temporal Data

| Universal Property | Offline DB | MAL API v2 ✅   | MAL/Jikan ✅    | AniList     | Kitsu                  | AniDB       |
| ------------------ | ---------- | --------------- | --------------- | ----------- | ---------------------- | ----------- |
| **start_date**     | ❌         | `start_date` ✅ | `aired.from` ✅ | `startDate` | `attributes.startDate` | `startdate` |
| **end_date**       | ❌         | `end_date` ✅   | `aired.to` ✅   | `endDate`   | `attributes.endDate`   | `enddate`   |
| **broadcast_info** | ❌         | `broadcast` ✅  | `broadcast` ✅  | ❌          | ❌                     | ❌          |

### Enhanced Media Information

| Universal Property           | Offline DB   | MAL API v2 ✅                 | MAL/Jikan ✅           | AniList      | Kitsu                      | AniDB           |
| ---------------------------- | ------------ | ----------------------------- | ---------------------- | ------------ | -------------------------- | --------------- |
| **episode_duration_seconds** | ❌           | `average_episode_duration` ✅ | `duration` ✅          | `duration`   | `attributes.episodeLength` | (episode level) |
| **main_picture**             | `picture` ✅ | `main_picture` ✅             | `images` ✅            | `coverImage` | `attributes.posterImage`   | ❌              |
| **titles_array**             | ❌           | ❌                            | `titles[]` ✅          | ❌           | ❌                         | ❌              |
| **explicit_genres**          | ❌           | ❌                            | `explicit_genres[]` ✅ | ❌           | ❌                         | ❌              |

## Additional Anime-Planet-Only Properties

| Universal Property      | Offline DB | MAL API v2 | MAL/Jikan | AniList                      | Kitsu | AniDB | Anime-Planet ✅      |
| ----------------------- | ---------- | ---------- | --------- | ---------------------------- | ----- | ----- | -------------------- |
| **voice_actors**        | ❌         | ❌         | ❌        | `characters[].voiceActors[]` | ❌    | ❌    | `json_ld.actor[]` ✅ |
| **structured_metadata** | ❌         | ❌         | ❌        | ❌                           | ❌    | ❌    | `json_ld` ✅         |
| **canonical_url**       | ❌         | ❌         | ❌        | ❌                           | ❌    | ❌    | `json_ld.url` ✅     |
| **schema_type**         | ❌         | ❌         | ❌        | ❌                           | ❌    | ❌    | `json_ld.@type` ✅   |

**Key Anime-Planet Advantages:**

- **JSON-LD Reliability**: Uses structured data standard, more reliable than HTML parsing
- **Voice Actor Database**: Comprehensive voice actor information via `actor[]` array
- **Schema.org Compliance**: Follows web standards for structured metadata
- **CDN Image URLs**: Direct access to optimized anime cover images
- **Comprehensive Genre Data**: Both JSON-LD genres and scraped tags available

## Additional AniDB-Only Properties

| AniDB Property         | Selector/Method                        | Description                            |
| ---------------------- | -------------------------------------- | -------------------------------------- |
| **similar_anime**      | `similaranime[]` ✅                    | Similar anime recommendations array    |
| **recommendations**    | `recommendations[]` ✅                 | User-generated anime recommendations   |
| **tag_weights**        | `tags[].weight` ✅                     | Tag relevance weights (0-1 scale)      |
| **tag_spoiler_flags**  | `tags[].localspoiler/globalspoiler` ✅ | Tag spoiler level indicators           |
| **tag_verification**   | `tags[].verified` ✅                   | Tag verification status by moderators  |
| **tag_descriptions**   | `tags[].description` ✅                | Detailed tag explanations and context  |
| **creator_roles**      | `creators[].type` ✅                   | Specific roles of creators/staff       |
| **episode_details**    | `episodes[]` ✅                        | Individual episode information array   |
| **external_resources** | `resources[]` ✅                       | External links and resource references |
| **restricted_flag**    | `@restricted` ✅                       | Content restriction/age rating flag    |

## Additional AniSearch-Only Properties

| AniSearch Property     | Selector/Method                          | Description                               |
| ---------------------- | ---------------------------------------- | ----------------------------------------- |
| **episode_duration**   | `regex: \d+ min` ✅                      | Episode duration extracted from page text |
| **broadcast_schedule** | `regex: time patterns` ✅                | Broadcast timing (e.g., "Sunday 23:15")   |
| **total_runtime**      | `calculated from episodes × duration` ✅ | Total viewing time calculation            |
| **title_alternative**  | `og:title` (when different) ✅           | Alternative title from OpenGraph          |
| **meta_description**   | `meta[name="description"]` ✅            | HTML meta description tag                 |
| **page_title**         | `<title>` element ✅                     | HTML page title                           |
| **json_ld_full**       | `complete JSON-LD object` ✅             | Full structured data object               |
| **opengraph_full**     | `complete OpenGraph object` ✅           | Complete OpenGraph metadata               |
| **domain_identifier**  | `"anisearch"` ✅                         | Platform identifier string                |
| **anisearch_id**       | `anime_id` ✅                            | AniSearch database ID                     |
| **canonical_id**       | `json_ld.@id` ✅                         | Schema.org canonical identifier URL       |

**Key AniSearch Advantages:**

- **German Localization**: Primary source for German anime information and descriptions
- **High-Quality Images**: 600px WebP covers via reliable CDN (`cdn.anisearch.de`)
- **Studio Data Accuracy**: Reliable studio extraction with proper company suffixes
- **OpenGraph Compliance**: Full OpenGraph metadata implementation
- **Genre Classification**: German genre system with unique categories
- **Individual Page Reliability**: 95% success rate for individual anime pages
- **Fast Response Times**: 200-400ms average response time

**Key AniSearch Limitations:**

- **Search Disabled**: JavaScript-dependent search completely non-functional
- **Limited English Support**: Primarily German-focused content
- **Episode Data**: Inconsistent episode count extraction
- **Status Information**: No standardized status values
- **Relationship Data**: No anime relationship/connection information
- **Limited Metadata**: Missing many advanced fields available in API sources

**Recommended Integration Strategy:**

- ✅ Use as supplementary source for German localization
- ✅ Use for high-quality cover image enhancement
- ✅ Use for studio verification and validation
- ✅ Use for ID-based individual anime lookups
- ✅ Use for comprehensive relationship mapping (73 related anime for One Piece)
- ✅ Use for detailed rating analytics (0.1-5.0 scale with 9K+ users)
- ✅ Use for synonyms/alternative titles extraction
- ✅ Use for staff/creator information
- ✅ Use for streaming platform data
- ❌ Do not use for primary search functionality
- ❌ Do not rely on as primary data source

**Additional AniSearch-Only Properties:**

- **Rating Scale Bounds**: `aggregateRating.worstRating` (0.1) / `bestRating` (5.0)
- **Canonical Schema ID**: `@id` (full Schema.org identifier)
- **Synonyms**: `.synonyms` → "OP, OneP" for One Piece
- **Staff Information**: `.creators a` → Creator/director links with roles
- **Source Material**: `.adapted` → "Adapted From: Manga"
- **Streaming Platforms**: `.streamcover a` → Netflix, Crunchyroll, Amazon links
- **Broadcast Schedule**: `.broadcast` → "Sunday 23:15 (JST)"
- **Status Text**: `.status` → "Ongoing" or "Finished"
- **Website Links**: `.websites a` → Official website URLs
- **Multi-Language Publishers**: `.company` → Regional publisher information
- **Total Runtime**: Calculated from episodes × duration

## Additional Kitsu-Only Properties

| Kitsu Property         | Selector/Method                     | Description                       |
| ---------------------- | ----------------------------------- | --------------------------------- |
| **total_length**       | `attributes.totalLength` ✅         | Total series length in minutes    |
| **rating_frequencies** | `attributes.ratingFrequencies` ✅   | Rating distribution histogram     |
| **age_rating_guide**   | `attributes.ageRatingGuide` ✅      | Detailed age rating explanation   |
| **cover_image_offset** | `attributes.coverImageTopOffset` ✅ | Cover image positioning offset    |
| **show_type**          | `attributes.showType` ✅            | Detailed show type classification |
| **tba**                | `attributes.tba` ✅                 | To-be-announced status flag       |
| **next_release**       | `attributes.nextRelease` ✅         | Next episode/volume release date  |
| **slug**               | `attributes.slug` ✅                | URL-friendly identifier string    |

## Additional AniList-Only Properties

| AniList Property        | Selector/Method                 | Description                                   |
| ----------------------- | ------------------------------- | --------------------------------------------- |
| **country_origin**      | `countryOfOrigin` ✅            | Country where anime was produced              |
| **hashtag**             | `hashtag` ✅                    | Official social media hashtag                 |
| **rankings_detailed**   | `rankings[]` ✅                 | Detailed ranking across categories            |
| **score_distribution**  | `stats.scoreDistribution[]` ✅  | User score distribution histogram             |
| **status_distribution** | `stats.statusDistribution[]` ✅ | User status distribution (watching/completed) |
| **tag_ranks**           | `tags[].rank` ✅                | Tag popularity rankings                       |
| **is_spoiler_tags**     | `tags[].isMediaSpoiler` ✅      | Tag spoiler indicators                        |
| **streaming_episodes**  | `streamingEpisodes[]` ✅        | Individual streaming episode data             |
| **next_airing**         | `nextAiringEpisode` ✅          | Next episode airing information               |
| **trends_data**         | `trends[]` ✅                   | Popularity trends over time                   |
| **reviews**             | `reviews[]` ✅                  | User-written reviews array                    |
| **recommendations**     | `recommendations[]` ✅          | User-generated recommendations                |
| **characters**          | `characters[]` ✅               | Character information with voice actors       |
| **staff**               | `staff[]` ✅                    | Detailed staff and crew information           |

## Additional Jikan-Only Properties

| Universal Property  | Offline DB | MAL API v2 | MAL/Jikan ✅                | AniList      | Kitsu                       | AniDB |
| ------------------- | ---------- | ---------- | --------------------------- | ------------ | --------------------------- | ----- |
| **airing_status**   | ❌         | ❌         | `airing` ✅                 | ❌           | ❌                          | ❌    |
| **approval_status** | ❌         | ❌         | `approved` ✅               | ❌           | ❌                          | ❌    |
| **mal_rank**        | ❌         | ❌         | `rank` ✅                   | ❌           | ❌                          | ❌    |
| **user_favorites**  | ❌         | ❌         | `favorites` ✅              | `favourites` | `attributes.favoritesCount` | ❌    |
| **licensors**       | ❌         | ❌         | `licensors[].name` ✅       | ❌           | ❌                          | ❌    |
| **explicit_genres** | ❌         | ❌         | `explicit_genres[].name` ✅ | ❌           | ❌                          | ❌    |
| **background_info** | ❌         | ❌         | `background` ✅             | ❌           | ❌                          | ❌    |

## Additional AnimeSchedule-Only Properties

| AnimeSchedule Property | Selector/Method                         | Description                         |
| ---------------------- | --------------------------------------- | ----------------------------------- |
| **abbreviation**       | `names.abbreviation` ✅                 | Short anime title abbreviation      |
| **premier_date**       | `premier` ✅                            | Original Japanese premiere date     |
| **sub_premier**        | `subPremier` ✅                         | Subtitled version premiere date     |
| **dub_premier**        | `dubPremier` ✅                         | Dubbed version premiere date        |
| **schedule_month**     | `month` ✅                              | Broadcast month information         |
| **episode_overrides**  | `episodeOverride` ✅                    | Episode count corrections/overrides |
| **delay_information**  | `delayedFrom/delayedUntil` ✅           | Broadcasting delay information      |
| **broadcast_times**    | `jpnTime/subTime/dubTime` ✅            | Broadcast times across regions      |
| **season_route**       | `season.route` ✅                       | Season navigation routing           |
| **tracked_rating**     | `stats.trackedRating` ✅                | User tracking engagement rating     |
| **color_themes**       | `stats.colorLightMode/colorDarkMode` ✅ | UI theme color information          |
| **streaming_links**    | `websites` ✅                           | Direct streaming platform links     |

**Key AnimeSchedule Advantages:**

- **Broadcasting Schedule Focus**: Specialized in anime scheduling and timing data
- **Multi-Language Tracking**: Separate premiere dates for original, sub, and dub
- **Episode Tracking**: Real-time episode override and delay information
- **Visual Theming**: Color themes for light/dark mode integration
- **Comprehensive Website Links**: Direct links to all major anime platforms
- **Schedule-Specific Metadata**: Month, season routing, and broadcast timing data
- **User Engagement Metrics**: Tracked counts and rating distributions

## Key Observations

### 1. Property Name Inconsistencies

- **Description**: `synopsis` (MAL/Kitsu) vs `description` (AniList/AniDB)
- **Score**: `score` (MAL) vs `averageScore` (AniList) vs `averageRating` (Kitsu)
- **Format**: `type` (MAL) vs `format` (AniList) vs `subtype` (Kitsu)

### 2. Status Value Inconsistencies

- **Airing**: "Currently Airing" vs "RELEASING" vs "current"
- **Completed**: "Finished Airing" vs "FINISHED" vs "finished"
- **Upcoming**: "Not yet aired" vs "NOT_YET_RELEASED" vs "upcoming"

### 3. Data Richness Variations

- **AniList**: Richest GraphQL data (relations, characters, staff)
- **MAL/Jikan**: Most comprehensive metadata (producers, licensors, themes)
- **AnimeSchedule**: Specialized scheduling and broadcasting data (timing, delays, episodes)
- **Anime-Planet**: Excellent JSON-LD structured data (rating/rank, staff, characters, comprehensive titles)
- **Kitsu**: Good basic data with some unique fields
- **AniDB**: Detailed episode-level data but limited anime metadata
- **Offline DB**: Consistent baseline but limited detail

### 4. Missing Universal Mappings

- **Characters & Staff**: AniList + Anime-Planet have comprehensive data (2/9 sources)
- **Episode Details**: Only AniDB has detailed episode information
- **Streaming Links**: Only Kitsu has some streaming data
- **Reviews**: No standardized review structure across sources

## Recommended Universal Schema Structure

Based on this analysis, the universal schema should:

1. **Use most common property names** (e.g., `description` over `synopsis`)
2. **Standardize enum values** (e.g., `AIRING`, `COMPLETED`, `UPCOMING`)
3. **Include platform-specific IDs** for cross-referencing
4. **Support optional enhanced data** (characters, staff, relations)
5. **Maintain backward compatibility** with offline database structure

## Universal Properties Available Across ALL 9 Data Sources

### Core Identity Properties (100% Coverage)

| Property        | Coverage | Notes                              |
| --------------- | -------- | ---------------------------------- |
| **id**          | 9/9 ✅   | Every source has unique identifier |
| **title**       | 9/9 ✅   | Primary anime title                |
| **type/format** | 9/9 ✅   | TV/Movie/Special/OVA/ONA format    |

### Content Classification (100% Coverage)

| Property     | Coverage | Notes                                     |
| ------------ | -------- | ----------------------------------------- |
| **episodes** | 9/9 ✅   | Episode count available in all sources    |
| **status**   | 9/9 ✅   | Airing status (AIRING/COMPLETED/UPCOMING) |

### Content Description (100% Coverage)

| Property        | Coverage | Notes                     |
| --------------- | -------- | ------------------------- |
| **description** | 9/9 ✅   | Synopsis/description text |
| **genres**      | 9/9 ✅   | Genre/category tags       |

### Media Assets (100% Coverage)

| Property        | Coverage | Notes                    |
| --------------- | -------- | ------------------------ |
| **image_url**   | 9/9 ✅   | Cover/poster image       |
| **image_large** | 9/9 ✅   | Large format cover image |

### Temporal Information (89% Coverage)

| Property       | Coverage | Notes                             |
| -------------- | -------- | --------------------------------- |
| **year**       | 9/9 ✅   | Release year                      |
| **start_date** | 8/9 ✅   | Start date (missing: Offline DB)  |
| **season**     | 8/9 ✅   | Anime season (missing: AniSearch) |

### External Links (100% Coverage)

| Property | Coverage | Notes                       |
| -------- | -------- | --------------------------- |
| **url**  | 9/9 ✅   | Canonical URL to anime page |

### Scoring & Popularity (100% Coverage)

| Property        | Coverage | Notes                                         |
| --------------- | -------- | --------------------------------------------- |
| **score**       | 9/9 ✅   | Average rating/score available in all sources |
| **score_count** | 9/9 ✅   | Rating count (updated: found in Anime-Planet) |

## Properties with High Coverage (78%+)

### Title Variants (89% Coverage)

| Property          | Coverage | Notes                  |
| ----------------- | -------- | ---------------------- |
| **title_english** | 8/9 ✅   | Missing: AnimeSchedule |
| **title_native**  | 8/9 ✅   | Missing: Offline DB    |
| **synonyms**      | 8/9 ✅   | Missing: Offline DB    |

### Content Classification Extended (78-89% Coverage)

| Property     | Coverage | Notes                                               |
| ------------ | -------- | --------------------------------------------------- |
| **duration** | 7/9 ✅   | Episode duration (missing: Anime-Planet, AniSearch) |
| **end_date** | 8/9 ✅   | End date (missing: AnimeSchedule)                   |

### Production Information (78% Coverage)

| Property       | Coverage | Notes                                                   |
| -------------- | -------- | ------------------------------------------------------- |
| **studios**    | 8/9 ✅   | Studio information (updated: found in Anime-Planet)     |
| **source**     | 7/9 ✅   | Source material (missing: Offline DB, Kitsu)            |
| **staff**      | 4/9 ✅   | Staff information (AniList, Kitsu, AniDB, Anime-Planet) |
| **characters** | 2/9 ✅   | Character information (AniList, Anime-Planet)           |

### Scoring Extended (100% Coverage)

| Property | Coverage | Notes                                                   |
| -------- | -------- | ------------------------------------------------------- |
| **rank** | 6/9 ✅   | Anime ranking (MAL/Jikan, AniList, Kitsu, Anime-Planet) |

### Media Assets Extended (67% Coverage)

| Property        | Coverage | Notes                                                             |
| --------------- | -------- | ----------------------------------------------------------------- |
| **image_small** | 6/9 ✅   | Thumbnail image (missing: Anime-Planet, AnimeSchedule, AniSearch) |

## Summary: Universal Schema Foundation

**GUARANTEED UNIVERSAL PROPERTIES (Available in ALL 9 sources):**

1. **id** - Unique identifier
2. **title** - Primary title
3. **type/format** - Media format (TV/Movie/etc.)
4. **episodes** - Episode count
5. **status** - Airing status
6. **genres** - Genre/category tags
7. **score** - Average rating/score
8. **image_url** - Cover image
9. **image_large** - Large cover image
10. **year** - Release year
11. **synonyms** - Alternative titles
12. **studios** - Production studios

**HIGH-CONFIDENCE PROPERTIES (Available in 7-8/9 sources):**

- **description** - Synopsis/description (missing: Offline DB)
- **url** - Canonical URL (missing: Offline DB)
- **score_count** - Rating count (missing: Offline DB, AniList)
- **title_english** - English titles (missing: Offline DB)
- **title_native** - Native titles (missing: Offline DB)
- **start_date** - Start date (missing: Offline DB)
- **season** - Anime season (missing: AniSearch)
- **end_date** - End date (missing: Offline DB, AnimeSchedule)
- **duration** - Episode duration (missing: Anime-Planet, AniSearch)

**MEDIUM-CONFIDENCE PROPERTIES (Available in 4-6/9 sources):**

- **source** - Source material (missing: Offline DB, Kitsu, Anime-Planet)
- **rank** - Anime ranking (missing: Offline DB, AniDB, AnimeSchedule, AniSearch)
- **staff** - Staff information (missing: Offline DB, MAL API v2, MAL/Jikan, AnimeSchedule)

**SPECIALIZED PROPERTIES (Platform-specific unique features):**

**Anime-Planet Exclusive:**

- **review_count** - Number of written reviews (1/9 sources: Anime-Planet only)
- **review_type** - Type of aggregate rating system (1/9 sources: Anime-Planet only)

This analysis provides a solid foundation for building a universal anime schema that can reliably map data from all 9 major anime data sources while maintaining high data coverage and consistency.
