# Comprehensive API Rate Limits for Anime Sources

This document contains thoroughly researched rate limiting information for all 9 anime API sources supported by the MCP server, based on official documentation and community best practices.

## Official API Sources

### 1. MyAnimeList (MAL) API v2
- **Documentation**: https://myanimelist.net/apiconfig/references/api/v2
- **Authentication**: OAuth2 required
- **Rate Limits**: 
  - **2 requests/second**
  - **60 requests/minute**
  - No daily limit specified
- **Implementation**: Token bucket with burst support
- **Backoff Strategy**: Standard exponential backoff
- **Note**: Official limits from comprehensive implementation plan, OAuth2 authentication required for all requests

### 2. AniList GraphQL API
- **Documentation**: https://docs.anilist.co/guide/rate-limiting
- **Authentication**: Optional OAuth2 (higher limits with auth)
- **Rate Limits**:
  - **Normal**: 90 requests/minute (1.5 req/sec)
  - **Degraded State**: 30 requests/minute (0.5 req/sec)
  - **Burst Limiter**: Prevents rapid consecutive requests
- **Headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `Retry-After`, `X-RateLimit-Reset`
- **Timeout**: 1-minute timeout for exceeded limits
- **Implementation**: Adaptive rate limiting recommended
- **Status Code**: 429 Too Many Requests with GraphQL error message

### 3. Kitsu JSON:API
- **Documentation**: https://kitsu.docs.apiary.io/
- **Authentication**: OAuth2 (optional for public endpoints)
- **Rate Limits**:
  - **No explicit rate limits documented**
  - **Conservative estimate**: 10 requests/second (based on comprehensive plan)
  - **Pagination limit**: 20 items max per request (500 for library entries)
- **Implementation**: Token bucket with generous limits
- **Note**: Uses JSON:API specification

### 4. AniDB HTTP API
- **Documentation**: https://wiki.anidb.net/HTTP_API_Definition
- **Authentication**: Client registration required
- **Rate Limits**:
  - **Official**: "Not more than one page every two seconds" (0.5 req/sec)
  - **Daily Limit**: ~100-200 requests per IP per 24 hours
  - **Recommended**: 2000ms between requests (short delay)
  - **Extended period**: 4000ms between requests (long delay)
- **Ban Duration**: 15 minutes to 24 hours
- **Implementation**: Very conservative token bucket
- **Critical**: Strictest rate limiting among all sources

### 5. Anime News Network API
- **Documentation**: https://www.animenewsnetwork.com/encyclopedia/api.php
- **Authentication**: No authentication required
- **Rate Limits**:
  - **1 request/second per IP address**
  - **Behavior**: Requests are delayed, not rejected
  - **No 429 errors**: Throttling through request delays
- **Implementation**: Token bucket with delay-based throttling
- **Note**: More forgiving than most APIs (delays vs rejections)

### 6. AnimeSchedule.net API v3
- **Documentation**: https://animeschedule.net/api/v3/documentation
- **Authentication**: Bearer token for private endpoints
- **Rate Limits**:
  - **Public endpoints**: "Far less forgiving rate limit" (not specified)
  - **Private endpoints**: More generous limits (not specified)
  - **Conservative estimate**: 5 requests/second for private, 1 req/sec for public
- **Implementation**: Token bucket with endpoint-based limits
- **Note**: V3+ only available to developers

## Unofficial API Sources

### 7. Jikan (MAL Unofficial) API v4
- **Documentation**: https://docs.api.jikan.moe/
- **Authentication**: No authentication required
- **Rate Limits**:
  - **v4**: 60 requests/minute, 3 requests/second
  - **v3**: 30 requests/minute, 2 requests/second (legacy)
  - **No monthly limit**
- **Implementation**: Token bucket with burst support
- **Note**: Open-source, handles 3M+ requests weekly

## Web Scraping Sources (Requires Aggressive Rate Limiting)

### 8. Anime-Planet
- **URL**: https://anime-planet.com
- **Method**: Web scraping with CloudScraper
- **Rate Limits**:
  - **Recommended**: 0.5 requests/second (1 request every 2 seconds)
  - **Burst**: Maximum 2 requests
  - **Max backoff**: 10 minutes
- **Implementation**: Adaptive rate limiting with aggressive backoff
- **Risk**: High blocking probability, requires conservative approach

### 9. AniSearch
- **URL**: https://anisearch.com
- **Method**: Web scraping (German site)
- **Rate Limits**:
  - **Recommended**: 0.33 requests/second (1 request every 3 seconds)
  - **Burst**: Maximum 1 request
  - **Max backoff**: 15 minutes
- **Implementation**: Most aggressive rate limiting
- **Risk**: Very strict anti-bot measures

### 10. AnimeCountdown
- **URL**: https://animecountdown.com
- **Method**: Web scraping
- **Rate Limits**:
  - **Recommended**: 0.5 requests/second (1 request every 2 seconds)
  - **Burst**: Maximum 2 requests
  - **Max backoff**: 10 minutes
- **Implementation**: Adaptive rate limiting
- **Risk**: Moderate blocking probability

## Implementation Strategy

### Rate Limiting Approaches by Source Type

1. **Official APIs with Clear Limits** (MAL, AniList, AniDB, AnimeNewsNetwork):
   - Use documented limits with 80% safety margin
   - Implement proper retry logic with exponential backoff
   - Monitor response headers for rate limit information

2. **APIs with Unclear Limits** (Kitsu, AnimeSchedule):
   - Start with conservative estimates
   - Implement adaptive rate limiting
   - Monitor for 429 responses and adjust accordingly

3. **Unofficial APIs** (Jikan):
   - Use documented limits but be prepared for changes
   - Implement graceful degradation if limits change

4. **Web Scraping Sources** (Anime-Planet, AniSearch, AnimeCountdown):
   - Use very conservative rate limits
   - Implement randomized delays
   - Use adaptive backoff on any blocking indicators
   - Rotate user agents and headers

### Priority Levels

- **High Priority**: User-facing search requests
- **Medium Priority**: Background data enrichment
- **Low Priority**: Bulk operations and maintenance

### Error Handling

- **429 Too Many Requests**: Exponential backoff with jitter
- **503 Service Unavailable**: Circuit breaker pattern
- **Connection errors**: Retry with increasing delays
- **Parsing errors**: Log and skip, don't retry

### Monitoring

- Track success rates by source
- Monitor average response times
- Alert on high error rates
- Log rate limit exhaustion events

## Configuration Summary

```python
RATE_LIMITS = {
    'mal': {'rps': 2.0, 'rpm': 60, 'strategy': 'token_bucket'},
    'anilist': {'rps': 1.5, 'rpm': 90, 'strategy': 'adaptive'},
    'kitsu': {'rps': 10.0, 'strategy': 'token_bucket'},
    'anidb': {'rps': 0.5, 'daily': 150, 'strategy': 'conservative'},
    'animenewsnetwork': {'rps': 1.0, 'strategy': 'delay_based'},
    'animeschedule': {'rps': 5.0, 'strategy': 'token_bucket'},
    'jikan': {'rps': 3.0, 'rpm': 60, 'strategy': 'token_bucket'},
    'animeplanet': {'rps': 0.5, 'strategy': 'adaptive_scraping'},
    'anisearch': {'rps': 0.33, 'strategy': 'aggressive_scraping'},
    'animecountdown': {'rps': 0.5, 'strategy': 'adaptive_scraping'}
}
```

## Best Practices

1. **Always implement local caching** to reduce API calls
2. **Use circuit breakers** for unreliable sources
3. **Implement request deduplication** for concurrent requests
4. **Monitor and log rate limit metrics** for optimization
5. **Have fallback strategies** when sources are unavailable
6. **Respect robots.txt** for scraping sources
7. **Use proper User-Agent headers** to identify your application

## References

- [MyAnimeList API Documentation](https://myanimelist.net/apiconfig/references/api/v2)
- [AniList Rate Limiting Guide](https://docs.anilist.co/guide/rate-limiting)
- [AniDB HTTP API Definition](https://wiki.anidb.net/HTTP_API_Definition)
- [Shoko Anime AniDB Ban Guide](https://docs.shokoanime.com/shoko-server/understanding-anidb-ban)
- [Jikan API Documentation](https://docs.api.jikan.moe/)
- [Anime News Network API](https://www.animenewsnetwork.com/encyclopedia/api.php)
- [Jellyfin AniDB Plugin Issues](https://github.com/jellyfin/jellyfin-plugin-anidb/issues/43)
- [FileBot AniDB Discussion](https://www.filebot.net/forums/viewtopic.php?t=11925)