# Platform Configuration Guide

This guide explains how to configure the separated MAL and Jikan platforms in the anime MCP server.

## Overview

The server now supports two separate anime platforms:

- **MAL API v2**: Official MyAnimeList API requiring OAuth2 authentication
- **Jikan API v4**: Unofficial MAL scraper requiring no authentication

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# MAL API v2 Configuration (Optional - requires MAL developer account)
MAL_CLIENT_ID=your_mal_client_id_here
MAL_CLIENT_SECRET=your_mal_client_secret_here

# No configuration needed for Jikan - it works out of the box
```

### Platform Priority

The ServiceManager uses this priority order:

1. **Jikan** (higher priority) - No authentication required
2. **MAL** (lower priority) - Requires authentication
3. Other platforms...

## MAL API v2 Setup

### 1. Get MAL API Credentials

1. Go to [MAL API Configuration](https://myanimelist.net/apiconfig)
2. Create a new application
3. Note your `Client ID` and `Client Secret`

### 2. Configure Environment

```env
MAL_CLIENT_ID=your_client_id_from_mal
MAL_CLIENT_SECRET=your_client_secret_from_mal
```

### 3. MAL Features

- **Search**: Limited to `q`, `limit`, `offset`, `fields` parameters
- **Field Selection**: Request specific fields to reduce response size
- **Authentication**: OAuth2 flow for user-specific data
- **Rate Limit**: ~2 requests/second
- **User Data**: Access to user anime lists, ratings, etc.

```python
# Example MAL API usage
results = await mal_service.search_anime(
    query="naruto",
    limit=10,
    fields="id,title,mean,genres"
)
```

## Jikan API v4 Setup

### 1. No Configuration Required

Jikan works immediately without any setup - no API keys needed!

### 2. Jikan Features

- **Advanced Search**: 17+ parameters including genres, status, scores, dates
- **No Authentication**: Public API, no signup required
- **Extensive Filtering**: Complex queries with multiple criteria
- **Rate Limit**: ~1 request/second
- **Seasonal Data**: Current/past/future seasonal anime
- **Statistics**: Anime statistics and rankings

```python
# Example Jikan API usage
results = await jikan_service.search_anime(
    query="action",
    limit=10,
    genres=[1, 2],  # Action, Adventure
    status="complete",
    min_score=8.0,
    anime_type="TV"
)
```

## Platform Comparison

| Feature | MAL API v2 | Jikan API v4 |
|---------|------------|--------------|
| **Authentication** | OAuth2 required | None |
| **Rate Limit** | ~2 req/sec | ~1 req/sec |
| **Search Parameters** | 4 basic | 17+ advanced |
| **Field Selection** | ✅ Yes | ❌ No |
| **Genre Filtering** | ❌ No | ✅ Yes |
| **Status Filtering** | ❌ No | ✅ Yes |
| **Score Filtering** | ❌ No | ✅ Yes |
| **Date Filtering** | ❌ No | ✅ Yes |
| **Seasonal Data** | ✅ Yes | ✅ Yes |
| **User Data** | ✅ Yes | ❌ No |
| **Official Data** | ✅ Yes | ✅ Yes (scraped) |

## Error Handling & Observability

Both platforms include comprehensive error handling:

### Correlation Tracking
- Auto-generated correlation IDs
- Request tracing through entire stack
- Parent/child correlation chains

### Error Context
- User-friendly error messages
- Technical debug information
- Complete execution traces

### Health Checks
- No unnecessary API calls
- Configuration validation
- Circuit breaker status

```python
# Health check example
jikan_health = await jikan_service.health_check()
# Returns: {"service": "jikan", "status": "healthy", "auth_required": False}

mal_health = await mal_service.health_check()
# Returns: {"service": "mal", "status": "healthy", "auth_configured": True}
```

## Usage Examples

### Universal Search (Auto-Platform Selection)

```python
from src.models.universal_anime import UniversalSearchParams
from src.integrations.service_manager import ServiceManager

# ServiceManager automatically selects best platform
params = UniversalSearchParams(query="demon slayer", limit=5)
results = await service_manager.search_anime_universal(params)
# Will use Jikan (higher priority, no auth needed)
```

### Direct Platform Usage

```python
# Use Jikan for advanced filtering
jikan_results = await jikan_service.search_anime(
    query="",
    genres=[1, 4],  # Action, Comedy
    status="complete",
    min_score=8.5,
    anime_type="TV"
)

# Use MAL for official data with field selection
mal_results = await mal_service.search_anime(
    query="one piece",
    fields="id,title,mean,rank,popularity"
)
```

## Troubleshooting

### MAL Issues

**Problem**: MAL service shows "unhealthy" status
- **Solution**: Check `MAL_CLIENT_ID` is set in environment
- **Check**: Verify credentials are valid at MAL API console

**Problem**: MAL authentication errors
- **Solution**: Implement OAuth2 flow for user tokens
- **Check**: Ensure `client_secret` is provided for token refresh

### Jikan Issues

**Problem**: Jikan rate limiting
- **Solution**: Reduce request frequency (max 1 req/sec)
- **Check**: Monitor logs for "429 rate limit" errors

**Problem**: No results from Jikan
- **Solution**: Check parameter formatting (genres as integers, dates as YYYY-MM-DD)
- **Check**: Verify search parameters match Jikan API specification

### General Issues

**Problem**: ServiceManager not finding clients
- **Solution**: Check client initialization in ServiceManager logs
- **Check**: Verify both platforms are properly imported

**Problem**: Correlation tracking not working
- **Solution**: Ensure correlation middleware is enabled in FastAPI app
- **Check**: Look for correlation IDs in log messages

## Migration from Old Hybrid Client

If you were using the old hybrid MAL client:

1. **Update code**: Replace `MALClient` usage with either `MALClient` (OAuth2) or `JikanClient` (no auth)
2. **Check parameters**: MAL API v2 has limited parameters vs Jikan's 17+ parameters
3. **Authentication**: Add MAL credentials for official API, or use Jikan for no-auth access
4. **Error handling**: Both clients now have proper error contexts and correlation tracking

The separation ensures cleaner, more maintainable code with platform-specific optimizations.