# API Parameters Reference - Anime MCP Server

## 01. Health & System Endpoints

### GET `/health`

**Parameters**: None
**Response**:

```json
{
  "status": "healthy",
  "qdrant": "connected",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### GET `/stats`

**Parameters**: None
**Response**:

```json
{
  "total_documents": 38894,
  "vector_size": 384,
  "status": "green",
  "collection_name": "anime_database"
}
```

### GET `/`

**Parameters**: None
**Response**: API overview and documentation

---

## 02. Basic Search Endpoints

### GET `/api/search/`

**Query Parameters**:

- `q` (string, required): Search query
- `limit` (integer, optional): Number of results (default: 10, max: 50)

**Example**: `?q=dragon ball&limit=5`

**Response**:

```json
{
  "query": "dragon ball",
  "results": [...],
  "total_results": 5,
  "processing_time_ms": 45.2
}
```

### POST `/api/search/semantic`

**Headers**:

- `Content-Type: application/json`

**Body Parameters**:

```json
{
  "query": "string (required) - Search query",
  "limit": "integer (optional) - Number of results (1-50, default: 10)"
}
```

**Example**:

```json
{
  "query": "mecha robots fighting in space",
  "limit": 10
}
```

### GET `/api/search/similar/{anime_id}`

**Path Parameters**:

- `anime_id` (string, required): Anime ID to find similar content

**Query Parameters**:

- `limit` (integer, optional): Number of results (default: 10, max: 50)

**Example**: `/api/search/similar/cac1eeaeddf7?limit=5`

---

## 03. Image Search Endpoints

### POST `/api/search/by-image`

**Headers**:

- `Content-Type: multipart/form-data`

**Form Data Parameters**:

- `image` (file, required): Image file (JPEG, PNG, WebP)
- `limit` (integer, optional): Number of results (1-50, default: 10)

**Example Form Data**:

```
image: [file upload]
limit: 5
```

### POST `/api/search/by-image-base64`

**Headers**:

- `Content-Type: application/x-www-form-urlencoded`

**Form Parameters**:

- `image_data` (string, required): Base64 encoded image data
- `limit` (integer, optional): Number of results (1-50, default: 10)

**Example**:

```
image_data=iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB...
limit=5
```

### GET `/api/search/visually-similar/{anime_id}`

**Path Parameters**:

- `anime_id` (string, required): Anime ID for visual similarity

**Query Parameters**:

- `limit` (integer, optional): Number of results (default: 10, max: 50)

### POST `/api/search/multimodal`

**Headers**:

- `Content-Type: multipart/form-data`

**Form Data Parameters**:

- `query` (string, required): Text search query
- `image` (file, required): Image file for visual similarity
- `limit` (integer, optional): Number of results (1-50, default: 10)
- `text_weight` (float, optional): Text vs image weight (0.0-1.0, default: 0.7)

**Example Form Data**:

```
query: mecha anime
image: [file upload]
limit: 10
text_weight: 0.7
```

---

## 04. Workflow Endpoints

### POST `/api/workflow/conversation`

**Headers**:

- `Content-Type: application/json`

**Body Parameters**:

```json
{
  "message": "string (required) - User message/query",
  "session_id": "string (optional) - Conversation session ID",
  "max_results": "integer (optional) - Max results per search (1-50, default: 10)"
}
```

**Example**:

```json
{
  "message": "Find me some good action anime",
  "session_id": "test-session-1"
}
```

### POST `/api/workflow/smart-conversation`

**Headers**:

- `Content-Type: application/json`

**Body Parameters**:

```json
{
  "message": "string (required) - Natural language query with AI processing",
  "session_id": "string (optional) - Conversation session ID",
  "limit": "integer (optional) - Override extracted limit (1-50)",
  "enable_smart_orchestration": "boolean (optional) - Enable complex query processing (default: true)",
  "max_discovery_depth": "integer (optional) - Max orchestration steps (1-5, default: 3)"
}
```

**Example**:

```json
{
  "message": "find me 5 mecha anime from 2020s but not too violent",
  "enable_smart_orchestration": true,
  "max_discovery_depth": 3
}
```

**AI Extraction Capabilities**:

- **Limits**: "find 5", "top 3", "show me 7" → limit: 5, 3, 7
- **Years**: "from 2020s", "90s anime" → year_range: [2020, 2029], [1990, 1999]
- **Genres**: "mecha", "action", "romance" → genres: ["mecha", "action", "romance"]
- **Exclusions**: "but not horror", "except isekai" → exclusions: ["horror", "isekai"]
- **Studios**: "Studio Ghibli", "Mappa" → studios: ["Studio Ghibli", "Mappa"]
- **Types**: "movies", "TV series" → anime_types: ["Movie", "TV"]

### POST `/api/workflow/multimodal`

**Headers**:

- `Content-Type: application/json`

**Body Parameters**:

```json
{
  "message": "string (required) - Text query with image context",
  "image_data": "string (required) - Base64 encoded image",
  "session_id": "string (optional) - Conversation session ID",
  "text_weight": "float (optional) - Text vs image weight (0.0-1.0, default: 0.7)",
  "limit": "integer (optional) - Number of results (1-50, default: 10)"
}
```

**Example**:

```json
{
  "message": "find me 5 anime similar to this image style but not too dark",
  "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB...",
  "text_weight": 0.6
}
```

### GET `/api/workflow/conversation/{session_id}`

**Path Parameters**:

- `session_id` (string, required): Conversation session ID

**Query Parameters**: None

### DELETE `/api/workflow/conversation/{session_id}`

**Path Parameters**:

- `session_id` (string, required): Conversation session ID to delete

**Query Parameters**: None

### GET `/api/workflow/stats`

**Parameters**: None

**Response**:

```json
{
  "total_conversations": 45,
  "active_sessions": 3,
  "total_workflow_steps": 234,
  "average_response_time_ms": 156.7,
  "smart_orchestration_usage": 67.2
}
```

### GET `/api/workflow/health`

**Parameters**: None

**Response**:

```json
{
  "status": "healthy",
  "langgraph_engine": "operational",
  "llm_service": "connected",
  "adapter_registry": "loaded"
}
```

---

## 05. Admin Operations

### POST `/api/admin/check-updates`

**Parameters**: None

**Response**:

```json
{
  "has_updates": false,
  "current_count": 38894,
  "last_check": "2024-01-01T12:00:00Z"
}
```

### POST `/api/admin/download-data`

**Parameters**: None

**Response**:

```json
{
  "status": "success",
  "downloaded_entries": 38894,
  "file_size_mb": 125.6,
  "download_time_ms": 2345
}
```

### POST `/api/admin/process-data`

**Parameters**: None

**Response**:

```json
{
  "status": "success",
  "processed_entries": 38894,
  "indexed_vectors": 38894,
  "processing_time_ms": 45600
}
```

---

## 06. MCP Server Endpoints

### GET `/health` (MCP Server)

**Port**: 8001
**Parameters**: None

**Response**:

```json
{
  "status": "healthy",
  "transport": "http",
  "tools_available": 8
}
```

### GET `/sse/` (MCP Server)

**Port**: 8001
**Headers**:

- `Accept: text/event-stream`

**Parameters**: None
**Response**: Server-Sent Events stream for MCP communication

---

## Common Response Structures

### Anime Result Object

```json
{
  "anime_id": "string - Unique anime identifier",
  "title": "string - Anime title",
  "synopsis": "string - Description",
  "type": "string - TV/Movie/OVA/Special/ONA",
  "episodes": "integer - Episode count",
  "tags": "array[string] - Genre tags",
  "studios": "array[string] - Animation studios",
  "year": "integer - Release year",
  "season": "string - Release season",
  "score": "float - Relevance/similarity score (0.0-1.0)",
  "picture": "string - Poster image URL",
  "myanimelist_id": "integer - MAL ID",
  "anilist_id": "integer - AniList ID"
}
```

### Workflow Response Structure

```json
{
  "session_id": "string - Session identifier",
  "messages": "array - Conversation history",
  "workflow_steps": "array - Executed workflow steps",
  "current_context": {
    "query": "string - Processed query",
    "limit": "integer - Result limit",
    "filters": "object - Applied filters",
    "results": "array[AnimeResult] - Search results"
  },
  "user_preferences": "object - Learned user preferences"
}
```

### Error Response Structure

```json
{
  "error": "string - Error message",
  "detail": "string - Detailed error information",
  "status_code": "integer - HTTP status code"
}
```

---

## Query Filter Examples

### Smart Conversation Filters

The AI can extract and apply these filters automatically:

```json
{
  "filters": {
    "year_range": [2020, 2029], // From "2020s"
    "year": 2019, // From "2019"
    "genres": ["mecha", "action"], // From "mecha action anime"
    "exclusions": ["horror", "violent"], // From "but not horror or violent"
    "studios": ["Studio Ghibli"], // From "Studio Ghibli movies"
    "anime_types": ["Movie"], // From "movies" or "films"
    "mood": ["light", "funny"] // From "light-hearted" or "funny"
  }
}
```

### Manual Filter Application

For standard endpoints, you can specify filters in the query:

```
"mecha anime 2020s -horror"      // Mecha anime from 2020s, exclude horror
"Studio Ghibli movies"           // Studio Ghibli movies only
"action adventure TV series"     // Action adventure TV series
```

---

## Rate Limits & Constraints

- **Search Limit**: 1-50 results per request
- **Image Size**: Max 10MB for image uploads
- **Session Timeout**: 1 hour of inactivity
- **Query Length**: Max 500 characters
- **Concurrent Requests**: 10 per client
- **Text Weight**: 0.0-1.0 (multimodal searches)

---

## Testing Sequences

### Basic API Testing

1. Health Check → Database Stats → Simple Search → Semantic Search

### AI Query Understanding Testing

1. Smart Conversation with limit extraction
2. Complex query with multiple filters
3. Studio + year + exclusion query
4. Verify extracted parameters in response

### Multimodal Testing

1. Base64 image search
2. Multimodal conversation with text + image
3. Visual similarity search
4. Verify image and text weights

### Workflow Testing

1. Standard conversation
2. Smart orchestration with complex query
3. Multimodal workflow
4. Session management (create, retrieve, delete)

