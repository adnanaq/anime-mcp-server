# Anime MCP Server

An AI-powered anime search and recommendation system built with **FastAPI**, **Qdrant vector database**, **FastMCP protocol**, and **LangGraph workflow orchestration**. Features semantic search capabilities over 38,000+ anime entries with MCP integration for AI assistants.

## Features

- **Semantic Search**: Natural language queries for anime discovery
- **High Performance**: Sub-200ms response times with vector embeddings
- **Comprehensive Database**: 38,894 anime entries with rich metadata
- **MCP Protocol Integration**: FastMCP server for AI assistant communication
- **Real-time Vector Search**: Qdrant-powered semantic search
- **Multi-Modal Search**: Visual similarity and combined text+image search with CLIP embeddings
- **Conversational Workflows**: LangGraph-powered intelligent conversation flows with ToolNode integration
- **Smart Orchestration**: Advanced multi-step query processing with complexity assessment
- **AI-Powered Query Understanding**: Natural language parameter extraction with LLM intelligence
- **Intelligent Parameter Extraction**: Automatic limit, genre, year, and exclusion detection
- **Native LangGraph Integration**: ToolNode-based workflow engine with ~200 lines less boilerplate
- **Docker Support**: Easy deployment with containerized services

## 🏗️ Architecture

```
anime-mcp-server/
├── src/
│   ├── main.py                  # FastAPI application entry point
│   ├── config.py                # Centralized configuration management
│   ├── api/
│   │   ├── search.py            # Search endpoints
│   │   ├── admin.py             # Admin endpoints
│   │   ├── recommendations.py   # Recommendation endpoints (basic)
│   │   └── workflow.py          # LangGraph workflow endpoints
│   ├── mcp/
│   │   ├── server.py            # FastMCP server implementation
│   │   └── tools.py             # MCP utility functions
│   ├── langgraph/
│   │   ├── langchain_tools.py   # LangChain tool creation & ToolNode workflow
│   │   └── workflow_engine.py   # Main anime workflow engine
│   ├── vector/
│   │   ├── qdrant_client.py     # Vector database operations
│   │   └── vision_processor.py  # CLIP image processing
│   ├── models/
│   │   └── anime.py             # Pydantic data models
│   ├── services/
│   │   ├── data_service.py      # Data processing pipeline
│   │   └── llm_service.py       # LLM service for AI-powered query understanding
│   └── exceptions.py            # Custom exception classes
├── scripts/
│   ├── test_mcp.py              # MCP server testing client
│   ├── migrate_to_multivector.py # Collection migration script
│   ├── add_image_embeddings.py  # Image processing pipeline
│   └── verify_mcp_server.py     # MCP functionality verification
├── data/
│   ├── raw/                     # Original anime database JSON
│   └── qdrant_storage/          # Qdrant vector database files
├── docker-compose.yml           # Service orchestration
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- 4GB+ RAM (for vector processing)

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd anime-mcp-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Services

**Full Stack (Recommended)**

```bash
# Start complete stack with Docker
docker compose up -d

# Services will be available at:
# - FastAPI REST API: http://localhost:8000
# - MCP Server (HTTP): http://localhost:8001 (if mcp-server container is enabled)
# - Qdrant Vector DB: http://localhost:6333
```

**Deployment Options:**

1. **REST API Only (Default)**
   ```bash
   # Start just FastAPI + Qdrant
   docker compose up -d fastapi qdrant
   ```

2. **REST API + MCP HTTP Server**
   ```bash
   # Start all services (REST API on :8000, MCP HTTP on :8001)
   docker compose up -d
   ```

3. **MCP stdio mode (for Claude Code integration)**
   ```bash
   # Start infrastructure only
   docker compose up -d qdrant
   
   # Run MCP server locally in stdio mode
   python -m src.mcp.server
   ```

**Local Development**

```bash
# Start Qdrant vector database
docker compose up -d qdrant

# Create environment file
cat > .env << EOF
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=anime_database
HOST=0.0.0.0
PORT=8000
DEBUG=True
ENABLE_MULTI_VECTOR=true
EOF

# Start FastAPI server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Initialize Database (First Time Setup)

**Production Indexing (Required for first run):**

```bash
# Start the full-update process (runs in background)
curl -X POST http://localhost:8000/api/admin/update-full

# Monitor indexing progress (check logs)
docker compose logs fastapi --tail 50 -f

# Expected output shows batch progress:
# INFO - Uploaded batch 29/389 (100 points)
# ...continues until batch 389/389
```

**Indexing Progress:** The system will process all 38,894 anime entries with dual image vectors. This typically takes 2-3 hours depending on network speed for image downloads.

### 4. Verify System Status

```bash
# Check system health
curl http://localhost:8000/health
# Response: {"status":"healthy","qdrant":"connected","timestamp":"..."}

# Check database stats (after indexing completes)
curl http://localhost:8000/stats
# Response: {"total_documents":38894,"vector_size":384,"status":"green"}

# Check indexing status
curl http://localhost:8000/api/admin/update-status
# Response: {"entry_count":38894,"last_full_update":"2025-06-29T19:00:00Z"}
```

## MCP Server Integration

### Transport Protocols

The MCP server supports multiple transport protocols for different use cases:

| Protocol       | Use Case                            | Port | Command                                |
| -------------- | ----------------------------------- | ---- | -------------------------------------- |
| **stdio**      | Local development, Claude Code      | N/A  | `python -m src.mcp.server`             |
| **HTTP (SSE)** | Web clients, Postman, remote access | 8001 | `python -m src.mcp.server --mode http` |

### Running the MCP Server

```bash
# Local development (stdio) - default mode
python -m src.mcp.server

# Web/remote access (HTTP)
python -m src.mcp.server --mode http --host 0.0.0.0 --port 8001

# With verbose logging
python -m src.mcp.server --mode http --verbose

# Test MCP functionality
python scripts/test_mcp.py
```

### Configuration Options

Environment variables for MCP server:

```bash
# MCP Server Configuration
SERVER_MODE=stdio          # Transport mode: stdio, http, sse, streamable
MCP_HOST=0.0.0.0          # HTTP server host (for HTTP modes)
MCP_PORT=8001             # HTTP server port (for HTTP modes)
```

### Client Integration

**Claude Code (stdio mode)**

```json
{
  "mcpServers": {
    "anime-search": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/anime-mcp-server"
    }
  }
}
```

**Postman or Web Clients (HTTP mode)**

- Start server: `python -m src.mcp.server --mode http --port 8001`
- MCP endpoint: `http://localhost:8001/sse/`
- Health check: `curl http://localhost:8001/health`

### MCP Tools Available

| Tool                          | Description                  | Parameters                              |
| ----------------------------- | ---------------------------- | --------------------------------------- |
| `search_anime`                | Semantic anime search        | `query` (string), `limit` (int)         |
| `get_anime_details`           | Get detailed anime info      | `anime_id` (string)                     |
| `find_similar_anime`          | Find similar anime           | `anime_id` (string), `limit` (int)      |
| `get_anime_stats`             | Database statistics          | None                                    |
| `search_anime_by_image`       | Image similarity search      | `image_data` (base64), `limit` (int)    |
| `find_visually_similar_anime` | Visual similarity            | `anime_id` (string), `limit` (int)      |
| `search_multimodal_anime`     | Text + image search          | `query`, `image_data`, `text_weight`    |

### MCP Resources

- `anime://database/stats` - Database statistics and health
- `anime://database/schema` - Database schema information

## 📡 API Reference

### 🏥 Core System Endpoints

| Endpoint  | Method | Purpose        | Example                             |
| --------- | ------ | -------------- | ----------------------------------- |
| `/`       | GET    | API overview   | `curl http://localhost:8000/`       |
| `/health` | GET    | Health check   | `curl http://localhost:8000/health` |
| `/stats`  | GET    | Database stats | `curl http://localhost:8000/stats`  |

### 🔍 Search & Discovery Endpoints

| Endpoint                         | Method | Purpose            | Parameters                                          |
| -------------------------------- | ------ | ------------------ | --------------------------------------------------- |
| `/api/search/`                   | GET    | Basic search       | `q` (required), `limit` (1-50, default: 10)        |
| `/api/search/semantic`           | POST   | Advanced search    | JSON body: `query` (required), `limit` (optional)  |
| `/api/search/similar/{anime_id}` | GET    | Find similar anime | `anime_id` (path), `limit` (1-50, default: 10)     |

### 🖼️ Image Search Endpoints

| Endpoint                                  | Method | Purpose             | Parameters                                          |
| ----------------------------------------- | ------ | ------------------- | --------------------------------------------------- |
| `/api/search/by-image`                    | POST   | Image upload search | Form: `image` (file), `limit` (1-50, default: 10)  |
| `/api/search/by-image-base64`             | POST   | Base64 image search | Form: `image_data` (base64), `limit` (optional)    |
| `/api/search/visually-similar/{anime_id}` | GET    | Visual similarity   | `anime_id` (path), `limit` (1-50, default: 10)     |
| `/api/search/multimodal`                  | POST   | Combined search     | Form: `query`, `image`, `limit`, `text_weight`     |

**Basic Search Examples:**

```bash
# Search by genre
curl "http://localhost:8000/api/search/?q=action%20adventure&limit=5"

# Search by studio
curl "http://localhost:8000/api/search/?q=studio%20ghibli&limit=3"

# Search by theme
curl "http://localhost:8000/api/search/?q=romantic%20comedy&limit=5"
```

**Advanced Semantic Search:**

```bash
curl -X POST http://localhost:8000/api/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mecha robots fighting in space",
    "limit": 10
  }'
```

**Image Search Examples:**

```bash
# Upload image file for visual similarity search
curl -X POST http://localhost:8000/api/search/by-image \
  -F "image=@anime_poster.jpg" \
  -F "limit=5"

# Search with base64 encoded image data
curl -X POST http://localhost:8000/api/search/by-image-base64 \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "image_data=iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB..." \
  -d "limit=5"

# Find visually similar anime by ID
curl "http://localhost:8000/api/search/visually-similar/cac1eeaeddf7?limit=5"

# Multimodal search (text + image)
curl -X POST http://localhost:8000/api/search/multimodal \
  -F "query=mecha anime" \
  -F "image=@robot_poster.jpg" \
  -F "limit=10" \
  -F "text_weight=0.7"
```

### 🤖 Conversational Workflow Endpoints

| Endpoint                           | Method | Purpose                      | Example                              |
| ---------------------------------- | ------ | ---------------------------- | ------------------------------------ |
| `/api/workflow/conversation`       | POST   | Start/continue conversation  | Standard conversation flows          |
| `/api/workflow/smart-conversation` | POST   | Smart orchestration workflow | Advanced multi-step query processing |
| `/api/workflow/multimodal`         | POST   | Multimodal conversation      | Text + image conversation            |
| `/api/workflow/conversation/{id}`  | GET    | Get conversation history     | Retrieve session with summary        |
| `/api/workflow/conversation/{id}`  | DELETE | Delete conversation          | Remove conversation session          |
| `/api/workflow/stats`              | GET    | Workflow statistics          | Get conversation metrics             |
| `/api/workflow/health`             | GET    | Workflow system health       | Check LangGraph engine status        |

**AI-Powered Query Understanding (Phase 6C):**

The system now features intelligent natural language processing that automatically extracts search parameters from user queries:

```bash
# AI-powered natural language understanding
curl -X POST http://localhost:8000/api/workflow/smart-conversation \
  -H "Content-Type: application/json" \
  -d '{"message": "find me 5 mecha anime from 2020s but not too violent"}'

# Response includes extracted parameters:
# {
#   "current_context": {
#     "query": "mecha anime",           # Cleaned query
#     "limit": 5,                      # Extracted from "find me 5"
#     "filters": {
#       "year_range": [2020, 2029],    # From "2020s"
#       "genres": ["mecha"],           # Detected genre
#       "exclusions": ["violent"]      # From "but not too violent"
#     }
#   }
# }
```

**Conversational Workflow Examples:**

```bash
# Start standard conversation
curl -X POST http://localhost:8000/api/workflow/conversation \
  -H "Content-Type: application/json" \
  -d '{"message": "Find me some good action anime"}'

# Smart orchestration for complex queries
curl -X POST http://localhost:8000/api/workflow/smart-conversation \
  -H "Content-Type: application/json" \
  -d '{
    "message": "find action anime but not horror and similar to attack on titan",
    "enable_smart_orchestration": true,
    "max_discovery_depth": 3
  }'

# Continue existing conversation
curl -X POST http://localhost:8000/api/workflow/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find something similar but more romantic",
    "session_id": "existing-session-id"
  }'

# Multimodal conversation with image
curl -X POST http://localhost:8000/api/workflow/multimodal \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find anime similar to this image",
    "image_data": "base64_encoded_image_data",
    "text_weight": 0.7
  }'

# Get conversation history
curl http://localhost:8000/api/workflow/conversation/session-id

# Check workflow system health
curl http://localhost:8000/api/workflow/health
```

### ⚙️ Admin Endpoints

| Endpoint                   | Method | Purpose                    | Example                                                      |
| -------------------------- | ------ | -------------------------- | ------------------------------------------------------------ |
| `/api/admin/check-updates` | POST   | Check for updates          | `curl -X POST http://localhost:8000/api/admin/check-updates` |
| `/api/admin/download-data` | POST   | Download latest anime data | `curl -X POST http://localhost:8000/api/admin/download-data` |
| `/api/admin/process-data`  | POST   | Process and index data     | `curl -X POST http://localhost:8000/api/admin/process-data`  |

### 🎯 Response Formats

#### **Standard Search Response**
```json
{
  "query": "dragon ball",
  "results": [
    {
      "anime_id": "cac1eeaeddf7",
      "title": "Dragon Ball Z", 
      "synopsis": "Description text",
      "type": "TV",
      "episodes": 291,
      "tags": ["action", "adventure", "fighting"],
      "studios": ["toei animation co., ltd."],
      "picture": "https://cdn.myanimelist.net/images/anime/1277/142022.jpg",
      "score": 0.7822759,
      "year": 1989,
      "season": "spring",
      "myanimelist_id": 813,
      "anilist_id": 813
    }
  ],
  "total_results": 1,
  "processing_time_ms": 45.2
}
```

#### **Workflow Response Structure**
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

#### **Error Response Structure**
```json
{
  "error": "string - Error message",
  "detail": "string - Detailed error information", 
  "status_code": "integer - HTTP status code"
}
```

### 🔧 API Constraints & Limits

- **Search Limit**: 1-50 results per request
- **Image Size**: Max 10MB for image uploads  
- **Session Timeout**: 1 hour of inactivity
- **Query Length**: Max 500 characters
- **Concurrent Requests**: 10 per client
- **Text Weight**: 0.0-1.0 (multimodal searches)

### 🎛️ Query Filters & AI Understanding

#### **AI-Extracted Filter Patterns**
The system automatically extracts these filters from natural language:

```json
{
  "filters": {
    "year_range": [2020, 2029],        // From "2020s"
    "year": 2019,                      // From "2019" 
    "genres": ["mecha", "action"],     // From "mecha action anime"
    "exclusions": ["horror", "violent"], // From "but not horror or violent"
    "studios": ["Studio Ghibli"],      // From "Studio Ghibli movies"
    "anime_types": ["Movie"],          // From "movies" or "films"
    "mood": ["light", "funny"]         // From "light-hearted" or "funny"
  }
}
```

#### **Manual Filter Syntax**
For direct API calls, use these patterns:
- `"mecha anime 2020s -horror"` → Mecha from 2020s, exclude horror
- `"Studio Ghibli movies"` → Studio Ghibli movies only  
- `"action adventure TV series"` → Action adventure TV series

## 🧪 Testing

### FastAPI Server Testing

```bash
# Health check
curl http://localhost:8000/health

# Test search
curl "http://localhost:8000/api/search/?q=dragon%20ball&limit=5"

# Stats
curl http://localhost:8000/stats
```

### 🔬 API Testing Sequences

#### **Basic API Testing Sequence**
1. Health Check → Database Stats → Simple Search → Semantic Search

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Database stats  
curl http://localhost:8000/stats

# 3. Simple search
curl "http://localhost:8000/api/search/?q=dragon%20ball&limit=5"

# 4. Semantic search
curl -X POST http://localhost:8000/api/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "mecha robots fighting in space", "limit": 10}'
```

#### **AI Query Understanding Testing** 
1. Smart Conversation with limit extraction
2. Complex query with multiple filters  
3. Studio + year + exclusion query
4. Verify extracted parameters in response

```bash
# Test natural language parameter extraction
curl -X POST http://localhost:8000/api/workflow/smart-conversation \
  -H "Content-Type: application/json" \
  -d '{"message": "find me 5 mecha anime from 2020s but not too violent"}'

# Test studio + year extraction
curl -X POST http://localhost:8000/api/workflow/smart-conversation \
  -H "Content-Type: application/json" \
  -d '{"message": "show me top 3 Studio Ghibli movies from 90s"}'

# Test complex exclusions
curl -X POST http://localhost:8000/api/workflow/smart-conversation \
  -H "Content-Type: application/json" \
  -d '{"message": "find action adventure anime but not romance or horror"}'
```

#### **Multimodal Testing Sequence**
1. Base64 image search
2. Multimodal conversation with text + image  
3. Visual similarity search
4. Verify image and text weights

```bash
# Upload image search
curl -X POST http://localhost:8000/api/search/by-image \
  -F "image=@anime_poster.jpg" -F "limit=5"

# Base64 image search  
curl -X POST http://localhost:8000/api/search/by-image-base64 \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "image_data=iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB..." -d "limit=5"

# Multimodal workflow
curl -X POST http://localhost:8000/api/workflow/multimodal \
  -H "Content-Type: application/json" \
  -d '{"message": "find anime similar to this style", "image_data": "base64_data", "text_weight": 0.7}'
```

#### **Workflow Testing Sequence**
1. Standard conversation
2. Smart orchestration with complex query
3. Multimodal workflow  
4. Session management (create, retrieve, delete)

```bash
# Standard conversation
curl -X POST http://localhost:8000/api/workflow/conversation \
  -H "Content-Type: application/json" \
  -d '{"message": "Find me some good action anime"}'

# Smart orchestration with complex query
curl -X POST http://localhost:8000/api/workflow/smart-conversation \
  -H "Content-Type: application/json" \
  -d '{
    "message": "find mecha anime but not too violent and similar to gundam",
    "enable_smart_orchestration": true,
    "max_discovery_depth": 3
  }'

# Session management
curl http://localhost:8000/api/workflow/conversation/session-id  # Get history
curl -X DELETE http://localhost:8000/api/workflow/conversation/session-id  # Delete session
```

### MCP Server Testing

**Comprehensive Testing**

```bash
# Test MCP server (comprehensive verification)
python scripts/verify_mcp_server.py

# With detailed output and image testing
python scripts/verify_mcp_server.py --detailed

# Skip image tests (if CLIP not available)
python scripts/verify_mcp_server.py --skip-image-tests

# Expected output:
# Starting comprehensive FastMCP Anime Server verification...
# MCP session initialized
# Available tools: ['search_anime', 'get_anime_details', 'find_similar_anime', 'get_anime_stats', 'search_anime_by_image', 'find_visually_similar_anime', 'search_multimodal_anime']
# All expected MCP tools are available
# Basic search test successful
# Stats test successful
# Testing image search functionality...
# Image search returned 3 results
# Visual similarity search returned 2 results
# Multimodal search returned 3 results
# Testing database health and statistics...
# Qdrant connection verified
# Total anime entries: 38,894
# All tests completed successfully!
```

**Protocol-Specific Testing**

**stdio mode (local development):**

```bash
# Start MCP server in one terminal
source venv/bin/activate
python -m src.mcp.server

# Use the automated test script (recommended approach)
python scripts/verify_mcp_server.py
```

**HTTP mode (web/remote access):**

```bash
# Start HTTP MCP server
python -m src.mcp.server --mode http --port 8001

# Test HTTP endpoint accessibility
curl http://localhost:8001/sse/

# Test with Postman or other HTTP MCP clients
# Endpoint: http://localhost:8001/sse/
```

### Unit Tests

```bash
# Run full test suite
python run_tests.py

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Test AI-powered query understanding (Phase 6C)
pytest tests/unit/services/test_llm_service.py -v
pytest tests/unit/langgraph/test_llm_integration.py -v

# Test smart orchestration features
pytest tests/unit/langgraph/test_smart_orchestration.py -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Configuration

### Environment Variables

```bash
# Qdrant Configuration
QDRANT_URL=http://localhost:6333          # Vector database URL
QDRANT_COLLECTION_NAME=anime_database     # Collection name

# Server Configuration
HOST=0.0.0.0                              # FastAPI bind host
PORT=8000                                 # FastAPI port
DEBUG=True                                # Debug mode

# Vector Search Configuration
FASTEMBED_MODEL=BAAI/bge-small-en-v1.5    # Embedding model
QDRANT_VECTOR_SIZE=384                    # Vector dimensions
QDRANT_DISTANCE_METRIC=cosine             # Distance function

# Multi-Vector Configuration (Image Search)
ENABLE_MULTI_VECTOR=true                  # Enable image search features
IMAGE_VECTOR_SIZE=512                     # CLIP embedding dimensions
CLIP_MODEL=ViT-B/32                       # CLIP model for image processing

# API Configuration
API_TITLE=Anime MCP Server                # API title
API_VERSION=1.0.0                         # API version
ALLOWED_ORIGINS=*                         # CORS origins

# LLM Configuration (Phase 6C - AI-Powered Query Understanding)
OPENAI_API_KEY=your_openai_key_here       # OpenAI API key for intelligent query parsing
ANTHROPIC_API_KEY=your_anthropic_key_here # Anthropic API key (alternative)
LLM_PROVIDER=openai                       # Default LLM provider: openai, anthropic
```

### Docker Configuration

The system uses Docker Compose for orchestration with support for both REST API and MCP server:

**Port Allocation:**

- **8000**: FastAPI REST API
- **8001**: MCP HTTP Server (when enabled)
- **6333**: Qdrant Vector Database
- **6334**: Qdrant gRPC (internal)

## 📊 Performance

- **Search Speed**: Sub-200ms text search, ~1s image search response times
- **Workflow Performance**: 150ms target response time (improved from 200ms via ToolNode optimization)
- **AI Query Understanding**: ~500ms LLM response time with structured output parsing
- **Smart Orchestration**: 50ms average response time (faster than standard workflows)
- **Vector Models**:
  - Text: BAAI/bge-small-en-v1.5 (384-dimensional embeddings)
  - Image: CLIP ViT-B/32 (512-dimensional embeddings)
  - LLM: OpenAI GPT-4o-mini / Anthropic Claude Haiku for query understanding
- **Database Size**: 38,894 anime entries with dual image vectors (picture + thumbnail)
- **Memory Usage**: ~3-4GB for full dataset with CLIP image embeddings
- **Indexing Time**: 2-3 hours for complete database with image downloads
- **Vector Storage**: Text (384D) + Picture (512D) + Thumbnail (512D) per anime
- **Concurrency**: Supports multiple simultaneous searches and complex query processing
- **MCP Protocol**: Full FastMCP 2.8.1 integration with 7 tools
- **Workflow Processing**: 2-5 workflow steps per query depending on complexity
- **Natural Language Processing**: Intelligent parameter extraction with graceful fallbacks

## 🔄 Data Pipeline

### Source Data

- **Provider**: [anime-offline-database](https://github.com/manami-project/anime-offline-database)
- **Format**: Comprehensive JSON with cross-platform references
- **Coverage**: MyAnimeList, AniList, Kitsu, AniDB, and 7 other sources
- **Total Entries**: 38,894 anime with rich metadata

### Processing Steps

1. **Download**: Fetch latest anime-offline-database JSON
2. **Validation**: Parse and validate entries with Pydantic models
3. **Enhancement**: Extract platform IDs, calculate quality scores
4. **Text Vectorization**: Create embeddings from title + synopsis + tags + studios
5. **Image Processing**: Download poster images and generate CLIP embeddings
6. **Indexing**: Store in Qdrant multi-vector collection with optimized batch processing

### Database Schema

Each anime entry contains:

- **Basic Info**: title, type, episodes, status, year, season
- **Metadata**: synopsis, tags, studios, producers, picture URLs
- **Platform IDs**: MyAnimeList, AniList, Kitsu, AniDB, etc.
- **Search Fields**: embedding_text, search_text
- **Vector Embeddings**: text (384-dim) + image (512-dim) in multi-vector collection
- **Quality Score**: Data completeness rating (0-1)

## 🎮 Interactive Testing

### Web Interface

Visit **http://localhost:8000** for:

- FastAPI automatic documentation (Swagger UI)
- Interactive API testing
- Real-time endpoint exploration

### MCP Testing

```bash
# Test MCP server communication
python scripts/test_mcp.py

# Start MCP server for AI integration
python -m src.mcp.server
```

## 🛠️ Development

### Important Scripts

```bash
# MCP Server Management
python -m src.mcp.server                            # Start MCP server (stdio mode)
python -m src.mcp.server --mode http --port 8001    # Start MCP server (HTTP mode)
python -m src.mcp.server --help                     # View all CLI options

# Data Management
python scripts/migrate_to_multivector.py --dry-run    # Test collection migration
python scripts/migrate_to_multivector.py             # Migrate to multi-vector
python scripts/add_image_embeddings.py --batch-size 100  # Process image embeddings

# Testing & Verification
python scripts/verify_mcp_server.py                 # Comprehensive MCP server testing
python scripts/verify_mcp_server.py --detailed      # Detailed testing with sample data
python scripts/verify_mcp_server.py --skip-image-tests  # Skip image tests if CLIP unavailable
python run_tests.py                                 # Run full test suite

# Data Pipeline
curl -X POST http://localhost:8000/api/admin/download-data  # Download latest data
curl -X POST http://localhost:8000/api/admin/process-data   # Process and index
```

### Code Quality & Formatting

```bash
# Code formatting and linting (recommended order)
autoflake --recursive --in-place --remove-all-unused-imports --remove-unused-variables src/ tests/ scripts/
isort src/ tests/ scripts/
black src/ tests/ scripts/

# Check formatting (CI/pre-commit)
autoflake --check --recursive --remove-all-unused-imports src/ tests/ scripts/
isort --check-only src/ tests/ scripts/
black --check src/ tests/ scripts/

# Type checking
mypy src/

# Run tests
pytest tests/ -v
```

**Formatting Tools Configuration:**
- **Black**: Code style formatting (88 char line length, Python 3.11+ target)
- **isort**: Import organization (Black-compatible profile)
- **autoflake**: Unused import removal (safe settings, preserves __init__.py imports)

All tools configured in `pyproject.toml` with modern best practices and compatibility.

### Project Structure

- **`src/main.py`**: FastAPI application entry point
- **`src/mcp/server.py`**: FastMCP server with 7 tools + 2 resources
- **`src/vector/qdrant_client.py`**: Multi-vector database operations with CLIP
- **`src/vector/vision_processor.py`**: CLIP image processing pipeline
- **`src/config.py`**: Centralized configuration management
- **`src/services/llm_service.py`**: AI-powered query understanding service
- **`scripts/test_mcp.py`**: MCP server testing client

## 🔮 Technology Stack

- **Backend**: FastAPI + Python 3.12
- **Vector Database**: Qdrant 1.11.3 (multi-vector support)
- **Text Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5)
- **Image Embeddings**: CLIP (ViT-B/32)
- **AI Integration**: OpenAI GPT-4o-mini / Anthropic Claude for query understanding
- **Workflow Engine**: LangGraph with native ToolNode integration for conversation orchestration
- **MCP Integration**: FastMCP 2.8.1
- **Image Processing**: PIL + torch + CLIP
- **Containerization**: Docker + Docker Compose
- **Data Validation**: Pydantic v2
- **Testing**: pytest + httpx

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [anime-offline-database](https://github.com/manami-project/anime-offline-database) for comprehensive anime data
- [Qdrant](https://qdrant.ai/) for vector search capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [FastMCP](https://github.com/jlowin/fastmcp) for MCP protocol integration

## Troubleshooting

### Common Issues

**Port Conflicts**

```bash
# Check if port is in use
netstat -tulpn | grep :8001

# Use different port
python -m src.mcp.server --mode http --port 8002
```

**MCP Connection Issues**

```bash
# Check server logs with verbose output
python -m src.mcp.server --mode http --verbose

# Verify Qdrant connection
curl http://localhost:6333/health
```

**Docker Issues**

```bash
# Check container logs
docker compose logs fastapi --tail 50
docker compose logs mcp-server --tail 50
docker compose logs qdrant --tail 50

# Monitor real-time logs (useful during indexing)
docker compose logs fastapi -f

# Check container status
docker compose ps

# Restart specific service
docker compose restart fastapi
docker compose restart mcp-server

# Rebuild containers after code changes
docker compose build --no-cache
docker compose up -d --force-recreate
```

**Indexing Issues**

```bash
# Check indexing progress
curl http://localhost:8000/api/admin/update-status

# Monitor indexing in real-time
docker compose logs fastapi -f | grep "batch"

# Check database statistics
curl http://localhost:8000/stats

# Restart indexing if it fails
curl -X POST http://localhost:8000/api/admin/update-full
```

**Invalid Transport Mode**

```bash
# Valid modes: stdio, http, sse, streamable
python -m src.mcp.server --mode invalid
# Error: argument --mode: invalid choice: 'invalid'
```

### Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Community discussions in GitHub Discussions
- **Documentation**: Full docs at `/docs` endpoint when server is running
- **Development**: See `CLAUDE.md` for detailed development guidance

---

**Status**: ✅ **Production Ready** - Complete anime search system with multi-modal capabilities, vector database, FastMCP integration, and comprehensive REST API.
