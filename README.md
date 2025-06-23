# 🚀 Anime MCP Server

An AI-powered anime search and recommendation system built with **FastAPI**, **Qdrant vector database**, and **FastMCP protocol**. Features semantic search capabilities over 38,000+ anime entries with MCP integration for AI assistants.

## ✨ Features

- 🔍 **Semantic Search**: Natural language queries for anime discovery
- ⚡ **High Performance**: Sub-200ms response times with vector embeddings
- 📊 **Comprehensive Database**: 38,894 anime entries with rich metadata
- 🤖 **MCP Protocol Integration**: FastMCP server for AI assistant communication
- 🎯 **Real-time Vector Search**: Qdrant-powered semantic search
- 🖼️ **Image Search** (Phase 4): Visual similarity search with CLIP embeddings *[In Development]*
- 🐳 **Docker Support**: Easy deployment with containerized services

## 🏗️ Architecture

```
anime-mcp-server/
├── src/
│   ├── main.py                  # FastAPI application entry point
│   ├── config.py                # Centralized configuration management
│   ├── api/
│   │   ├── search.py            # Search endpoints
│   │   ├── admin.py             # Admin endpoints
│   │   └── recommendations.py   # Recommendation endpoints (basic)
│   ├── mcp/
│   │   ├── server.py            # FastMCP server implementation
│   │   └── tools.py             # MCP utility functions
│   ├── vector/
│   │   └── qdrant_client.py     # Vector database operations
│   ├── models/
│   │   └── anime.py             # Pydantic data models
│   ├── services/
│   │   └── data_service.py      # Data processing pipeline
│   └── exceptions.py            # Custom exception classes
├── scripts/
│   └── test_mcp.py              # MCP server testing client
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
# - FastAPI: http://localhost:8000
# - Qdrant: http://localhost:6333
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
EOF

# Start FastAPI server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Verify System Status

```bash
# Check system health
curl http://localhost:8000/health
# Response: {"status":"healthy","qdrant":"connected","timestamp":"..."}

# Check database stats
curl http://localhost:8000/stats
# Response: {"total_documents":38894,"vector_size":384,"status":"green"}

# Check for updates
curl -X POST http://localhost:8000/api/admin/check-updates
# Response: {"has_updates":false,"entry_count":38894}
```

## 🤖 MCP Server Integration

### Running the MCP Server

The project includes a **FastMCP server** for AI assistant integration:

```bash
# Start MCP server
python -m src.mcp.server

# Test MCP functionality
python scripts/test_mcp.py
```

### MCP Tools Available

| Tool                 | Description                  | Parameters                              |
| -------------------- | ---------------------------- | --------------------------------------- |
| `search_anime`       | Semantic anime search        | `query` (string), `limit` (int)         |
| `get_anime_details`  | Get detailed anime info      | `anime_id` (string)                     |
| `find_similar_anime` | Find similar anime           | `anime_id` (string), `limit` (int)      |
| `get_anime_stats`    | Database statistics          | None                                    |
| `recommend_anime`    | Personalized recommendations | `genres`, `year`, `anime_type`, `limit` |
| *`search_anime_by_image`* | *Image similarity search* | *`image_data` (base64), `limit` (int)* |
| *`search_multimodal_anime`* | *Text + image search* | *`query`, `image_data`, `text_weight`* |

> **Note**: Image search tools (*italicized*) are implemented but require multi-vector collection setup. See Phase 4 development section below.

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

| Endpoint               | Method | Purpose         | Example                                                            |
| ---------------------- | ------ | --------------- | ------------------------------------------------------------------ |
| `/api/search/`         | GET    | Basic search    | `curl "http://localhost:8000/api/search/?q=dragon%20ball&limit=5"` |
| `/api/search/semantic` | POST   | Advanced search | See examples below                                                 |

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

### ⚙️ Admin Endpoints

| Endpoint                   | Method | Purpose           | Example                                                      |
| -------------------------- | ------ | ----------------- | ------------------------------------------------------------ |
| `/api/admin/check-updates` | POST   | Check for updates | `curl -X POST http://localhost:8000/api/admin/check-updates` |

### 🎯 Response Format

```json
{
  "query": "dragon ball",
  "results": [
    {
      "anime_id": "cac1eeaeddf7",
      "title": "Dragon Ball Z",
      "synopsis": "",
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
  "processing_time_ms": 0.0
}
```

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

### MCP Server Testing

**Automated Testing**

```bash
# Test MCP server (comprehensive)
python scripts/test_mcp.py

# Expected output:
# ✅ MCP session initialized
# ✅ Available tools: ['search_anime', 'get_anime_details', 'find_similar_anime', 'get_anime_stats', 'recommend_anime']
# ✅ Search result: Found Dragon Ball Z & Dragon Ball with full metadata
# ✅ Stats result: 38,894 documents, healthy database
# ✅ Available resources: ['anime://database/stats', 'anime://database/schema']
# ✅ All MCP tests completed successfully!
```

**Manual MCP Testing**

For manual testing without the Python script, use the automated test as reference since MCP protocol requires proper session management:

```bash
# Start MCP server in one terminal
source venv/bin/activate
python -m src.mcp.server

# Use the automated test script (recommended approach)
python scripts/test_mcp.py
```

### Unit Tests

```bash
# Run full test suite
python run_tests.py

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
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

# API Configuration
API_TITLE=Anime MCP Server                # API title
API_VERSION=1.0.0                         # API version
ALLOWED_ORIGINS=*                         # CORS origins
```

### Docker Configuration

The system uses Docker Compose for orchestration:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./data/qdrant_storage:/qdrant/storage
```

## 📊 Performance

- **Search Speed**: Sub-second response times
- **Vector Model**: BAAI/bge-small-en-v1.5 (384-dimensional embeddings)
- **Database Size**: 38,894 anime entries when fully indexed
- **Memory Usage**: ~2-3GB for full dataset
- **Concurrency**: Supports multiple simultaneous searches
- **MCP Protocol**: Full FastMCP 2.8.1 integration

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
4. **Vectorization**: Create embeddings from title + synopsis + tags + studios
5. **Indexing**: Store in Qdrant with optimized batch processing

### Database Schema

Each anime entry contains:

- **Basic Info**: title, type, episodes, status, year, season
- **Metadata**: synopsis, tags, studios, producers
- **Platform IDs**: MyAnimeList, AniList, Kitsu, AniDB, etc.
- **Search Fields**: embedding_text, search_text
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

### Code Quality

```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Run tests
pytest tests/ -v
```

### Project Structure

- **`src/main.py`**: FastAPI application entry point
- **`src/mcp/server.py`**: FastMCP server with 5 tools + 2 resources
- **`src/vector/qdrant_client.py`**: Vector database operations
- **`src/config.py`**: Centralized configuration management
- **`scripts/test_mcp.py`**: MCP server testing client

## 🔮 Technology Stack

- **Backend**: FastAPI + Python 3.12
- **Vector Database**: Qdrant 1.14.1
- **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5)
- **MCP Integration**: FastMCP 2.8.1
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

## 📞 Support

- 🐛 **Issues**: Report bugs via GitHub Issues
- 💬 **Discussions**: Community discussions in GitHub Discussions
- 📖 **Documentation**: Full docs at `/docs` endpoint when server is running
- 🔧 **Development**: See `CLAUDE.md` for detailed development guidance

---

## 🚧 Phase 4: Image Search Development

### Current Implementation Status

**🔧 Infrastructure**: ✅ Complete
- Multi-vector Qdrant client with CLIP integration
- Vision processor with ViT-B/32 model  
- MCP tools for image search functionality
- Environment configuration (`ENABLE_MULTI_VECTOR=true`)

**🗄️ Database State**: ❌ Requires Setup
- Current: Single-vector collection (text embeddings only)
- Required: Multi-vector collection (text + image vectors)
- Action needed: Collection migration + image processing

**🔗 API Availability**: ⚠️ MCP Only
- Image search available through MCP protocol
- REST API endpoints not yet implemented
- MCP tools ready: `search_anime_by_image`, `search_multimodal_anime`

### Quick Setup for Image Search Testing

```bash
# Enable multi-vector support
echo "ENABLE_MULTI_VECTOR=true" >> .env

# Restart server to load CLIP model
docker-compose restart

# Migrate collection to multi-vector (TODO: implement)
python scripts/migrate_to_multivector.py

# Process anime images (TODO: implement) 
python scripts/add_image_embeddings.py
```

**Note**: Image search is architecturally complete but requires database migration and image processing to become functional.

---

**Status**: ✅ **Production Ready** - Fully functional anime search with vector database, FastMCP integration, and comprehensive API. Image search infrastructure ready for Phase 4 completion.
