# 🚀 Anime MCP Server

An AI-powered anime search and recommendation system built with FastAPI and Qdrant vector database. Features semantic search capabilities over 38,000+ anime entries from the anime-offline-database.

## ✨ Features

- 🔍 **Semantic Search**: Natural language queries for anime discovery
- ⚡ **High Performance**: Sub-200ms response times with vector embeddings  
- 📊 **Comprehensive Database**: 38,894 anime entries with rich metadata
- 🤖 **MCP Protocol Ready**: Designed for AI assistant integration
- 🐳 **Docker Support**: Easy deployment with containerized services
- 🎯 **Real-time Indexing**: Background processing with live status updates

## 🏗️ Architecture

```
anime-mcp-server/
├── src/
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── search.py        # Search endpoints
│   │   └── recommendations.py # Recommendation endpoints
│   ├── vector/
│   │   └── qdrant_client.py  # Vector database operations
│   ├── models/
│   │   └── anime.py         # Pydantic data models
│   └── services/
│       └── data_service.py  # Data processing pipeline
├── data/
│   ├── raw/                 # Original anime database JSON
│   └── processed/           # Vector-ready anime data
├── docker-compose.yml       # Service orchestration
└── requirements.txt         # Python dependencies
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

**Option A: Full Docker Stack (Recommended)**
```bash
# Start complete stack with Docker
docker compose up -d

# Services will be available at:
# - FastAPI: http://localhost:8000
# - Qdrant: http://localhost:6333
```

**Option B: Local Development**
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
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Initialize Database

**For Docker Stack:**
```bash
# Trigger full database indexing via API
curl -X POST http://localhost:8000/api/admin/update-full

# Monitor indexing progress
curl http://localhost:8000/stats
```

**For Local Development:**
```bash
# Download and process anime data (one-time setup)
python download_data.py
```

The system will:
- Download 38,894 anime entries from anime-offline-database
- Process entries into vector embeddings (~6 seconds)
- Index data in Qdrant for semantic search (~3-4 hours)
- Continue indexing in background while API remains functional
- Use optimized batch processing with model preloading

## 📡 Complete API Reference

The server provides three main categories of endpoints: **Core System**, **Search & Discovery**, and **Admin & Management**.

### 🏥 Core System Endpoints

Essential endpoints for system health and information.

```http
GET  /                   # API overview and available endpoints
GET  /health            # System health check (Qdrant connectivity)
GET  /stats             # Database statistics (entry count, health)
```

**Examples:**
```bash
# Check if system is running
curl http://localhost:8000/health
# Response: {"status":"healthy","qdrant":"connected","timestamp":"2025-01-21"}

# Get database stats
curl http://localhost:8000/stats  
# Response: {"total_anime":3056,"indexed_anime":3056,"last_updated":"...","index_health":"healthy"}
```

---

### 🔍 Search & Discovery Endpoints

Core functionality for finding and discovering anime.

#### **Basic Search**
```http
GET  /api/search/        # Simple text-based search
```

**Parameters:**
- `q` (required): Search query 
- `limit` (optional): Number of results (1-100, default: 20)

**Examples:**
```bash
# Search by genre
curl "http://localhost:8000/api/search/?q=action%20adventure&limit=5"

# Search by theme  
curl "http://localhost:8000/api/search/?q=high%20school%20romance&limit=3"

# Search by studio
curl "http://localhost:8000/api/search/?q=studio%20ghibli&limit=10"
```

#### **Advanced Semantic Search**  
```http
POST /api/search/semantic # Advanced search with filters
```

**Request Body:**
```json
{
  "query": "mecha robots fighting in space",
  "limit": 10,
  "filters": {"type": "TV"}  // Optional filters
}
```

**Examples:**
```bash
# Complex semantic search
curl -X POST http://localhost:8000/api/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "dark psychological thriller", "limit": 5}'

# Search with filters
curl -X POST http://localhost:8000/api/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "romantic comedy", "limit": 3, "filters": {"type": "TV"}}'
```

#### **Similarity & Recommendations**
```http
GET  /api/search/similar/{anime_id}           # Find similar anime
GET  /api/recommendations/similar/{anime_id}  # Get recommendations  
POST /api/recommendations/based-on-preferences # Preference-based recommendations
```

**Examples:**
```bash
# Find similar anime
curl "http://localhost:8000/api/search/similar/abc123def456?limit=10"

# Get recommendations
curl "http://localhost:8000/api/recommendations/similar/abc123def456"
```

---

### ⚙️ Admin & Management Endpoints

Administrative endpoints for database management and updates.

#### **Update Status & Information**
```http
GET  /api/admin/update-status         # Current update metadata
POST /api/admin/check-updates         # Check for available updates
```

**Examples:**
```bash
# Check update status
curl http://localhost:8000/api/admin/update-status
# Response: {"last_check":"...","last_update":"...","entry_count":38894,...}

# Check for new updates
curl -X POST http://localhost:8000/api/admin/check-updates
# Response: {"has_updates":true,"last_check":"...","content_hash":"abc123..."}
```

#### **Manual Updates**
```http
POST /api/admin/update-incremental    # Smart incremental update
POST /api/admin/update-full          # Full database refresh (emergency)
POST /api/admin/schedule-weekly-update # Trigger weekly update process
```

**Examples:**
```bash
# Perform incremental update (recommended)
curl -X POST http://localhost:8000/api/admin/update-incremental
# Response: {"message":"Incremental update started in background","status":"processing"}

# Emergency full update (takes 2-3 hours)
curl -X POST http://localhost:8000/api/admin/update-full
# Response: {"message":"Full update started...","warning":"This will re-index all 38,000+ entries"}
```

#### **🧠 Smart Scheduling (Advanced)**
```http
GET  /api/admin/update-safety-check      # Check if safe to update now
GET  /api/admin/smart-schedule-analysis  # Analyze optimal update timing
POST /api/admin/smart-update            # Update only if safe
```

**Examples:**
```bash
# Check if it's safe to update right now
curl http://localhost:8000/api/admin/update-safety-check
# Response: {"safe_to_update":true,"reasons":["Release 29.7 hours ago - safe to update"],...}

# Get optimal schedule analysis
curl http://localhost:8000/api/admin/smart-schedule-analysis
# Response: {"analysis":{"most_common_release_day":"Thursday",...},"recommendation":{...}}

# Smart update (only proceeds if safe)
curl -X POST http://localhost:8000/api/admin/smart-update
# Response: {"message":"Smart update started - safety checks passed","status":"processing"}
```

---

## 🎯 Common Workflows

### 🔍 **Searching for Anime**

**Quick Search:**
```bash
# Start with basic search
curl "http://localhost:8000/api/search/?q=your_query&limit=5"
```

**Advanced Search:**
```bash
# Use semantic search for complex queries
curl -X POST http://localhost:8000/api/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "your complex query", "limit": 10}'
```

### ⚙️ **Managing Database Updates**

**Check System Status:**
```bash
# 1. Check system health
curl http://localhost:8000/health

# 2. Check database stats  
curl http://localhost:8000/stats

# 3. Check update status
curl http://localhost:8000/api/admin/update-status
```

**Smart Update Process:**
```bash
# 1. Check if it's safe to update
curl http://localhost:8000/api/admin/update-safety-check

# 2. If safe, perform smart update
curl -X POST http://localhost:8000/api/admin/smart-update

# 3. Monitor progress
curl http://localhost:8000/api/admin/update-status
```

**Manual Update Process:**
```bash
# 1. Check for available updates
curl -X POST http://localhost:8000/api/admin/check-updates

# 2. If updates available, run incremental update
curl -X POST http://localhost:8000/api/admin/update-incremental

# 3. Monitor progress
curl http://localhost:8000/api/admin/update-status
```

### 🌐 **Web Interface**

Visit **http://localhost:8000** in your browser for:
- Interactive API documentation (Swagger UI)
- Live endpoint testing
- Request/response examples
- Real-time API exploration

---

## 📋 Quick Reference Table

| Category | Endpoint | Method | Purpose | Use Case |
|----------|----------|--------|---------|----------|
| **🏥 System** | `/health` | GET | Health check | Monitoring, uptime checks |
| **🏥 System** | `/stats` | GET | Database stats | Check index size, last update |
| **🏥 System** | `/` | GET | API overview | Discover available endpoints |
| **🔍 Search** | `/api/search/` | GET | Basic search | Quick anime lookup |
| **🔍 Search** | `/api/search/semantic` | POST | Advanced search | Complex queries with filters |
| **🔍 Search** | `/api/search/similar/{id}` | GET | Find similar | Discover related anime |
| **🎯 Recommendations** | `/api/recommendations/similar/{id}` | GET | Get recommendations | Personalized suggestions |
| **⚙️ Admin** | `/api/admin/update-status` | GET | Update metadata | Check update history |
| **⚙️ Admin** | `/api/admin/check-updates` | POST | Check for updates | See if new data available |
| **⚙️ Admin** | `/api/admin/update-incremental` | POST | Smart update | Update only changed data |
| **⚙️ Admin** | `/api/admin/update-full` | POST | Full refresh | Emergency complete rebuild |
| **🧠 Smart** | `/api/admin/update-safety-check` | GET | Safety check | Verify safe to update |
| **🧠 Smart** | `/api/admin/smart-update` | POST | Intelligent update | Update only when safe |
| **🧠 Smart** | `/api/admin/smart-schedule-analysis` | GET | Schedule analysis | Optimize update timing |

### 🎯 Endpoint Categories Explained

- **🏥 System**: Basic health and information
- **🔍 Search**: Core anime discovery functionality  
- **🎯 Recommendations**: AI-powered suggestions
- **⚙️ Admin**: Manual database management
- **🧠 Smart**: Intelligent automation features

### ⭐ Most Commonly Used Endpoints

**For End Users:**
```bash
# Basic anime search
curl "http://localhost:8000/api/search/?q=your_query&limit=5"

# Check system status
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

**For Administrators:**
```bash
# Smart update (recommended)
curl -X POST http://localhost:8000/api/admin/smart-update

# Check update status
curl http://localhost:8000/api/admin/update-status

# Safety check before updating
curl http://localhost:8000/api/admin/update-safety-check
```

---

## 🎯 Response Formats

### Basic Search
```bash
# Search by genre
curl "http://localhost:8000/api/search/?q=action%20adventure&limit=5"

# Search by theme
curl "http://localhost:8000/api/search/?q=high%20school%20romance&limit=3"

# Search by mood
curl "http://localhost:8000/api/search/?q=dark%20psychological&limit=3"
```

### Advanced Search (POST)
```bash
curl -X POST http://localhost:8000/api/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mecha robots fighting in space",
    "limit": 5,
    "filters": {"type": "TV"}
  }'
```

### Response Format
```json
{
  "query": "action adventure",
  "results": [
    {
      "anime_id": "abc123def456",
      "title": "Attack on Titan",
      "synopsis": "Humanity fights for survival...",
      "type": "TV",
      "episodes": 25,
      "tags": ["action", "drama", "fantasy"],
      "studios": ["Studio Pierrot"],
      "picture": "https://cdn.myanimelist.net/...",
      "score": 0.95,
      "year": 2013,
      "season": "spring"
    }
  ],
  "total_results": 1,
  "processing_time_ms": 45.2
}
```

## 🔧 Configuration

### Environment Variables
```bash
QDRANT_URL=http://localhost:6333     # Qdrant server URL (use http://qdrant:6333 in Docker)
QDRANT_COLLECTION_NAME=anime_database     # Vector collection name
HOST=0.0.0.0                        # FastAPI bind host
PORT=8000                           # FastAPI port
DEBUG=True                          # Enable debug mode

# Additional settings
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173  # CORS origins
```

### Database Schema
Each anime entry contains:
- **Basic Info**: title, type, episodes, status, year, season
- **Metadata**: synopsis, tags, studios, producers, duration
- **References**: MyAnimeList ID, AniList ID, source URLs
- **Search Fields**: embedding_text, search_text (for vector search)
- **Quality Score**: Data completeness rating (0-1)

## 🎮 Interactive Testing

### Web Interface
Visit http://localhost:8000 in your browser for:
- FastAPI automatic documentation (Swagger UI)
- Interactive API testing
- Real-time endpoint exploration

### Command Line Testing
```bash
# Health check
curl http://localhost:8000/health

# Current statistics
curl http://localhost:8000/stats

# Search examples
curl "http://localhost:8000/api/search/?q=studio%20ghibli&limit=3"
curl "http://localhost:8000/api/search/?q=romantic%20comedy&limit=5"
curl "http://localhost:8000/api/search/?q=mecha%20robots&limit=3"
```

## 📊 Performance

- **Search Speed**: <200ms average response time
- **Vector Model**: hf/e5-base-v2 (multilingual embeddings) with model preloading
- **Index Size**: 38,894 anime entries when fully loaded
- **Memory Usage**: ~2GB for full dataset
- **Concurrency**: Supports multiple simultaneous searches
- **Indexing Speed**: ~3 docs/second with optimized batch processing (client_batch_size=32)
- **Full Indexing Time**: ~3-4 hours for complete database
- **Infrastructure**: Full Docker deployment with container networking optimization

## 🔄 Data Pipeline

### Source Data
- **Provider**: [anime-offline-database](https://github.com/manami-project/anime-offline-database)
- **Format**: Comprehensive JSON with cross-references
- **Coverage**: MyAnimeList, AniList, Kitsu, and 8 other sources
- **Updates**: Manual refresh via `python download_data.py`

### Processing Steps
1. **Download**: Fetch latest anime-offline-database JSON (~38,894 entries)
2. **Validation**: Parse and validate entries with Pydantic
3. **Enhancement**: Extract IDs, calculate quality scores (6+ seconds with async processing)
4. **Vectorization**: Create embeddings from title + synopsis + tags + studios
5. **Indexing**: Store in Qdrant with optimized batch processing:
   - `client_batch_size=32` for optimal performance
   - 100 documents per API batch with 3 concurrent workers
   - Vector embeddings with 384-dimensional cosine similarity
   - Background task networking fixes for containerized deployment

### Monitoring
```bash
# Check current indexing status
curl http://localhost:8000/stats

# Monitor Docker container logs
docker logs anime-mcp-server --follow

# Watch indexing progress (local development)
tail -f indexing.log

# Monitor server logs (local development)
tail -f server.log
```

## 🔮 Future Features

### Planned Enhancements
- [ ] Enhanced synopsis data from multiple sources
- [ ] Advanced recommendation algorithms  
- [ ] Model Context Protocol (MCP) integration for AI assistants
- [ ] Authentication and rate limiting
- [ ] Caching layer for improved performance
- [ ] Monitoring and alerting capabilities
- [ ] Horizontal scaling support

## 🛠️ Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run test suite
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/
isort src/

# Type checking
mypy src/

# Linting
flake8 src/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [anime-offline-database](https://github.com/manami-project/anime-offline-database) for comprehensive anime data
- [Qdrant](https://qdrant.ai/) for vector search capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation

## 📞 Support

- 🐛 **Issues**: Report bugs via GitHub Issues
- 💬 **Discussions**: Community discussions in GitHub Discussions  
- 📖 **Documentation**: Full docs at `/docs` endpoint when server is running
- 🔧 **Development**: See CLAUDE.md for detailed development guidance

---

**Status**: ✅ **Production Ready** - Fully functional anime search with vector database, optimized Docker deployment, and comprehensive API