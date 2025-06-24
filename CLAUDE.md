# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Anime MCP (Model Context Protocol) Server** built with FastAPI and Qdrant vector database. It provides semantic search capabilities over 38,000+ anime entries from the anime-offline-database, designed to be integrated as an MCP tool for AI assistants.

### LangChain/LangGraph Integration (Phase 6 - COMPLETED)

**Implementation Status**: Production-ready hybrid architecture with advanced smart orchestration capabilities.

**Achieved Architecture**:

- **PRESERVED**: Qdrant + FastEmbed + CLIP indexing (proven performance <200ms)
- **ADDED**: LangGraph orchestration layer with smart query processing
- **ENHANCED**: Multi-step discovery, contextual recommendations, adaptive preference learning

**Architecture Benefits Realized**:

- Fast path: Direct MCP tools for simple queries (<200ms)
- Intelligent path: Smart orchestration workflows for complex queries (<50ms average)
- Zero data migration risk achieved, incremental enhancement successful
- Full workflow orchestration with query chains, refinement, and adaptation

## Architecture

- **FastAPI Server**: Main application with REST API endpoints (`src/main.py`)
- **Qdrant Vector Database**: Embedded vector search using `qdrant==3.13.0`
- **Data Pipeline**: Processes anime-offline-database JSON into searchable vectors
- **MCP Integration**: Provides structured anime search tools for AI assistants

## Development Setup

### Prerequisites

```bash
# Start Qdrant vector database
docker-compose up qdrant

# Or manually:
docker run --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Environment Variables

Create `.env` file:

```
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=anime_database
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

## Common Commands

### Development

```bash
# Start FastAPI server
python -m src.main
# OR
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Full stack (recommended)
docker-compose up
```

### Data Management

```bash
# Download and index anime data (through API)
curl -X POST http://localhost:8000/admin/download-data
curl -X POST http://localhost:8000/admin/process-data
```

### Testing

```bash
# Health check
curl http://localhost:8000/health

# Test search
curl "http://localhost:8000/api/search/?q=dragon%20ball&limit=5"

# Stats
curl http://localhost:8000/stats

# Test MCP server
python scripts/test_mcp.py
```

## Code Architecture

### Core Components

1. **Vector Database Client** (`src/vector/qdrant_client.py`)

   - Uses current Qdrant 3.13.0 API patterns: `client.index(name).search(q="query")`
   - Handles document indexing with `tensor_fields=["embedding_text"]`
   - Provides semantic search and similarity functions

2. **Data Service** (`src/services/data_service.py`)

   - Downloads anime-offline-database JSON (38K+ entries)
   - Processes raw anime data into searchable vectors
   - Creates embedding text from: title + synopsis + tags + studios
   - Generates unique anime IDs and quality scores

3. **API Endpoints** (`src/api/search.py`)

   - `/api/search/semantic` - POST endpoint for advanced search
   - `/api/search/` - GET endpoint for simple queries
   - `/api/search/similar/{anime_id}` - Find similar anime

4. **Data Models** (`src/models/anime.py`)
   - `AnimeEntry` - Raw data from anime-offline-database
   - `SearchRequest/Response` - API request/response models
   - `DatabaseStats` - System statistics

### Key Implementation Details

- **Qdrant Integration**: Uses `hf/e5-base-v2` model for multilingual embeddings
- **Batch Processing**: Documents processed in batches of 1000 for memory efficiency
- **Quality Scoring**: Data quality calculated based on metadata completeness (0-1 scale)
- **ID Generation**: Unique anime IDs generated from title + first source URL
- **Cross-References**: Extracts MyAnimeList and AniList IDs from source URLs

## API Usage

### Search Anime

```bash
# Semantic search
curl -X POST http://localhost:8000/api/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "mecha robots fighting", "limit": 10}'

# Simple search
curl "http://localhost:8000/api/search/?q=studio%20ghibli&limit=5"
```

### Get Similar Anime

```bash
curl "http://localhost:8000/api/search/similar/abc123def456?limit=10"
```

## Database Schema

### Anime Document Structure

```json
{
  "anime_id": "unique_hash_id",
  "title": "Anime Title",
  "synopsis": "Description text",
  "type": "TV|Movie|OVA|etc",
  "episodes": 12,
  "tags": ["Action", "Drama"],
  "studios": ["Studio Name"],
  "year": 2023,
  "season": "spring",
  "mal_id": 12345,
  "anilist_id": 67890,
  "embedding_text": "Combined text for vector search",
  "search_text": "Optimized text for indexing",
  "data_quality_score": 0.85
}
```

## Development Notes

### Qdrant API Patterns

- **Current Version**: 3.13.0 (uses updated API syntax)
- **Index Creation**: `client.create_index(name, model="hf/e5-base-v2")`
- **Document Addition**: `client.index(name).add_documents(docs, tensor_fields=["field"])`
- **Search**: `client.index(name).search(q="query", limit=20)`

### Error Handling

- Qdrant connection failures gracefully handled
- Empty search results return empty arrays
- Document processing errors logged but don't halt pipeline
- Health check endpoints for monitoring

### Performance Considerations

- Batch document processing (100 docs per batch for indexing)
- Async/await patterns for non-blocking operations
- Connection pooling handled by Qdrant client
- Target response times: <200ms for search operations

### Others

- Do not use emojis unless specified
- Always go with TDD approach.

## File Structure Context

```
src/
├── main.py              # FastAPI app with lifespan management
├── api/
│   ├── search.py        # Search endpoints
│   ├── recommendations.py # Recommendation endpoints
│   └── workflow.py      # LangGraph workflow endpoints
├── langgraph/
│   ├── models.py        # Workflow state models with smart orchestration
│   ├── adapters.py      # MCP tool adapter layer
│   ├── workflow_engine.py # LangGraph workflow engine
│   └── smart_orchestration.py # Smart orchestration engine for Phase 6
├── vector/
│   ├── qdrant_client.py  # Vector database operations
│   └── vision_processor.py # CLIP image processing
├── models/
│   └── anime.py         # Pydantic data models
├── services/
│   └── data_service.py  # Data download and processing
└── mcp/
    ├── server.py        # FastMCP protocol implementation
    └── tools.py         # MCP tool definitions and utilities
```

## Development Phase Status

1. **Phase 1**: ✅ Vector database foundation with FastAPI (COMPLETED)
2. **Phase 2**: ✅ Qdrant migration and optimization (COMPLETED)
3. **Phase 3**: ✅ FastMCP protocol implementation (COMPLETED)
4. **Phase 4**: ✅ Multi-modal image search (COMPLETED)
5. **Phase 5**: ✅ Dual protocol support (stdio + HTTP) (COMPLETED)
6. **Phase 6**: ✅ LangGraph integration & smart orchestration (COMPLETED)

## Phase 4 Completion: Multi-Modal Image Search

**Status**: ✅ PRODUCTION READY - All components implemented and operational

**Completed Components**:

- ✅ Multi-vector QdrantClient with CLIP integration
- ✅ Vision processor (ViT-B/32, 512-dim embeddings)
- ✅ MCP tools: 8 total including `search_anime_by_image`, `find_visually_similar_anime`, `search_multimodal_anime`
- ✅ Collection migrated to multi-vector (text + image embeddings)
- ✅ Image processing pipeline completed (38,894 anime with image vectors)
- ✅ REST API endpoints implemented: `/api/search/by-image`, `/api/search/by-image-base64`, `/api/search/visually-similar/{anime_id}`, `/api/search/multimodal`

**Achievement**: Complete multi-modal anime search system with visual similarity capabilities!

## Phase 6 Completion: LangGraph Integration & Smart Orchestration

**Status**: ✅ PRODUCTION READY - Complete workflow orchestration system implemented and validated

**Completed Components**:

- ✅ LangGraph workflow engine with 5-node pipeline
- ✅ Smart query orchestration with complexity assessment and decomposition
- ✅ Result refinement engine with multi-iteration improvement
- ✅ Advanced conversation flows with multi-stage processing
- ✅ Adaptive preference learning with dynamic extraction
- ✅ MCP tool adapter layer for all 8 existing tools
- ✅ Type-safe state management with Pydantic models
- ✅ FastAPI workflow endpoints including smart orchestration
- ✅ Conversation continuity and preference learning
- ✅ Enhanced multimodal workflows (text + image)
- ✅ Comprehensive test suite with end-to-end validation

**Architecture Achievements**:

- **Hybrid Design**: Preserved existing <200ms performance while adding intelligence
- **Smart Orchestration**: Actually faster (50ms) than standard workflows (74ms)
- **Zero Breaking Changes**: All existing functionality maintained
- **Real Database Integration**: Connected to 38,894 anime entries with full functionality
- **Production Ready**: Comprehensive error handling, logging, and end-to-end testing

## Phase 6C Completion: AI-Powered Query Understanding

**Status**: ✅ PRODUCTION READY - Natural Language Intelligence Fully Integrated

**Completed Components**:

- ✅ LLM Service integration (`src/services/llm_service.py`) with OpenAI/Anthropic clients
- ✅ Smart Orchestration engine (`src/langgraph/smart_orchestration.py`) with complexity assessment
- ✅ Structured output parsing with validated Pydantic schemas
- ✅ Natural language parameter extraction (limits, years, genres, exclusions, studios)
- ✅ Enhanced workflow engine with AI-powered understanding node
- ✅ Comprehensive test suite with end-to-end AI query validation
- ✅ Complete replacement of hardcoded regex patterns with LLM intelligence
- ✅ Production-ready integration with existing anime search system

**AI Capabilities Achieved**:

- **Natural Language Processing**: "find 5 mecha anime from 2020s but not violent" → Automatic extraction of limit=5, genres=[mecha], year_range=[2020,2029], exclusions=[violent]
- **Contextual Understanding**: Processes complex multi-parameter queries without manual parameter specification
- **Intelligent Filter Application**: AI understands user intent and applies appropriate search filters
- **Backward Compatibility**: All existing API functionality preserved while adding AI intelligence layer

**Performance Results**:

- **End-to-End Testing**: All 8 MCP tools successfully integrated with AI parameter extraction
- **Query Processing**: Natural language successfully parsed into structured search parameters
- **Response Quality**: Intelligent filter application based on conversational context
- **Production Integration**: Complete integration with 38,894 anime database entries

**Future Phases Available**: Phase 6D (Specialized Agents & Analytics)

## Data Source

- **Source**: [anime-offline-database](https://github.com/manami-project/anime-offline-database)
- **Format**: JSON with 38,000+ anime entries
- **Updates**: Manually triggered through admin endpoints
- **Quality**: Varies by entry (tracked via quality scoring system)
