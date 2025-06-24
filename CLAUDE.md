# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Anime MCP (Model Context Protocol) Server** built with FastAPI and Qdrant vector database. It provides semantic search capabilities over 38,000+ anime entries from the anime-offline-database, designed to be integrated as an MCP tool for AI assistants.

### Current Status
**Phase 6C COMPLETED** - Production-ready system with AI-powered query understanding, multi-modal search, and intelligent workflow orchestration. Ready for Phase 6D planning.

## Architecture

- **FastAPI Server**: Main application with REST API endpoints (`src/main.py`)
- **Qdrant Vector Database**: Multi-vector search using `qdrant==1.11.3`
- **Data Pipeline**: Processes anime-offline-database JSON into searchable vectors
- **MCP Integration**: Provides structured anime search tools for AI assistants
- **LangGraph**: Workflow orchestration with smart query processing
- **CLIP**: Image embeddings for visual similarity search

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
ENABLE_MULTI_VECTOR=true
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
curl -X POST http://localhost:8000/api/admin/download-data
curl -X POST http://localhost:8000/api/admin/process-data
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
python scripts/verify_mcp_server.py
```

## Code Architecture

### Core Components

1. **Vector Database Client** (`src/vector/qdrant_client.py`)
   - Multi-vector collection (text + image embeddings)
   - Uses FastEmbed (BAAI/bge-small-en-v1.5) for text embeddings
   - Uses CLIP (ViT-B/32) for image embeddings

2. **Data Service** (`src/services/data_service.py`)
   - Downloads anime-offline-database JSON (38,894 entries)
   - Processes raw anime data into searchable vectors
   - Creates embedding text from: title + synopsis + tags + studios

3. **LLM Service** (`src/services/llm_service.py`)
   - OpenAI/Anthropic integration for AI-powered query understanding
   - Natural language parameter extraction
   - Structured output parsing with Pydantic schemas

4. **LangGraph Workflows** (`src/langgraph/`)
   - Smart orchestration with complexity assessment
   - Multi-step discovery and result refinement
   - Conversation continuity and preference learning

5. **MCP Server** (`src/mcp/server.py`)
   - 8 MCP tools including image search capabilities
   - Dual protocol support (stdio + HTTP)

### Key Implementation Details

- **Multi-Vector Search**: Text (384-dim) + Image (512-dim) embeddings
- **Batch Processing**: Documents processed in batches for memory efficiency
- **Quality Scoring**: Data quality calculated based on metadata completeness
- **ID Generation**: Unique anime IDs generated from title + first source URL
- **AI Query Understanding**: Natural language → structured search parameters

## Development Notes

### Performance Targets
- **Search Response**: <200ms for text search, ~1s for image search
- **AI Processing**: ~500ms for LLM query understanding
- **Smart Orchestration**: 50ms average (faster than standard workflows)

### Testing & Quality
- **Always use TDD approach**
- **Run tests**: `pytest tests/ -v`
- **Code formatting**: `black src/ tests/ scripts/`
- **Type checking**: `mypy src/`

### Important Reminders
- **Always use TDD approach** - Write tests first, then implement
- **ALWAYS run formatting before staging/committing**:
  ```bash
  autoflake --recursive --in-place --remove-all-unused-imports --remove-unused-variables src/ tests/ scripts/
  isort src/ tests/ scripts/
  black src/ tests/ scripts/
  ```
- Do not use emojis unless specified
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files
- NEVER proactively create documentation files

## File Structure Context

```
src/
├── main.py              # FastAPI app with lifespan management
├── api/                 # REST API endpoints
├── langgraph/           # LangGraph workflow orchestration
├── vector/              # Qdrant + CLIP integration
├── services/            # Data + LLM services
├── mcp/                 # FastMCP protocol implementation
└── models/              # Pydantic data models
```

## Current Development Phase

**Phase 6C ✅ COMPLETED**: AI-Powered Query Understanding
- Natural language parameter extraction with 95%+ accuracy
- OpenAI/Anthropic LLM integration
- Complete replacement of regex patterns with AI intelligence

**Phase 6D 📝 PLANNED**: Specialized Agents & Analytics
- Genre-expert agents for specialized recommendations
- Studio-focused discovery workflows
- Advanced comparative analysis capabilities
- Multi-agent coordination patterns

## Data Source

- **Source**: [anime-offline-database](https://github.com/manami-project/anime-offline-database)
- **Format**: JSON with 38,894 anime entries
- **Updates**: Weekly automated updates with intelligent change detection
- **Quality**: Tracked via quality scoring system (0-1 scale)