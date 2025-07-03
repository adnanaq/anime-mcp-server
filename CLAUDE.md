# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Anime MCP (Model Context Protocol) Server** built with FastAPI and Qdrant vector database. It provides semantic search capabilities over 38,000+ anime entries from the anime-offline-database, designed to be integrated as an MCP tool for AI assistants.

### Current Status

**Phase 7 COMPLETED** - Production-ready system with modern ReactAgent architecture, AI-powered query understanding, multi-modal search, and comprehensive code cleanup. System optimized and ready for specialized agents development.

## Documentation File Responsibilities

**IMPORTANT**: Keep clear separation between documentation files:

### `project_context.md` - STRATEGIC OVERVIEW

- **Purpose**: Condensed historical context, strategic roadmap, system overview
- **Content**: Completed phases (condensed), architecture, capabilities, vision
- **Updates**: Updated when major phases complete, strategic changes occur
- **Scope**: High-level context for understanding the project's current state and direction

**Rule**: If work is COMPLETED → Move condensed summary to `project_context.md`.

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
# Start FastAPI server (development)
python -m src.main
# OR with auto-reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Full stack with Docker (recommended)
docker-compose up

# Start only required services
docker-compose up -d qdrant
docker-compose up -d fastapi qdrant
```

### Data Management

```bash
# Full data update (downloads + processes + indexes)
curl -X POST http://localhost:8000/api/admin/update-full

# Individual steps
curl -X POST http://localhost:8000/api/admin/download-data
curl -X POST http://localhost:8000/api/admin/process-data

# Check update status
curl http://localhost:8000/api/admin/update-status
```

### Testing & Verification

```bash
# System health
curl http://localhost:8000/health

# Database stats  
curl http://localhost:8000/stats

# Basic search test
curl "http://localhost:8000/api/search/?q=dragon%20ball&limit=5"

# MCP server verification (comprehensive)
python scripts/verify_mcp_server.py

# Run test suite
pytest tests/ -v
pytest -m unit -v        # Unit tests only
pytest -m integration -v # Integration tests only
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
   - 7 MCP tools including image search capabilities
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
- **Run all tests**: `pytest tests/ -v`
- **Run single test**: `pytest tests/unit/services/test_data_service.py::test_specific_function -v`
- **Run test by marker**: `pytest -m unit -v` or `pytest -m integration -v`
- **Code formatting**: `black src/ tests/ scripts/`
- **Type checking**: `mypy src/`
- **Coverage report**: `pytest tests/ --cov=src --cov-report=html`

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

## File Structure Context & Key Files

```
src/
├── main.py              # FastAPI app with lifespan management
├── config.py            # Centralized configuration with environment settings
├── api/                 # REST API endpoints
│   ├── search.py        # Search endpoints (/api/search/*)
│   ├── admin.py         # Admin endpoints (/api/admin/*)
│   └── workflow.py      # LangGraph workflow endpoints (/api/workflow/*)
├── langgraph/           # LangGraph workflow orchestration
│   ├── langchain_tools.py    # LangChain tool creation & ToolNode workflow
│   └── react_agent_workflow.py    # Main ReactAgent workflow engine
├── vector/              # Qdrant + CLIP integration
│   ├── qdrant_client.py      # Multi-vector database operations
│   └── vision_processor.py  # CLIP image processing
├── services/            # Core business logic
│   ├── data_service.py       # Data processing & indexing pipeline
│   ├── smart_scheduler.py    # Rate limiting & service coordination
│   └── update_service.py     # Database update management
├── mcp/                 # FastMCP protocol implementation
│   ├── server.py             # FastMCP server with 7 tools
│   └── fastmcp_client_adapter.py  # MCP client integration
├── models/              # Pydantic data models
│   ├── anime.py              # Core anime data models
│   └── universal_anime.py    # Universal schema for platform mapping
└── integrations/        # External platform integrations (9 platforms)
    ├── clients/              # HTTP clients for each platform
    ├── mappers/              # Data transformation layers
    ├── scrapers/             # Web scraping components
    └── rate_limiting/        # Multi-tier rate limiting system
```

**Critical Files for Understanding:**
- `src/config.py` - All environment variables and settings
- `src/services/data_service.py` - Core data processing pipeline
- `src/vector/qdrant_client.py` - Vector database operations
- `src/langgraph/react_agent_workflow.py` - AI workflow orchestration

## Current Development Phase

**Phase 6C ✅ COMPLETED**: AI-Powered Query Understanding

- Natural language parameter extraction with 95%+ accuracy
- OpenAI/Anthropic LLM integration
- Complete replacement of regex patterns with AI intelligence

**Phase 7 ✅ COMPLETED**: Production-Ready ReactAgent Architecture

- Comprehensive ReactAgent workflow with conversational memory
- Multi-platform data integration (9 anime platforms)
- Advanced rate limiting system with multi-tier architecture
- Complete test coverage with 60+ passing tests
- Production-ready system optimization

**Current System Status**: Fully operational production system ready for specialized agent development.

## Data Source

- **Source**: [anime-offline-database](https://github.com/manami-project/anime-offline-database)
- **Format**: JSON with 38,894 anime entries
- **Updates**: Weekly automated updates with intelligent change detection
- **Quality**: Tracked via quality scoring system (0-1 scale)

