# CLAUDE.md

This file provides guidance to Claude Code when working with this anime MCP server codebase.

## 🎯 CRITICAL CONTEXT (Read First)

**Project**: Anime MCP Server - FastAPI + Qdrant vector database providing semantic search over 38,000+ anime entries
**Purpose**: MCP tool integration for AI assistants with advanced search capabilities
**Key Tech**: FastAPI, Qdrant, LangGraph, CLIP, FastEmbed, Pydantic

### Essential Pre-Work Checklist (MANDATORY RULE COMPLIANCE)

**Rule Loading (ALWAYS FIRST):**
- [ ] Read `rules/rules.mdc` (base improvement rules)
- [ ] Read `rules/memory.mdc` (memory management workflow)
- [ ] Read `rules/plan.mdc` (if planning) OR `rules/implement.mdc` (if coding)

**Memory Hierarchy (Follow rules/memory.mdc sequence):**
- [ ] Read `docs/product_requirement_docs.md` (foundation)
- [ ] Read `docs/architecture.md` (system design)
- [ ] Read `docs/technical.md` (implementation details)
- [ ] Read `tasks/tasks_plan.md` (project progress)
- [ ] Read `tasks/active_context.md` (current state)

**Current State Assessment:**
- [ ] Use `TodoRead` to check current tasks
- [ ] Identify current MODE: PLAN (architect) vs ACT (code)
- [ ] Get required code context from `src/` if needed

**Development Environment:**
- [ ] Always use `source venv/bin/activate` before Python commands
- [ ] Ensure Qdrant is running: `docker-compose up -d qdrant`

## 🏗️ ARCHITECTURE OVERVIEW

### Core Stack

```
FastAPI Server (src/main.py)
├── Qdrant Vector DB (multi-vector: text + image)
├── LangGraph Workflows (smart orchestration)
├── MCP Server (modern_server.py + platform-specific tools)
└── External APIs (9 platforms)
```

### Critical Files to Know

- `src/config.py` - All environment variables and settings
- `src/services/data_service.py` - Core data processing pipeline
- `src/vector/qdrant_client.py` - Vector database operations
- `src/langgraph/react_agent_workflow.py` - AI workflow orchestration
- `src/anime_mcp/modern_server.py` - Modern MCP server with LangGraph workflows
- `src/anime_mcp/server.py` - Core MCP server implementation
- `src/anime_mcp/tools/` - Platform-specific MCP tools

### Data Flow

1. **Ingest**: anime-offline-database JSON → Data Service
2. **Process**: Text + image embeddings → Qdrant multi-vector
3. **Query**: Natural language → LangGraph → Structured search
4. **Results**: Ranked anime with quality scores

## 🚀 QUICK START

### Development Setup

```bash
# Start services
docker-compose up -d qdrant
source venv/bin/activate
pip install -r requirements.txt

# Start server
python -m src.main
# OR with auto-reload
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Essential Commands

```bash
# Testing
pytest tests/ -v                    # All tests
pytest -m unit -v                   # Unit tests only
pytest -m integration -v            # Integration tests only

# Data management
curl -X POST http://localhost:8000/api/admin/update-full
curl http://localhost:8000/stats

# MCP server modes (choose based on use case)
python -m src.anime_mcp.modern_server               # Modern workflow server (stdio mode)
python -m src.anime_mcp.modern_server --mode sse --port 8001  # Modern server (SSE mode)
python -m src.anime_mcp.server                     # Core MCP server (stdio mode)
python -m src.anime_mcp.server --mode sse --port 8001       # Core server (SSE mode)
```

## 🔗 MCP PROTOCOL INTEGRATION

### MCP Servers (Two Available)

**Core Server** (`src.anime_mcp.server`) - **8 core tools + platform tools**
- **Use for**: Basic search, image search, detailed anime data
- **Transport modes**: stdio, http, sse, streamable
- **Tools**: search_anime, get_anime_details, find_similar_anime, search_anime_by_image, etc.

**Modern Server** (`src.anime_mcp.modern_server`) - **4 workflow tools + LangGraph**
- **Use for**: AI-powered discovery, multi-agent workflows, complex queries
- **Transport modes**: stdio, sse
- **Tools**: discover_anime, get_currently_airing_anime, find_similar_anime_workflow

### Quick MCP Commands

```bash
# Local development (stdio mode) - recommended for AI assistants
python -m src.anime_mcp.modern_server           # Workflow tools + LangGraph
python -m src.anime_mcp.server                 # Core tools + platform tools

# Web/remote clients - core server only (modern server only supports sse)
python -m src.anime_mcp.server --mode http --port 8001     # HTTP transport
python -m src.anime_mcp.server --mode sse --port 8001      # Server-Sent Events
python -m src.anime_mcp.server --mode streamable --port 8001  # Streamable HTTP

# Test MCP functionality
python scripts/verify_mcp_server.py            # Comprehensive testing
```

### MCP Development Notes

- **Default choice**: `modern_server` for AI assistants, `core_server` for basic tools/testing
- **Protocol guide**: `stdio` (local), `http` (testing), `sse` (web clients), `streamable` (advanced)
- **Testing**: Always run `verify_mcp_server.py` after MCP changes
- **Tool count**: 31 total (8 core + 4 workflow + 14 platform + 5 enrichment)

## 📁 FILE STRUCTURE

```
src/
├── main.py                 # FastAPI app entry point
├── config.py              # Environment settings
├── api/                   # REST endpoints
│   ├── search.py          # Search endpoints
│   ├── admin.py           # Admin endpoints
│   └── workflow.py        # LangGraph workflow endpoints
├── langgraph/             # AI workflow orchestration
│   ├── langchain_tools.py        # LangChain tool creation
│   └── react_agent_workflow.py  # Main ReactAgent workflow
├── vector/                # Qdrant + CLIP integration
│   ├── qdrant_client.py          # Multi-vector database ops
│   └── vision_processor.py      # CLIP image processing
├── services/              # Core business logic
│   ├── data_service.py           # Data processing pipeline
│   ├── smart_scheduler.py        # Rate limiting coordination
│   └── update_service.py         # Database update management
├── anime_mcp/             # MCP implementation
│   ├── modern_server.py          # LangGraph workflow MCP server
│   ├── server.py                 # Core MCP server
│   ├── handlers/                 # MCP request handlers
│   └── tools/                    # Platform-specific MCP tools
├── models/                # Pydantic data models
│   ├── anime.py                  # Core anime models
│   └── universal_anime.py        # Universal schema mapping
└── integrations/          # External platform integrations
    ├── clients/                  # HTTP clients (9 platforms)
    ├── mappers/                  # Data transformation
    ├── scrapers/                 # Web scraping
    └── rate_limiting/            # Multi-tier rate limiting
```

## 💻 DEVELOPMENT RULES

### 🔄 SYSTEMATIC CODE PROTOCOL (Implementation Tasks)

**Step 1: ANALYZE CODE**
- Dependency analysis: Which components affected?
- Flow analysis: End-to-end impact assessment
- Document dependencies thoroughly

**Step 2: PLAN CODE**
- Use CLARIFICATION process for unclear requirements
- Provide STRUCTURED PROPOSALS: files changed, why, impacts, tradeoffs
- Present REASONING for validation

**Step 3: MAKE CHANGES**
- INCREMENTAL ROLLOUTS: One logical change at a time
- SIMULATION TESTING: Dry runs before implementation
- Architecture preservation: Integrate with existing structure

**Step 4: TESTING**
- Write tests for new functionality
- Run dependency-based testing
- NO BREAKAGE ASSERTION: Verify no regressions

**Step 5: LOOP** - Implement all changes systematically
**Step 6: OPTIMIZE** - After all changes tested and verified

### 🎯 Task Execution Rules

**BEFORE Every Task:**
- **Implementation**: Read docs/ + tasks/ + get src/ context + validate architecture
- **Planning**: Read docs/ + tasks/ + get src/ context + deeper analysis

**AFTER Every Task:**
- **Implementation**: Update src/ + docs/ + tasks/ + complete testing + update lessons learned
- **Planning**: Update docs/ + tasks/ + plans and context

**Critical Rules**: 
- "Stop only when you're done till successfully testing, not before"
- Always validate changes against docs/architecture.md constraints

### Code Quality Standards

- **File Size Limit**: Never exceed 500 lines per file (from rules/implement.mdc)
- **Module Organization**: Break into atomic parts (modularity principle)
- **Testing**: MANDATORY for any functionality (proactive_testing principle)
- **Systematic Sequence**: Complete one step before starting another
- **Code Preservation**: Don't modify working components without necessity

### Python Conventions

- Use type hints and Google-style docstrings
- Use Pydantic for data validation
- Prefer relative imports within packages
- Use `.venv` and `load_env()` for environment variables

### Testing Requirements (Critical Rule Compliance)

- **Test Structure**: Mirror main app structure in `/tests`
- **Mandatory Testing**: Any functionality MUST have tests (rules/implement.mdc)
- **Test Types**: Unit tests for components, integration for workflows
- **Coverage Target**: >80% (currently at 6% - major violation)
- **Test Before Complete**: Never finish implementation without testing
- **Markers**: Use `pytest -m unit` or `pytest -m integration`

### 📋 Task Completion Validation (After Every Task)

- [ ] All affected code updated in src/
- [ ] Documentation updated in docs/ and tasks/
- [ ] Testing completed (for implementation tasks)
- [ ] TodoWrite updated with progress
- [ ] Memory files updated if significant changes made
- [ ] rules/lessons-learned.mdc updated if new patterns discovered
- [ ] rules/error-documentation.mdc updated if errors resolved

### 🐛 Debug Protocol (When Stuck)

**DIAGNOSE:**
- Gather error messages, logs, behavioral symptoms
- Add relevant context from files
- Retrieve project architecture, plan, current working task from memory files

**DEBUGGING SEQUENCE:**
1. Add context using DIAGNOSE
2. Explain OBSERVATIONS and REASONINGS 
3. Use STEP BY STEP REASONING for all possible causes
4. Look for similar patterns in rules/error-documentation.mdc
5. Present fix using REASONING PRESENTATION
6. Implement using SYSTEMATIC CODE PROTOCOL
7. Document solution in rules/error-documentation.mdc

## 🎛️ ENVIRONMENT SETUP

### Required .env Variables

```
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=anime_database
HOST=0.0.0.0
PORT=8000
DEBUG=True
ENABLE_MULTI_VECTOR=true
```

### Docker Services

```bash
# Full stack
docker-compose up

# Qdrant only
docker-compose up -d qdrant

# Manual Qdrant
docker run --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

## 📊 KEY IMPLEMENTATION DETAILS

### Vector Database

- **Text Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5, 384-dim)
- **Image Embeddings**: CLIP (ViT-B/32, 512-dim)
- **Multi-Vector**: Combined text + image search capabilities
- **Batch Processing**: Memory-efficient document processing

### Data Processing

- **Source**: anime-offline-database (38,894 entries)
- **ID Generation**: title + first source URL
- **Quality Scoring**: Metadata completeness (0-1 scale)
- **Embedding Text**: title + synopsis + tags + studios

### AI Integration

- **LangGraph**: Smart orchestration with complexity assessment
- **Query Understanding**: Natural language → structured parameters
- **Multi-step Discovery**: Result refinement and preference learning


## 🚨 AI BEHAVIOR RULES

### Safety & Accuracy

- Never assume missing context - ask questions if uncertain
- Only use verified Python packages - no hallucination
- Confirm file paths and module names exist before referencing
- Never delete/overwrite code unless explicitly instructed

### Documentation Updates

- Update README.md when features/dependencies change
- Comment non-obvious code for mid-level developer understanding
- Add `# Reason:` comments for complex logic explaining why, not what

## 🔍 VERIFICATION COMMANDS

### Health Checks

```bash
curl http://localhost:8000/health         # System health
curl http://localhost:8000/stats          # Database stats
curl "http://localhost:8000/api/search/?q=dragon%20ball&limit=5"  # Search test
python scripts/verify_mcp_server.py      # MCP server verification
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov-report=html  # Coverage report
mypy src/                                   # Type checking
```
