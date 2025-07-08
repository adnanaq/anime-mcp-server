# CLAUDE.md

## SESSION START PROTOCOL

### MANDATORY RULE LOADING (ALWAYS FIRST)

- Read `Rules/memory.md` (memory management workflow)
- Read `Rules/plan.md` (if planning) OR `Rules/implement.md` (if coding)

### MEMORY HIERARCHY (Load in sequence)

- Read `docs/product_requirement_docs.md` (foundation)
- Read `docs/architecture.md` (system design)
- Read `docs/technical.md` (implementation details)
- Read `tasks/tasks_plan.md` (project progress)
- Read `tasks/active_context.md` (current state)

### ENVIRONMENT SETUP

```bash
source venv/bin/activate                    # Always activate venv first
docker compose up -d qdrant                 # Start Qdrant vector DB
```

### EXECUTION MODES

**PLAN MODE**: Strategy→Present→Document  
**ACT MODE**: 6-Step Protocol (Analyze→Plan→Change→Test→Loop→Optimize)

### CRITICAL RULES

- **File Size Limit**: Never exceed 500 lines per file
- **Testing**: MANDATORY for any functionality
- **Code Preservation**: Don't modify working components without necessity
- **Stop Rule**: "Stop only when you're done till successfully testing, not before"

### ESSENTIAL COMMANDS

```bash
# Testing
pytest tests/ -v                           # All tests
pytest -m unit -v                          # Unit tests only
pytest -m integration -v                   # Integration tests only

# MCP Servers
python -m src.anime_mcp.modern_server      # Modern workflow server (stdio)
python -m src.anime_mcp.server             # Core MCP server (stdio)
python scripts/verify_mcp_server.py        # MCP verification

# Health Checks
curl http://localhost:8000/health          # System health
curl http://localhost:8000/stats           # Database stats
```

### TASK COMPLETION VALIDATION (After Every Task)

- [ ] All affected code updated in src/
- [ ] Documentation updated in docs/ and tasks/
- [ ] Testing completed (for implementation tasks)
- [ ] TodoWrite updated with progress
- [ ] Memory files updated if significant changes made

---

## PROJECT CONTEXT

**Project**: Anime MCP Server - FastAPI + Qdrant vector database providing semantic search over 38,000+ anime entries  
**Purpose**: MCP tool integration for AI assistants with advanced search capabilities  
**Key Tech**: FastAPI, Qdrant, LangGraph, CLIP, FastEmbed, Pydantic

## ARCHITECTURE OVERVIEW

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

## SERVER STARTUP

```bash
# Start FastAPI server
python -m src.main                                 # Basic startup
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000  # With auto-reload

# Data management
curl -X POST http://localhost:8000/api/admin/update-full   # Update database
```

## MCP PROTOCOL INTEGRATION

### MCP Servers (Two Available)

**Core Server** (`src.anime_mcp.server`) - **8 core tools + platform tools**

- **Use for**: Basic search, image search, detailed anime data
- **Transport modes**: stdio, http, sse, streamable
- **Tools**: search_anime, get_anime_details, find_similar_anime, search_anime_by_image, etc.

**Modern Server** (`src.anime_mcp.modern_server`) - **4 workflow tools + LangGraph**

- **Use for**: AI-powered discovery, multi-agent workflows, complex queries
- **Transport modes**: stdio, sse
- **Tools**: discover_anime, get_currently_airing_anime, find_similar_anime_workflow

### MCP Transport Modes

```bash
# Web/remote clients - core server only (modern server only supports sse)
python -m src.anime_mcp.server --mode http --port 8001     # HTTP transport
python -m src.anime_mcp.server --mode sse --port 8001      # Server-Sent Events
python -m src.anime_mcp.server --mode streamable --port 8001  # Streamable HTTP
```

### MCP Development Notes

- **Default choice**: `modern_server` for AI assistants, `core_server` for basic tools/testing
- **Protocol guide**: `stdio` (local), `http` (testing), `sse` (web clients), `streamable` (advanced)
- **Testing**: Always run `verify_mcp_server.py` after MCP changes
- **Tool count**: 31 total (8 core + 4 workflow + 14 platform + 5 enrichment)

## FILE STRUCTURE

```
src/
├── main.py                 # FastAPI app entry point
├── config.py              # Environment settings
├── api/                   # REST endpoints
│   ├── search.py          # Search endpoints
│   ├── admin.py           # Admin endpoints
│   └── query.py           # Universal query endpoint
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

## DEVELOPMENT RULES

### SYSTEMATIC CODE PROTOCOL (Implementation Tasks)

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

### Task Execution Rules

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

- **File Size Limit**: Never exceed 500 lines per file (from Rules/implement.md)
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
- **Mandatory Testing**: Any functionality MUST have tests (Rules/implement.md)
- **Test Types**: Unit tests for components, integration for workflows
- **Coverage Target**: >80%
- **Test Before Complete**: Never finish implementation without testing
- **Markers**: Use `pytest -m unit` or `pytest -m integration`

### Debug Protocol (When Stuck)

**DIAGNOSE:**

- Gather error messages, logs, behavioral symptoms
- Add relevant context from files
- Retrieve project architecture, plan, current working task from memory files

**DEBUGGING SEQUENCE:**

1. Add context using DIAGNOSE defined in Rules/debug.md
2. Explain OBSERVATIONS and REASONINGS
3. Use STEP BY STEP REASONING for all possible causes, defined in Rules/plan.md
4. Look for similar patterns in Rules/error-documentation.md
5. Present fix using REASONING PRESENTATION, defined in Rules/plan.md
6. Implement using SYSTEMATIC CODE PROTOCOL, defined in Rules/implement.md
7. Document solution in Rules/error-documentation.md

## ENVIRONMENT SETUP

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
docker compose up -d qdrant

# Manual Qdrant
docker run --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

## KEY IMPLEMENTATION DETAILS

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

## AI BEHAVIOR RULES

### Safety & Accuracy

- Never assume missing context - ask questions if uncertain
- Only use verified Python packages - no hallucination
- Confirm file paths and module names exist before referencing
- Never delete/overwrite code unless explicitly instructed

### Documentation Updates

- Update README.md when features/dependencies change
- Comment non-obvious code for mid-level developer understanding
- Add `# Reason:` comments for complex logic explaining why, not what

## VERIFICATION & TESTING

```bash
# Quick verification
curl "http://localhost:8000/api/search/?q=dragon%20ball&limit=5"  # Search test

# Comprehensive testing
pytest tests/ --cov=src --cov-report=html  # Coverage report
mypy src/                                   # Type checking
```
