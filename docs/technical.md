# Technical Documentation
# Anime MCP Server

## 1. Technology Stack

### 1.1 Core Technologies
- **Python 3.11+**: Primary development language
- **FastAPI**: High-performance REST API framework with automatic OpenAPI documentation
- **Uvicorn**: ASGI server for production deployment
- **Pydantic v2**: Data validation, settings management, and type safety

### 1.2 Vector Database & AI Stack
- **Qdrant**: Vector database for semantic search with multi-vector support
- **FastEmbed**: Efficient text embeddings (BAAI/bge-small-en-v1.5, 384-dim)
- **OpenAI CLIP**: Image embeddings for visual similarity (ViT-B/32, 512-dim) 
- **PyTorch**: ML framework for CLIP operations

### 1.3 AI Workflow & LLM Integration
- **LangGraph**: Workflow orchestration and multi-agent systems
- **LangChain**: LLM framework and tool integration
- **OpenAI/Anthropic APIs**: LLM providers for intelligent query processing
- **langgraph-swarm**: Multi-agent coordination (dependency resolved)
- **Stateful Routing Memory**: Conversation memory and context learning system

### 1.4 MCP Protocol
- **FastMCP**: Modern MCP server implementation with multiple transport modes
- **stdio/http/sse/streamable**: Transport protocols for different client types

## 2. Development Environment

### 2.1 Setup Requirements
- **Python 3.11+** with virtual environment
- **Docker & Docker Compose** for Qdrant vector database
- **Environment Variables**: Configure via `.env` file (see `.env.example`)

### 2.2 Key Configuration
- **Settings Management**: Centralized Pydantic settings with environment variable support
- **Vector Database**: Qdrant collection with 384-dim text + 512-dim image vectors
- **API Configuration**: FastAPI with CORS, automatic docs, and validation
- **Multi-Modal**: Optional image processing with CLIP embeddings

## 3. Key Technical Decisions

### 3.1 Vector Search Architecture  
- **Multi-Vector Design**: Text + image embeddings in single collection for hybrid search
- **Embedding Models**: FastEmbed for efficiency, CLIP for visual similarity
- **Distance Metric**: Cosine similarity for both text and image vectors
- **Batch Processing**: 100-point batches for optimal memory usage

### 3.2 MCP Server Design (Modernized 2025)
- **Dual Server Architecture**: Core server (basic tools) + Modern server (LangGraph workflows) 
- **31 MCP Tools**: 4-tier architecture with structured response models
- **Transport Flexibility**: Multiple protocols (stdio/http/sse/streamable) for different clients
- **Tool Integration**: FastMCP wrapper functions for LangChain compatibility
- **Dependency Isolation**: Core server avoids LangGraph dependencies for stability
- **Modern LLM Architecture**: 90% complexity reduction with direct tool calls

### 3.3 LangGraph Workflow Integration
- **ReactAgent Pattern**: create_react_agent for native LLM tool calling
- **Memory Persistence**: MemorySaver for conversation continuity  
- **Tool Wrapper Strategy**: FastMCP tools wrapped as LangChain-compatible functions
- **Query Processing**: AI-powered intent extraction and parameter routing

### 3.4 External Platform Integration (Modernized 2025)
- **9 Anime Platforms**: Mix of APIs (MAL, AniList, Jikan) and scraping targets
- **Rate Limiting**: Platform-specific limits with fallback strategies
- **Direct Tool Calls**: Structured response models replace Universal parameter system
- **4-Tier Architecture**: Progressive complexity (Basic → Standard → Detailed → Comprehensive)
- **Error Handling**: Circuit breakers and graceful degradation

### 3.5 Stateful Routing Memory System (Task #89 Implementation)
- **Memory Architecture**: Multi-tier memory system with conversation persistence
- **Pattern Learning**: Query pattern recognition and similarity matching
- **Agent Optimization**: Handoff sequence learning for improved coordination
- **User Preferences**: Personalized routing based on interaction history
- **Performance Caching**: 50%+ response time improvement via optimal route selection
- **Memory Management**: Automatic cleanup with configurable limits (10,000 patterns, 1,000 sessions)

### 3.6 AI-Powered Data Standardization System
- **Multi-Source Integration**: AI intelligently fetches and merges data from 4 anime platforms
- **Schema Standardization**: AI normalizes different API schemas into uniform properties
- **Character Deduplication**: AI identifies and merges same characters across platforms using fuzzy matching
- **Statistics Harmonization**: AI converts different rating scales and field names to consistent format
- **Fallback Strategy**: Single-source fields use primary + fallback, multi-source leaves empty if unavailable
- **Schema Compliance**: Output strictly matches enhanced_anime_schema_example.json structure

## 4. Design Patterns in Use

### 4.1 Service Layer Patterns
- **Dependency Injection**: Settings and clients injected via configuration
- **Repository Pattern**: Data service abstracts vector database operations
- **Factory Pattern**: Client creation with automatic configuration
- **Adapter Pattern**: Platform-specific clients behind unified interface

### 4.2 Async & Concurrency Patterns
- **Async/Await**: Full async stack from FastAPI to database operations
- **Connection Pooling**: Reused connections for Qdrant and HTTP clients
- **Circuit Breaker**: Automatic failover for external API failures
- **Rate Limiting**: Token bucket algorithm for platform API compliance

### 4.3 Error Handling Patterns
- **Custom Exception Hierarchy**: Structured errors for different failure types
- **Graceful Degradation**: Fallback strategies maintain service availability
- **Correlation Tracking**: Request correlation IDs for debugging and monitoring
- **Structured Logging**: JSON-formatted logs with contextual information

## 5. Technical Constraints

### 5.1 Performance Constraints
- **Vector Search**: Target <200ms for semantic search operations
- **API Response**: Target <500ms for enhanced queries with external APIs
- **Memory Usage**: Embedding models cached but size-limited
- **Concurrent Users**: Designed for 100+ concurrent connections

### 5.2 External API Constraints
- **Rate Limits**: Platform-specific (MAL: 2/sec, AniList: 90/min, Jikan: 1/sec)
- **Authentication**: OAuth2 required for MAL, optional for AniList
- **Data Freshness**: Cached data with TTL to balance performance vs freshness
- **Fallback Strategy**: Multiple data sources for resilience

### 5.3 Deployment Constraints
- **Container Requirements**: Docker with sufficient memory for embeddings
- **GPU Optional**: CLIP processing benefits from GPU but not required  
- **Network Requirements**: Outbound access to anime platform APIs
- **Storage Requirements**: Qdrant vector storage scales with anime database size

## 6. Development Setup

### 6.1 Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start services
docker compose up -d qdrant

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 6.2 Essential Commands
```bash
# Development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# MCP servers
python -m src.anime_mcp.modern_server          # LangGraph workflows
python -m src.anime_mcp.server                 # Core tools

# Testing
pytest tests/ -v                               # All tests
pytest -m unit -v                              # Unit tests only

# Data management  
curl -X POST http://localhost:8000/api/admin/update-full
```

### 6.3 API Keys Required
- **OPENAI_API_KEY**: For LangGraph LLM operations (optional)
- **ANTHROPIC_API_KEY**: Alternative LLM provider (optional)
- **MAL_CLIENT_ID/SECRET**: MyAnimeList API access (optional)
- **Other Platform APIs**: See platform documentation for requirements

## 7. AI-Powered Data Standardization Architecture

### 7.1 Source Mapping Strategy
The AI enrichment agent uses a hardcoded source mapping based on the `_sources` schema to determine which APIs to call for each property:

```python
# Single-source fields (primary + fallback)
SINGLE_SOURCE_MAPPING = {
    "genres": {"primary": "anilist", "fallback": "jikan"},
    "demographics": {"primary": "jikan", "fallback": "anilist"},
    "themes": {"primary": "anilist", "fallback": "jikan"},
    "source_material": {"primary": "jikan", "fallback": "anilist"},
    "rating": {"primary": "kitsu", "fallback": "jikan"},
    "aired_dates": {"primary": "jikan", "fallback": "anilist"},
    "broadcast": {"primary": "animeschedule", "fallback": "jikan"},
    "staff": {"primary": "jikan", "fallback": "anilist"},
    "opening_themes": {"primary": "jikan", "fallback": "anilist"},
    "ending_themes": {"primary": "jikan", "fallback": "anilist"},
    "episode_details": {"primary": "jikan", "fallback": "anilist"},
    "relations": {"primary": "jikan", "fallback": "anilist"},
    "awards": {"primary": "jikan", "fallback": "anilist"},
    "external_links": {"primary": "animeschedule", "fallback": "jikan"},
    "licensors": {"primary": "animeschedule", "fallback": "jikan"},
    "streaming_licenses": {"primary": "animeschedule", "fallback": "jikan"}
}

# Multi-source fields (fetch from all applicable sources)
MULTI_SOURCE_MAPPING = {
    "statistics": ["jikan", "anilist", "kitsu", "animeschedule"],
    "images": ["jikan", "anilist", "kitsu", "animeschedule"],
    "characters": ["jikan", "anilist"],  # Special character merging
    "streaming_info": ["animeschedule", "kitsu"],
    "popularity_trends": ["jikan", "anilist", "kitsu"]
}
```

### 7.2 AI Standardization Process

**Statistics Harmonization**: AI maps different field names and scales to uniform properties:
```python
# Example: Converting AniList data to standardized format
{
    "anilist": {
        "score": 8.2,            # AI converts averageScore 82 → 8.2 (10-point scale)
        "scored_by": null,       # Not available in AniList
        "rank": null,            # Not available in AniList  
        "popularity_rank": null, # Different concept in AniList
        "members": 147329,       # AI maps "popularity" field to members
        "favorites": 53821       # AI maps "favourites" to favorites
    }
}
```

**Character Deduplication Logic**: AI identifies same characters across platforms:
- **Name Matching**: Fuzzy matching on character names and variations
- **Role Consistency**: Ensures consistent character roles across sources
- **Image Collection**: Gathers character images from all available sources
- **Voice Actor Merging**: Deduplicates voice actors across different sources

### 7.3 Multi-Source Data Handling

**Key Principles**:
- **Single-Source Fields**: Use primary source, fallback to alternative if primary fails
- **Multi-Source Fields**: Fetch from ALL relevant APIs, include only available data
- **No Cross-Substitution**: For multi-source fields, don't substitute missing data with alternative sources
- **Empty Handling**: Leave platform entries empty if source doesn't have the data

**Example Multi-Source Output**:
```json
{
    "statistics": {
        "mal": {"score": 8.43, "scored_by": 2251158, "rank": 68},
        "anilist": {"score": 8.2, "members": 147329, "favorites": 53821},
        "kitsu": {},  // Empty - Kitsu didn't have relevant data
        "animeschedule": {"average_score": 90.66, "rating_count": 338}
    }
}
```

### 7.4 Schema Compliance & Validation

**Output Requirements**:
- **Exact Schema Match**: Output must match `enhanced_anime_schema_example.json` structure
- **Property Preservation**: All properties present even if empty
- **Type Consistency**: Maintain correct data types for all fields
- **Nested Structure**: Preserve complex nested objects and arrays

**Validation Process**:
1. **AI Generates Structured Response**: Based on comprehensive prompt with schema definition
2. **Schema Validation**: Pydantic models ensure type safety and required fields
3. **Completeness Check**: Verify all expected properties are present
4. **Error Recovery**: Graceful handling of malformed AI responses

### 7.5 Performance & Efficiency

**API Call Optimization**:
- **Targeted Fetching**: Only call APIs that provide specific required data
- **Existing Infrastructure**: Leverage current rate limiting and pagination logic
- **Character Chunking**: Maintain existing large dataset processing for characters
- **No Additional Calls**: Work within existing fetch mechanisms

**Memory Management**:
- **Streaming Processing**: Process large character datasets in chunks
- **Rate Limiting**: Respect existing platform rate limits
- **Error Handling**: Continue processing with available sources if some fail

## 8. Testing Infrastructure

### 8.1 Test Categories
- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: Cross-component testing with real dependencies
- **MCP Tests**: Protocol compliance and tool execution testing
- **Performance Tests**: Load testing and response time validation

### 8.2 Testing Patterns
- **Mock Strategy**: Global external mocks + specific business logic mocks
- **Async Testing**: Full async test suite with proper cleanup
- **Fixture Management**: Reusable test data and client fixtures
- **Coverage Tracking**: Target >80% code coverage

### 8.3 CI/CD Considerations
- **Dependency Isolation**: Tests run without external API dependencies
- **Container Testing**: Docker-based test environments
- **Platform Testing**: Multiple Python versions and OS compatibility
- **Security Scanning**: Dependency vulnerability scanning