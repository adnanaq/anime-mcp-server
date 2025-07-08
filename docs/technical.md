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

### 3.2 MCP Server Design
- **Dual Server Architecture**: Core server (basic tools) + Modern server (LangGraph workflows) 
- **Transport Flexibility**: Multiple protocols (stdio/http/sse/streamable) for different clients
- **Tool Integration**: FastMCP wrapper functions for LangChain compatibility
- **Dependency Isolation**: Core server avoids LangGraph dependencies for stability

### 3.3 LangGraph Workflow Integration
- **ReactAgent Pattern**: create_react_agent for native LLM tool calling
- **Memory Persistence**: MemorySaver for conversation continuity  
- **Tool Wrapper Strategy**: FastMCP tools wrapped as LangChain-compatible functions
- **Query Processing**: AI-powered intent extraction and parameter routing

### 3.4 External Platform Integration
- **9 Anime Platforms**: Mix of APIs (MAL, AniList, Jikan) and scraping targets
- **Rate Limiting**: Platform-specific limits with fallback strategies
- **Data Mapping**: Universal anime schema with platform-specific enrichment
- **Error Handling**: Circuit breakers and graceful degradation

### 3.5 Stateful Routing Memory System (Task #89 Implementation)
- **Memory Architecture**: Multi-tier memory system with conversation persistence
- **Pattern Learning**: Query pattern recognition and similarity matching
- **Agent Optimization**: Handoff sequence learning for improved coordination
- **User Preferences**: Personalized routing based on interaction history
- **Performance Caching**: 50%+ response time improvement via optimal route selection
- **Memory Management**: Automatic cleanup with configurable limits (10,000 patterns, 1,000 sessions)

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

## 7. Testing Infrastructure

### 7.1 Test Categories
- **Unit Tests**: Individual component testing with mocks
- **Integration Tests**: Cross-component testing with real dependencies
- **MCP Tests**: Protocol compliance and tool execution testing
- **Performance Tests**: Load testing and response time validation

### 7.2 Testing Patterns
- **Mock Strategy**: Global external mocks + specific business logic mocks
- **Async Testing**: Full async test suite with proper cleanup
- **Fixture Management**: Reusable test data and client fixtures
- **Coverage Tracking**: Target >80% code coverage

### 7.3 CI/CD Considerations
- **Dependency Isolation**: Tests run without external API dependencies
- **Container Testing**: Docker-based test environments
- **Platform Testing**: Multiple Python versions and OS compatibility
- **Security Scanning**: Dependency vulnerability scanning