# 🚀 AnimeTrackr MCP Server Integration - Refined Implementation Plan

## 📋 Executive Summary

This document presents a refined, iterative strategy for building an AnimeTrackr MCP server ecosystem, starting with vector database foundation and evolving to full AI assistant integration. The approach prioritizes rapid iteration, proven foundations, and gradual enhancement.

### 🎯 Vision Statement

Create a world-class MCP server that allows AI assistants (Claude, GPT, etc.) to naturally interact with anime data through semantic search and intelligent recommendations, while providing the frontend with enhanced discovery capabilities through advanced vector search.

### 🌟 Key Objectives

1. **Vector-First Foundation**: Start with robust vector database using anime-offline-database (38K+ entries)
2. **Iterative Development**: Build and validate each component before proceeding
3. **Enhanced Data Pipeline**: Fork and enhance modb-app for synopsis inclusion
4. **MCP Integration**: Layer AI assistant tools on proven vector foundation
5. **Stateless Architecture**: No user data storage - API calls for user modifications

## 🔄 **UPDATED APPROACH: Iterative Vector-First Strategy**

### **📁 New Repository Structure**

```
../anime-mcp-server/
├── plan.md                 # This comprehensive plan
├── tasks.md               # Detailed task breakdown
├── current_sprint.md      # Current sprint tracking
├── src/                   # FastAPI MCP server
├── data/                  # Anime database processing
├── vector/               # Marqo integration
├── modb-enhanced/        # Forked modb-app with synopsis
└── docs/                 # Technical documentation
```

## 🏗️ **REFINED DEVELOPMENT PHASES**

### **Phase 1: Vector Database Foundation (Week 1-2)**

**Objective**: Build searchable anime database with vector embeddings

**Deliverables**:

1. FastAPI server setup with basic structure
2. Anime-offline-database integration (38K+ entries)
3. Marqo vector database setup and configuration
4. Data ingestion pipeline with basic embeddings
5. Semantic search API endpoints
6. Basic recommendation algorithms

**Success Criteria**:

- ✅ All 38K anime entries indexed in Marqo
- ✅ Semantic search returning relevant results
- ✅ API response times < 200ms
- ✅ Basic similarity recommendations working

### **Phase 2: Enhanced Data Pipeline (Week 3)**

**Objective**: Fork modb-app and add synopsis extraction

**Deliverables**:

1. Fork manami-project/modb-app repository
2. Add synopsis field to Anime data model
3. Implement synopsis extraction for all 11 sources
4. Enhanced JSON schema with synopsis support
5. Regenerate anime database with synopsis data
6. Update vector embeddings with synopsis content

**Success Criteria**:

- ✅ 95%+ anime entries have synopsis data
- ✅ Enhanced embeddings show improved search quality
- ✅ Automated weekly sync pipeline working
- ✅ Backward compatibility maintained

### **Phase 3: MCP Server Integration (Week 4-5)**

**Objective**: Layer MCP protocol on vector foundation

**Deliverables**:

1. MCP protocol implementation
2. AI assistant tool definitions
3. Natural language query processing
4. Claude/GPT integration and testing
5. Advanced recommendation tools
6. Contextual search capabilities

**Success Criteria**:

- ✅ Claude can search and recommend anime
- ✅ Natural language queries working
- ✅ MCP tools respond accurately
- ✅ Integration with existing frontend

### **Phase 4: Production Optimization (Week 6+)**

**Objective**: Performance, monitoring, and deployment

**Deliverables**:

1. Performance optimization and caching
2. Monitoring and logging setup
3. Production deployment pipeline
4. Documentation and API guides
5. Testing and quality assurance
6. Frontend integration enhancements

---

## 🎯 **KEY ARCHITECTURAL DECISIONS**

### **✅ Data Strategy**

- **Primary Source**: anime-offline-database (38K+ entries with 11 source cross-references)
- **Enhancement**: Fork modb-app to add synopsis extraction for richer embeddings
- **No User Storage**: Stateless architecture - API calls for user modifications
- **Weekly Sync**: Automated updates from upstream anime-offline-database

### **✅ Technology Stack**

- **Backend**: FastAPI (Python) for performance and type safety
- **Vector DB**: Marqo for semantic search and embeddings
- **MCP**: Model Context Protocol for AI assistant integration
- **Data Processing**: Enhanced modb-app (Kotlin) for comprehensive data collection

### **✅ Vector Strategy**

```python
# Phase 1: Basic embeddings with available data
embedding_text = f"{title} {' '.join(synonyms)} {' '.join(tags)} {' '.join(studios)}"

# Phase 2: Enhanced embeddings with synopsis
enhanced_text = f"{title} {synopsis} {' '.join(tags)} {' '.join(studios)} {' '.join(synonyms)}"
```

### **✅ Integration Approach**

- **Frontend Compatibility**: Maintain existing API endpoints during transition
- **Gradual Migration**: Layer new capabilities without breaking existing functionality
- **AI Assistant Ready**: MCP tools designed for natural language interaction

---

## 🔍 Current State Ultra-Analysis

### 📊 Existing Architecture Assessment

#### **Frontend Excellence (React + TypeScript)**

- **Technology Stack**: React 18, TypeScript, Vite, Tailwind CSS
- **State Management**: Zustand with devtools (95% TypeScript coverage)
- **Animation System**: Anime.js v4 + Three.js + Framer Motion
- **Testing Infrastructure**: 756 tests across 31 files (98.9% pass rate)

---

## 🚀 **IMMEDIATE NEXT STEPS**

### **Step 1: Create New Repository Structure**

```bash
cd /home/dani/code
mkdir anime-mcp-server
cd anime-mcp-server

# Initialize project structure
mkdir -p {src,data,vector,modb-enhanced,docs,tests}
touch {plan.md,tasks.md,current_sprint.md}
```

### **Step 2: Initialize FastAPI Project**

```bash
# Setup Python environment
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn marqo-py pydantic

# Create basic FastAPI structure
touch src/{main.py,__init__.py}
mkdir -p src/{api,core,models,services}
```

### **Step 3: Start Vector Database Implementation**

- Download anime-offline-database JSON
- Setup Marqo vector database
- Implement basic data ingestion pipeline
- Create semantic search endpoints

---

## 📋 **PROJECT TRACKING FILES**

### **plan.md**

- Complete technical architecture
- Phase-by-phase implementation guide
- Success criteria and milestones
- Technology stack decisions

### **tasks.md**

- Detailed task breakdown for each phase
- Dependencies and blockers
- Estimated effort and priority
- Assignment and status tracking

### **current_sprint.md**

- Current sprint objectives
- Daily progress tracking
- Blockers and decisions needed
- Sprint retrospective and learnings

---

## 🎯 **SUCCESS METRICS**

### **Phase 1 Metrics**

- Vector database performance: < 200ms search response
- Data coverage: 100% of 38K anime entries indexed
- Search quality: Relevant results for common anime queries
- API reliability: 99%+ uptime during testing

### **Phase 2 Metrics**

- Synopsis coverage: 95%+ anime entries with synopsis
- Enhanced search quality: Improved relevance scores
- Data freshness: Weekly automated updates working
- Pipeline reliability: Zero data corruption incidents

### **Phase 3 Metrics**

- MCP functionality: All defined tools working correctly
- AI integration: Claude can successfully find and recommend anime
- Natural language: Complex queries processed accurately
- Response quality: Contextually appropriate recommendations

### **Phase 4 Metrics**

- Production readiness: Monitoring, logging, error handling
- Performance: < 100ms API response times at scale
- Documentation: Complete API guides and integration docs
- Frontend integration: Seamless experience for existing users
- **Component Architecture**: 39 UI components with Storybook integration
- **Performance**: 60-80% cache hit rates with multi-tiered caching

#### **Current Backend Limitations (Express.js Proxy)**

```javascript
// Current proxy-server.js (limited scope)
const express = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");

// Basic CORS proxy - needs evolution to full MCP server
app.use(
  "/api/mal",
  createProxyMiddleware({
    target: "https://api.myanimelist.net",
    changeOrigin: true,
    pathRewrite: { "^/api/mal": "" },
  }),
);
```

**Limitations**:

- No persistent data storage
- Limited to CORS proxying
- No AI/ML capabilities
- No vector search or semantic understanding
- No MCP protocol support

#### **API Integration Analysis**

```typescript
// Current triple-source integration
interface ServiceArchitecture {
  sources: ["MyAnimeList", "AniList", "Jikan"];
  authentication: ["OAuth 2.0 PKCE", "OAuth 2.0 S256"];
  caching: "Multi-tiered (Memory + IndexedDB + Request deduplication)";
  normalization: "Unified AnimeBase interface";
}
```

**Strengths to Preserve**:

- Excellent OAuth authentication flows
- Robust caching system (60-80% hit rates)
- Comprehensive data normalization
- Strong error handling and fallbacks

#### **Data Flow Mapping**

```mermaid
graph TD
    A[Frontend React App] --> B[Express.js Proxy]
    B --> C[MyAnimeList API]
    B --> D[AniList GraphQL]
    B --> E[Jikan API]
    F[Zustand Store] --> G[Multi-tier Cache]
    G --> H[IndexedDB]
```

**Current Limitations**:

- No persistent user data storage
- No recommendation engine
- No semantic search capabilities
- No AI assistant integration
- Limited to real-time API calls

---

## 🏗️ MCP Server Architecture Design

### 🎯 Target Architecture Vision

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[React + TypeScript Frontend]
        B[Zustand Store]
        C[Multi-tier Cache]
    end

    subgraph "MCP Server Layer (FastAPI)"
        D[MCP Protocol Handler]
        E[Authentication Service]
        F[Anime Management Tools]
        G[Recommendation Engine]
        H[Vector Search Service]
    end

    subgraph "Data Layer"
        I[PostgreSQL User Data]
        J[Marqo Vector Database]
        K[Anime Offline Database]
        L[Redis Cache]
    end

    subgraph "External APIs"
        M[MyAnimeList API]
        N[AniList GraphQL]
        O[Jikan API]
    end

    subgraph "AI Assistants"
        P[Claude via MCP]
        Q[GPT via MCP]
        R[Custom AI Agents]
    end

    A --> D
    D --> E
    D --> F
    F --> G
    G --> H
    H --> J
    F --> I
    E --> M
    E --> N
    E --> O
    H --> K
    P --> D
    Q --> D
    R --> D
```

### 🛠️ FastAPI MCP Server Core Architecture

#### **Project Structure**

```
anime_mcp_server/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py           # MCP protocol implementation
│   │   ├── tools/              # MCP tools for AI assistants
│   │   │   ├── __init__.py
│   │   │   ├── anime_search.py
│   │   │   ├── recommendations.py
│   │   │   ├── watchlist_management.py
│   │   │   ├── user_analytics.py
│   │   │   └── semantic_queries.py
│   │   └── resources/          # MCP resources
│   │       ├── __init__.py
│   │       ├── user_data.py
│   │       └── anime_database.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── anime.py        # Anime CRUD operations
│   │   │   ├── users.py        # User management
│   │   │   ├── recommendations.py
│   │   │   └── auth.py         # OAuth handlers
│   │   └── legacy/
│   │       └── proxy.py        # Maintain frontend compatibility
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   ├── database.py         # Database connections
│   │   ├── security.py         # Authentication & authorization
│   │   └── exceptions.py       # Custom exceptions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── anime.py            # Pydantic models
│   │   ├── user.py
│   │   └── recommendations.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── anime_service.py    # Business logic
│   │   ├── vector_service.py   # Marqo integration
│   │   ├── recommendation_service.py
│   │   ├── auth_service.py     # OAuth service
│   │   └── cache_service.py    # Redis caching
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── mal/                # MyAnimeList integration
│   │   ├── anilist/            # AniList integration
│   │   ├── jikan/              # Jikan integration
│   │   ├── marqo/              # Vector database
│   │   └── offline_db/         # Anime offline database
│   └── utils/
│       ├── __init__.py
│       ├── langchain_integration.py
│       └── data_normalization.py
├── tests/
├── migrations/
├── docker/
├── requirements.txt
└── pyproject.toml
```

#### **Core MCP Server Implementation**

```python
# app/mcp/server.py
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    CallToolRequest,
    GetToolRequest,
    ListToolsRequest,
    Tool,
)
from typing import Any, Dict, List
import asyncio

class AnimeTrackrMCPServer:
    def __init__(self):
        self.server = Server("anime-trackr")
        self.setup_tools()
        self.setup_resources()

    def setup_tools(self):
        """Register all MCP tools for AI assistants"""

        @self.server.call_tool()
        async def search_anime(arguments: Dict[str, Any]) -> List[Dict]:
            """Search for anime across multiple sources with semantic understanding"""
            query = arguments.get("query", "")
            filters = arguments.get("filters", {})

            # Implement multi-source search with vector similarity
            results = await self.anime_service.semantic_search(
                query=query,
                filters=filters,
                limit=arguments.get("limit", 20)
            )
            return results

        @self.server.call_tool()
        async def get_recommendations(arguments: Dict[str, Any]) -> List[Dict]:
            """Get personalized anime recommendations using vector similarity"""
            user_id = arguments.get("user_id")
            context = arguments.get("context", "")

            recommendations = await self.recommendation_service.get_contextual_recommendations(
                user_id=user_id,
                context=context,
                limit=arguments.get("limit", 10)
            )
            return recommendations

        @self.server.call_tool()
        async def update_watchlist(arguments: Dict[str, Any]) -> Dict:
            """Add or update anime in user's watchlist"""
            user_id = arguments.get("user_id")
            anime_id = arguments.get("anime_id")
            status = arguments.get("status")
            score = arguments.get("score")

            result = await self.anime_service.update_user_anime(
                user_id=user_id,
                anime_id=anime_id,
                status=status,
                score=score
            )
            return result

        @self.server.call_tool()
        async def analyze_user_preferences(arguments: Dict[str, Any]) -> Dict:
            """Analyze user's anime preferences and provide insights"""
            user_id = arguments.get("user_id")

            analysis = await self.analytics_service.analyze_user_preferences(
                user_id=user_id
            )
            return analysis

        @self.server.call_tool()
        async def natural_language_query(arguments: Dict[str, Any]) -> List[Dict]:
            """Process natural language queries about anime"""
            query = arguments.get("query")
            user_context = arguments.get("user_context", {})

            # Use LangChain + Marqo for semantic understanding
            results = await self.langchain_service.process_natural_query(
                query=query,
                user_context=user_context
            )
            return results

    async def run(self, transport: str = "stdio"):
        """Run the MCP server"""
        if transport == "stdio":
            from mcp.server.stdio import stdio_server
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream, write_stream, InitializationOptions(
                        server_name="anime-trackr",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    )
                )
```

#### **FastAPI Application Structure**

```python
# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
import asyncio

from app.core.config import settings
from app.core.database import init_db
from app.mcp.server import AnimeTrackrMCPServer
from app.api.v1 import anime, users, recommendations, auth
from app.services.vector_service import VectorService
from app.integrations.marqo.client import MarqoClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    # Initialize database
    await init_db()

    # Initialize Marqo vector database
    marqo_client = MarqoClient(settings.MARQO_URL)
    await marqo_client.initialize_indexes()

    # Start MCP server in background
    mcp_server = AnimeTrackrMCPServer()
    mcp_task = asyncio.create_task(mcp_server.run())

    yield

    # Cleanup
    mcp_task.cancel()

app = FastAPI(
    title="AnimeTrackr MCP Server",
    description="AI-powered anime tracking and recommendation system",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(anime.router, prefix="/api/v1/anime", tags=["anime"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(recommendations.router, prefix="/api/v1/recommendations", tags=["recommendations"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])

# Legacy proxy endpoints for frontend compatibility
from app.api.legacy.proxy import router as legacy_router
app.include_router(legacy_router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/mcp/capabilities")
async def mcp_capabilities():
    """Expose MCP server capabilities for AI assistants"""
    return {
        "tools": [
            {
                "name": "search_anime",
                "description": "Search for anime with semantic understanding",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "filters": {"type": "object"},
                        "limit": {"type": "integer", "default": 20}
                    }
                }
            },
            {
                "name": "get_recommendations",
                "description": "Get personalized anime recommendations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "context": {"type": "string"},
                        "limit": {"type": "integer", "default": 10}
                    }
                }
            },
            {
                "name": "update_watchlist",
                "description": "Update user's anime watchlist",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "anime_id": {"type": "string"},
                        "status": {"type": "string"},
                        "score": {"type": "integer"}
                    }
                }
            }
        ],
        "resources": [
            {
                "uri": "anime://database",
                "name": "Anime Database",
                "description": "Complete anime database with metadata"
            },
            {
                "uri": "user://watchlist/{user_id}",
                "name": "User Watchlist",
                "description": "User's personal anime watchlist"
            }
        ]
    }
```

---

## 📚 Data Integration Strategy: Anime Offline Database

### 🎯 Integration Architecture

#### **Data Source Analysis**

- **Size**: 38,804 anime entries (70% reviewed)
- **Format**: Structured JSON with comprehensive metadata
- **Update Frequency**: Weekly updates (current: week 24, 2025)
- **Coverage**: Multi-source aggregation (MAL, Anime-Planet, Kitsu, etc.)

#### **Data Schema Mapping**

```python
# app/models/anime_offline_db.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class AnimeOfflineDBEntry(BaseModel):
    """Anime Offline Database entry structure"""
    sources: List[str] = Field(..., description="Source URLs from various providers")
    title: str = Field(..., description="Primary anime title")
    type: str = Field(..., description="TV, Movie, OVA, etc.")
    episodes: int = Field(default=0, description="Number of episodes")
    status: str = Field(..., description="Airing status")
    animeSeason: Optional[Dict[str, Any]] = Field(None, description="Season and year")
    picture: Optional[str] = Field(None, description="Cover image URL")
    thumbnail: Optional[str] = Field(None, description="Thumbnail URL")
    duration: Optional[int] = Field(None, description="Episode duration in minutes")
    synonyms: List[str] = Field(default_factory=list, description="Alternative titles")
    relatedAnime: List[str] = Field(default_factory=list, description="Related anime URLs")
    tags: List[str] = Field(default_factory=list, description="Genre and theme tags")

class EnrichedAnimeEntry(BaseModel):
    """Enhanced anime entry with vector embeddings"""
    # Original offline DB data
    offline_db_data: AnimeOfflineDBEntry

    # Enhanced fields
    mal_id: Optional[int] = None
    anilist_id: Optional[int] = None
    jikan_id: Optional[int] = None

    # Vector embeddings for recommendations
    content_embedding: Optional[List[float]] = None
    genre_embedding: Optional[List[float]] = None
    description_embedding: Optional[List[float]] = None

    # Computed fields
    popularity_score: Optional[float] = None
    recommendation_tags: List[str] = Field(default_factory=list)
    similarity_cluster: Optional[int] = None

    # Metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    data_quality_score: float = Field(default=0.0)
```

#### **Data Integration Pipeline**

```python
# app/services/offline_db_integration.py
import aiohttp
import asyncio
from typing import List, Dict, Any
import json
from app.integrations.marqo.client import MarqoClient
from app.services.vector_service import VectorService

class OfflineDBIntegrator:
    def __init__(self, marqo_client: MarqoClient, vector_service: VectorService):
        self.marqo_client = marqo_client
        self.vector_service = vector_service
        self.base_url = "https://raw.githubusercontent.com/manami-project/anime-offline-database/master"

    async def download_latest_database(self) -> Dict[str, Any]:
        """Download the latest anime offline database"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/anime-offline-database-minified.json") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to download database: {response.status}")

    async def process_and_enrich_data(self, raw_data: Dict[str, Any]) -> List[EnrichedAnimeEntry]:
        """Process raw database and enrich with vector embeddings"""
        anime_list = raw_data.get("data", [])
        enriched_entries = []

        # Process in batches for memory efficiency
        batch_size = 100
        for i in range(0, len(anime_list), batch_size):
            batch = anime_list[i:i + batch_size]
            enriched_batch = await self._process_batch(batch)
            enriched_entries.extend(enriched_batch)

            # Progress logging
            print(f"Processed {min(i + batch_size, len(anime_list))}/{len(anime_list)} entries")

        return enriched_entries

    async def _process_batch(self, batch: List[Dict]) -> List[EnrichedAnimeEntry]:
        """Process a batch of anime entries"""
        tasks = [self._enrich_single_entry(entry) for entry in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _enrich_single_entry(self, entry: Dict) -> EnrichedAnimeEntry:
        """Enrich a single anime entry with embeddings and metadata"""
        # Create base entry
        offline_db_entry = AnimeOfflineDBEntry(**entry)

        # Extract IDs from sources
        mal_id = self._extract_mal_id(offline_db_entry.sources)
        anilist_id = self._extract_anilist_id(offline_db_entry.sources)

        # Generate text for embedding
        embedding_text = self._create_embedding_text(offline_db_entry)

        # Generate embeddings using Marqo
        embeddings = await self.vector_service.generate_embeddings(embedding_text)

        # Create enriched entry
        enriched = EnrichedAnimeEntry(
            offline_db_data=offline_db_entry,
            mal_id=mal_id,
            anilist_id=anilist_id,
            content_embedding=embeddings.get("content"),
            genre_embedding=embeddings.get("genre"),
            description_embedding=embeddings.get("description"),
            data_quality_score=self._calculate_quality_score(offline_db_entry)
        )

        return enriched

    def _create_embedding_text(self, entry: AnimeOfflineDBEntry) -> Dict[str, str]:
        """Create text representations for different embedding types"""
        return {
            "content": f"{entry.title} {' '.join(entry.synonyms)} {entry.type} {' '.join(entry.tags)}",
            "genre": " ".join(entry.tags),
            "description": f"{entry.title} is a {entry.type} anime with {entry.episodes} episodes. Genres: {', '.join(entry.tags)}"
        }

    async def populate_vector_database(self, enriched_entries: List[EnrichedAnimeEntry]):
        """Populate Marqo with enriched anime data"""
        # Prepare documents for Marqo
        documents = []
        for entry in enriched_entries:
            doc = {
                "anime_id": f"offline_db_{entry.mal_id or entry.anilist_id or hash(entry.offline_db_data.title)}",
                "title": entry.offline_db_data.title,
                "type": entry.offline_db_data.type,
                "episodes": entry.offline_db_data.episodes,
                "tags": entry.offline_db_data.tags,
                "synonyms": entry.offline_db_data.synonyms,
                "content_text": self._create_embedding_text(entry.offline_db_data)["content"],
                "mal_id": entry.mal_id,
                "anilist_id": entry.anilist_id,
                "quality_score": entry.data_quality_score
            }
            documents.append(doc)

        # Batch insert into Marqo
        await self.marqo_client.add_documents("anime_database", documents, tensor_fields=["content_text"])

    async def full_integration_pipeline(self):
        """Complete integration pipeline from download to vector database"""
        print("Starting anime offline database integration...")

        # Step 1: Download latest data
        print("Downloading latest database...")
        raw_data = await self.download_latest_database()

        # Step 2: Process and enrich
        print("Processing and enriching data...")
        enriched_entries = await self.process_and_enrich_data(raw_data)

        # Step 3: Populate vector database
        print("Populating vector database...")
        await self.populate_vector_database(enriched_entries)

        # Step 4: Create indexes and optimizations
        print("Creating search indexes...")
        await self.marqo_client.optimize_indexes()

        print(f"Integration complete! Processed {len(enriched_entries)} anime entries.")

        return {
            "total_entries": len(enriched_entries),
            "successful_embeddings": sum(1 for e in enriched_entries if e.content_embedding),
            "mal_mappings": sum(1 for e in enriched_entries if e.mal_id),
            "anilist_mappings": sum(1 for e in enriched_entries if e.anilist_id)
        }
```

#### **Data Synchronization Strategy**

```python
# app/services/data_sync_service.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.services.offline_db_integration import OfflineDBIntegrator

class DataSyncService:
    def __init__(self, integrator: OfflineDBIntegrator):
        self.integrator = integrator
        self.scheduler = AsyncIOScheduler()

    def start_sync_scheduler(self):
        """Start automated data synchronization"""
        # Weekly sync on Sundays at 2 AM
        self.scheduler.add_job(
            self.sync_anime_database,
            'cron',
            day_of_week=6,  # Sunday
            hour=2,
            minute=0
        )
        self.scheduler.start()

    async def sync_anime_database(self):
        """Perform incremental database sync"""
        try:
            # Check for updates
            latest_version = await self._check_database_version()
            current_version = await self._get_current_version()

            if latest_version != current_version:
                print(f"Database update detected: {current_version} -> {latest_version}")

                # Perform incremental update
                result = await self.integrator.full_integration_pipeline()

                # Update version tracking
                await self._update_version_tracking(latest_version)

                print(f"Database sync completed: {result}")
            else:
                print("Database is up to date")

        except Exception as e:
            print(f"Database sync failed: {e}")
            # Send notification to monitoring system
```

---

## 🔍 Vector Database Architecture: Marqo Integration

### 🎯 Marqo Configuration & Setup

#### **Multi-Modal Search Architecture**

```python
# app/integrations/marqo/client.py
import marqo
from typing import List, Dict, Any, Optional
import asyncio
from app.core.config import settings

class MarqoClient:
    def __init__(self, url: str = "http://localhost:8882"):
        self.client = marqo.Client(url=url)
        self.indexes = {
            "anime_database": {
                "model": "hf/e5-base-v2",  # High-quality multilingual embeddings
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": 2,
                    "split_overlap": 0,
                    "split_method": "sentence"
                }
            },
            "user_preferences": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",  # Lightweight for user data
                "normalize_embeddings": True
            },
            "anime_descriptions": {
                "model": "hf/e5-large-v2",  # Large model for detailed descriptions
                "normalize_embeddings": True,
                "treat_urls_and_pointers_as_text": False
            },
            "anime_images": {
                "model": "ViT-L/14",  # Vision transformer for image search
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": 1,
                    "split_overlap": 0
                }
            }
        }

    async def initialize_indexes(self):
        """Initialize all required Marqo indexes"""
        for index_name, config in self.indexes.items():
            try:
                # Check if index exists
                existing_indexes = self.client.get_indexes()["results"]
                if index_name not in [idx["indexName"] for idx in existing_indexes]:
                    print(f"Creating index: {index_name}")
                    await self._create_index(index_name, config)
                else:
                    print(f"Index already exists: {index_name}")
            except Exception as e:
                print(f"Error initializing index {index_name}: {e}")

    async def _create_index(self, index_name: str, config: Dict):
        """Create a Marqo index with specific configuration"""
        self.client.create_index(
            index_name=index_name,
            model=config["model"],
            normalize_embeddings=config["normalize_embeddings"],
            text_preprocessing=config.get("text_preprocessing", {}),
            image_preprocessing=config.get("image_preprocessing", {}),
            video_preprocessing=config.get("video_preprocessing", {})
        )

    async def add_documents(self, index_name: str, documents: List[Dict], tensor_fields: List[str]):
        """Add documents to a Marqo index with specified tensor fields"""
        try:
            response = self.client.index(index_name).add_documents(
                documents,
                tensor_fields=tensor_fields,
                device="cpu"  # Use "cuda" if GPU available
            )
            return response
        except Exception as e:
            print(f"Error adding documents to {index_name}: {e}")
            raise

    async def semantic_search(
        self,
        index_name: str,
        query: str,
        filter_string: Optional[str] = None,
        limit: int = 20,
        search_method: str = "TENSOR"
    ) -> List[Dict]:
        """Perform semantic search with optional filtering"""
        try:
            search_params = {
                "q": query,
                "limit": limit,
                "search_method": search_method
            }

            if filter_string:
                search_params["filter_string"] = filter_string

            response = self.client.index(index_name).search(**search_params)
            return response["hits"]
        except Exception as e:
            print(f"Error performing search on {index_name}: {e}")
            return []

    async def multimodal_search(
        self,
        index_name: str,
        query: Dict[str, Any],  # Can include text, image, and weightings
        limit: int = 20
    ) -> List[Dict]:
        """Perform multimodal search combining text and images"""
        try:
            response = self.client.index(index_name).search(
                q=query,
                limit=limit,
                search_method="TENSOR"
            )
            return response["hits"]
        except Exception as e:
            print(f"Error performing multimodal search: {e}")
            return []

    async def get_similar_anime(
        self,
        anime_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """Find similar anime based on vector similarity"""
        try:
            # Get the anime document first
            anime_doc = await self.get_document("anime_database", anime_id)
            if not anime_doc:
                return []

            # Use the anime's content as query for similarity search
            query_text = anime_doc.get("content_text", anime_doc.get("title", ""))

            similar = await self.semantic_search(
                index_name="anime_database",
                query=query_text,
                filter_string=f"anime_id != '{anime_id}'",  # Exclude the original anime
                limit=limit
            )

            # Filter by similarity score
            return [
                result for result in similar
                if result.get("_score", 0) >= similarity_threshold
            ]
        except Exception as e:
            print(f"Error finding similar anime: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        user_preferences: Dict[str, Any],
        limit: int = 20
    ) -> List[Dict]:
        """Combine semantic search with user preference filtering"""
        try:
            # Build dynamic filter based on user preferences
            filters = []

            if user_preferences.get("preferred_genres"):
                genre_filter = " OR ".join([
                    f"tags:'{genre}'" for genre in user_preferences["preferred_genres"]
                ])
                filters.append(f"({genre_filter})")

            if user_preferences.get("min_episodes"):
                filters.append(f"episodes >= {user_preferences['min_episodes']}")

            if user_preferences.get("anime_types"):
                type_filter = " OR ".join([
                    f"type:'{anime_type}'" for anime_type in user_preferences["anime_types"]
                ])
                filters.append(f"({type_filter})")

            filter_string = " AND ".join(filters) if filters else None

            # Perform search with combined semantic + preference filtering
            results = await self.semantic_search(
                index_name="anime_database",
                query=query,
                filter_string=filter_string,
                limit=limit
            )

            return results
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    async def optimize_indexes(self):
        """Optimize all indexes for better performance"""
        for index_name in self.indexes.keys():
            try:
                # Marqo automatically optimizes, but we can trigger refresh
                stats = self.client.index(index_name).get_stats()
                print(f"Index {index_name} stats: {stats}")
            except Exception as e:
                print(f"Error optimizing index {index_name}: {e}")
```

#### **Advanced Vector Search Service**

```python
# app/services/vector_service.py
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from app.integrations.marqo.client import MarqoClient
from app.models.anime import EnrichedAnimeEntry
from app.services.user_preference_service import UserPreferenceService

class VectorService:
    def __init__(self, marqo_client: MarqoClient, user_service: UserPreferenceService):
        self.marqo = marqo_client
        self.user_service = user_service

    async def generate_embeddings(self, text_dict: Dict[str, str]) -> Dict[str, List[float]]:
        """Generate embeddings for different text types"""
        embeddings = {}

        for text_type, text in text_dict.items():
            try:
                # Use appropriate index/model for each text type
                if text_type == "content":
                    index_name = "anime_database"
                elif text_type == "genre":
                    index_name = "user_preferences"
                else:
                    index_name = "anime_descriptions"

                # Add temporary document to get embedding
                temp_doc = {"text": text, "temp_id": "embedding_gen"}
                result = await self.marqo.add_documents(
                    index_name=index_name,
                    documents=[temp_doc],
                    tensor_fields=["text"]
                )

                # Extract embedding (this would need to be adapted based on Marqo's API)
                # For now, we'll use the search similarity approach
                search_result = await self.marqo.semantic_search(
                    index_name=index_name,
                    query=text,
                    limit=1
                )

                if search_result:
                    embeddings[text_type] = search_result[0].get("_tensor", [])

            except Exception as e:
                print(f"Error generating embedding for {text_type}: {e}")
                embeddings[text_type] = []

        return embeddings

    async def contextual_anime_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Perform contextual anime search considering user preferences and context"""

        # Get user preferences if user_id provided
        user_preferences = {}
        if user_id:
            user_preferences = await self.user_service.get_user_preferences(user_id)

        # Enhance query with context
        enhanced_query = self._enhance_query_with_context(query, context, user_preferences)

        # Perform hybrid search
        results = await self.marqo.hybrid_search(
            query=enhanced_query,
            user_preferences=user_preferences,
            limit=limit
        )

        # Re-rank results based on user history and preferences
        if user_id:
            results = await self._rerank_with_user_preferences(results, user_id)

        return results

    def _enhance_query_with_context(
        self,
        query: str,
        context: Optional[str],
        user_preferences: Dict
    ) -> str:
        """Enhance search query with context and user preferences"""
        enhanced_parts = [query]

        if context:
            enhanced_parts.append(context)

        # Add implicit preferences
        if user_preferences.get("preferred_genres"):
            genre_boost = " ".join(user_preferences["preferred_genres"][:3])  # Top 3 genres
            enhanced_parts.append(genre_boost)

        return " ".join(enhanced_parts)

    async def _rerank_with_user_preferences(
        self,
        results: List[Dict],
        user_id: str
    ) -> List[Dict]:
        """Re-rank search results based on user preferences and history"""
        user_history = await self.user_service.get_user_watch_history(user_id)
        user_preferences = await self.user_service.get_user_preferences(user_id)

        for result in results:
            score_adjustments = []

            # Boost based on preferred genres
            anime_genres = result.get("tags", [])
            preferred_genres = user_preferences.get("preferred_genres", [])
            genre_overlap = len(set(anime_genres) & set(preferred_genres))
            if genre_overlap > 0:
                score_adjustments.append(genre_overlap * 0.1)

            # Boost based on similar anime in watch history
            similar_anime_boost = await self._calculate_similarity_boost(
                result, user_history
            )
            score_adjustments.append(similar_anime_boost)

            # Penalize if already watched
            anime_id = result.get("anime_id")
            if anime_id in [item["anime_id"] for item in user_history]:
                score_adjustments.append(-0.5)

            # Apply adjustments
            original_score = result.get("_score", 0)
            adjustment = sum(score_adjustments)
            result["_score"] = max(0, original_score + adjustment)
            result["_score_adjustment"] = adjustment

        # Re-sort by adjusted score
        return sorted(results, key=lambda x: x.get("_score", 0), reverse=True)

    async def _calculate_similarity_boost(
        self,
        anime_result: Dict,
        user_history: List[Dict]
    ) -> float:
        """Calculate similarity boost based on user's watch history"""
        if not user_history:
            return 0.0

        # Get embeddings for current anime
        current_anime_text = anime_result.get("content_text", "")

        similarity_scores = []
        for watched_anime in user_history[-10:]:  # Consider last 10 watched
            if watched_anime.get("score", 0) >= 7:  # Only consider well-rated anime
                watched_text = watched_anime.get("content_text", "")

                # Calculate similarity (simplified - in real implementation, use vector similarity)
                similarity = self._calculate_text_similarity(current_anime_text, watched_text)
                similarity_scores.append(similarity * (watched_anime.get("score", 5) / 10.0))

        if similarity_scores:
            return max(similarity_scores) * 0.2  # Max 0.2 boost
        return 0.0

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (placeholder for vector similarity)"""
        # This is a simplified implementation
        # In practice, use vector cosine similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    async def generate_user_preference_vector(self, user_id: str) -> List[float]:
        """Generate a vector representation of user preferences"""
        user_history = await self.user_service.get_user_watch_history(user_id)

        if not user_history:
            return []

        # Collect text from highly-rated anime
        preference_texts = []
        for anime in user_history:
            if anime.get("score", 0) >= 7:
                preference_texts.append(anime.get("content_text", ""))

        if not preference_texts:
            return []

        # Generate embedding for combined preference text
        combined_text = " ".join(preference_texts)
        embeddings = await self.generate_embeddings({"preference": combined_text})

        return embeddings.get("preference", [])

    async def find_anime_clusters(self, limit: int = 1000) -> Dict[str, List[str]]:
        """Find anime clusters based on vector similarity"""
        # This would involve more complex clustering algorithms
        # For now, return a simplified version

        # Get all anime from database
        all_anime = await self.marqo.semantic_search(
            index_name="anime_database",
            query="*",  # Get all
            limit=limit
        )

        # Simple clustering based on genres (placeholder for vector clustering)
        clusters = {}
        for anime in all_anime:
            primary_genre = anime.get("tags", ["Other"])[0]
            if primary_genre not in clusters:
                clusters[primary_genre] = []
            clusters[primary_genre].append(anime.get("anime_id"))

        return clusters
```

---

## 🤖 LangChain Integration Patterns

### 🎯 LangChain + MCP Architecture

#### **LangChain Service Integration**

```python
# app/utils/langchain_integration.py
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import List, Dict, Any, Optional
import asyncio

class AnimeTrackrLangChainService:
    def __init__(self, openai_api_key: str, mcp_server_url: str):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=openai_api_key
        )

        self.mcp_client = MultiServerMCPClient({
            "anime_trackr": {
                "transport": "streamable_http",
                "url": mcp_server_url
            }
        })

        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 exchanges
            return_messages=True,
            output_key="output"
        )

        self.tools = []
        self.agent = None

    async def initialize(self):
        """Initialize LangChain agent with MCP tools"""
        # Get MCP tools
        mcp_tools = await self.mcp_client.get_tools()

        # Convert MCP tools to LangChain tools
        self.tools = await self._convert_mcp_tools_to_langchain(mcp_tools)

        # Add custom tools
        self.tools.extend(self._create_custom_tools())

        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    async def _convert_mcp_tools_to_langchain(self, mcp_tools: List[Dict]) -> List[Tool]:
        """Convert MCP tools to LangChain Tool format"""
        langchain_tools = []

        for mcp_tool in mcp_tools:
            tool = Tool(
                name=mcp_tool["name"],
                description=mcp_tool["description"],
                func=self._create_mcp_tool_wrapper(mcp_tool["name"])
            )
            langchain_tools.append(tool)

        return langchain_tools

    def _create_mcp_tool_wrapper(self, tool_name: str):
        """Create a wrapper function for MCP tool calls"""
        async def wrapper(input_str: str) -> str:
            try:
                # Parse input (simplified - would need better parsing)
                args = {"query": input_str}

                # Call MCP tool
                result = await self.mcp_client.call_tool(tool_name, args)

                # Format result for LangChain
                return str(result)
            except Exception as e:
                return f"Error calling {tool_name}: {str(e)}"

        return wrapper

    def _create_custom_tools(self) -> List[Tool]:
        """Create custom LangChain tools for enhanced functionality"""
        return [
            Tool(
                name="anime_recommendation_explainer",
                description="Explain why specific anime are recommended to a user",
                func=self._explain_recommendations
            ),
            Tool(
                name="anime_comparison",
                description="Compare multiple anime and highlight differences",
                func=self._compare_anime
            ),
            Tool(
                name="user_taste_analyzer",
                description="Analyze user's anime preferences and provide insights",
                func=self._analyze_user_taste
            ),
            Tool(
                name="seasonal_anime_curator",
                description="Curate seasonal anime based on user preferences",
                func=self._curate_seasonal_anime
            )
        ]

    async def process_natural_query(
        self,
        query: str,
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process natural language query about anime"""
        try:
            # Enhance query with user context
            enhanced_query = self._enhance_query_with_user_context(query, user_context)

            # Run agent
            response = await self.agent.arun(enhanced_query)

            return {
                "response": response,
                "query": query,
                "enhanced_query": enhanced_query,
                "user_context": user_context,
                "success": True
            }
        except Exception as e:
            return {
                "response": f"I encountered an error processing your query: {str(e)}",
                "query": query,
                "error": str(e),
                "success": False
            }

    def _enhance_query_with_user_context(self, query: str, user_context: Dict) -> str:
        """Enhance query with user context for better results"""
        context_parts = [f"User query: {query}"]

        if user_context.get("user_id"):
            context_parts.append(f"User ID: {user_context['user_id']}")

        if user_context.get("current_watching"):
            watching_list = ", ".join(user_context["current_watching"][:5])
            context_parts.append(f"Currently watching: {watching_list}")

        if user_context.get("favorite_genres"):
            genres = ", ".join(user_context["favorite_genres"][:3])
            context_parts.append(f"Favorite genres: {genres}")

        if user_context.get("recent_ratings"):
            ratings = user_context["recent_ratings"][:3]
            rating_info = ", ".join([f"{r['title']} ({r['score']}/10)" for r in ratings])
            context_parts.append(f"Recent ratings: {rating_info}")

        return "\n".join(context_parts)

    async def _explain_recommendations(self, input_str: str) -> str:
        """Explain anime recommendations"""
        # This would integrate with the recommendation service
        # For now, return a placeholder
        return f"Analyzing recommendations for: {input_str}"

    async def _compare_anime(self, input_str: str) -> str:
        """Compare multiple anime"""
        # Parse anime titles from input
        # Get anime details from MCP server
        # Generate comparison
        return f"Comparing anime: {input_str}"

    async def _analyze_user_taste(self, input_str: str) -> str:
        """Analyze user's anime taste"""
        # Get user watch history and ratings
        # Perform taste analysis
        return f"Analyzing taste for user: {input_str}"

    async def _curate_seasonal_anime(self, input_str: str) -> str:
        """Curate seasonal anime"""
        # Get current season anime
        # Filter by user preferences
        return f"Curating seasonal anime: {input_str}"

class ConversationalAnimeAgent:
    def __init__(self, langchain_service: AnimeTrackrLangChainService):
        self.langchain_service = langchain_service
        self.conversation_history = []

    async def chat(self, message: str, user_id: str) -> Dict[str, Any]:
        """Handle conversational anime queries"""
        # Get user context
        user_context = await self._get_user_context(user_id)

        # Process with LangChain
        result = await self.langchain_service.process_natural_query(
            query=message,
            user_context=user_context
        )

        # Store conversation
        self.conversation_history.append({
            "user_message": message,
            "assistant_response": result["response"],
            "timestamp": datetime.utcnow(),
            "user_id": user_id
        })

        return result

    async def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context for conversation"""
        # This would integrate with user service
        return {
            "user_id": user_id,
            "current_watching": [],  # Get from user service
            "favorite_genres": [],   # Get from user service
            "recent_ratings": []     # Get from user service
        }
```

#### **Advanced Query Processing Pipeline**

```python
# app/services/query_processing_service.py
from typing import Dict, List, Any, Optional
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from app.utils.langchain_integration import AnimeTrackrLangChainService

class QueryProcessingService:
    def __init__(self, langchain_service: AnimeTrackrLangChainService):
        self.langchain_service = langchain_service
        self.query_patterns = self._initialize_query_patterns()
        self.intent_classifier = self._initialize_intent_classifier()

    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for different query types"""
        return {
            "recommendation": [
                r"recommend.*anime",
                r"what.*should.*watch",
                r"suggest.*anime",
                r"similar.*to",
                r"like.*but.*different"
            ],
            "search": [
                r"find.*anime",
                r"anime.*about",
                r"anime.*with",
                r"show.*me.*anime"
            ],
            "status_update": [
                r"mark.*as.*watched",
                r"add.*to.*watchlist",
                r"rate.*anime",
                r"completed.*anime",
                r"currently.*watching"
            ],
            "information": [
                r"tell.*me.*about",
                r"what.*is.*anime",
                r"how.*many.*episodes",
                r"when.*did.*air"
            ],
            "comparison": [
                r"compare.*anime",
                r"difference.*between",
                r"better.*anime",
                r"versus",
                r"vs\\.?"
            ]
        }

    def _initialize_intent_classifier(self):
        """Initialize intent classification model"""
        # In a real implementation, this would be a trained model
        # For now, we'll use pattern matching
        pass

    async def process_query(self, query: str, user_context: Dict) -> Dict[str, Any]:
        """Process user query and route to appropriate handler"""
        # Step 1: Classify intent
        intent = self._classify_intent(query)

        # Step 2: Extract entities
        entities = await self._extract_entities(query)

        # Step 3: Route to appropriate handler
        if intent == "recommendation":
            result = await self._handle_recommendation_query(query, entities, user_context)
        elif intent == "search":
            result = await self._handle_search_query(query, entities, user_context)
        elif intent == "status_update":
            result = await self._handle_status_update_query(query, entities, user_context)
        elif intent == "information":
            result = await self._handle_information_query(query, entities, user_context)
        elif intent == "comparison":
            result = await self._handle_comparison_query(query, entities, user_context)
        else:
            # Fallback to general LangChain processing
            result = await self.langchain_service.process_natural_query(query, user_context)

        return {
            **result,
            "intent": intent,
            "entities": entities,
            "processing_method": f"handle_{intent}_query"
        }

    def _classify_intent(self, query: str) -> str:
        """Classify the intent of the user query"""
        query_lower = query.lower()

        for intent, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        return "general"

    async def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from the query"""
        entities = {
            "anime_titles": [],
            "genres": [],
            "years": [],
            "ratings": [],
            "statuses": []
        }

        # Extract anime titles (this would use NER in production)
        # For now, simple pattern matching
        anime_title_patterns = [
            r"(?:anime|series|show)?\s*[\"']([^\"']+)[\"']",
            r"(?:Attack on Titan|Naruto|One Piece|Death Note)",  # Common anime names
        ]

        for pattern in anime_title_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["anime_titles"].extend(matches)

        # Extract genres
        genre_keywords = [
            "action", "adventure", "comedy", "drama", "fantasy", "horror",
            "mystery", "romance", "sci-fi", "slice of life", "sports", "thriller"
        ]

        for genre in genre_keywords:
            if genre in query.lower():
                entities["genres"].append(genre)

        # Extract years
        year_matches = re.findall(r"\b(19|20)\d{2}\b", query)
        entities["years"] = [int(year) for year in year_matches]

        # Extract ratings
        rating_matches = re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10\b", query)
        entities["ratings"] = [float(rating) for rating in rating_matches]

        # Extract status keywords
        status_keywords = ["watching", "completed", "dropped", "plan to watch", "on hold"]
        for status in status_keywords:
            if status in query.lower():
                entities["statuses"].append(status)

        return entities

    async def _handle_recommendation_query(
        self,
        query: str,
        entities: Dict,
        user_context: Dict
    ) -> Dict[str, Any]:
        """Handle recommendation-specific queries"""

        # Build recommendation request
        recommendation_params = {
            "user_id": user_context.get("user_id"),
            "context": query,
            "limit": 10
        }

        # Add entity-based filters
        if entities["genres"]:
            recommendation_params["preferred_genres"] = entities["genres"]

        if entities["anime_titles"]:
            recommendation_params["similar_to"] = entities["anime_titles"][0]

        # Use MCP tool for recommendations
        recommendations = await self.langchain_service.mcp_client.call_tool(
            "get_recommendations",
            recommendation_params
        )

        # Format response
        if recommendations:
            response_parts = [
                f"Based on your query '{query}', here are my recommendations:",
                ""
            ]

            for i, anime in enumerate(recommendations[:5], 1):
                response_parts.append(
                    f"{i}. **{anime.get('title', 'Unknown Title')}** "
                    f"({anime.get('type', 'TV')}, {anime.get('episodes', '?')} episodes)"
                )
                if anime.get('tags'):
                    response_parts.append(f"   Genres: {', '.join(anime['tags'][:3])}")
                if anime.get('similarity_score'):
                    response_parts.append(f"   Match: {anime['similarity_score']:.1%}")
                response_parts.append("")

            return {
                "response": "\n".join(response_parts),
                "recommendations": recommendations,
                "success": True
            }
        else:
            return {
                "response": "I couldn't find any recommendations matching your criteria. Could you provide more details about what you're looking for?",
                "recommendations": [],
                "success": False
            }

    async def _handle_search_query(
        self,
        query: str,
        entities: Dict,
        user_context: Dict
    ) -> Dict[str, Any]:
        """Handle search-specific queries"""

        # Build search parameters
        search_params = {
            "query": query,
            "limit": 20
        }

        # Add filters from entities
        if entities["genres"]:
            search_params["filters"] = {"genres": entities["genres"]}

        if entities["years"]:
            search_params["filters"] = search_params.get("filters", {})
            search_params["filters"]["year"] = entities["years"][0]

        # Use MCP tool for search
        search_results = await self.langchain_service.mcp_client.call_tool(
            "search_anime",
            search_params
        )

        # Format response
        if search_results:
            response_parts = [
                f"I found {len(search_results)} anime matching '{query}':",
                ""
            ]

            for i, anime in enumerate(search_results[:10], 1):
                response_parts.append(
                    f"{i}. **{anime.get('title', 'Unknown Title')}** "
                    f"({anime.get('year', '?')}) - {anime.get('type', 'TV')}"
                )
                if anime.get('synopsis'):
                    synopsis = anime['synopsis'][:100] + "..." if len(anime['synopsis']) > 100 else anime['synopsis']
                    response_parts.append(f"   {synopsis}")
                response_parts.append("")

            return {
                "response": "\n".join(response_parts),
                "search_results": search_results,
                "success": True
            }
        else:
            return {
                "response": f"I couldn't find any anime matching '{query}'. Try a different search term or check the spelling.",
                "search_results": [],
                "success": False
            }

    async def _handle_status_update_query(
        self,
        query: str,
        entities: Dict,
        user_context: Dict
    ) -> Dict[str, Any]:
        """Handle status update queries"""

        if not entities["anime_titles"]:
            return {
                "response": "I need to know which anime you want to update. Could you specify the anime title?",
                "success": False
            }

        anime_title = entities["anime_titles"][0]
        status = entities["statuses"][0] if entities["statuses"] else "watching"
        score = entities["ratings"][0] if entities["ratings"] else None

        # Use MCP tool for status update
        update_params = {
            "user_id": user_context.get("user_id"),
            "anime_title": anime_title,
            "status": status
        }

        if score:
            update_params["score"] = score

        result = await self.langchain_service.mcp_client.call_tool(
            "update_watchlist",
            update_params
        )

        if result.get("success"):
            response = f"✅ Updated '{anime_title}' to '{status}'"
            if score:
                response += f" with a score of {score}/10"
            return {
                "response": response,
                "update_result": result,
                "success": True
            }
        else:
            return {
                "response": f"❌ Failed to update '{anime_title}'. {result.get('error', 'Unknown error')}",
                "update_result": result,
                "success": False
            }

    async def _handle_information_query(
        self,
        query: str,
        entities: Dict,
        user_context: Dict
    ) -> Dict[str, Any]:
        """Handle information requests about anime"""

        if not entities["anime_titles"]:
            # General information query - use LangChain
            return await self.langchain_service.process_natural_query(query, user_context)

        anime_title = entities["anime_titles"][0]

        # Search for the anime first
        search_result = await self.langchain_service.mcp_client.call_tool(
            "search_anime",
            {"query": anime_title, "limit": 1}
        )

        if search_result:
            anime = search_result[0]

            response_parts = [
                f"**{anime.get('title', 'Unknown Title')}**",
                f"Type: {anime.get('type', 'Unknown')}",
                f"Episodes: {anime.get('episodes', 'Unknown')}",
                f"Status: {anime.get('status', 'Unknown')}",
                f"Year: {anime.get('year', 'Unknown')}",
            ]

            if anime.get('genres'):
                response_parts.append(f"Genres: {', '.join(anime['genres'])}")

            if anime.get('synopsis'):
                response_parts.append(f"\nSynopsis: {anime['synopsis']}")

            return {
                "response": "\n".join(response_parts),
                "anime_info": anime,
                "success": True
            }
        else:
            return {
                "response": f"I couldn't find information about '{anime_title}'. Please check the spelling or try a different title.",
                "success": False
            }

    async def _handle_comparison_query(
        self,
        query: str,
        entities: Dict,
        user_context: Dict
    ) -> Dict[str, Any]:
        """Handle anime comparison queries"""

        if len(entities["anime_titles"]) < 2:
            return {
                "response": "I need at least two anime titles to make a comparison. Could you specify which anime you'd like to compare?",
                "success": False
            }

        anime_titles = entities["anime_titles"][:2]  # Compare first two

        # Get information for both anime
        comparison_data = []
        for title in anime_titles:
            search_result = await self.langchain_service.mcp_client.call_tool(
                "search_anime",
                {"query": title, "limit": 1}
            )
            if search_result:
                comparison_data.append(search_result[0])

        if len(comparison_data) == 2:
            anime1, anime2 = comparison_data

            response_parts = [
                f"**Comparison: {anime1.get('title')} vs {anime2.get('title')}**",
                "",
                f"**{anime1.get('title')}:**",
                f"• Type: {anime1.get('type', 'Unknown')}",
                f"• Episodes: {anime1.get('episodes', 'Unknown')}",
                f"• Year: {anime1.get('year', 'Unknown')}",
                f"• Genres: {', '.join(anime1.get('genres', []))}",
                "",
                f"**{anime2.get('title')}:**",
                f"• Type: {anime2.get('type', 'Unknown')}",
                f"• Episodes: {anime2.get('episodes', 'Unknown')}",
                f"• Year: {anime2.get('year', 'Unknown')}",
                f"• Genres: {', '.join(anime2.get('genres', []))}",
                "",
                "**Key Differences:**"
            ]

            # Compare genres
            genres1 = set(anime1.get('genres', []))
            genres2 = set(anime2.get('genres', []))
            common_genres = genres1 & genres2
            unique_genres1 = genres1 - genres2
            unique_genres2 = genres2 - genres1

            if common_genres:
                response_parts.append(f"• Common genres: {', '.join(common_genres)}")
            if unique_genres1:
                response_parts.append(f"• {anime1.get('title')} unique genres: {', '.join(unique_genres1)}")
            if unique_genres2:
                response_parts.append(f"• {anime2.get('title')} unique genres: {', '.join(unique_genres2)}")

            return {
                "response": "\n".join(response_parts),
                "comparison_data": comparison_data,
                "success": True
            }
        else:
            return {
                "response": f"I couldn't find complete information for both anime. Please check the titles and try again.",
                "success": False
            }
```

---

## 🔄 FastAPI Migration Strategy

### 🎯 Migration Philosophy: Zero-Downtime Transition

#### **Parallel Development Approach**

```mermaid
graph TD
    subgraph "Phase 1: Parallel Development"
        A[Current Express.js Proxy] --> B[Frontend React App]
        C[New FastAPI Server] --> D[MCP Protocol Layer]
        E[Shared Database Layer]
    end

    subgraph "Phase 2: Gradual Migration"
        F[Express.js Proxy] --> G[Traffic Router]
        G --> H[FastAPI Server]
        G --> I[Legacy Endpoints]
    end

    subgraph "Phase 3: Complete Migration"
        J[FastAPI MCP Server] --> K[Frontend + AI Assistants]
        L[Enhanced Features]
    end
```

#### **Migration Timeline & Strategy**

```python
# Migration configuration management
class MigrationConfig:
    """Configuration for managing the migration process"""

    def __init__(self):
        self.phases = {
            "phase_1_parallel": {
                "duration": "2 weeks",
                "description": "Develop FastAPI server alongside existing Express.js",
                "features": ["Basic API endpoints", "Database setup", "MCP protocol"],
                "risk_level": "low"
            },
            "phase_2_gradual": {
                "duration": "3 weeks",
                "description": "Gradual migration with traffic splitting",
                "features": ["Traffic router", "A/B testing", "Monitoring"],
                "risk_level": "medium"
            },
            "phase_3_complete": {
                "duration": "1 week",
                "description": "Complete migration and cleanup",
                "features": ["Remove legacy code", "Full MCP integration"],
                "risk_level": "low"
            }
        }

        self.compatibility_matrix = {
            "frontend_apis": {
                "/api/mal/*": "migrate_week_1",
                "/api/anilist/*": "migrate_week_2",
                "/api/jikan/*": "migrate_week_3",
                "/api/auth/*": "migrate_week_4",
                "/api/cache/*": "migrate_week_5"
            },
            "new_endpoints": {
                "/api/v1/mcp/*": "week_1",
                "/api/v1/vector/*": "week_2",
                "/api/v1/recommendations/*": "week_3"
            }
        }
```

#### **Traffic Router Implementation**

```javascript
// migration-router.js - Gradual traffic migration
const express = require("express");
const httpProxy = require("http-proxy-middleware");

class MigrationRouter {
  constructor() {
    this.fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000";
    this.legacyUrl = process.env.LEGACY_URL || "http://localhost:3002";
    this.migrationConfig = this.loadMigrationConfig();
  }

  setupRoutes(app) {
    // Route specific endpoints based on migration status
    app.use("/api/v1/*", this.createFastApiProxy());

    // Legacy routes with gradual migration
    app.use("/api/mal/*", this.createGradualMigrationProxy("mal"));
    app.use("/api/anilist/*", this.createGradualMigrationProxy("anilist"));
    app.use("/api/jikan/*", this.createGradualMigrationProxy("jikan"));

    // Health check for both systems
    app.get("/health/migration", this.getMigrationStatus.bind(this));
  }

  createGradualMigrationProxy(service) {
    return (req, res, next) => {
      const migrationPercentage = this.getMigrationPercentage(service);
      const shouldUseFastApi = Math.random() < migrationPercentage;

      const targetUrl = shouldUseFastApi ? this.fastApiUrl : this.legacyUrl;
      const proxyMiddleware = httpProxy({
        target: targetUrl,
        changeOrigin: true,
        onError: (err, req, res) => {
          // Fallback to legacy on FastAPI error
          if (shouldUseFastApi) {
            console.log(
              `FastAPI error, falling back to legacy: ${err.message}`,
            );
            this.routeToLegacy(req, res);
          } else {
            res.status(500).json({ error: "Service unavailable" });
          }
        },
      });

      proxyMiddleware(req, res, next);
    };
  }

  getMigrationPercentage(service) {
    // Progressive migration percentages
    const week = this.getCurrentMigrationWeek();
    const percentages = {
      1: { mal: 0.1, anilist: 0, jikan: 0 },
      2: { mal: 0.3, anilist: 0.1, jikan: 0 },
      3: { mal: 0.6, anilist: 0.3, jikan: 0.1 },
      4: { mal: 0.8, anilist: 0.6, jikan: 0.3 },
      5: { mal: 1.0, anilist: 0.8, jikan: 0.6 },
      6: { mal: 1.0, anilist: 1.0, jikan: 1.0 },
    };

    return percentages[week]?.[service] || 0;
  }
}
```

#### **Database Migration Strategy**

```python
# Database schema migration with Alembic
# migrations/versions/001_initial_schema.py

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

def upgrade():
    """Create initial database schema for MCP server"""

    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('mal_user_id', sa.String(50), unique=True, nullable=True),
        sa.Column('anilist_user_id', sa.String(50), unique=True, nullable=True),
        sa.Column('username', sa.String(100), nullable=False),
        sa.Column('email', sa.String(255), unique=True, nullable=True),
        sa.Column('preferences', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Anime database table (from offline DB)
    op.create_table(
        'anime_database',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('mal_id', sa.Integer, unique=True, nullable=True),
        sa.Column('anilist_id', sa.Integer, unique=True, nullable=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('title_synonyms', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('type', sa.String(50), nullable=True),
        sa.Column('episodes', sa.Integer, nullable=True),
        sa.Column('status', sa.String(50), nullable=True),
        sa.Column('season', sa.String(20), nullable=True),
        sa.Column('year', sa.Integer, nullable=True),
        sa.Column('genres', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('synopsis', sa.Text, nullable=True),
        sa.Column('cover_image', sa.String(500), nullable=True),
        sa.Column('sources', postgresql.JSONB, nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('content_embedding', postgresql.ARRAY(sa.Float), nullable=True),
        sa.Column('genre_embedding', postgresql.ARRAY(sa.Float), nullable=True),
        sa.Column('quality_score', sa.Float, default=0.0),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # User anime lists
    op.create_table(
        'user_anime_lists',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('anime_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('anime_database.id'), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),  # watching, completed, dropped, etc.
        sa.Column('score', sa.Integer, nullable=True),  # 1-10 rating
        sa.Column('episodes_watched', sa.Integer, default=0),
        sa.Column('start_date', sa.Date, nullable=True),
        sa.Column('finish_date', sa.Date, nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('source_platform', sa.String(20), nullable=True),  # mal, anilist
        sa.Column('sync_status', sa.String(20), default='pending'),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint('user_id', 'anime_id', name='unique_user_anime')
    )

    # User preferences and recommendations
    op.create_table(
        'user_recommendations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('anime_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('anime_database.id'), nullable=False),
        sa.Column('recommendation_score', sa.Float, nullable=False),
        sa.Column('recommendation_reason', sa.Text, nullable=True),
        sa.Column('recommendation_type', sa.String(50), nullable=True),  # vector, collaborative, content
        sa.Column('context', sa.String(500), nullable=True),
        sa.Column('generated_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime, nullable=True)
    )

    # MCP conversation history
    op.create_table(
        'mcp_conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('session_id', sa.String(100), nullable=False),
        sa.Column('message_type', sa.String(20), nullable=False),  # user, assistant, system
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('intent', sa.String(50), nullable=True),
        sa.Column('entities', postgresql.JSONB, nullable=True),
        sa.Column('response_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now())
    )

    # Create indexes for performance
    op.create_index('idx_anime_mal_id', 'anime_database', ['mal_id'])
    op.create_index('idx_anime_anilist_id', 'anime_database', ['anilist_id'])
    op.create_index('idx_anime_title', 'anime_database', ['title'])
    op.create_index('idx_anime_genres', 'anime_database', ['genres'], postgresql_using='gin')
    op.create_index('idx_user_anime_user_id', 'user_anime_lists', ['user_id'])
    op.create_index('idx_user_anime_status', 'user_anime_lists', ['status'])
    op.create_index('idx_recommendations_user_id', 'user_recommendations', ['user_id'])
    op.create_index('idx_conversations_user_id', 'mcp_conversations', ['user_id'])
    op.create_index('idx_conversations_session', 'mcp_conversations', ['session_id'])

def downgrade():
    """Drop all tables"""
    op.drop_table('mcp_conversations')
    op.drop_table('user_recommendations')
    op.drop_table('user_anime_lists')
    op.drop_table('anime_database')
    op.drop_table('users')
```

#### **Legacy API Compatibility Layer**

```python
# app/api/legacy/compatibility.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Any, Dict, List
from app.services.legacy_adapter import LegacyAPIAdapter
from app.core.auth import get_current_user

router = APIRouter()
legacy_adapter = LegacyAPIAdapter()

@router.get("/api/mal/{path:path}")
async def mal_proxy(path: str, request: Request):
    """Maintain compatibility with existing MAL API calls"""
    try:
        # Extract query parameters
        params = dict(request.query_params)

        # Route to appropriate new endpoint
        if "user" in path and "animelist" in path:
            return await legacy_adapter.handle_mal_user_animelist(path, params)
        elif "anime" in path:
            return await legacy_adapter.handle_mal_anime_details(path, params)
        else:
            # Fallback to direct proxy
            return await legacy_adapter.proxy_to_mal(path, params)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MAL API error: {str(e)}")

@router.get("/api/anilist/{path:path}")
async def anilist_proxy(path: str, request: Request):
    """Maintain compatibility with existing AniList API calls"""
    try:
        params = dict(request.query_params)

        # Convert GraphQL queries to new API format
        if request.method == "POST":
            body = await request.json()
            return await legacy_adapter.handle_anilist_graphql(body)
        else:
            return await legacy_adapter.handle_anilist_rest(path, params)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AniList API error: {str(e)}")

@router.get("/api/jikan/{path:path}")
async def jikan_proxy(path: str, request: Request):
    """Maintain compatibility with existing Jikan API calls"""
    try:
        params = dict(request.query_params)
        return await legacy_adapter.handle_jikan_request(path, params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Jikan API error: {str(e)}")

class LegacyAPIAdapter:
    """Adapter to maintain compatibility with existing frontend API calls"""

    def __init__(self):
        self.anime_service = AnimeService()
        self.user_service = UserService()

    async def handle_mal_user_animelist(self, path: str, params: Dict) -> Dict:
        """Convert MAL user animelist calls to new format"""
        # Extract user_id from path
        user_id = self.extract_user_id_from_path(path)

        # Get user anime list from new service
        anime_list = await self.user_service.get_user_anime_list(
            user_id=user_id,
            source="mal",
            status=params.get("status"),
            limit=params.get("limit", 100)
        )

        # Convert to legacy format
        return self.convert_to_mal_format(anime_list)

    async def handle_mal_anime_details(self, path: str, params: Dict) -> Dict:
        """Convert MAL anime details calls to new format"""
        anime_id = self.extract_anime_id_from_path(path)

        # Get anime details from new service
        anime_details = await self.anime_service.get_anime_details(
            anime_id=anime_id,
            source="mal"
        )

        return self.convert_anime_to_mal_format(anime_details)

    def convert_to_mal_format(self, anime_list: List[Dict]) -> Dict:
        """Convert new format to legacy MAL format"""
        return {
            "data": [
                {
                    "node": {
                        "id": anime["mal_id"],
                        "title": anime["title"],
                        "main_picture": {
                            "medium": anime["cover_image"],
                            "large": anime["cover_image"]
                        }
                    },
                    "list_status": {
                        "status": anime["status"],
                        "score": anime["score"],
                        "num_episodes_watched": anime["episodes_watched"]
                    }
                }
                for anime in anime_list
            ]
        }
```

---

## 📋 Implementation Roadmap

### 🎯 Comprehensive Development Timeline

#### **Phase 1: Foundation Setup (Weeks 1-2)**

```mermaid
gantt
    title MCP Server Implementation Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1: Foundation
    FastAPI Setup           :active, setup, 2025-06-19, 3d
    Database Schema         :db, after setup, 4d
    MCP Protocol Basic      :mcp1, after setup, 5d
    Legacy Compatibility    :legacy, after db, 4d

    section Phase 2: Core Features
    Anime Offline DB        :anime-db, 2025-06-30, 5d
    Marqo Integration       :marqo, after anime-db, 4d
    Vector Embeddings       :vectors, after marqo, 3d
    Basic MCP Tools         :mcp-tools, after mcp1, 6d

    section Phase 3: Advanced Features
    LangChain Integration   :langchain, 2025-07-14, 4d
    Query Processing        :queries, after langchain, 3d
    Recommendation Engine   :recs, after vectors, 5d
    User Preference System  :prefs, after recs, 3d

    section Phase 4: Migration
    Traffic Router          :router, 2025-07-28, 2d
    Gradual Migration       :migration, after router, 7d
    Testing & Validation    :testing, after migration, 5d
    Full Cutover           :cutover, after testing, 2d
```

#### **Phase 1: Foundation Setup (Weeks 1-2)**

**Week 1: FastAPI Infrastructure**

```python
# Daily breakdown for Week 1
class Week1Implementation:
    def __init__(self):
        self.daily_tasks = {
            "day_1": {
                "tasks": [
                    "Initialize FastAPI project structure",
                    "Setup Docker configuration",
                    "Configure PostgreSQL database",
                    "Setup Alembic for migrations"
                ],
                "deliverables": ["Basic FastAPI app", "Database connection", "Docker setup"],
                "time_estimate": "8 hours",
                "dependencies": []
            },
            "day_2": {
                "tasks": [
                    "Create database models with SQLAlchemy",
                    "Setup Pydantic schemas",
                    "Implement basic CRUD operations",
                    "Create database migration scripts"
                ],
                "deliverables": ["Database models", "API schemas", "Basic endpoints"],
                "time_estimate": "8 hours",
                "dependencies": ["day_1"]
            },
            "day_3": {
                "tasks": [
                    "Setup MCP protocol handler",
                    "Implement basic MCP server",
                    "Create tool registration system",
                    "Setup MCP client testing"
                ],
                "deliverables": ["MCP server framework", "Tool registration", "Basic tests"],
                "time_estimate": "8 hours",
                "dependencies": ["day_2"]
            },
            "day_4": {
                "tasks": [
                    "Implement OAuth authentication services",
                    "Create JWT token management",
                    "Setup user registration/login",
                    "Integrate with MAL/AniList auth"
                ],
                "deliverables": ["Auth system", "Token management", "OAuth integration"],
                "time_estimate": "8 hours",
                "dependencies": ["day_3"]
            },
            "day_5": {
                "tasks": [
                    "Create legacy API compatibility layer",
                    "Setup traffic routing",
                    "Implement API versioning",
                    "Create health check endpoints"
                ],
                "deliverables": ["Legacy compatibility", "API routing", "Health checks"],
                "time_estimate": "8 hours",
                "dependencies": ["day_4"]
            }
        }
```

**Week 2: Data Integration**

```python
class Week2Implementation:
    def __init__(self):
        self.daily_tasks = {
            "day_6": {
                "tasks": [
                    "Setup Marqo vector database",
                    "Configure embedding models",
                    "Create vector index schemas",
                    "Test basic vector operations"
                ],
                "deliverables": ["Marqo setup", "Vector indexes", "Basic search"],
                "time_estimate": "8 hours",
                "dependencies": ["week_1"]
            },
            "day_7": {
                "tasks": [
                    "Download anime offline database",
                    "Create data processing pipeline",
                    "Implement embedding generation",
                    "Setup data validation"
                ],
                "deliverables": ["Data pipeline", "Embedding system", "Validation"],
                "time_estimate": "8 hours",
                "dependencies": ["day_6"]
            },
            "day_8": {
                "tasks": [
                    "Populate vector database",
                    "Create search indexes",
                    "Optimize query performance",
                    "Setup data refresh pipeline"
                ],
                "deliverables": ["Populated database", "Search optimization", "Refresh system"],
                "time_estimate": "8 hours",
                "dependencies": ["day_7"]
            },
            "day_9": {
                "tasks": [
                    "Implement anime service layer",
                    "Create recommendation engine",
                    "Setup user preference tracking",
                    "Test recommendation accuracy"
                ],
                "deliverables": ["Anime service", "Recommendation engine", "Preference system"],
                "time_estimate": "8 hours",
                "dependencies": ["day_8"]
            },
            "day_10": {
                "tasks": [
                    "Create MCP tools for anime operations",
                    "Implement search and recommendation tools",
                    "Setup user management tools",
                    "Test MCP tool functionality"
                ],
                "deliverables": ["MCP tools", "Search/recommendation", "User tools"],
                "time_estimate": "8 hours",
                "dependencies": ["day_9"]
            }
        }
```

#### **Phase 2: Core Features (Weeks 3-4)**

**Advanced MCP Tools Implementation**

```python
# Detailed implementation plan for core MCP tools
class MCPToolsImplementation:
    def __init__(self):
        self.tool_specifications = {
            "search_anime": {
                "complexity": "medium",
                "time_estimate": "2 days",
                "features": [
                    "Multi-source search (MAL/AniList/Jikan)",
                    "Semantic vector search",
                    "Advanced filtering",
                    "Fuzzy matching",
                    "Search result ranking"
                ],
                "dependencies": ["vector_service", "anime_service"],
                "test_coverage": ["unit_tests", "integration_tests", "performance_tests"]
            },
            "get_recommendations": {
                "complexity": "high",
                "time_estimate": "3 days",
                "features": [
                    "Vector similarity recommendations",
                    "Collaborative filtering",
                    "Content-based filtering",
                    "Hybrid recommendation system",
                    "Contextual recommendations",
                    "Explanation generation"
                ],
                "dependencies": ["vector_service", "user_service", "marqo_client"],
                "test_coverage": ["unit_tests", "integration_tests", "accuracy_tests"]
            },
            "update_watchlist": {
                "complexity": "medium",
                "time_estimate": "1.5 days",
                "features": [
                    "Multi-platform sync (MAL/AniList)",
                    "Batch updates",
                    "Conflict resolution",
                    "Update validation",
                    "History tracking"
                ],
                "dependencies": ["user_service", "auth_service"],
                "test_coverage": ["unit_tests", "integration_tests", "sync_tests"]
            },
            "analyze_user_preferences": {
                "complexity": "high",
                "time_estimate": "2.5 days",
                "features": [
                    "Preference vector generation",
                    "Genre analysis",
                    "Rating pattern analysis",
                    "Recommendation accuracy tracking",
                    "Preference evolution tracking"
                ],
                "dependencies": ["vector_service", "analytics_service"],
                "test_coverage": ["unit_tests", "analysis_tests", "accuracy_tests"]
            },
            "natural_language_query": {
                "complexity": "very_high",
                "time_estimate": "4 days",
                "features": [
                    "Intent classification",
                    "Entity extraction",
                    "Query processing pipeline",
                    "LangChain integration",
                    "Context awareness",
                    "Multi-turn conversations"
                ],
                "dependencies": ["langchain_service", "query_processor", "all_other_tools"],
                "test_coverage": ["unit_tests", "nlp_tests", "conversation_tests"]
            }
        }
```

#### **Phase 3: Advanced Features (Weeks 5-6)**

**LangChain Integration Deep Dive**

```python
class LangChainImplementationPlan:
    def __init__(self):
        self.implementation_phases = {
            "phase_3a_langchain_setup": {
                "duration": "3 days",
                "components": [
                    {
                        "name": "LangChain Agent Setup",
                        "tasks": [
                            "Initialize ChatOpenAI with GPT-4",
                            "Setup conversation memory",
                            "Create tool conversion system",
                            "Implement agent initialization"
                        ],
                        "time": "1 day"
                    },
                    {
                        "name": "Query Processing Pipeline",
                        "tasks": [
                            "Intent classification system",
                            "Entity extraction (NER)",
                            "Query routing logic",
                            "Context enhancement"
                        ],
                        "time": "1.5 days"
                    },
                    {
                        "name": "Custom Tools Development",
                        "tasks": [
                            "Anime comparison tool",
                            "Recommendation explainer",
                            "User taste analyzer",
                            "Seasonal curator"
                        ],
                        "time": "0.5 days"
                    }
                ]
            },
            "phase_3b_advanced_nlp": {
                "duration": "2 days",
                "components": [
                    {
                        "name": "Conversational AI",
                        "tasks": [
                            "Multi-turn conversation handling",
                            "Context persistence",
                            "Conversation history analysis",
                            "Personality and tone consistency"
                        ],
                        "time": "1 day"
                    },
                    {
                        "name": "Advanced Query Understanding",
                        "tasks": [
                            "Complex query parsing",
                            "Ambiguity resolution",
                            "Preference inference",
                            "Implicit intent detection"
                        ],
                        "time": "1 day"
                    }
                ]
            }
        }

    def get_detailed_langchain_architecture(self):
        return {
            "core_components": {
                "agent_executor": {
                    "type": "CHAT_CONVERSATIONAL_REACT_DESCRIPTION",
                    "memory": "ConversationBufferWindowMemory(k=10)",
                    "tools": ["mcp_tools", "custom_tools"],
                    "llm": "ChatOpenAI(model='gpt-4', temperature=0.7)"
                },
                "query_processor": {
                    "intent_classifier": "Pattern-based + ML hybrid",
                    "entity_extractor": "spaCy + custom NER",
                    "query_router": "Rule-based routing with fallback",
                    "context_enhancer": "User history + preferences"
                },
                "conversation_manager": {
                    "memory_type": "Persistent + Window",
                    "context_tracking": "Multi-session awareness",
                    "personality": "Anime-enthusiast assistant",
                    "response_formatting": "Structured + conversational"
                }
            },
            "integration_patterns": {
                "mcp_tool_calling": {
                    "async_tool_execution": True,
                    "error_handling": "Graceful degradation",
                    "result_formatting": "Structured JSON + natural language",
                    "caching": "Tool result caching for performance"
                },
                "vector_search_integration": {
                    "embedding_consistency": "Same models as Marqo",
                    "similarity_thresholds": "Dynamic based on query type",
                    "result_reranking": "LLM-based relevance scoring",
                    "explanation_generation": "Vector similarity explanations"
                }
            }
        }
```

#### **Phase 4: Migration & Testing (Weeks 7-8)**

**Migration Strategy Implementation**

```python
class MigrationExecutionPlan:
    def __init__(self):
        self.migration_schedule = {
            "week_7_preparation": {
                "day_1": {
                    "tasks": [
                        "Deploy FastAPI server to staging",
                        "Setup monitoring and logging",
                        "Create migration dashboard",
                        "Prepare rollback procedures"
                    ],
                    "metrics": ["Response times", "Error rates", "Resource usage"],
                    "success_criteria": ["<100ms avg response", "<1% error rate"]
                },
                "day_2": {
                    "tasks": [
                        "Implement traffic router",
                        "Setup A/B testing framework",
                        "Create feature flags",
                        "Configure load balancing"
                    ],
                    "traffic_split": "0% FastAPI, 100% Express.js",
                    "monitoring": "Full observability stack"
                },
                "day_3": {
                    "tasks": [
                        "Begin 5% traffic migration",
                        "Monitor error rates and performance",
                        "Validate data consistency",
                        "Test MCP functionality"
                    ],
                    "traffic_split": "5% FastAPI, 95% Express.js",
                    "rollback_trigger": "Error rate > 2% or latency > 200ms"
                },
                "day_4": {
                    "tasks": [
                        "Increase to 15% traffic",
                        "Load testing and optimization",
                        "Performance tuning",
                        "Database query optimization"
                    ],
                    "traffic_split": "15% FastAPI, 85% Express.js",
                    "optimization_targets": ["Query performance", "Memory usage"]
                },
                "day_5": {
                    "tasks": [
                        "Scale to 30% traffic",
                        "Monitor user experience metrics",
                        "Validate MCP tool performance",
                        "Test recommendation accuracy"
                    ],
                    "traffic_split": "30% FastAPI, 70% Express.js",
                    "quality_metrics": ["Recommendation accuracy", "Search relevance"]
                }
            },
            "week_8_completion": {
                "day_1": {
                    "tasks": [
                        "Increase to 60% traffic",
                        "Full feature validation",
                        "Performance stress testing",
                        "Data integrity verification"
                    ],
                    "traffic_split": "60% FastAPI, 40% Express.js"
                },
                "day_2": {
                    "tasks": [
                        "Scale to 85% traffic",
                        "Final optimization round",
                        "User acceptance testing",
                        "AI assistant integration testing"
                    ],
                    "traffic_split": "85% FastAPI, 15% Express.js"
                },
                "day_3": {
                    "tasks": [
                        "Complete migration to 100%",
                        "Decommission Express.js proxy",
                        "Clean up legacy code",
                        "Update documentation"
                    ],
                    "traffic_split": "100% FastAPI, 0% Express.js"
                },
                "day_4": {
                    "tasks": [
                        "Post-migration validation",
                        "Performance optimization",
                        "Monitor MCP usage patterns",
                        "Gather user feedback"
                    ],
                    "validation": "Full system validation"
                },
                "day_5": {
                    "tasks": [
                        "Final cleanup and optimization",
                        "Documentation updates",
                        "Team training on new system",
                        "Plan Phase 2 enhancements"
                    ],
                    "deliverable": "Fully migrated MCP server"
                }
            }
        }
```

---

## 🧪 Testing Strategy

### 🎯 Comprehensive Testing Framework

#### **Multi-Layer Testing Architecture**

```python
# Testing framework configuration
class TestingArchitecture:
    def __init__(self):
        self.testing_layers = {
            "unit_tests": {
                "framework": "pytest",
                "coverage_target": "95%",
                "focus": ["Business logic", "Data transformations", "Utility functions"],
                "tools": ["pytest", "pytest-asyncio", "pytest-mock", "coverage.py"]
            },
            "integration_tests": {
                "framework": "pytest + testcontainers",
                "coverage_target": "90%",
                "focus": ["API endpoints", "Database operations", "Service interactions"],
                "tools": ["testcontainers", "httpx", "sqlalchemy-utils"]
            },
            "mcp_protocol_tests": {
                "framework": "custom MCP test suite",
                "coverage_target": "100%",
                "focus": ["MCP tool functionality", "Protocol compliance", "AI assistant integration"],
                "tools": ["mcp-test-client", "anthropic-sdk", "openai-sdk"]
            },
            "vector_search_tests": {
                "framework": "pytest + custom metrics",
                "coverage_target": "95%",
                "focus": ["Search accuracy", "Recommendation quality", "Vector similarity"],
                "tools": ["scikit-learn", "numpy", "custom similarity metrics"]
            },
            "performance_tests": {
                "framework": "locust + pytest-benchmark",
                "targets": ["<100ms API response", "1000+ concurrent users", "99.9% uptime"],
                "focus": ["Load testing", "Stress testing", "Endurance testing"],
                "tools": ["locust", "pytest-benchmark", "memory-profiler"]
            },
            "e2e_tests": {
                "framework": "playwright + pytest",
                "coverage": "Critical user journeys",
                "focus": ["Frontend integration", "MCP workflows", "User scenarios"],
                "tools": ["playwright", "selenium", "custom test harness"]
            }
        }
```

#### **MCP Protocol Testing Suite**

```python
# tests/mcp/test_mcp_protocol.py
import pytest
from mcp.client import ClientSession
from mcp.client.stdio import stdio_client
from mcp.types import CallToolRequest, GetToolRequest

class TestMCPProtocol:
    """Comprehensive MCP protocol compliance testing"""

    @pytest.fixture
    async def mcp_client(self):
        """Setup MCP client for testing"""
        from app.mcp.server import AnimeTrackrMCPServer

        server = AnimeTrackrMCPServer()
        # Setup test client connection
        async with stdio_client() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    @pytest.mark.asyncio
    async def test_search_anime_tool(self, mcp_client):
        """Test anime search tool functionality"""
        # Test basic search
        result = await mcp_client.call_tool(
            CallToolRequest(
                name="search_anime",
                arguments={"query": "Attack on Titan", "limit": 5}
            )
        )

        assert result.content[0].type == "text"
        search_results = json.loads(result.content[0].text)
        assert len(search_results) <= 5
        assert any("Attack on Titan" in anime["title"] for anime in search_results)

        # Test with filters
        result = await mcp_client.call_tool(
            CallToolRequest(
                name="search_anime",
                arguments={
                    "query": "action anime",
                    "filters": {"genres": ["action"], "min_score": 8.0},
                    "limit": 10
                }
            )
        )

        filtered_results = json.loads(result.content[0].text)
        assert all(
            "action" in [genre.lower() for genre in anime.get("genres", [])]
            for anime in filtered_results
        )

    @pytest.mark.asyncio
    async def test_recommendation_tool(self, mcp_client):
        """Test recommendation tool with various scenarios"""
        # Test user-based recommendations
        result = await mcp_client.call_tool(
            CallToolRequest(
                name="get_recommendations",
                arguments={
                    "user_id": "test_user_123",
                    "context": "I liked Death Note and Code Geass",
                    "limit": 10
                }
            )
        )

        recommendations = json.loads(result.content[0].text)
        assert len(recommendations) <= 10
        assert all("similarity_score" in rec for rec in recommendations)
        assert all(rec["similarity_score"] > 0 for rec in recommendations)

        # Test contextual recommendations
        result = await mcp_client.call_tool(
            CallToolRequest(
                name="get_recommendations",
                arguments={
                    "user_id": "test_user_123",
                    "context": "something light and funny for a weekend",
                    "limit": 5
                }
            )
        )

        contextual_recs = json.loads(result.content[0].text)
        # Should return comedy/slice-of-life anime
        assert any(
            "comedy" in [genre.lower() for genre in anime.get("genres", [])]
            for anime in contextual_recs
        )

    @pytest.mark.asyncio
    async def test_natural_language_query_tool(self, mcp_client):
        """Test natural language processing capabilities"""
        test_queries = [
            "Find me anime similar to Naruto but shorter",
            "What should I watch if I liked Your Name?",
            "Recommend some romance anime from 2023",
            "Add Demon Slayer to my completed list with a score of 9"
        ]

        for query in test_queries:
            result = await mcp_client.call_tool(
                CallToolRequest(
                    name="natural_language_query",
                    arguments={
                        "query": query,
                        "user_context": {"user_id": "test_user_123"}
                    }
                )
            )

            response = json.loads(result.content[0].text)
            assert response["success"] is True
            assert len(response["response"]) > 0
            assert response["query"] == query

    @pytest.mark.asyncio
    async def test_watchlist_management_tool(self, mcp_client):
        """Test watchlist management functionality"""
        # Test adding anime to watchlist
        result = await mcp_client.call_tool(
            CallToolRequest(
                name="update_watchlist",
                arguments={
                    "user_id": "test_user_123",
                    "anime_id": "test_anime_456",
                    "status": "watching",
                    "score": 8
                }
            )
        )

        update_result = json.loads(result.content[0].text)
        assert update_result["success"] is True
        assert update_result["anime_id"] == "test_anime_456"
        assert update_result["status"] == "watching"
        assert update_result["score"] == 8

        # Test batch updates
        batch_updates = [
            {"anime_id": "anime_1", "status": "completed", "score": 9},
            {"anime_id": "anime_2", "status": "dropped", "score": 4},
            {"anime_id": "anime_3", "status": "plan_to_watch"}
        ]

        for update in batch_updates:
            result = await mcp_client.call_tool(
                CallToolRequest(
                    name="update_watchlist",
                    arguments={"user_id": "test_user_123", **update}
                )
            )
            response = json.loads(result.content[0].text)
            assert response["success"] is True
```

#### **Vector Search Accuracy Testing**

```python
# tests/vector/test_recommendation_accuracy.py
import pytest
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from app.services.vector_service import VectorService
from app.integrations.marqo.client import MarqoClient

class TestRecommendationAccuracy:
    """Test vector-based recommendation accuracy"""

    @pytest.fixture
    def recommendation_test_data(self):
        """Load test dataset with known good recommendations"""
        return {
            "test_users": [
                {
                    "user_id": "accuracy_test_1",
                    "watch_history": [
                        {"anime_id": "death_note", "score": 10},
                        {"anime_id": "code_geass", "score": 9},
                        {"anime_id": "psycho_pass", "score": 8}
                    ],
                    "expected_recommendations": [
                        "monster", "paranoia_agent", "serial_experiments_lain"
                    ]
                },
                {
                    "user_id": "accuracy_test_2",
                    "watch_history": [
                        {"anime_id": "your_name", "score": 10},
                        {"anime_id": "spirited_away", "score": 9},
                        {"anime_id": "weathering_with_you", "score": 8}
                    ],
                    "expected_recommendations": [
                        "a_silent_voice", "garden_of_words", "5_cm_per_second"
                    ]
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_recommendation_precision(self, recommendation_test_data):
        """Test recommendation precision for known user preferences"""
        vector_service = VectorService()

        precision_scores = []

        for test_user in recommendation_test_data["test_users"]:
            # Generate recommendations
            recommendations = await vector_service.contextual_anime_search(
                query="",  # Empty query to rely on user history
                user_id=test_user["user_id"],
                limit=10
            )

            recommended_ids = [rec["anime_id"] for rec in recommendations]
            expected_ids = test_user["expected_recommendations"]

            # Calculate precision
            true_positives = len(set(recommended_ids) & set(expected_ids))
            precision = true_positives / len(recommended_ids) if recommended_ids else 0
            precision_scores.append(precision)

        avg_precision = np.mean(precision_scores)
        assert avg_precision >= 0.3  # At least 30% precision

    @pytest.mark.asyncio
    async def test_similarity_consistency(self):
        """Test that similar anime consistently appear in recommendations"""
        vector_service = VectorService()

        # Test anime known to be similar
        similar_anime_groups = [
            ["attack_on_titan", "fullmetal_alchemist", "demon_slayer"],
            ["your_name", "spirited_away", "howls_moving_castle"],
            ["death_note", "code_geass", "monster"]
        ]

        for group in similar_anime_groups:
            similarity_matrix = []

            for anime_id in group:
                similar_anime = await vector_service.marqo.get_similar_anime(
                    anime_id=anime_id,
                    limit=20
                )

                similar_ids = [anime["anime_id"] for anime in similar_anime]
                group_overlap = len(set(similar_ids) & set(group))
                similarity_matrix.append(group_overlap)

            # Each anime should find at least 1 other anime from its group
            assert all(overlap >= 1 for overlap in similarity_matrix)

    @pytest.mark.asyncio
    async def test_diversity_in_recommendations(self):
        """Test that recommendations provide good diversity"""
        vector_service = VectorService()

        # Test user with diverse preferences
        test_user_id = "diversity_test_user"
        recommendations = await vector_service.contextual_anime_search(
            query="recommend diverse anime",
            user_id=test_user_id,
            limit=20
        )

        # Extract genres from recommendations
        all_genres = []
        for rec in recommendations:
            all_genres.extend(rec.get("genres", []))

        unique_genres = set(all_genres)

        # Should have at least 5 different genres
        assert len(unique_genres) >= 5

        # No single genre should dominate (>50% of recommendations)
        genre_counts = {genre: all_genres.count(genre) for genre in unique_genres}
        max_genre_percentage = max(genre_counts.values()) / len(recommendations)
        assert max_genre_percentage <= 0.5
```

#### **Performance Testing Suite**

```python
# tests/performance/test_load_performance.py
import asyncio
import time
import pytest
from locust import HttpUser, task, between
import statistics

class MCPServerLoadTest(HttpUser):
    """Load testing for MCP server endpoints"""

    wait_time = between(1, 3)

    def on_start(self):
        """Setup for each test user"""
        self.test_user_id = f"load_test_user_{self.user_id}"

    @task(3)
    def search_anime(self):
        """Test anime search performance"""
        queries = [
            "Naruto", "One Piece", "Attack on Titan",
            "Death Note", "Dragon Ball", "Pokemon"
        ]

        query = self.environment.random.choice(queries)

        response = self.client.post("/api/v1/mcp/call_tool", json={
            "name": "search_anime",
            "arguments": {"query": query, "limit": 10}
        })

        if response.status_code == 200:
            response_time = response.elapsed.total_seconds()
            assert response_time < 0.5  # Should respond within 500ms

    @task(2)
    def get_recommendations(self):
        """Test recommendation performance"""
        response = self.client.post("/api/v1/mcp/call_tool", json={
            "name": "get_recommendations",
            "arguments": {
                "user_id": self.test_user_id,
                "context": "something interesting",
                "limit": 10
            }
        })

        if response.status_code == 200:
            response_time = response.elapsed.total_seconds()
            assert response_time < 1.0  # Recommendations can take up to 1s

    @task(1)
    def natural_language_query(self):
        """Test NLP query performance"""
        queries = [
            "Find me something like Naruto",
            "What should I watch next?",
            "Recommend comedy anime",
            "Add One Piece to my watchlist"
        ]

        query = self.environment.random.choice(queries)

        response = self.client.post("/api/v1/mcp/call_tool", json={
            "name": "natural_language_query",
            "arguments": {
                "query": query,
                "user_context": {"user_id": self.test_user_id}
            }
        })

        if response.status_code == 200:
            response_time = response.elapsed.total_seconds()
            assert response_time < 2.0  # NLP can take up to 2s

@pytest.mark.performance
class TestVectorSearchPerformance:
    """Test vector search and embedding performance"""

    @pytest.mark.asyncio
    async def test_batch_embedding_performance(self):
        """Test performance of batch embedding generation"""
        from app.services.vector_service import VectorService

        vector_service = VectorService()

        # Test with different batch sizes
        batch_sizes = [10, 50, 100, 500]
        performance_results = {}

        for batch_size in batch_sizes:
            test_texts = [f"Test anime description {i}" for i in range(batch_size)]

            start_time = time.time()
            embeddings = await vector_service.generate_embeddings({
                f"text_{i}": text for i, text in enumerate(test_texts)
            })
            end_time = time.time()

            processing_time = end_time - start_time
            per_item_time = processing_time / batch_size

            performance_results[batch_size] = {
                "total_time": processing_time,
                "per_item_time": per_item_time,
                "throughput": batch_size / processing_time
            }

            # Performance assertions
            assert per_item_time < 0.1  # Less than 100ms per item
            assert processing_time < 30  # Total time under 30s

        # Log performance results for analysis
        print(f"Embedding performance results: {performance_results}")

    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self):
        """Test performance under concurrent search loads"""
        from app.services.vector_service import VectorService

        vector_service = VectorService()

        async def search_task(query_id):
            start_time = time.time()
            results = await vector_service.contextual_anime_search(
                query=f"test query {query_id}",
                limit=10
            )
            end_time = time.time()
            return end_time - start_time

        # Test with increasing concurrency levels
        concurrency_levels = [5, 10, 25, 50]

        for concurrency in concurrency_levels:
            tasks = [search_task(i) for i in range(concurrency)]

            start_time = time.time()
            response_times = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            # Performance metrics
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            throughput = concurrency / total_time

            # Assertions
            assert avg_response_time < 1.0  # Average under 1s
            assert max_response_time < 3.0  # Max under 3s
            assert throughput > 2.0  # At least 2 requests/second

            print(f"Concurrency {concurrency}: avg={avg_response_time:.3f}s, "
                  f"max={max_response_time:.3f}s, throughput={throughput:.1f}/s")
```

---

## 🚀 Deployment Architecture

### 🎯 Multi-Environment Deployment Strategy

#### **Container-First Architecture**

```yaml
# docker-compose.yml - Complete deployment stack
version: "3.8"

services:
  # FastAPI MCP Server
  mcp-server:
    build:
      context: ./anime_mcp_server
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/anime_mcp
      - REDIS_URL=redis://redis:6379
      - MARQO_URL=http://marqo:8882
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MAL_CLIENT_ID=${MAL_CLIENT_ID}
      - ANILIST_CLIENT_ID=${ANILIST_CLIENT_ID}
    depends_on:
      - postgres
      - redis
      - marqo
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: anime_mcp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  # Marqo Vector Database
  marqo:
    image: marqoai/marqo:latest
    ports:
      - "8882:8882"
    environment:
      - MARQO_MAX_CPU_MODEL_MEMORY=4000000000
      - MARQO_MAX_CUDA_MODEL_MEMORY=4000000000
    volumes:
      - marqo_data:/opt/marqo

  # Frontend React App
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://mcp-server:8000
      - VITE_MAL_CLIENT_ID=${VITE_MAL_CLIENT_ID}
      - VITE_ANILIST_CLIENT_ID=${VITE_ANILIST_CLIENT_ID}
    depends_on:
      - mcp-server

  # Legacy Express Proxy (during migration)
  legacy-proxy:
    build:
      context: .
      dockerfile: Dockerfile.proxy
    ports:
      - "3002:3002"
    environment:
      - FASTAPI_URL=http://mcp-server:8000
    depends_on:
      - mcp-server

  # Traffic Router (migration phase)
  traffic-router:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - mcp-server
      - legacy-proxy

  # Monitoring Stack
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards

volumes:
  postgres_data:
  redis_data:
  marqo_data:
  prometheus_data:
  grafana_data:
```

#### **Kubernetes Deployment**

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: anime-mcp

---
# k8s/mcp-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  namespace: anime-mcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
        - name: mcp-server
          image: anime-mcp-server:latest
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: mcp-secrets
                  key: database-url
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: mcp-secrets
                  key: openai-api-key
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5

---
# k8s/mcp-server-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-service
  namespace: anime-mcp
spec:
  selector:
    app: mcp-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mcp-server-ingress
  namespace: anime-mcp
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
    - hosts:
        - api.animetrackr.com
      secretName: mcp-server-tls
  rules:
    - host: api.animetrackr.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: mcp-server-service
                port:
                  number: 80

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server-hpa
  namespace: anime-mcp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

#### **Environment-Specific Configurations**

```python
# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 3600  # 1 hour

    # Marqo Vector Database
    MARQO_URL: str = "http://localhost:8882"
    MARQO_INDEX_PREFIX: str = "anime"

    # External APIs
    MAL_CLIENT_ID: str
    MAL_CLIENT_SECRET: str
    ANILIST_CLIENT_ID: str
    ANILIST_CLIENT_SECRET: str
    OPENAI_API_KEY: str

    # MCP Configuration
    MCP_SERVER_NAME: str = "anime-trackr"
    MCP_SERVER_VERSION: str = "1.0.0"
    MCP_TOOL_TIMEOUT: int = 30

    # Security
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 hours

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None

    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    SENTRY_DSN: Optional[str] = None

    # Performance
    ASYNC_POOL_SIZE: int = 50
    MAX_CONCURRENT_REQUESTS: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = True

# Environment-specific settings
class DevelopmentSettings(Settings):
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"

class StagingSettings(Settings):
    ENVIRONMENT: str = "staging"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

class ProductionSettings(Settings):
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    ASYNC_POOL_SIZE: int = 100
    MAX_CONCURRENT_REQUESTS: int = 500

def get_settings() -> Settings:
    env = os.getenv("ENVIRONMENT", "development")
    if env == "development":
        return DevelopmentSettings()
    elif env == "staging":
        return StagingSettings()
    elif env == "production":
        return ProductionSettings()
    else:
        return Settings()

settings = get_settings()
```

---

## 📊 Monitoring & Observability

### 🎯 Comprehensive Monitoring Stack

#### **Application Metrics & Instrumentation**

```python
# app/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from functools import wraps
from typing import Callable

# Prometheus metrics registry
REGISTRY = CollectorRegistry()

# MCP Tool Metrics
MCP_TOOL_CALLS_TOTAL = Counter(
    'mcp_tool_calls_total',
    'Total number of MCP tool calls',
    ['tool_name', 'status'],
    registry=REGISTRY
)

MCP_TOOL_DURATION = Histogram(
    'mcp_tool_duration_seconds',
    'Duration of MCP tool calls',
    ['tool_name'],
    registry=REGISTRY,
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Vector Search Metrics
VECTOR_SEARCH_REQUESTS = Counter(
    'vector_search_requests_total',
    'Total vector search requests',
    ['search_type', 'status'],
    registry=REGISTRY
)

VECTOR_SEARCH_DURATION = Histogram(
    'vector_search_duration_seconds',
    'Vector search duration',
    ['search_type'],
    registry=REGISTRY
)

VECTOR_SEARCH_RESULTS = Histogram(
    'vector_search_results_count',
    'Number of results returned by vector search',
    ['search_type'],
    registry=REGISTRY,
    buckets=[1, 5, 10, 20, 50, 100]
)

# Recommendation Metrics
RECOMMENDATION_REQUESTS = Counter(
    'recommendation_requests_total',
    'Total recommendation requests',
    ['recommendation_type', 'status'],
    registry=REGISTRY
)

RECOMMENDATION_ACCURACY = Gauge(
    'recommendation_accuracy_score',
    'Recommendation accuracy score',
    ['recommendation_type'],
    registry=REGISTRY
)

# Database Metrics
DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Number of active database connections',
    registry=REGISTRY
)

DATABASE_QUERY_DURATION = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['table', 'operation'],
    registry=REGISTRY
)

# Cache Metrics
CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type'],
    registry=REGISTRY
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type'],
    registry=REGISTRY
)

# Error Metrics
ERROR_RATE = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'component'],
    registry=REGISTRY
)

def monitor_mcp_tool(tool_name: str):
    """Decorator to monitor MCP tool performance"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                ERROR_RATE.labels(
                    error_type=type(e).__name__,
                    component=f"mcp_tool_{tool_name}"
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                MCP_TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status=status).inc()
                MCP_TOOL_DURATION.labels(tool_name=tool_name).observe(duration)

        return wrapper
    return decorator

def monitor_vector_search(search_type: str):
    """Decorator to monitor vector search performance"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            result_count = 0

            try:
                result = await func(*args, **kwargs)
                result_count = len(result) if isinstance(result, list) else 1
                return result
            except Exception as e:
                status = "error"
                ERROR_RATE.labels(
                    error_type=type(e).__name__,
                    component=f"vector_search_{search_type}"
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                VECTOR_SEARCH_REQUESTS.labels(search_type=search_type, status=status).inc()
                VECTOR_SEARCH_DURATION.labels(search_type=search_type).observe(duration)
                if status == "success":
                    VECTOR_SEARCH_RESULTS.labels(search_type=search_type).observe(result_count)

        return wrapper
    return decorator
```

#### **Structured Logging**

```python
# app/monitoring/logging.py
import logging
import json
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
from app.core.config import settings

# Context variables for request tracing
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar('user_id', default=None)

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context information
        request_id = request_id_context.get()
        user_id = user_id_context.get()

        if request_id:
            log_entry["request_id"] = request_id
        if user_id:
            log_entry["user_id"] = user_id

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in log_entry and not key.startswith('_'):
                if isinstance(value, (str, int, float, bool, list, dict)):
                    log_entry[key] = value

        return json.dumps(log_entry)

def setup_logging():
    """Setup application logging configuration"""

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # File handler (if specified)
    if settings.LOG_FILE:
        file_handler = logging.FileHandler(settings.LOG_FILE)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    # Disable some noisy loggers in production
    if settings.ENVIRONMENT == "production":
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

class MCPLogger:
    """Specialized logger for MCP operations"""

    def __init__(self):
        self.logger = logging.getLogger("mcp")

    def tool_call_start(self, tool_name: str, arguments: Dict[str, Any]):
        """Log MCP tool call start"""
        self.logger.info(
            "MCP tool call started",
            extra={
                "event_type": "mcp_tool_start",
                "tool_name": tool_name,
                "arguments": arguments
            }
        )

    def tool_call_complete(self, tool_name: str, duration: float, result_size: int):
        """Log MCP tool call completion"""
        self.logger.info(
            "MCP tool call completed",
            extra={
                "event_type": "mcp_tool_complete",
                "tool_name": tool_name,
                "duration": duration,
                "result_size": result_size
            }
        )

    def tool_call_error(self, tool_name: str, error: Exception, duration: float):
        """Log MCP tool call error"""
        self.logger.error(
            "MCP tool call failed",
            exc_info=error,
            extra={
                "event_type": "mcp_tool_error",
                "tool_name": tool_name,
                "duration": duration,
                "error_type": type(error).__name__
            }
        )

    def recommendation_generated(self, user_id: str, recommendation_count: int, context: str):
        """Log recommendation generation"""
        self.logger.info(
            "Recommendations generated",
            extra={
                "event_type": "recommendation_generated",
                "user_id": user_id,
                "recommendation_count": recommendation_count,
                "context": context
            }
        )

    def vector_search_performed(self, query: str, results_count: int, duration: float):
        """Log vector search operation"""
        self.logger.info(
            "Vector search performed",
            extra={
                "event_type": "vector_search",
                "query": query,
                "results_count": results_count,
                "duration": duration
            }
        )

# Global logger instance
mcp_logger = MCPLogger()
```

#### **Health Checks & Diagnostics**

```python
# app/monitoring/health.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import asyncio
import time
from sqlalchemy import text
from app.core.database import get_db
from app.integrations.marqo.client import MarqoClient
from app.services.cache_service import CacheService

router = APIRouter()

class HealthChecker:
    def __init__(self):
        self.checks = {
            "database": self._check_database,
            "redis": self._check_redis,
            "marqo": self._check_marqo,
            "external_apis": self._check_external_apis,
            "mcp_tools": self._check_mcp_tools
        }

    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            from app.core.database import engine

            start_time = time.time()
            async with engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

            duration = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": round(duration * 1000, 2),  # ms
                "details": "Database connection successful"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Database connection failed"
            }

    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        try:
            cache_service = CacheService()

            start_time = time.time()
            test_key = "health_check_test"
            test_value = "test_value"

            await cache_service.set(test_key, test_value, ttl=10)
            retrieved_value = await cache_service.get(test_key)
            await cache_service.delete(test_key)

            duration = time.time() - start_time

            if retrieved_value == test_value:
                return {
                    "status": "healthy",
                    "response_time": round(duration * 1000, 2),
                    "details": "Redis read/write successful"
                }
            else:
                return {
                    "status": "unhealthy",
                    "details": "Redis read/write failed"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Redis connection failed"
            }

    async def _check_marqo(self) -> Dict[str, Any]:
        """Check Marqo vector database connectivity"""
        try:
            marqo_client = MarqoClient()

            start_time = time.time()
            # Try to get index stats (lightweight operation)
            indexes = marqo_client.client.get_indexes()
            duration = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": round(duration * 1000, 2),
                "details": f"Marqo responding, {len(indexes['results'])} indexes available"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Marqo connection failed"
            }

    async def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API availability"""
        try:
            import aiohttp

            apis_to_check = [
                ("MAL", "https://api.myanimelist.net/v2/anime/ranking"),
                ("Jikan", "https://api.jikan.moe/v4/anime/1"),
            ]

            results = {}
            overall_status = "healthy"

            async with aiohttp.ClientSession() as session:
                for api_name, url in apis_to_check:
                    try:
                        start_time = time.time()
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            duration = time.time() - start_time

                            if response.status == 200:
                                results[api_name] = {
                                    "status": "healthy",
                                    "response_time": round(duration * 1000, 2),
                                    "status_code": response.status
                                }
                            else:
                                results[api_name] = {
                                    "status": "degraded",
                                    "status_code": response.status
                                }
                                overall_status = "degraded"
                    except Exception as e:
                        results[api_name] = {
                            "status": "unhealthy",
                            "error": str(e)
                        }
                        overall_status = "unhealthy"

            return {
                "status": overall_status,
                "details": results
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "External API checks failed"
            }

    async def _check_mcp_tools(self) -> Dict[str, Any]:
        """Check MCP tools functionality"""
        try:
            from app.mcp.tools.anime_search import search_anime

            # Test basic search tool
            start_time = time.time()
            test_result = await search_anime({
                "query": "test",
                "limit": 1
            })
            duration = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": round(duration * 1000, 2),
                "details": "MCP tools responding"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "MCP tools check failed"
            }

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks concurrently"""
        results = await asyncio.gather(
            *[check() for check in self.checks.values()],
            return_exceptions=True
        )

        check_results = {}
        overall_status = "healthy"

        for i, (check_name, result) in enumerate(zip(self.checks.keys(), results)):
            if isinstance(result, Exception):
                check_results[check_name] = {
                    "status": "unhealthy",
                    "error": str(result)
                }
                overall_status = "unhealthy"
            else:
                check_results[check_name] = result
                if result["status"] != "healthy":
                    overall_status = result["status"]

        return {
            "status": overall_status,
            "timestamp": time.time(),
            "checks": check_results
        }

health_checker = HealthChecker()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all system components"""
    results = await health_checker.run_all_checks()

    if results["status"] != "healthy":
        raise HTTPException(status_code=503, detail=results)

    return results

@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe endpoint"""
    results = await health_checker.run_all_checks()

    # For readiness, we're more strict - all components must be healthy
    critical_checks = ["database", "redis", "marqo"]

    for check_name in critical_checks:
        check_result = results["checks"].get(check_name, {})
        if check_result.get("status") != "healthy":
            raise HTTPException(
                status_code=503,
                detail=f"Critical component {check_name} is not ready"
            )

    return {"status": "ready", "timestamp": time.time()}
```

---

## ⚠️ Risk Assessment & Mitigation

### 🎯 Comprehensive Risk Analysis

#### **Technical Risks**

```python
class RiskAssessment:
    def __init__(self):
        self.risks = {
            "migration_risks": {
                "data_loss_during_migration": {
                    "probability": "low",
                    "impact": "high",
                    "description": "Potential data loss during Express.js to FastAPI migration",
                    "mitigation_strategies": [
                        "Implement comprehensive backup strategy",
                        "Use parallel deployment with traffic routing",
                        "Maintain rollback capability throughout migration",
                        "Perform extensive testing with production data copies"
                    ],
                    "detection_methods": [
                        "Automated data integrity checks",
                        "Real-time monitoring of data consistency",
                        "User-facing error rate monitoring"
                    ],
                    "recovery_procedures": [
                        "Immediate rollback to Express.js proxy",
                        "Restore from automated backups",
                        "Manual data reconciliation if needed"
                    ]
                },
                "performance_degradation": {
                    "probability": "medium",
                    "impact": "medium",
                    "description": "Potential performance issues with new FastAPI implementation",
                    "mitigation_strategies": [
                        "Extensive load testing before deployment",
                        "Gradual traffic migration with monitoring",
                        "Performance baselines and SLA definitions",
                        "Auto-scaling configuration"
                    ],
                    "detection_methods": [
                        "Response time monitoring",
                        "Throughput measurements",
                        "Resource utilization alerts"
                    ],
                    "recovery_procedures": [
                        "Traffic throttling",
                        "Horizontal scaling",
                        "Rollback to previous version"
                    ]
                }
            },
            "integration_risks": {
                "marqo_dependency": {
                    "probability": "medium",
                    "impact": "high",
                    "description": "Heavy dependency on Marqo vector database for core functionality",
                    "mitigation_strategies": [
                        "Implement fallback search using traditional methods",
                        "Regular backup of vector indexes",
                        "Marqo cluster deployment for redundancy",
                        "Alternative vector database evaluation (Pinecone, Weaviate)"
                    ],
                    "detection_methods": [
                        "Marqo health checks",
                        "Vector search response time monitoring",
                        "Search accuracy degradation alerts"
                    ],
                    "recovery_procedures": [
                        "Failover to backup Marqo instance",
                        "Switch to traditional search temporarily",
                        "Rebuild vector indexes from source data"
                    ]
                },
                "external_api_rate_limits": {
                    "probability": "high",
                    "impact": "medium",
                    "description": "Rate limiting from MAL, AniList, or Jikan APIs",
                    "mitigation_strategies": [
                        "Implement intelligent caching strategies",
                        "Rate limiting and request queuing",
                        "Multiple API key rotation",
                        "Graceful degradation to cached data"
                    ],
                    "detection_methods": [
                        "API response status monitoring",
                        "Rate limit header tracking",
                        "Request queue length monitoring"
                    ],
                    "recovery_procedures": [
                        "Switch to alternative data sources",
                        "Serve from cache with staleness warnings",
                        "Implement exponential backoff"
                    ]
                }
            },
            "operational_risks": {
                "vector_database_corruption": {
                    "probability": "low",
                    "impact": "high",
                    "description": "Corruption of vector indexes affecting recommendations",
                    "mitigation_strategies": [
                        "Regular automated backups of vector indexes",
                        "Checksums and integrity verification",
                        "Blue-green deployment for vector database updates",
                        "Source data retention for index rebuilding"
                    ],
                    "detection_methods": [
                        "Search result quality monitoring",
                        "Vector similarity score validation",
                        "Automated consistency checks"
                    ],
                    "recovery_procedures": [
                        "Restore from last known good backup",
                        "Rebuild indexes from source data",
                        "Temporary fallback to non-vector recommendations"
                    ]
                },
                "ai_model_availability": {
                    "probability": "medium",
                    "impact": "medium",
                    "description": "OpenAI API or embedding model unavailability",
                    "mitigation_strategies": [
                        "Multi-provider strategy (OpenAI + local models)",
                        "Local embedding model fallback",
                        "Caching of previous AI responses",
                        "Circuit breaker pattern implementation"
                    ],
                    "detection_methods": [
                        "AI API response monitoring",
                        "Model response quality validation",
                        "Latency spike detection"
                    ],
                    "recovery_procedures": [
                        "Switch to backup AI provider",
                        "Use local embedding models",
                        "Serve cached/pre-computed responses"
                    ]
                }
            },
            "security_risks": {
                "api_key_exposure": {
                    "probability": "low",
                    "impact": "high",
                    "description": "Exposure of OpenAI, MAL, or AniList API keys",
                    "mitigation_strategies": [
                        "Secrets management system (HashiCorp Vault, K8s secrets)",
                        "API key rotation procedures",
                        "Environment-specific key isolation",
                        "Access logging and monitoring"
                    ],
                    "detection_methods": [
                        "Unusual API usage patterns",
                        "Rate limit breach alerts",
                        "Unauthorized access attempts"
                    ],
                    "recovery_procedures": [
                        "Immediate key revocation and rotation",
                        "API usage audit",
                        "Security incident response protocol"
                    ]
                },
                "user_data_breach": {
                    "probability": "low",
                    "impact": "high",
                    "description": "Unauthorized access to user anime lists and preferences",
                    "mitigation_strategies": [
                        "Database encryption at rest and in transit",
                        "OAuth token encryption",
                        "Regular security audits",
                        "Minimal data retention policies"
                    ],
                    "detection_methods": [
                        "Unusual database access patterns",
                        "Failed authentication alerts",
                        "Data access auditing"
                    ],
                    "recovery_procedures": [
                        "Immediate system isolation",
                        "User notification procedures",
                        "Forensic analysis and remediation"
                    ]
                }
            }
        }
```

#### **Risk Monitoring Dashboard**

```python
# app/monitoring/risk_monitor.py
from typing import Dict, List, Any
import asyncio
import time
from enum import Enum
from dataclasses import dataclass
from app.monitoring.metrics import REGISTRY

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskIndicator:
    name: str
    level: RiskLevel
    description: str
    current_value: float
    threshold: float
    timestamp: float

class RiskMonitor:
    def __init__(self):
        self.risk_thresholds = {
            "api_error_rate": {"medium": 5.0, "high": 10.0, "critical": 20.0},
            "response_time_p95": {"medium": 1000, "high": 2000, "critical": 5000},
            "database_connection_pool": {"medium": 70, "high": 85, "critical": 95},
            "vector_search_failure_rate": {"medium": 2.0, "high": 5.0, "critical": 10.0},
            "recommendation_accuracy": {"medium": 0.6, "high": 0.4, "critical": 0.2},
            "cache_hit_rate": {"medium": 70, "high": 50, "critical": 30}
        }

    async def assess_current_risks(self) -> List[RiskIndicator]:
        """Assess current system risks based on metrics"""
        risks = []

        # Get current metrics
        current_metrics = await self._collect_current_metrics()

        for metric_name, value in current_metrics.items():
            if metric_name in self.risk_thresholds:
                thresholds = self.risk_thresholds[metric_name]
                level = self._determine_risk_level(value, thresholds, metric_name)

                if level != RiskLevel.LOW:
                    risks.append(RiskIndicator(
                        name=metric_name,
                        level=level,
                        description=self._get_risk_description(metric_name, level),
                        current_value=value,
                        threshold=self._get_threshold_for_level(thresholds, level),
                        timestamp=time.time()
                    ))

        return risks

    def _determine_risk_level(self, value: float, thresholds: Dict, metric_name: str) -> RiskLevel:
        """Determine risk level based on value and thresholds"""
        # Different metrics have different direction (higher is worse vs lower is worse)
        reverse_metrics = {"recommendation_accuracy", "cache_hit_rate"}

        if metric_name in reverse_metrics:
            # For these metrics, lower values are worse
            if value <= thresholds["critical"]:
                return RiskLevel.CRITICAL
            elif value <= thresholds["high"]:
                return RiskLevel.HIGH
            elif value <= thresholds["medium"]:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        else:
            # For these metrics, higher values are worse
            if value >= thresholds["critical"]:
                return RiskLevel.CRITICAL
            elif value >= thresholds["high"]:
                return RiskLevel.HIGH
            elif value >= thresholds["medium"]:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics for risk assessment"""
        # This would integrate with your metrics collection system
        # For now, returning mock data
        return {
            "api_error_rate": 2.5,  # percentage
            "response_time_p95": 800,  # milliseconds
            "database_connection_pool": 60,  # percentage
            "vector_search_failure_rate": 1.0,  # percentage
            "recommendation_accuracy": 0.75,  # ratio
            "cache_hit_rate": 80  # percentage
        }

    def _get_risk_description(self, metric_name: str, level: RiskLevel) -> str:
        """Get human-readable risk description"""
        descriptions = {
            "api_error_rate": f"{level.value.title()} API error rate detected",
            "response_time_p95": f"{level.value.title()} response time latency",
            "database_connection_pool": f"{level.value.title()} database connection usage",
            "vector_search_failure_rate": f"{level.value.title()} vector search failures",
            "recommendation_accuracy": f"{level.value.title()} recommendation quality degradation",
            "cache_hit_rate": f"{level.value.title()} cache performance degradation"
        }
        return descriptions.get(metric_name, f"{level.value.title()} risk detected")
```

---

## 📈 Success Metrics & KPIs

### 🎯 Comprehensive Success Framework

#### **Technical Performance Metrics**

```python
class SuccessMetrics:
    def __init__(self):
        self.kpis = {
            "migration_success": {
                "zero_downtime_achievement": {
                    "target": "100%",
                    "measurement": "Uptime during migration period",
                    "critical_threshold": "99.9%",
                    "monitoring_method": "Continuous uptime monitoring"
                },
                "data_integrity_maintained": {
                    "target": "100%",
                    "measurement": "Data consistency checks",
                    "critical_threshold": "100%",
                    "monitoring_method": "Automated data validation"
                },
                "performance_improvement": {
                    "target": ">20% faster response times",
                    "measurement": "P95 response time comparison",
                    "critical_threshold": "No regression",
                    "monitoring_method": "APM tools comparison"
                }
            },
            "mcp_adoption": {
                "tool_usage_rate": {
                    "target": ">80% of AI interactions use MCP tools",
                    "measurement": "MCP tool calls / total AI requests",
                    "critical_threshold": ">50%",
                    "monitoring_method": "Tool usage analytics"
                },
                "recommendation_accuracy": {
                    "target": ">75% user satisfaction",
                    "measurement": "User feedback scores on recommendations",
                    "critical_threshold": ">60%",
                    "monitoring_method": "User surveys and implicit feedback"
                },
                "natural_language_success_rate": {
                    "target": ">90% successful query resolution",
                    "measurement": "NLP query success rate",
                    "critical_threshold": ">80%",
                    "monitoring_method": "Query resolution tracking"
                }
            },
            "vector_search_performance": {
                "search_accuracy": {
                    "target": ">80% relevance score",
                    "measurement": "Average search result relevance",
                    "critical_threshold": ">70%",
                    "monitoring_method": "Automated relevance scoring"
                },
                "search_speed": {
                    "target": "<200ms average response time",
                    "measurement": "Vector search latency",
                    "critical_threshold": "<500ms",
                    "monitoring_method": "Performance monitoring"
                },
                "recommendation_diversity": {
                    "target": ">5 different genres in top 20",
                    "measurement": "Genre diversity in recommendations",
                    "critical_threshold": ">3 genres",
                    "monitoring_method": "Diversity analysis"
                }
            },
            "system_reliability": {
                "uptime": {
                    "target": "99.9%",
                    "measurement": "System availability",
                    "critical_threshold": "99.5%",
                    "monitoring_method": "Uptime monitoring"
                },
                "error_rate": {
                    "target": "<0.1%",
                    "measurement": "Error rate across all endpoints",
                    "critical_threshold": "<1%",
                    "monitoring_method": "Error tracking"
                },
                "scalability": {
                    "target": "1000+ concurrent users",
                    "measurement": "Concurrent user capacity",
                    "critical_threshold": "500+ users",
                    "monitoring_method": "Load testing"
                }
            },
            "user_experience": {
                "frontend_performance": {
                    "target": "<100ms UI response time",
                    "measurement": "Frontend interaction latency",
                    "critical_threshold": "<200ms",
                    "monitoring_method": "Real User Monitoring (RUM)"
                },
                "ai_interaction_satisfaction": {
                    "target": ">85% user satisfaction",
                    "measurement": "User feedback on AI interactions",
                    "critical_threshold": ">70%",
                    "monitoring_method": "User surveys and analytics"
                },
                "feature_adoption": {
                    "target": ">60% users try MCP features",
                    "measurement": "Feature usage analytics",
                    "critical_threshold": ">40%",
                    "monitoring_method": "Feature analytics"
                }
            }
        }

        self.success_phases = {
            "phase_1_foundation": {
                "duration": "2 weeks",
                "success_criteria": [
                    "FastAPI server deployed and stable",
                    "Database migration completed successfully",
                    "Basic MCP tools functional",
                    "Legacy API compatibility maintained"
                ]
            },
            "phase_2_migration": {
                "duration": "4 weeks",
                "success_criteria": [
                    "100% traffic migrated to FastAPI",
                    "No performance regression",
                    "All existing functionality preserved",
                    "MCP tools integrated and tested"
                ]
            },
            "phase_3_enhancement": {
                "duration": "2 weeks",
                "success_criteria": [
                    "Vector recommendations active",
                    "LangChain integration complete",
                    "AI assistant integration functional",
                    "User adoption metrics positive"
                ]
            }
        }
```

#### **Business Impact Metrics**

```python
class BusinessMetrics:
    def __init__(self):
        self.roi_metrics = {
            "development_efficiency": {
                "development_time_reduction": {
                    "target": "30% faster feature development",
                    "measurement": "Time to implement new anime features",
                    "baseline": "Current feature development time",
                    "tracking_method": "Development velocity tracking"
                },
                "bug_reduction": {
                    "target": "50% fewer production bugs",
                    "measurement": "Bug reports per release",
                    "baseline": "Current bug rate",
                    "tracking_method": "Bug tracking system"
                },
                "code_maintainability": {
                    "target": "40% reduction in technical debt",
                    "measurement": "Code complexity metrics",
                    "baseline": "Current codebase metrics",
                    "tracking_method": "Static code analysis"
                }
            },
            "user_engagement": {
                "session_duration": {
                    "target": "25% increase in average session time",
                    "measurement": "Time spent using anime features",
                    "baseline": "Current session analytics",
                    "tracking_method": "User analytics"
                },
                "feature_discovery": {
                    "target": "60% improvement in feature adoption",
                    "measurement": "New feature usage rate",
                    "baseline": "Current feature adoption rates",
                    "tracking_method": "Feature analytics"
                },
                "user_retention": {
                    "target": "20% improvement in 30-day retention",
                    "measurement": "Users returning within 30 days",
                    "baseline": "Current retention metrics",
                    "tracking_method": "Cohort analysis"
                }
            },
            "operational_efficiency": {
                "infrastructure_cost": {
                    "target": "20% reduction in per-user costs",
                    "measurement": "Infrastructure cost per active user",
                    "baseline": "Current operational costs",
                    "tracking_method": "Cost monitoring"
                },
                "support_ticket_reduction": {
                    "target": "40% fewer user support requests",
                    "measurement": "Support tickets related to anime features",
                    "baseline": "Current support volume",
                    "tracking_method": "Support system analytics"
                },
                "automated_operations": {
                    "target": "80% of routine tasks automated",
                    "measurement": "Manual vs automated operations",
                    "baseline": "Current manual processes",
                    "tracking_method": "Operations audit"
                }
            }
        }

        self.success_timeline = {
            "week_1_2": {
                "milestones": [
                    "FastAPI deployment successful",
                    "Database migration complete",
                    "Health monitoring active"
                ],
                "metrics_to_track": [
                    "System uptime",
                    "Migration data integrity",
                    "Performance baselines"
                ]
            },
            "week_3_6": {
                "milestones": [
                    "Traffic migration complete",
                    "MCP tools operational",
                    "Vector search active"
                ],
                "metrics_to_track": [
                    "Response times",
                    "Error rates",
                    "Tool usage rates"
                ]
            },
            "week_7_8": {
                "milestones": [
                    "AI integration complete",
                    "User training finished",
                    "Full system optimization"
                ],
                "metrics_to_track": [
                    "User satisfaction",
                    "Feature adoption",
                    "Performance optimization"
                ]
            },
            "month_2_3": {
                "milestones": [
                    "Business impact measurement",
                    "ROI validation",
                    "Continuous improvement planning"
                ],
                "metrics_to_track": [
                    "Business KPIs",
                    "User engagement",
                    "Cost efficiency"
                ]
            }
        }
```

---

## 🎯 Final Recommendations & Next Steps

### 🚀 Implementation Priority Matrix

#### **Immediate Actions (Week 1)**

1. **Setup Development Environment**

   - Clone and analyze current React codebase architecture
   - Setup FastAPI project structure with proper documentation
   - Configure PostgreSQL and Redis for development
   - Initialize Marqo vector database with basic configuration

2. **Data Foundation**

   - Download and analyze anime-offline-database
   - Design database schema for MCP server
   - Create data migration scripts
   - Setup automated data quality validation

3. **Basic MCP Framework**
   - Implement core MCP protocol handler
   - Create basic tool registration system
   - Setup authentication and security framework
   - Develop comprehensive testing strategy

#### **Short-term Goals (Weeks 2-4)**

1. **Core Integration**

   - Complete Marqo integration with embedding generation
   - Implement basic vector search functionality
   - Create fundamental MCP tools (search, recommend, update)
   - Establish monitoring and logging infrastructure

2. **Migration Preparation**
   - Develop traffic routing system
   - Create legacy API compatibility layer
   - Setup comprehensive testing framework
   - Prepare rollback procedures

#### **Medium-term Objectives (Weeks 5-8)**

1. **Advanced Features**

   - Complete LangChain integration
   - Implement natural language query processing
   - Develop sophisticated recommendation algorithms
   - Create AI assistant integration

2. **Production Deployment**
   - Execute gradual migration strategy
   - Monitor performance and reliability
   - Optimize system performance
   - Complete user training and documentation

### 🎯 Critical Success Factors

#### **Technical Excellence**

- **Zero-Downtime Migration**: Ensure seamless transition without service interruption
- **Performance Improvement**: Achieve >20% performance improvement over current system
- **Reliability**: Maintain 99.9% uptime with comprehensive monitoring
- **Scalability**: Support 1000+ concurrent users with auto-scaling

#### **User Experience**

- **Seamless Integration**: Users should not notice the backend migration
- **Enhanced Features**: AI-powered recommendations should provide clear value
- **Fast Response Times**: <200ms average response for search and recommendations
- **High Accuracy**: >80% relevance in search results and recommendations

#### **Business Impact**

- **Development Efficiency**: 30% faster feature development with new architecture
- **User Engagement**: 25% increase in session duration and feature usage
- **Cost Optimization**: 20% reduction in infrastructure costs per user
- **Innovation Platform**: Foundation for future AI-powered features

### 🔮 Future Enhancement Roadmap

#### **Phase 2: Advanced AI Features (Months 2-3)**

- **Multi-modal Search**: Image-based anime search using computer vision
- **Voice Integration**: Voice-activated anime queries and updates
- **Personalized Chatbot**: Dedicated anime assistant with personality
- **Social Features**: AI-powered anime discussion and community features

#### **Phase 3: Ecosystem Expansion (Months 4-6)**

- **Third-party Integrations**: Streaming platforms, review sites, merchandise
- **API Marketplace**: Expose MCP tools for other developers
- **Mobile App Integration**: Dedicated mobile app with AI features
- **Content Creation**: AI-assisted review writing and recommendation posts

### 📋 Implementation Checklist

#### **Pre-Implementation Requirements**

- [ ] **Team Preparation**: Ensure development team has FastAPI and vector database experience
- [ ] **Infrastructure Setup**: Prepare development, staging, and production environments
- [ ] **API Keys and Credentials**: Secure all required API keys and credentials
- [ ] **Monitoring Tools**: Setup Prometheus, Grafana, and logging infrastructure
- [ ] **Backup Strategy**: Implement comprehensive backup and recovery procedures

#### **Implementation Validation**

- [ ] **Performance Baselines**: Establish current system performance metrics
- [ ] **User Acceptance Criteria**: Define clear success criteria with stakeholders
- [ ] **Rollback Procedures**: Test and validate rollback mechanisms
- [ ] **Security Audit**: Complete security assessment of new architecture
- [ ] **Documentation**: Create comprehensive operational and user documentation

#### **Post-Implementation Review**

- [ ] **Metrics Analysis**: Compare actual vs. expected performance improvements
- [ ] **User Feedback**: Collect and analyze user satisfaction data
- [ ] **System Optimization**: Identify and implement performance optimizations
- [ ] **Lessons Learned**: Document lessons learned for future projects
- [ ] **Continuous Improvement**: Plan next phase of enhancements

### 🎖️ Success Criteria Summary

**Migration Success**:

- Zero downtime during migration
- 100% data integrity maintained
- No functionality regression
- > 20% performance improvement

**MCP Integration Success**:

- > 90% successful AI query resolution
- > 80% user satisfaction with recommendations
- <200ms average vector search response time
- > 75% recommendation accuracy

**Business Impact Success**:

- 30% faster feature development
- 25% increase in user engagement
- 20% reduction in operational costs
- 99.9% system reliability

---

## 📚 Conclusion

This ultra-comprehensive MCP server integration plan provides a detailed roadmap for transforming AnimeTrackr into a cutting-edge AI-powered anime tracking platform. The plan balances ambitious technical goals with practical implementation strategies, ensuring a successful migration while laying the foundation for innovative AI features.

**Key Takeaways**:

1. **Gradual Migration Strategy**: The zero-downtime migration approach minimizes risk while ensuring business continuity
2. **Comprehensive Architecture**: The FastAPI + Marqo + LangChain stack provides a solid foundation for AI features
3. **Data-Driven Approach**: Integration with anime-offline-database provides rich metadata for superior recommendations
4. **Monitoring and Observability**: Comprehensive monitoring ensures system reliability and performance
5. **Scalable Foundation**: The architecture supports future enhancements and third-party integrations

**Success Depends On**:

- Careful planning and execution of the migration strategy
- Comprehensive testing at every phase
- Strong monitoring and alerting infrastructure
- User-focused approach to feature development
- Continuous optimization based on real-world usage

This plan provides the framework for creating a world-class anime tracking platform that leverages the latest AI technologies while maintaining the reliability and performance users expect. The modular architecture and comprehensive testing strategy ensure that the platform can evolve and scale as new opportunities emerge in the AI ecosystem.
