# Anime MCP Server - Planning Scratchpad

## Vision Statement

Transform the current static MCP server into a fully LLM-driven, dynamic system that can handle ANY anime-related query, no matter how complex or unpredictable.

## Current State Analysis

### Strengths 

- **ReactAgent Architecture**: Modern LangGraph integration with create_react_agent pattern
- **Multi-modal Search**: Text + image search via CLIP embeddings (512-dim)
- **AI Parameter Extraction**: 95% accuracy with GPT-4/Claude for query understanding
- **Comprehensive Database**: 38,894 anime entries from offline database
- **Platform Integration**: ID extraction for 11 anime services (MAL, AniList, Kitsu, etc.)
- **Performance**: <200ms text search, ~1s image search

### Critical Gaps =4

1. **Static Tool Limitations**

   - Only 7 predefined tools
   - Cannot adapt to novel query types
   - No dynamic tool generation

2. **No External API Integration**

   - Only uses offline database (updates weekly)
   - No real-time data (airing schedules, streaming availability)
   - No user account data access

3. **No User Management**

   - No account linking/authentication
   - No personalization or preference learning
   - No cross-platform sync

4. **Limited Query Understanding**

   - Can't handle complex narrative queries
   - No temporal understanding ("anime from my childhood")
   - No ambiguity resolution

[Content moved to Phase 4 documentation]

## Proposed Architecture: LLM-Native MCP

### 1. Dynamic Tool Generation System

**Status**: ❌ CANCELLED (MCP protocol incompatible)

**Original Concept**: Generate MCP tools on-demand based on queries

**Why Cancelled**:

- MCP protocol requires static tool definitions
- Tools must be declared at server startup
- Dynamic generation violates protocol specification

**Alternative Approach**: Use flexible, parameterized static tools that LLM can adapt

### 2. Multi-Service Integration Layer

### 3. Advanced Query Understanding Pipeline

[Content moved to Phase 3 documentation]

### 4. User Profile & Preference System

**Status**: 🔮 FUTURE ENHANCEMENT

[Content moved to Phase 1 documentation]

### 5. Real-Time Data Pipeline

## Frontend Services Migration

[Content moved to Phase 1 documentation]

- Enhanced data with vector search
- Single deployment target
  [Frontend services and authentication details moved to Phase 1 documentation]

## Implementation Phases

### Phase 1: Frontend Services Migration & Authentication (2-3 weeks)

**Priority**: HIGH

[Content moved to Phase 1 documentation]

### Phase 2: Dynamic Tool System (3-4 weeks)

**Priority**: =4 HIGH

- [ ] Design DynamicToolGenerator architecture
- [ ] LLM-based tool specification generation
- [ ] Tool composition for complex queries
- [ ] Caching and optimization layer
- [ ] Safety and validation framework

### Phase 3: External Service Integration (4-5 weeks)

### Phase 4: User Personalization (2-3 weeks)

**Priority**: =� MEDIUM

- [ ] UserProfileManager implementation
- [ ] OAuth2 account linking flows
- [ ] Preference learning algorithms
- [ ] Cross-platform sync logic
- [ ] Privacy controls

### Phase 5: Advanced Features (3-4 weeks)

**Priority**: =� LOW

- [ ] Real-time episode notifications
- [ ] Regional streaming availability
- [ ] Social features (shared lists)
- [ ] Voice/audio query support
- [ ] Mobile app integration

[Content moved to Phase 4 documentation]

### Performance Targets

- Query understanding: <500ms
- Simple search: <200ms
- Complex multi-service: <2s
- Streaming responses for long queries

## Example: Handling Unpredictable Queries

### Query: "I vaguely remember an anime from 2010ish with underground tunnels and someone talking through dreams"

**Processing Flow**:

1. **Intent Detection**: Memory-based search
2. **Temporal Extraction**: Year H 2010 � 2 years
3. **Key Elements**: ["underground", "tunnels", "dreams", "communication"]
4. **Expand Search**:
   - Genres: Psychological, Mystery, Sci-Fi
   - Similar themes: Mind control, parallel worlds
5. **Multi-Source Query**:
   - Offline DB: Full-text search
   - AniList: GraphQL with fuzzy matching
   - MAL: Tag-based search
6. **LLM Ranking**: Score matches by description similarity
7. **Results with Confidence**:
   [Content moved to Phase 3 documentation]

## Open Questions & Decisions Needed

### 1. Priority Services

- Which external APIs are MUST-HAVE vs nice-to-have?
- Should we focus on data completeness or user features first?

### 2. User Authentication

- Build our own user system or integrate only?
- How to handle API keys for external services?

### 3. Real-Time Requirements

- How critical are live updates?
- Webhook support needed?

### 4. Regional Considerations

- Multi-region streaming availability?
- Language preferences (sub/dub)?

### 5. Performance vs Completeness

- For complex queries: fast partial results or complete slow results?
- Caching strategy for external APIs?

### 6. Privacy & Data

- How to handle user watch history?
- Data retention policies?
- GDPR compliance needs?

## Current API Analysis & LLM-Driven Architecture

### Existing FastAPI Routes Inventory

#### 1. Search API (`/api/search`) - 🔴 MOSTLY REDUNDANT

- `/api/search/semantic` - POST semantic search
- `/api/search/` - GET search wrapper
- `/api/search/similar/{anime_id}` - Find similar anime
- `/api/search/by-image` - POST image search
- `/api/search/by-image-base64` - POST base64 image search
- `/api/search/visually-similar/{anime_id}` - Visual similarity
- `/api/search/multimodal` - Combined text+image search

**Assessment**: These are static, parameter-based endpoints that don't leverage LLM intelligence. In an LLM-driven system, ALL of these would be replaced by the conversational workflow API that understands intent and dynamically chooses the right search method.

#### 2. Recommendations API (`/api/recommendations`) - 🔴 NOT IMPLEMENTED

- `/api/recommendations/similar/{anime_id}` - TODO placeholder
- `/api/recommendations/based-on-preferences` - TODO placeholder

**Assessment**: Never implemented. Would be redundant in LLM-driven system where recommendations happen through natural conversation.

#### 3. Admin API (`/api/admin`) - 🟢 KEEP ESSENTIAL

- `/api/admin/check-updates` - Check for database updates
- `/api/admin/update-incremental` - Incremental update
- `/api/admin/update-full` - Full database refresh
- `/api/admin/update-status` - Get update status
- `/api/admin/schedule-weekly-update` - Trigger weekly update
- `/api/admin/smart-schedule-analysis` - Analyze update patterns
- `/api/admin/update-safety-check` - Check if safe to update
- `/api/admin/smart-update` - Smart update with safety checks

**Assessment**: KEEP ALL. These are essential for data management and don't need LLM intelligence. They maintain the offline database that powers the system.

#### 4. Workflow API (`/api/workflow`) - 🟡 EVOLVE & EXPAND

- `/api/workflow/conversation` - Process conversation with LLM
- `/api/workflow/multimodal` - Process with image+text
- `/api/workflow/smart-conversation` - Smart orchestration
- `/api/workflow/conversation/{session_id}` - Get history
- `/api/workflow/stats` - Get statistics
- `/api/workflow/health` - Health check

**Assessment**: This is the CORE of the LLM-driven system. Keep and expand significantly.

#### 5. Root/Utility Endpoints - 🟢 KEEP

- `/` - Root info endpoint
- `/health` - System health check
- `/stats` - Database statistics

**Assessment**: Keep for monitoring and operational needs.

### Proposed Architecture: LLM-First Design

#### What to REMOVE (Static APIs):

```
❌ /api/search/* - All 7 endpoints
❌ /api/recommendations/* - All 2 endpoints (never implemented anyway)
```

#### What to KEEP:

```
✅ /api/admin/* - All 8 endpoints (data management)
✅ /api/workflow/* - All 6 endpoints (expand these)
✅ Root utilities - All 3 endpoints (/, /health, /stats)
```

#### New Architecture Benefits:

1. **Single Entry Point**: `/api/workflow/conversation` handles ALL queries
2. **No API Documentation Needed**: Users just describe what they want
3. **Infinite Flexibility**: No predefined parameters or endpoints
4. **Context Awareness**: Maintains conversation state
5. **Multimodal Native**: Image+text handled naturally

### Example Query Transformations

**OLD Static API Way**:

```
GET /api/search?q=action%20anime&limit=10
GET /api/search/similar/12345
POST /api/search/by-image (with file upload)
```

**NEW LLM-Driven Way**:

```
POST /api/workflow/conversation
{
  "message": "find me some action anime",
  "session_id": "user-123"
}

POST /api/workflow/conversation
{
  "message": "show me anime similar to the first one",
  "session_id": "user-123"
}

POST /api/workflow/multimodal
{
  "message": "find anime with this art style",
  "image_data": "base64...",
  "session_id": "user-123"
}
```

### Implications for MCP Protocol

Since MCP tools are meant to be consumed by LLMs:

- **Static MCP tools become internal only** - Not exposed as REST endpoints
- **LLM decides which MCP tool to use** - Based on natural language understanding
- **Dynamic tool generation** - Create new tools on-the-fly for novel queries
- **Tool composition** - Combine multiple tools for complex requests

## Data Architecture Analysis

### Current Offline Database Structure

From anime-offline-database (38,894 entries):

**Available Fields:**

- `sources`: URLs to 11 different anime platforms
- `title`: Main title
- `type`: TV, MOVIE, ONA, OVA, SPECIAL
- `episodes`: Episode count
- `status`: FINISHED, UPCOMING, ONGOING, UNKNOWN
- `animeSeason`: Season (SPRING/SUMMER/FALL/WINTER) + Year
- `picture`/`thumbnail`: MAL image URLs
- `duration`: Episode length in seconds
- `score`: Aggregated scores (arithmetic/geometric mean, median)
- `synonyms`: Alternative titles in multiple languages
- `studios`: Animation studios (often empty)
- `producers`: Production companies (often empty)
- `relatedAnime`: Related anime URLs
- `tags`: Genre/theme tags (limited set)

**Critical Missing Data for Complex Queries:**

- ❌ **Synopsis/Description** - Cannot understand plot or story
- ❌ **Episode Air Dates** - No individual episode schedule
- ❌ **Characters** - No character names or descriptions
- ❌ **Staff** - No directors, writers, voice actors
- ❌ **Source Material** - No info on manga/novel/game origins
- ❌ **Detailed Genres** - Limited tag system
- ❌ **Themes/Tropes** - No narrative elements
- ❌ **Content Warnings** - No violence/mature content indicators
- ❌ **Popularity Metrics** - No member counts or rankings
- ❌ **Streaming Availability** - No platform/region info

### Supported Platforms & Their APIs

**From Sources Array:**

1. **MyAnimeList** - Jikan API (REST)
2. **AniList** - GraphQL API (most comprehensive)
3. **Kitsu** - JSON:API spec
4. **AniDB** - XML API (requires auth)
5. **Anime-Planet** - No official API (scraping)
6. **AniSearch** - No API
7. **Notify.moe** - GraphQL API
8. **LiveChart.me** - No official API (has schedule data)
9. **AnimeNewsNetwork** - XML API
10. **Simkl** - REST API
11. **AnimeCountdown** - No API

### Data Enhancement Strategy

#### Option 1: Fork & Enhance modb-app

**Pros:**

- Control over data structure
- Can add all missing fields
- Unified data source

**Cons:**

- Massive engineering effort
- Need to maintain scrapers for 11 sites
- Storage and processing costs
- Still won't have real-time data

#### Option 2: Hybrid Approach (Recommended)

**Base Layer**: Keep offline database for core metadata
**Enhancement Layer**: Real-time API calls for missing data
**Cache Layer**: Store enhanced data with TTL

**Enhanced Data Structure**:

- **Base Data**: From offline database (always available)
  - Title, year, genres, studios, picture URLs
- **API Enhanced Fields** (fetched on-demand):
  - Synopsis (MAL/AniList)
  - Characters & Staff (AniList)
  - Episode details (AniList/MAL)
  - Streaming availability (Kitsu/platforms)
- **LLM Computed Fields**:
  - Extracted themes from synopsis
  - Content warnings analysis
  - Narrative elements identification

### Priority API Integrations for Complex Queries

1. **AniList (GraphQL)** - HIGHEST PRIORITY

   - Has synopsis, characters, staff, episodes
   - Rich GraphQL schema
   - Good rate limits
   - Example queries: character relationships, source material

2. **MyAnimeList/Jikan** - HIGH PRIORITY

   - Largest user base = best recommendations
   - Synopsis, reviews, forum discussions
   - User statistics for popularity

3. **Kitsu** - MEDIUM PRIORITY

   - Streaming links
   - Episode data
   - Good for availability queries

4. **AnimeSchedule.net API v3** - MEDIUM PRIORITY (CONFIRMED)

   - Comprehensive scheduling API already in use
   - Episode air times (Raw, Sub, Dub) with timezone support
   - Weekly timetables
   - Streaming platform availability
   - Rich filtering (genre, studio, season, duration)
   - No authentication required
   - Example: "What's airing tomorrow on Crunchyroll"

5. **LiveChart.me** - LOW PRIORITY (NO API)
   - No official API available
   - Would require scraping
   - Consider as future enhancement only

### Query Examples & Required Data

**"Anime where MC dies and becomes villain"**

- Needs: Synopsis + LLM analysis
- Sources: AniList/MAL synopsis → LLM extraction

**"Anime from 2010 with tunnels and dreams"**

- Needs: Year + Synopsis + Theme analysis
- Sources: Offline DB year + API synopsis → LLM matching

**"What's airing tomorrow that's similar to what I watched"**

- Needs: User history + Airing schedule + Similarity
- Sources: User profile + LiveChart API + Vector similarity

**"Dark psychological anime but not too violent"**

- Needs: Genre tags + Content warnings + Mood analysis
- Sources: Tags + LLM content analysis of synopsis
  """Handles API failures gracefully"""

  async def execute_with_fallback(self, primary_strategy, query):
  try:
  return await primary_strategy.execute()
  except SourceUnavailable as e: # Fallback to alternative sources
  if e.source == "AniList":
  return await self.use_mal_instead()
  elif e.source == "AnimeSchedule":
  return await self.estimate_from_patterns()
  except RateLimitExceeded: # Use cached data with disclaimer
  return await self.serve_from_cache_with_warning()
  except Exception: # Final fallback to offline DB only
  return await self.offline_only_search()

````


[Content moved to Phase 4 documentation]

## Monitoring & Observability Architecture

### Core Metrics Categories

#### 1. Query Intelligence Metrics

Track how users interact with our LLM system:

- **Query Complexity**: Distribution of simple/medium/complex/multi-source queries
- **Intent Classification**: search/compare/recommend/discover patterns
- **Ambiguity Resolution**: How often clarification is needed
- **Source Preferences**: Which sources users explicitly request
- **LLM Performance**: Response time, token usage, provider selection, retry rates
- **Natural Language Features**: Temporal queries, narrative descriptions, multimodal

#### 2. Multi-Source Orchestration Metrics

Track multi-source query execution:

- **Sources per Request**: Distribution of single vs multi-source queries
- **Parallel Execution Time**: Latency for concurrent API calls
- **Orchestration Strategy**: Which strategies are used most
- **Per-Source Performance**: Response time, success rate, data quality by source
- **Scraping Metrics**: Trigger frequency, Cloudflare challenges, fallback usage

#### 3. Cost & Resource Tracking

Monitor operational expenses:

- **LLM API Costs**: Actual dollars spent by provider/model
- **Token Efficiency**: Tokens per dollar by provider
- **Query Cost Analysis**: Which query types are most expensive
- **Resource Usage**: CPU time, memory usage, concurrent requests

#### 4. User Experience Metrics

Track user-facing performance:

- **Response Format Distribution**: JSON vs narrative vs hybrid
- **Response Completeness**: Percentage of requested data returned
- **Performance SLIs**: End-to-end latency (p50, p95, p99)
- **Failure Patterns**: Empty results, partial results, timeouts
- **User Satisfaction Signals**: Query refinements needed

### Implementation with OpenTelemetry

```python
# Comprehensive observability setup
- FastAPI instrumentation for automatic HTTP metrics
- Custom spans for LLM calls, API fetches, scraping operations
- Prometheus metrics exporter for time-series data
- OTLP trace exporter for distributed tracing
- Structured logging with trace correlation
````

### Real-time Analytics Dashboard

1. **Query Pattern Analytics**

   - Hot query detection and caching opportunities
   - Source preference trends over time
   - Failure pattern identification
   - Cost optimization insights

2. **Distributed Tracing**

   - Full request flow visualization
   - Multi-source fetch parallelization
   - LLM processing breakdown
   - Bottleneck identification

3. **Alerting Rules**
   - High LLM latency (>2s for 5m) → Switch to faster model
   - Source degradation (<80% success) → Activate fallbacks
   - Cost spike (>$10/5m) → Rate limit expensive operations
   - Scraping blocked (>100 challenges/5m) → Review scraper config
   - Memory pressure (>4GB) → Clear caches

### Grafana Dashboard Panels

1. **Query Intelligence Panel**

   - Query rate and complexity distribution
   - Intent classification breakdown
   - LLM provider usage and costs

2. **Source Health Panel**

   - Availability heatmap by source
   - Response time percentiles
   - Scraping success rates

3. **User Experience Panel**

   - End-to-end latency histogram
   - Error rate by type
   - Response format preferences

4. **Cost Analysis Panel**
   - Running costs by component
   - Token usage trends
   - Cost per query type

This monitoring architecture ensures complete visibility into our LLM-driven system, enabling data-driven optimizations and proactive issue detection.

## MCP Protocol Compatibility

### The Core Challenge

MCP (Model Context Protocol) is designed for **static tool definitions**, but our LLM-driven vision requires **dynamic adaptation**. This creates fundamental conflicts:

**MCP Protocol Expectations:**

- Tools defined at server startup
- Fixed tool list throughout session
- Predictable schemas for type safety
- Client-side tool caching

**LLM-Driven Requirements:**

- Dynamic tool generation per query
- Adaptive parameters based on context
- Novel query handling without predefined tools
- Continuous learning and adaptation

### Compatibility Strategies

#### 1. Hybrid Static-Dynamic Approach (Recommended)

Expose static MCP tools that internally leverage dynamic LLM logic:
[Content moved to Phase 4 documentation]

- 100+ curated queries with expected behaviors
- Categories: Simple lookups, complex narratives, multi-source, schedules
- Track semantic correctness, not exact responses
- Version control for regression detection

**Semantic Validation Examples**:

- Query: "dark psychological anime" → Must extract genre="psychological", mood="dark"
- Query: "anime id 12345" → Must return exact ID match
- Query: "airing tomorrow" → Results must have future dates
- Query: "like Death Note" → Must call similarity tool

**Mock LLM Service**:

- Deterministic responses for testing
- Configurable behavior for edge cases
- Cost-effective testing without API calls
- Test tool selection and parameter extraction

#### 3. **Integration Testing Levels**

**Level 1: MCP Tool Testing**

- Each tool tested with various parameter combinations
- Error handling for missing/invalid parameters
- Service failure scenarios
- Response schema validation

**Level 2: Orchestration Testing**

- Strategy selection based on query patterns
- Multi-source aggregation logic
- Fallback behavior verification
- Cache coordination

**Level 3: End-to-End Testing**

- Real LLM → ReactAgent → Tools → Services
- Test accounts on real APIs
- Snapshot testing for structure
- Performance benchmarking

#### 4. **MCP Protocol Testing**

**Protocol Compliance**:

- Tool discovery consistency
- Parameter schema validation
- Both stdio and HTTP transport testing
- Multi-client compatibility (Postman for HTTP)

**Concurrent Request Testing**:

- Request coalescing behavior
- Thread safety verification
- Resource pool management

#### 5. **Chaos Engineering**

**Service Degradation Scenarios**:

- API rate limiting (429 responses)
- Service timeouts (30s+)
- Partial service failures
- Network interruptions

**Expected Behaviors**:

- Graceful degradation
- Partial results with warnings
- No cascade failures
- Automatic recovery

#### 6. **Time-Based Testing**

**Time Injection Strategy**:

- Injectable "current time" for predictable tests
- Timezone boundary testing
- DST transition handling
- Relative time calculations

**Schedule Test Scenarios**:

- "What's airing tomorrow" at different times
- Episode countdowns
- Cross-timezone queries
- Historical schedule queries

#### 7. **Performance & Load Testing**

**Performance Benchmarks**:

- Simple queries < 500ms (95th percentile)
- Complex multi-source < 3s
- Image search < 2s
- Batch operations scale linearly

**Load Patterns**:

- Burst similar queries (coalescing test)
- Mixed query types
- Cache warming scenarios
- API rate limit approaching

#### 8. **Response Validation**

**Schema Testing**:

- All responses match defined schemas
- Required fields always present
- Correct data types
- Nested object validation

**Semantic Testing**:

- Results match query intent
- Proper tool was selected
- Parameters correctly extracted
- Error responses follow schema

### Test Data Management

**Fixtures**:

- Recorded API responses (VCR pattern)
- Synthetic anime database
- Generated test images
- Time-series mock data

**Privacy & Safety**:

- No real user data
- Public anime data only
- Anonymized patterns
- Synthetic preferences

### Continuous Integration

**Test Pyramid**:

- 70% Unit tests (fast, isolated)
- 20% Integration tests (service interactions)
- 10% E2E tests (full flow validation)

**Parallel Execution**:

- Isolated test environments
- Independent databases
- No shared state
- Containerized testing

### Regression Detection

**Golden Query Monitoring**:

- Automated daily runs
- Semantic diff detection
- Alert on behavior changes
- Track model performance

**A/B Testing Framework**:

- Compare LLM models
- Test prompt strategies
- Measure response quality
- Cost/performance analysis

### Error Boundary Testing

**LLM Error Scenarios**:

- Malformed tool calls
- Infinite chaining prevention
- Refusal handling
- Timeout management

**Recovery Testing**:

- Graceful error messages
- Fallback strategies
- State cleanup
- User-friendly errors

This comprehensive testing strategy ensures reliability while accepting LLM non-determinism, focusing on behavior validation over exact matching.

## Error Handling & Resilience

### Core Challenge

LangGraph's execution model creates unique debugging challenges:

- Non-linear graph execution flow
- LLM decision opacity
- Async parallel operations
- Limited visibility into ReactAgent decisions

### Multi-Layer Error Handling Architecture

#### 1. **Execution Tracing System**

**Comprehensive Trace Structure**:

- Unique execution ID per graph traversal
- Node entry/exit timestamps
- Tool selection reasoning capture
- State snapshots at each decision point

**Trace Example**:

```
ExecutionID: abc123
├─ Query: "find dark anime like Death Note"
├─ Intent Analysis: {type: "similarity", mood: "dark"}
├─ Node: ReactAgent (2ms)
│  ├─ Tool Selected: search_anime
│  ├─ LLM Reasoning: "Similarity search with mood filter"
│  └─ Parameters Extracted: {enhance: true}
├─ Node: ToolExecution (245ms)
│  └─ Service Calls: Qdrant → AniList → Cache
└─ Response Generation (12ms)
```

#### 2. **Contextual Error Preservation**

**Error Context Layers**:

- **User Layer**: Friendly, actionable message
- **Debug Layer**: Technical context for developers
- **Trace Layer**: Complete execution path

**Error Example**:

- **User Sees**: "Unable to fetch anime details. Try again in a few moments."
- **Developer Sees**: `ToolError: search_anime → AniList → RateLimit (429)`
- **Trace Shows**: Full graph execution, timings, state at failure

#### 3. **LangGraph-Specific Debugging**

**Decision Point Logging**:

- LLM reasoning for each tool selection
- Prompt/response pairs for analysis
- Token usage per decision
- Confidence scores when available

**State Management**:

- Snapshot before/after each node
- Track all state mutations
- Checkpoint for rollback capability
- State corruption detection

#### 4. **Structured Logging Strategy**

**Log Levels**:

- **USER**: Clean error messages with suggested actions
- **DEVELOPER**: Technical context, service details
- **DEBUG**: LLM interactions, parameters, responses
- **TRACE**: Every call, state change, timing

**Correlation System**:

- Request ID (user request)
- Execution ID (graph run)
- Tool Call ID (individual tool)
- Service Call ID (external API)

#### 5. **Common Failure Patterns**

**LangGraph-Specific Failures**:

1. **Tool Selection Loop**: LLM repeatedly selects wrong tool
   - Detection: Same tool called 3+ times
   - Recovery: Force different tool or fallback
2. **Parameter Extraction Failure**: Can't parse intent
   - Detection: Empty or invalid parameters
   - Recovery: Simplified parameter extraction
3. **State Corruption**: Invalid state blocks progress
   - Detection: State validation fails
   - Recovery: Rollback to last checkpoint
4. **Graph Timeout**: Execution exceeds limits
   - Detection: >30s execution time
   - Recovery: Return partial results
5. **Memory Explosion**: Context too large
   - Detection: State size > threshold
   - Recovery: Prune conversation history

#### 6. **Developer Tools**

**Debug Mode Features**:

- Graph execution visualization
- Step-through debugging
- Node breakpoints
- State inspection UI
- LLM decision replay

**Local Development**:

- Mock LLM for predictable testing
- Error injection framework
- Production error replay
- Performance profiling

#### 7. **Production Observability**

**Real-time Dashboards**:

- Graph execution flow visualization
- Tool success/failure rates
- LLM decision patterns
- Error clustering and trends

**Smart Alerting**:

- New error pattern detection
- Performance degradation
- Anomalous tool selection
- Cost spike detection

#### 8. **Error Recovery Strategies**

**Graceful Degradation**:

1. **Tool Failure**: Try alternative tool
2. **Service Failure**: Use cached/offline data
3. **LLM Failure**: Fallback to simple search
4. **Complete Failure**: Clear error with retry

**User Communication**:

- What happened (simplified)
- Why it happened (if relevant)
- What they can do
- When to try again

#### 9. **Error Testing Strategy**

**Failure Simulation**:

- Service-specific failures
- LLM refusal scenarios
- Timeout conditions
- State corruption

**Validation Points**:

- Error messages are helpful
- Debug info is complete
- No data leakage
- Recovery works correctly

### Implementation Benefits

This approach provides:

- **Full visibility** into LangGraph execution
- **Rapid debugging** with comprehensive traces
- **User confidence** through clear communication
- **Developer efficiency** with proper tooling
- **System reliability** through pattern detection

## Data Flow Architecture

### Overview

Data flows through multiple layers in our LLM-driven system, each with specific responsibilities for transformation, caching, and error handling.

### 1. **Request Lifecycle**

**Phase 1: Query Ingestion**

- Request ID generation for tracing
- Input validation and sanitization
- Rate limit verification
- Query normalization for consistency

**Phase 2: LLM Processing**

- Intent analysis and classification
- Tool selection based on query type
- Parameter extraction from natural language
- Strategy determination for execution

**Phase 3: Orchestration**

- Tool chain planning by ReactAgent
- Parallel vs sequential execution decision
- Cache strategy selection
- Execution plan optimization

**Phase 4: Data Retrieval**

- Multi-level cache checks
- Offline database queries
- API enhancement when needed
- Result aggregation from sources

**Phase 5: Response Generation**

- Data merging from multiple sources
- Schema validation
- Format selection (JSON/narrative)
- Streaming decision for large results

### 2. **Multi-Level Caching Strategy**

**L1 Cache (In-Memory)**

- Hot data: Popular anime, recent queries
- TTL: 5-15 minutes
- Size limit: ~1GB
- Use case: Extremely fast repeated queries

**L2 Cache (Redis)**

- Warm data: Enhanced details, API responses
- TTL: Field-specific (5 min to 7 days)
- Size limit: ~10GB
- Use case: Cross-request caching

**L3 Cache (Offline DB)**

- Cold data: Complete anime database
- TTL: Weekly refresh
- Size: Unlimited
- Use case: Fallback data source

**Cache Key Patterns**:

- Simple: `anime:12345`
- Enhanced: `anime:12345:enhanced:mal`
- Query: `query:hash(normalized):strategy`
- Schedule: `schedule:2024-01-29:platform:tz`

### 3. **State Management**

**Request State**

- Immutable within request lifecycle
- Contains query, context, preferences
- Thread-safe context propagation

**Conversation State**

- Managed by LangGraph MemorySaver
- Previous queries and results
- Automatic pruning on size limits

**System State**

- Service health monitoring
- Cache statistics
- Rate limit counters
- Circuit breaker states

### 4. **Transaction Boundaries**

**Read Operations**

- No explicit transactions
- Eventually consistent model
- Stale cache serving during updates

**Write Operations (Future)**

- User preference updates
- Watchlist modifications
- Atomic operation guarantees
- Compensation on failure

### 5. **Data Transformation Pipeline**

**Input Transformations**

- Query normalization
- Image preprocessing
- Parameter validation
- Security sanitization

**Processing Transformations**

- Embedding generation
- Score normalization
- Timezone conversions
- Language handling

**Output Transformations**

- Field filtering
- Format conversions
- Null handling
- Pagination support

### 6. **Streaming Architecture**

**Streaming Triggers**:

- Large result sets (>50 items)
- Multi-source aggregation
- Real-time updates
- Progressive enhancement

**Stream Flow**:

1. Initial fast response with basic data
2. Progressive updates with enhancements
3. Completion signal when done

### 7. **Data Consistency Model**

**Eventual Consistency**

- Weekly offline DB updates
- TTL-based cache expiry
- No strong consistency guarantees

**Conflict Resolution**

- Latest timestamp wins for dynamic fields
- Aggregation for scores
- Priority-based merging for metadata

### 8. **Error Propagation**

**Partial Failure Handling**

- Service failures isolated
- Partial results with warnings
- Graceful degradation

**Failure Boundaries**

- No cascade failures
- Stale cache on errors
- User-friendly error context

### 9. **Performance Optimizations**

**Query Planning**

- Complexity analysis
- Parallel opportunity detection
- Service call ordering
- Cache hit prediction

**Resource Management**

- Connection pooling
- Thread pool sizing
- Memory management
- Rate limit budgeting

### 10. **Privacy Boundaries**

**Public Data**

- Anime metadata
- Aggregated stats
- Cacheable freely

**Private Data (Future)**

- User preferences
- Watch history
- Isolated caching
- No cross-user sharing
  [Content moved to Phase 4 documentation]

[Content moved to Phase 5 documentation]

### Caching Strategy for Schedule Data

Since schedule data changes frequently, we use smart caching:

```python
CACHE_DURATION = {
    # Very short cache for currently airing
    'next_episode_airing': 15 * 60,         # 15 minutes
    'episode_just_aired': 5 * 60,           # 5 minutes

    # Medium cache for weekly schedules
    'weekly_schedule': 6 * 3600,            # 6 hours
    'streaming_availability': 3600,          # 1 hour

    # Longer cache for finished anime
    'completed_anime_schedule': 24 * 3600,   # 24 hours
    'seasonal_schedule': 12 * 3600          # 12 hours
}
```

### Real-World Query Examples:

1. **"What psychological anime are airing tomorrow?"**

   - Search offline DB for psychological genre
   - Filter for ONGOING status
   - Fetch tomorrow's schedule from AnimeSchedule
   - Merge and return time-sorted results

2. **"When does the next dubbed episode release?"**

   - Identify anime from context/query
   - Check AnimeSchedule for dub release times
   - Calculate timezone-adjusted times
   - Show platform-specific availability

3. **"Show me Friday's anime lineup on Crunchyroll"**
   - Query AnimeSchedule for Friday broadcasts
   - Filter by Crunchyroll availability
   - Add synopsis from AniList/MAL
   - Sort by air time

The beauty is that ALL of this happens automatically through the workflow APIs - the LLM decides when schedule data is needed and fetches it seamlessly!

## Error Handling & Fallback Strategy

### The Rate Limit Reality

Based on real-world experience:

- **AniList**: Often hits rate limits (90 requests/minute)
- **Jikan/MAL**: Frequently rate limited (2 requests/second)
- **Impact**: Degraded user experience, incomplete data

### Innovative Multi-Layer Resilience Strategy

Instead of simple offline DB fallback, implement a sophisticated degradation system:

#### 1. **Collaborative Community Cache**

```python
class CollaborativeCacheSystem:
    """
    Users unknowingly help each other by sharing API responses
    """

    def __init__(self):
        self.shared_cache = RedisCluster()  # Distributed cache
        self.contribution_scores = {}  # Track user contributions

    async def get_enhanced_data(self, anime_id: str, user_id: str) -> dict:
        # Check personal quota first
        if await self.has_personal_quota(user_id):
            return await self.fetch_with_user_quota(anime_id, user_id)

        # Check community cache (other users' recent fetches)
        if cached := await self.shared_cache.get(f"community:{anime_id}"):
            # Found data fetched by another user recently!
            self.track_cache_hit(user_id, cached['contributor_id'])
            return cached['data']

        # Check if any user in pool has quota
        if donor_user := await self.find_quota_donor():
            # Use volunteer's quota (with permission)
            data = await self.fetch_with_user_quota(anime_id, donor_user)
            await self.shared_cache.set(
                f"community:{anime_id}",
                {'data': data, 'contributor_id': donor_user},
                ttl=3600
            )
            return data

        # Fallback to offline DB
        return await self.get_offline_data(anime_id)
```

#### 2. **Smart Request Coalescing**

```python
class RequestCoalescer:
    """
    Multiple users requesting same anime? Make ONE API call!
    """

    def __init__(self):
        self.pending_requests = {}  # anime_id -> Future
        self.request_queues = {}    # anime_id -> List[user_callbacks]

    async def get_anime_data(self, anime_id: str, user_id: str) -> dict:
        # Check if someone else is already fetching this anime
        if anime_id in self.pending_requests:
            # Piggyback on existing request!
            future = self.pending_requests[anime_id]
            self.request_queues[anime_id].append(user_id)
            return await future

        # First request for this anime
        self.pending_requests[anime_id] = asyncio.Future()
        self.request_queues[anime_id] = [user_id]

        try:
            # Make the actual API call
            data = await self.fetch_from_api(anime_id)

            # Notify all waiting users
            self.pending_requests[anime_id].set_result(data)

            # Cache for everyone
            await self.cache_for_all_users(
                anime_id,
                data,
                self.request_queues[anime_id]
            )

            return data

        finally:
            # Cleanup
            del self.pending_requests[anime_id]
            del self.request_queues[anime_id]
```

#### 3. **Predictive Pre-warming with ML**

```python
class PredictivePreWarmer:
    """
    Use ML to predict what users will search for and pre-fetch during low traffic
    """

    async def run_predictive_warming(self):
        # Analyze patterns
        patterns = await self.analyze_search_patterns()

        # Time-based predictions
        if self.is_friday_evening():
            # Pre-fetch popular currently airing anime
            await self.prefetch_airing_anime()

        # Seasonal predictions
        if self.is_new_season_start():
            # Pre-fetch all new season anime
            await self.prefetch_seasonal_anime()

        # User-based predictions
        for user_id in self.get_active_users():
            # Predict what user might search next
            predictions = await self.ml_model.predict_user_interests(user_id)
            await self.prefetch_user_predictions(user_id, predictions[:5])

        # Trending predictions
        trending = await self.get_trending_topics()
        await self.prefetch_trending_anime(trending)
```

#### 4. **Progressive Enhancement with Priority Queues**

```python
class ProgressiveEnhancer:
    """
    Return basic data immediately, enhance in background based on priority
    """

    async def search_with_progressive_enhancement(self, query: str, user_id: str):
        # 1. Return offline data IMMEDIATELY
        basic_results = await self.offline_search(query)

        # 2. Start enhancement jobs with smart prioritization
        enhancement_jobs = []
        for i, anime in enumerate(basic_results):
            priority = self.calculate_priority(
                anime=anime,
                position=i,  # Higher position = higher priority
                user_tier=self.get_user_tier(user_id),
                anime_popularity=anime.get('popularity', 0),
                is_airing=anime['status'] == 'ONGOING'
            )

            job = EnhancementJob(anime['id'], priority)
            enhancement_jobs.append(job)

        # 3. Queue jobs with rate limit awareness
        await self.enhancement_queue.add_batch(enhancement_jobs)

        # 4. Return results with enhancement promise
        return {
            'results': basic_results,
            'enhancement_status': 'in_progress',
            'enhancement_urls': {
                # WebSocket endpoints for live updates
                'websocket': f'/ws/enhancement/{query_id}',
                # Polling endpoint
                'poll': f'/api/enhancement-status/{query_id}'
            }
        }
```

#### 5. **API Credit Pool System**

```python
class APICreditPool:
    """
    Users can donate unused API credits to a community pool
    """

    def __init__(self):
        self.credit_pools = {
            'anilist': CreditPool(rate_limit=90/min),
            'mal': CreditPool(rate_limit=2/sec),
            'kitsu': CreditPool(rate_limit=100/min)
        }

    async def donate_credits(self, user_id: str, api: str, credits: int):
        """Power users can donate their unused API credits"""
        pool = self.credit_pools[api]
        await pool.add_credits(user_id, credits)

        # Reward donors with perks
        await self.grant_donor_perks(user_id, credits)

    async def borrow_credits(self, user_id: str, api: str) -> bool:
        """Borrow from pool when personal credits exhausted"""
        pool = self.credit_pools[api]
        if await pool.has_available_credits():
            await pool.use_credit(borrower=user_id)
            return True
        return False
```

#### 6. **Graceful Degradation Levels**

```python
class GracefulDegradation:
    """
    Multiple levels of degradation, not just "API or offline"
    """

    DEGRADATION_LEVELS = [
        # Level 0: Full enhancement (all APIs available)
        {
            'name': 'full',
            'features': ['synopsis', 'schedule', 'characters', 'streaming', 'reviews']
        },

        # Level 1: Partial enhancement (some APIs limited)
        {
            'name': 'partial',
            'features': ['synopsis', 'schedule'],  # Only critical data
            'message': 'Some features limited due to high traffic'
        },

        # Level 2: Cache only (no new API calls)
        {
            'name': 'cache_only',
            'features': [],  # Only cached enhancements
            'message': 'Using cached data only'
        },

        # Level 3: Offline (pure database)
        {
            'name': 'offline',
            'features': [],
            'message': 'Enhanced data temporarily unavailable'
        },

        # Level 4: Emergency (even DB is slow)
        {
            'name': 'emergency',
            'features': [],
            'max_results': 10,  # Limit results
            'message': 'System under heavy load'
        }
    ]

    async def get_current_level(self) -> dict:
        # Dynamically determine degradation level
        api_health = await self.check_api_health()
        system_load = await self.get_system_load()
        cache_hit_rate = await self.get_cache_stats()

        return self.calculate_optimal_level(api_health, system_load, cache_hit_rate)
```

#### 7. **Distributed Rate Limit Tracking**

```python
class DistributedRateLimiter:
    """
    Track rate limits across all instances and users
    """

    def __init__(self):
        self.redis = RedisCluster()
        self.circuit_breakers = {}

    async def can_call_api(self, api: str, user_id: str) -> tuple[bool, str]:
        # Check circuit breaker first
        if self.circuit_breakers.get(api, {}).get('open', False):
            return False, "API temporarily disabled due to errors"

        # Check global rate limit
        global_key = f"rate_limit:{api}:global"
        global_count = await self.redis.incr(global_key)
        await self.redis.expire(global_key, 60)  # Reset every minute

        if global_count > API_LIMITS[api]['global']:
            return False, "Global rate limit reached"

        # Check user rate limit
        user_key = f"rate_limit:{api}:user:{user_id}"
        user_count = await self.redis.incr(user_key)
        await self.redis.expire(user_key, 60)

        if user_count > API_LIMITS[api]['per_user']:
            return False, "Personal rate limit reached"

        return True, "OK"

    async def record_api_result(self, api: str, success: bool, response_time: float):
        """Track API health for circuit breaker"""
        if not success or response_time > 5000:  # Failed or slow
            self.circuit_breakers[api]['failures'] += 1

            if self.circuit_breakers[api]['failures'] > 5:
                # Open circuit breaker
                self.circuit_breakers[api]['open'] = True
                self.circuit_breakers[api]['retry_after'] = time.time() + 300  # 5 min
```

### Implementation Priority for Error Handling:

1. **Phase 1**: Request Coalescing + Smart Degradation

   - Immediate impact on rate limits
   - No user-facing changes needed

2. **Phase 2**: Collaborative Cache + Progressive Enhancement

   - Community benefit
   - Better UX with progressive loading

3. **Phase 3**: Predictive Pre-warming + Credit Pool

   - Advanced optimization
   - Requires user participation

4. **Phase 4**: ML-based Predictions
   - Long-term optimization
   - Requires training data

The key insight: **Don't just handle errors - prevent them through intelligent request management and community collaboration!**

### Implementation Details: Request Tracking & Privacy

#### How to Track Concurrent Requests

```python
class RequestTracker:
    """
    Track concurrent requests with privacy-aware design
    """

    def __init__(self):
        # In-memory tracking for active requests
        self.active_requests = {}  # normalized_query -> RequestInfo
        self.request_lock = asyncio.Lock()

    def normalize_request(self, query: str, params: dict) -> str:
        """
        Create a unique but privacy-safe cache key
        """
        # For anime by ID - straightforward
        if anime_id := params.get('anime_id'):
            return f"anime:id:{anime_id}"

        # For search queries - normalize but exclude user-specific data
        if search_query := params.get('query'):
            # Normalize: lowercase, sort words, remove user preferences
            normalized = ' '.join(sorted(search_query.lower().split()))

            # Add non-personal filters
            if genres := params.get('genres'):
                normalized += f":genres:{','.join(sorted(genres))}"
            if year := params.get('year'):
                normalized += f":year:{year}"
            if status := params.get('status'):
                normalized += f":status:{status}"

            # EXCLUDE personal preferences like:
            # - user_excluded_genres
            # - user_minimum_score
            # - user_watch_history_filter

            return f"search:{hash(normalized)}"

        return None
```

#### Privacy-Aware Caching Strategy

```python
class PrivacyAwareCache:
    """
    Share only non-personal data between users
    """

    def __init__(self):
        self.public_cache = RedisCache()    # Shareable data
        self.private_cache = RedisCache()   # User-specific data

    async def get_anime_data(self, anime_id: str, user_id: Optional[str] = None):
        cache_key = f"anime:{anime_id}"

        # Check if request is in-flight
        if pending := self.active_requests.get(cache_key):
            # Wait for the in-flight request
            return await pending.future

        # Start new request tracking
        request_info = RequestInfo(
            key=cache_key,
            future=asyncio.Future(),
            started_at=datetime.now(),
            waiting_users=[]  # Track count, not IDs for privacy
        )
        self.active_requests[cache_key] = request_info

        try:
            # Check public cache first
            if cached := await self.public_cache.get(cache_key):
                # Add user-specific enhancements if logged in
                if user_id:
                    cached = await self.enhance_with_user_data(cached, user_id)
                request_info.future.set_result(cached)
                return cached

            # Fetch from API
            api_data = await self.fetch_from_api(anime_id)

            # Separate public and private data
            public_data, private_data = self.separate_data(api_data)

            # Cache public data for everyone
            await self.public_cache.set(
                cache_key,
                public_data,
                ttl=3600  # 1 hour
            )

            # Cache user-specific data separately if applicable
            if user_id and private_data:
                private_key = f"user:{user_id}:anime:{anime_id}"
                await self.private_cache.set(
                    private_key,
                    private_data,
                    ttl=3600
                )

            # Combine for response
            result = public_data.copy()
            if user_id and private_data:
                result.update(private_data)

            # Notify all waiting requests
            request_info.future.set_result(result)
            return result

        finally:
            # Clean up tracking
            del self.active_requests[cache_key]

    def separate_data(self, api_data: dict) -> tuple[dict, dict]:
        """
        Separate shareable vs private data
        """
        # Public data - safe to share
        public_fields = {
            'title', 'synopsis', 'genres', 'studios',
            'episodes', 'status', 'score', 'popularity',
            'air_date', 'picture', 'characters', 'staff',
            'streaming_platforms', 'next_episode'
        }

        # Private data - user-specific
        private_fields = {
            'user_score', 'user_status', 'user_episodes_watched',
            'user_notes', 'user_tags', 'personalized_recommendations',
            'watch_history', 'user_review'
        }

        public_data = {k: v for k, v in api_data.items() if k in public_fields}
        private_data = {k: v for k, v in api_data.items() if k in private_fields}

        return public_data, private_data
```

#### Real-Time Request Coalescing with Privacy

```python
class PrivacyAwareCoalescer:
    """
    Coalesce requests while maintaining privacy
    """

    async def search_anime(self, query: str, user_id: Optional[str], filters: dict):
        # Create normalized key without user data
        public_filters = self.extract_public_filters(filters)
        cache_key = self.create_cache_key(query, public_filters)

        # Check if similar search is in progress
        async with self.request_lock:
            if pending := self.active_requests.get(cache_key):
                # Someone is already searching for this!
                pending.waiting_count += 1

                # Wait for result
                public_results = await pending.future

                # Apply user-specific filtering if needed
                if user_id:
                    return self.apply_user_filters(
                        public_results,
                        filters,  # Original filters with user preferences
                        user_id
                    )
                return public_results

            # First request for this search
            request_info = RequestInfo(
                key=cache_key,
                future=asyncio.Future(),
                waiting_count=0
            )
            self.active_requests[cache_key] = request_info

        try:
            # Perform the search with public filters only
            results = await self.perform_search(query, public_filters)

            # Cache the public results
            await self.public_cache.set(cache_key, results, ttl=300)  # 5 min

            # Notify waiting requests
            request_info.future.set_result(results)

            # Apply user filters for original requester
            if user_id:
                return self.apply_user_filters(results, filters, user_id)
            return results

        finally:
            async with self.request_lock:
                del self.active_requests[cache_key]

    def extract_public_filters(self, filters: dict) -> dict:
        """Extract only non-personal filters"""
        public_filter_keys = {
            'genres', 'year', 'season', 'status',
            'type', 'min_episodes', 'max_episodes'
        }
        return {k: v for k, v in filters.items() if k in public_filter_keys}

    def apply_user_filters(self, public_results: list, user_filters: dict, user_id: str) -> list:
        """Apply user-specific filtering on public results"""
        filtered = public_results.copy()

        # Apply user's excluded genres
        if excluded := user_filters.get('excluded_genres'):
            filtered = [a for a in filtered if not any(g in a['genres'] for g in excluded)]

        # Apply user's minimum score preference
        if min_score := user_filters.get('user_min_score'):
            filtered = [a for a in filtered if a.get('score', 0) >= min_score]

        # Apply watch history filter
        if user_filters.get('hide_watched'):
            watched = self.get_user_watched_anime(user_id)
            filtered = [a for a in filtered if a['id'] not in watched]

        return filtered
```

#### Practical Example

```python
# Scenario: 10 users search for "psychological anime" within 2 seconds

# User 1 searches
GET /api/workflow/conversation
{
    "message": "psychological anime",
    "user_id": "user_1",
    "filters": {
        "excluded_genres": ["horror"],  # Personal preference
        "hide_watched": true            # Personal filter
    }
}
→ Creates cache key: "search:anime:psychological" (excludes personal filters)
→ Starts API call

# User 2 searches (100ms later)
GET /api/workflow/conversation
{
    "message": "psychological anime",
    "user_id": "user_2",
    "filters": {
        "excluded_genres": ["mecha"],   # Different preference
        "min_score": 8.0               # Different threshold
    }
}
→ Same cache key: "search:anime:psychological"
→ Waits for User 1's request

# Users 3-10 search (within 2 seconds)
→ All wait for the same request

# Result:
- 1 API call instead of 10
- Each user gets the same base results
- Each user's personal filters applied client-side
- User 1 doesn't see horror anime
- User 2 doesn't see mecha anime
- All users' privacy maintained
```

#### Privacy Rules

1. **Never share**:

   - User watch history
   - User ratings/reviews
   - Personal filters/preferences
   - Account-linked data

2. **Safe to share**:

   - Anime metadata (title, synopsis, etc.)
   - Public scores/rankings
   - Schedule information
   - General search results

3. **Separate caching**:
   - Public cache: 1-6 hours (anime data)
   - Private cache: 30 minutes (user-specific)
   - Request coalescing: 5 minutes (search results)

## API Authentication & Key Management

[Content moved to Phase 1 documentation]

## Smart Source Integration Strategy

### The Realization: Offline DB Has Source URLs!

The anime-offline-database already contains URLs to all 11 sources for each anime:

```json
{
  "sources": [
    "https://anidb.net/anime/4563",
    "https://anilist.co/anime/1535",
    "https://anime-planet.com/anime/death-note",
    "https://kitsu.io/anime/1376",
    "https://myanimelist.net/anime/1535",
    "https://notify.moe/anime/0-A-5Fimg"
  ],
  "title": "Death Note",
  ...
}
```

This changes EVERYTHING! We can use these URLs for intelligent source-specific operations.

### Proposed Hybrid Strategy: Offline-First with Selective Enhancement

```python
class SourceAwareDataFetcher:
    """
    Intelligently combines offline DB with source-specific fetching
    """

    async def get_anime_from_source(
        self,
        anime_id: str,
        source: str,
        enhance: bool = False,
        required_fields: List[str] = None
    ):
        # Step 1: ALWAYS start with offline DB
        offline_data = await self.offline_db.get_anime(anime_id)

        if not offline_data:
            return None

        # Step 2: Check if source URL exists
        source_url = self.find_source_url(offline_data['sources'], source)

        if not source_url:
            # Source doesn't have this anime
            return {
                'error': f'Anime not found on {source}',
                'available_sources': self.extract_source_names(offline_data['sources'])
            }

        # Step 3: Determine if enhancement needed
        if not enhance and not self.needs_enhancement(required_fields):
            # Offline data is sufficient
            return {
                **offline_data,
                '_source': 'offline_database',
                '_source_url': source_url
            }

        # Step 4: Enhance from source if needed
        enhanced_data = await self.enhance_from_source(
            source,
            source_url,
            offline_data,
            required_fields
        )

        return enhanced_data
```

### Three-Tier Data Strategy

#### Tier 1: Offline Database Only (Fastest)

```python
class OfflineOnlyStrategy:
    """
    Use when basic metadata is sufficient
    """

    OFFLINE_FIELDS = [
        'title', 'type', 'episodes', 'status',
        'animeSeason', 'picture', 'thumbnail',
        'tags', 'sources', 'synonyms', 'relations'
    ]

    async def search_from_source(self, query: str, source: str):
        # Search offline DB
        results = await self.offline_db.search(query)

        # Filter to only anime available on requested source
        filtered = []
        for anime in results:
            source_url = self.find_source_url(anime['sources'], source)
            if source_url:
                filtered.append({
                    **anime,
                    '_source_url': source_url,
                    '_has_on_source': True
                })

        return filtered
```

#### Tier 2: API Enhancement (When Available)

```python
class APIEnhancementStrategy:
    """
    Use APIs for rich data when available
    """

    API_SOURCES = ['mal', 'anilist', 'kitsu', 'simkl', 'notify', 'anidb']

    async def enhance_if_api_available(self, anime: dict, source: str, fields: List[str]):
        if source not in self.API_SOURCES:
            return anime  # No API, return offline data

        # Extract source-specific ID from URL
        source_id = self.extract_id_from_url(anime['_source_url'])

        # Fetch via API
        api_client = self.get_api_client(source)
        enhanced = await api_client.get_anime(source_id, fields=fields)

        # Merge with offline data
        return self.merge_data(anime, enhanced, source)
```

#### Tier 3: Selective Scraping (Last Resort)

```python
class SelectiveScrapingStrategy:
    """
    Scrape only when explicitly needed and no API available
    """

    SCRAPING_TRIGGERS = {
        'synopsis': ['animeplanet', 'anisearch', 'livechart'],
        'reviews': ['animeplanet', 'anisearch'],
        'schedule': ['livechart', 'animecountdown'],
        'streaming': ['livechart']
    }

    async def should_scrape(self, source: str, required_fields: List[str]) -> bool:
        # Only scrape if:
        # 1. User explicitly requested fields not in offline DB
        # 2. Source has no API
        # 3. Source is known to have the required data

        if source in self.API_SOURCES:
            return False  # Use API instead

        for field in required_fields:
            if field in self.SCRAPING_TRIGGERS.get(source, []):
                return True

        return False

    async def scrape_selectively(self, source_url: str, fields: List[str]):
        # Check cache first (respect the sites!)
        cache_key = f"scrape:{source_url}:{','.join(sorted(fields))}"
        if cached := await self.cache.get(cache_key):
            return cached

        # Scrape only required sections
        scraper = self.get_scraper(source)
        data = await scraper.scrape_fields(source_url, fields)

        # Cache for a long time
        await self.cache.set(cache_key, data, ttl=7*24*3600)  # 1 week

        return data
```

### Smart Query Routing

```python
class SourceAwareQueryRouter:
    """
    Route queries intelligently based on source capabilities
    """

    async def process_source_query(self, query: str, context: dict):
        # Extract source preference
        source = context.get('source')  # e.g., "mal", "animeplanet"

        # Determine data needs
        needs = self.analyze_query_needs(query)

        # Decision tree
        if not source:
            # No source specified - use best available
            return await self.use_best_sources(query, needs)

        elif source in self.API_SOURCES:
            # Has API - prefer API over scraping
            return await self.use_api_with_fallback(query, source, needs)

        elif needs.requires_only_basic_data:
            # Non-API source but basic data sufficient
            return await self.use_offline_only(query, source)

        elif needs.requires_rich_data:
            # Non-API source needs scraping
            if self.user_consents_to_scraping:
                return await self.use_selective_scraping(query, source, needs)
            else:
                return await self.use_offline_with_disclaimer(query, source)
```

### Source URL Parsing

```python
class SourceURLParser:
    """
    Extract IDs and build URLs for all sources
    """

    PATTERNS = {
        'mal': r'myanimelist\.net/anime/(\d+)',
        'anilist': r'anilist\.co/anime/(\d+)',
        'kitsu': r'kitsu\.io/anime/(\d+)',
        'animeplanet': r'anime-planet\.com/anime/([\w-]+)',
        'anidb': r'anidb\.net/anime/(\d+)',
        'anisearch': r'anisearch\.com/anime/(\d+)',
        'notify': r'notify\.moe/anime/([\w-]+)',
        'simkl': r'simkl\.com/anime/(\d+)',
        'livechart': r'livechart\.me/anime/(\d+)',
        'ann': r'animenewsnetwork\.com/encyclopedia/anime\.php\?id=(\d+)',
        'animecountdown': r'animecountdown\.com/anime/([\w-]+)'
    }

    def extract_source_mapping(self, source_urls: List[str]) -> dict:
        """Extract source -> ID mapping from URLs"""
        mapping = {}

        for url in source_urls:
            for source, pattern in self.PATTERNS.items():
                if match := re.search(pattern, url):
                    mapping[source] = {
                        'id': match.group(1),
                        'url': url
                    }
                    break

        return mapping
```

### Example Flow: "Show me Death Note from Anime-Planet"

```python
# 1. Query offline DB
anime = offline_db.search("Death Note")[0]
# Found with sources array

# 2. Check if Anime-Planet URL exists
anime_planet_url = "https://anime-planet.com/anime/death-note"
# ✓ Found in sources

# 3. User wants synopsis (not in offline DB)
if "synopsis" in required_fields:
    # Anime-Planet has no API
    if should_enhance:
        # Scrape just the synopsis section
        synopsis = await scrape_field(anime_planet_url, "synopsis")
        anime['synopsis'] = synopsis
        anime['_enhancement'] = {
            'source': 'anime-planet',
            'method': 'scraping',
            'fields': ['synopsis'],
            'cached_until': '2024-02-23'
        }

# 4. Return enriched data
return anime
```

### Benefits of This Approach

1. **Respects User Intent**: When user says "from MAL", we actually use MAL
2. **Minimizes External Calls**: Offline DB first, enhance only when needed
3. **Handles Missing Sources**: Gracefully reports when anime not on requested source
4. **Smart Caching**: Scrape once, cache for a week
5. **Progressive Enhancement**: Start fast, enhance on demand
6. **Source Validation**: Verify anime exists on source before operations

### Implementation Priority

1. **Phase 1**: Offline + API sources only
2. **Phase 2**: Add source URL parsing and validation
3. **Phase 3**: Selective field scraping for non-API sources
4. **Phase 4**: Full source parity with intelligent routing

## Integration with Existing Workflow - NO New Endpoints Needed!

### The Beautiful Realization

We DON'T need new workflow endpoints! Source-specific queries work within existing architecture:

```python
# Existing endpoint handles it all
POST /api/workflow/conversation
{
    "message": "Show me psychological anime from Anime-Planet"
}

# LLM extracts:
# - intent: search
# - genre: psychological
# - source: anime-planet
# - required_fields: auto-detected based on query
```

### How It Works Within Current Architecture

#### 1. LLM Parameter Extraction (Existing)

```python
class EnhancedLLMExtractor:
    """
    LLM already extracts parameters, just add source awareness
    """

    async def extract_parameters(self, query: str) -> dict:
        # Existing extractions
        params = {
            'intent': 'search',
            'genres': ['psychological'],
            'limit': 10
        }

        # NEW: Source extraction
        if source := self.extract_source_preference(query):
            params['source'] = source  # 'mal', 'anime-planet', etc.

        return params
```

#### 2. Tool Enhancement (Modified Existing Tools)

```python
@tool("search_anime")
async def search_anime(
    query: str,
    genres: List[str] = None,
    limit: int = 10,
    enhance: bool = None,  # Existing
    source: Optional[str] = None,  # NEW parameter
    fields: Optional[List[str]] = None  # Existing
) -> List[AnimeResult]:
    """
    Same tool, now source-aware
    """

    # Step 1: Search offline DB (existing logic)
    results = await offline_db.search(
        query=query,
        genres=genres,
        limit=limit * 3  # Get more for filtering
    )

    # Step 2: NEW - Filter by source if specified
    if source:
        filtered = []
        for anime in results:
            # Check if anime exists on requested source
            source_url = find_source_url(anime['sources'], source)
            if source_url:
                anime['_source_url'] = source_url
                anime['_requested_source'] = source
                filtered.append(anime)
        results = filtered[:limit]

    # Step 3: Enhancement decision (existing logic, now source-aware)
    if should_enhance(enhance, fields, source):
        results = await enhance_results(results, fields, source)

    return results
```

### When Does Scraping/API Enhancement Trigger?

The SAME triggers as before, just source-aware:

```python
def should_enhance(enhance: bool, fields: List[str], source: str) -> bool:
    """
    Enhancement triggers - same logic, source-aware
    """

    # Explicit enhancement request
    if enhance is True:
        return True

    # Required fields not in offline DB
    if fields:
        missing_fields = set(fields) - OFFLINE_DB_FIELDS
        if missing_fields:
            return True

    # Auto-detect based on query patterns (existing)
    if needs_synopsis or needs_schedule or needs_streaming:
        return True

    # NEW: Source-specific enhancement rules
    if source == 'livechart' and needs_seasonal_data:
        return True  # LiveChart specializes in seasonal

    return False
```

### Enhancement Flow (Same as Current)

```python
async def enhance_results(results: List[dict], fields: List[str], source: str) -> List[dict]:
    """
    Enhancement logic - now respects source preference
    """
    enhanced = []

    for anime in results:
        if source and anime.get('_requested_source'):
            # User wants specific source
            if source in API_SOURCES:
                # Use API
                data = await enhance_via_api(anime, source, fields)
            else:
                # Scraping needed (Anime-Planet, etc.)
                data = await enhance_via_scraping(anime, source, fields)
        else:
            # No source preference - use best available (existing logic)
            data = await enhance_best_source(anime, fields)

        enhanced.append(data)

    return enhanced
```

### Real Examples - How Queries Flow

#### Example 1: "Show me Death Note from MAL"

```python
# 1. LLM extracts
params = {
    'query': 'Death Note',
    'source': 'mal'
}

# 2. Tool searches offline DB
anime = offline_db.get('death-note')

# 3. Validates MAL URL exists
mal_url = "https://myanimelist.net/anime/1535"  # ✓ Found

# 4. No enhancement needed (basic query)
return anime  # Fast response with MAL URL
```

#### Example 2: "What's the plot of Steins;Gate on Anime-Planet?"

```python
# 1. LLM extracts
params = {
    'query': 'Steins;Gate',
    'source': 'anime-planet',
    'fields': ['synopsis']  # Plot = synopsis
}

# 2. Tool searches offline DB
anime = offline_db.get('steins-gate')

# 3. Validates Anime-Planet URL exists
ap_url = "https://anime-planet.com/anime/steins-gate"  # ✓

# 4. Enhancement triggered (synopsis not in offline DB)
if 'synopsis' not in anime:
    # Anime-Planet has no API → Scraping
    synopsis = await scrape_field(ap_url, 'synopsis')
    anime['synopsis'] = synopsis
    anime['_enhanced_from'] = 'anime-planet'
```

#### Example 3: Frontend Always-Enhanced Request

```python
POST /api/workflow/frontend
{
    "query": "currently airing",
    "required_fields": ["synopsis", "schedule", "streaming"]
}

# Frontend API ALWAYS enhances
# But now it can be source-aware if user filtered by platform
```

### The Beauty of This Approach

1. **NO new endpoints** - Works within existing `/api/workflow/*`
2. **NO new complexity** - Just one new parameter: `source`
3. **Same enhancement flow** - Just source-aware
4. **Backward compatible** - Old queries work exactly the same
5. **Progressive enhancement** - Still returns fast, enhances on demand

### Scraping Triggers Summary

Scraping ONLY happens when ALL are true:

1. User needs data not in offline DB (synopsis, reviews, etc.)
2. Requested source has no API (Anime-Planet, LiveChart, etc.)
3. Enhancement is justified (not just browsing)
4. Data is not already cached

Otherwise, we use:

- Offline DB for basic data (always)
- APIs for sources that have them
- Cached scraped data when available

## Low-Level Hybrid Execution Plan

### Decision Point Architecture

The hybrid decision happens **INSIDE THE TOOL**, not in the LLM. Here's the exact flow:

````
User Query → LLM (LangGraph) → Tool Call → [DECISION POINT] → Response
                                              ↓
                                    Tool Internal Logic:
                                    1. Analyze what's needed
                                    2. Check offline DB first
                                    3. Decide if enhancement needed
                                    4. Route to appropriate source

## Cost & Rate Limiting Strategy

### The Financial Reality Check

Let's calculate real costs and rate limit impacts:

#### API Rate Limits & Costs
```yaml
MAL/Jikan:
  - Rate: 2 req/sec, 60 req/min
  - Cost: Free (but strict limits)
  - Daily max: ~86,400 requests

AniList:
  - Rate: 90 req/min (1.5 req/sec)
  - Cost: Free
  - Daily max: ~129,600 requests

Kitsu:
  - Rate: 36,000 req/hour (10 req/sec)
  - Cost: Free
  - Daily max: ~864,000 requests

AnimeSchedule:
  - Rate: Unknown (be reasonable)
  - Cost: Free API

SIMKL:
  - Rate: Requires API key
  - Cost: Free tier available

Scraping Costs:
  - Bandwidth: ~$0.09/GB
  - Proxy services: $10-50/month
  - Captcha solving: $1-3 per 1000
````

### Cost Optimization Strategies

#### 1. **Request Prioritization by ROI**

```python
class CostAwareRequestManager:
    """
    Prioritize requests based on value vs cost
    """

    REQUEST_VALUE_SCORES = {
        # High value (user-facing, real-time needs)
        'user_search': 10,
        'trending_anime': 9,
        'airing_schedule': 9,
        'frontend_enhancement': 8,

        # Medium value (nice to have)
        'recommendations': 6,
        'similar_anime': 5,
        'character_details': 5,

        # Low value (background tasks)
        'metadata_refresh': 3,
        'image_updates': 2,
        'score_updates': 2
    }

    async def should_make_request(self, request_type: str, source: str) -> bool:
        # Check rate limit budget
        current_usage = await self.get_usage_percentage(source)
        value_score = self.REQUEST_VALUE_SCORES.get(request_type, 5)

        # High-value requests: Allow up to 80% usage
        if value_score >= 8 and current_usage < 80:
            return True

        # Medium-value: Allow up to 60% usage
        if value_score >= 5 and current_usage < 60:
            return True

        # Low-value: Only if under 30% usage
        if current_usage < 30:
            return True

        return False
```

#### 2. **Dynamic Source Selection by Cost**

```python
class CostOptimizedSourceSelector:
    """
    Choose cheapest source that meets requirements
    """

    SOURCE_COSTS = {
        # Free, high limits
        'kitsu': {'cost': 0, 'limit': 864000, 'reliability': 0.95},

        # Free, medium limits
        'anilist': {'cost': 0, 'limit': 129600, 'reliability': 0.98},

        # Free, low limits
        'mal': {'cost': 0, 'limit': 86400, 'reliability': 0.99},

        # Scraping costs
        'animeplanet': {'cost': 0.001, 'limit': 10000, 'reliability': 0.80},
        'livechart': {'cost': 0.002, 'limit': 5000, 'reliability': 0.75}
    }

    def select_source_for_data(self, required_fields: List[str]) -> str:
        """Select cheapest source that has required data"""

        candidates = []
        for source, capabilities in SOURCE_CAPABILITIES.items():
            if all(field in capabilities for field in required_fields):
                cost_info = self.SOURCE_COSTS[source]
                if self.has_remaining_quota(source):
                    candidates.append({
                        'source': source,
                        'cost': cost_info['cost'],
                        'reliability': cost_info['reliability']
                    })

        # Sort by cost, then reliability
        candidates.sort(key=lambda x: (x['cost'], -x['reliability']))

        return candidates[0]['source'] if candidates else None
```

#### 3. **Intelligent Caching with TTL by Value**

```python
class ValueBasedCachingStrategy:
    """
    Cache duration based on data value and change frequency
    """

    CACHE_STRATEGIES = {
        # Long cache - rarely changes
        'anime_metadata': {
            'ttl': 30 * 24 * 3600,  # 30 days
            'cost_per_miss': 0.001
        },

        # Medium cache - changes weekly
        'anime_scores': {
            'ttl': 7 * 24 * 3600,   # 7 days
            'cost_per_miss': 0.001
        },

        # Short cache - changes daily
        'airing_schedule': {
            'ttl': 6 * 3600,        # 6 hours
            'cost_per_miss': 0.002
        },

        # Very short - real-time data
        'streaming_availability': {
            'ttl': 3600,            # 1 hour
            'cost_per_miss': 0.003
        }
    }

    def calculate_cache_ttl(self, data_type: str, source: str) -> int:
        base_ttl = self.CACHE_STRATEGIES[data_type]['ttl']

        # Adjust by source reliability
        if source in ['mal', 'anilist']:
            return int(base_ttl * 1.2)  # More reliable, cache longer
        elif source in ['scraping']:
            return int(base_ttl * 0.8)  # Less reliable, refresh sooner

        return base_ttl
```

#### 4. **User Tier System for Rate Limit Distribution**

```python
class UserTierRateLimiting:
    """
    Distribute rate limits fairly across user tiers
    """

    TIER_ALLOCATIONS = {
        'free': {
            'percentage': 20,  # 20% of total capacity
            'requests_per_hour': 100,
            'enhancement_limit': 10,  # Enhanced results per hour
            'cache_priority': 'low'
        },
        'premium': {
            'percentage': 50,  # 50% of total capacity
            'requests_per_hour': 1000,
            'enhancement_limit': 100,
            'cache_priority': 'high'
        },
        'enterprise': {
            'percentage': 30,  # 30% of total capacity
            'requests_per_hour': 10000,
            'enhancement_limit': 'unlimited',
            'cache_priority': 'highest'
        }
    }

    async def check_user_quota(self, user_id: str, request_type: str) -> bool:
        tier = await self.get_user_tier(user_id)
        usage = await self.get_user_usage(user_id, request_type)

        limits = self.TIER_ALLOCATIONS[tier]

        if request_type == 'enhancement':
            return usage < limits['enhancement_limit']
        else:
            return usage < limits['requests_per_hour']
```

#### 5. **Cost Monitoring & Alerts**

```python
class CostMonitor:
    """
    Track and alert on API usage costs
    """

    def __init__(self):
        self.daily_budget = 10.00  # $10/day
        self.alert_thresholds = [0.5, 0.8, 0.95]  # 50%, 80%, 95%

    async def track_request(self, source: str, request_type: str):
        cost = self.calculate_request_cost(source, request_type)

        await self.increment_daily_cost(cost)

        # Check thresholds
        usage_percent = await self.get_daily_usage_percent()

        for threshold in self.alert_thresholds:
            if usage_percent >= threshold * 100:
                await self.send_alert(
                    f"Daily budget {int(threshold*100)}% used",
                    source,
                    cost
                )
```

### Smart Rate Limit Distribution

```python
class IntelligentRateLimiter:
    """
    Distribute rate limits intelligently across operations
    """

    async def allocate_request_budget(self, hour: int) -> dict:
        """Allocate hourly budget based on usage patterns"""

        # Peak hours (6PM - 10PM local time)
        if 18 <= hour <= 22:
            return {
                'user_searches': 0.40,      # 40% for user searches
                'frontend_enhance': 0.30,   # 30% for frontend
                'background_jobs': 0.10,    # 10% for background
                'api_updates': 0.10,        # 10% for updates
                'buffer': 0.10              # 10% emergency buffer
            }

        # Off-peak hours (2AM - 6AM)
        elif 2 <= hour <= 6:
            return {
                'user_searches': 0.10,      # Few users
                'frontend_enhance': 0.10,
                'background_jobs': 0.50,    # Heavy background processing
                'api_updates': 0.20,        # Update caches
                'buffer': 0.10
            }

        # Normal hours
        else:
            return {
                'user_searches': 0.30,
                'frontend_enhance': 0.25,
                'background_jobs': 0.20,
                'api_updates': 0.15,
                'buffer': 0.10
            }
```

### Graceful Degradation Under Load

```python
class LoadBasedDegradation:
    """
    Reduce service quality gracefully under high load
    """

    DEGRADATION_LEVELS = [
        {
            'load': 0.9,  # 90% capacity
            'actions': [
                'disable_image_searches',
                'reduce_enhancement_fields',
                'increase_cache_ttl_2x'
            ]
        },
        {
            'load': 0.8,  # 80% capacity
            'actions': [
                'limit_results_to_10',
                'disable_recommendations',
                'use_stale_cache_if_available'
            ]
        },
        {
            'load': 0.7,  # 70% capacity
            'actions': [
                'prefer_offline_db_only',
                'batch_similar_requests'
            ]
        }
    ]
```

### Estimated Costs at Scale

```python
# Assumptions: 10,000 daily active users
Daily Requests:
- Searches: 50,000 (5 per user)
- Enhancements: 20,000 (2 per user)
- Background: 10,000

Cost Breakdown:
- API calls: $0 (using free tiers smartly)
- Scraping bandwidth: ~$5/day (50GB)
- Proxy services: $1.67/day ($50/month)
- Cache storage: $2/day (Redis)
- Total: ~$8.67/day or $260/month

Revenue Needed:
- Free tier: 70% of users
- Premium ($5/mo): 25% of users = $1,250/mo
- Enterprise ($50/mo): 5% of users = $2,500/mo
- Total: $3,750/mo (healthy margin)
```

### Key Takeaways for Cost Management

1. **Kitsu First**: 10x more generous rate limits than MAL
2. **Cache Aggressively**: 90% of requests should hit cache
3. **Time-Shift Load**: Background jobs during off-peak
4. **User Tiers**: Let premium users subsidize free tier
5. **Degrade Gracefully**: Better slow than error 2. Query offline DB (always) 3. Enhance if needed (conditional) 4. Consolidate results 5. Return unified data

## Data Conflict Resolution Strategy

### The Reality of Multi-Source Conflicts

[Content moved to Phase 5 documentation]

### Radical Modernization Approach (No Backward Compatibility)

Since you want everything modern and don't care about backward compatibility, here's the **ultra-aggressive** migration strategy:

#### 1. **New API Structure - Extreme Simplification**

```
/api/query              # Single unified LLM endpoint for EVERYTHING
/api/conversation       # Stateful conversations (renamed from workflow)
/api/admin/*           # Keep admin endpoints as-is
/api/data/status       # Keep data status endpoints
```

**Complete Removal:**

- `/api/search/*` - All search through LLM
- `/api/recommendations/*` - Absorbed into LLM
- `/api/workflow/*` - Renamed to conversation

#### 2. **Critical Breaking Points**

**Performance Degradation:**

- **Before**: ~50-200ms direct vector search
- **After**: ~500-2000ms with LLM processing
- **Impact**: 10x latency increase
- **Mitigation**: Edge LLM deployment + aggressive caching

**Cost Explosion:**

- **Before**: ~$0 per request (just compute)
- **After**: $0.001-0.01 per request (LLM tokens)
- **Impact**: At 100k requests/day = $100-1000/day
- **Mitigation**: Request coalescing + response caching

**MCP Tool Compatibility:**

- **Issue**: MCP tools expect structured responses
- **Solution**: LLM output formatters that maintain tool contracts

```python
class MCPResponseFormatter:
    def format_for_tool(self, llm_response: str, tool_name: str) -> dict:
        # Convert LLM narrative to structured MCP response
```

**Error Handling Paradigm Shift:**

- **Before**: Deterministic errors (404, validation)
- **After**: Probabilistic responses (LLM confusion)
- **Solution**: Confidence thresholds + retry logic

```python
if llm_confidence < 0.7:
    return {"error": "Query unclear", "suggestions": [...]}
```

**Rate Limiting Cascade:**

- **OpenAI**: 90k tokens/minute
- **Anthropic**: 100k tokens/minute
- **Impact**: ~50-100 requests/minute max
- **Solution**: Multi-provider load balancing

#### 3. **Week-by-Week Implementation**

**Week 1: Unified Query Handler**

```python
@router.post("/api/query")
async def unified_query(request: UnifiedRequest):
    """
    Single endpoint that replaces ALL search/recommendation endpoints
    """
    # LLM determines intent and executes
    llm_response = await llm.process(
        query=request.query,
        context=request.context,
        tools=ALL_MCP_TOOLS  # Give LLM access to everything
    )
    return format_response(llm_response)
```

**Week 2: Delete Legacy Code**

```bash
# Aggressive removal
rm src/api/search.py
rm src/api/recommendations.py
git rm -r tests/unit/api/test_search_api.py
# Update main.py to remove route mounting
```

**Week 3: Performance Optimization**

```python
# Redis caching for common queries
@cache(ttl=3600)
async def cached_llm_query(query_hash: str):
    return await llm.process(query)

# Request deduplication
PENDING_QUERIES = {}  # query_hash -> Future
async def deduplicated_query(query: str):
    query_hash = hash(query)
    if query_hash in PENDING_QUERIES:
        return await PENDING_QUERIES[query_hash]
```

#### 4. **New Request/Response Format**

**Unified Request:**

```python
class UnifiedRequest(BaseModel):
    query: str  # Natural language, anything goes
    context: Optional[Dict]  # Session/conversation context
    source_preference: Optional[str]  # "MAL", "AniList", etc.
    mode: Literal["fast", "thorough", "creative"]  # LLM behavior
```

**Streaming Response:**

```python
# WebSocket for real-time streaming
@router.websocket("/api/stream")
async def stream_query(websocket: WebSocket):
    await websocket.accept()

    # Stream LLM reasoning process
    async for chunk in llm.stream_process(query):
        await websocket.send_json({
            "type": "reasoning",
            "content": chunk
        })

    # Final results
    await websocket.send_json({
        "type": "results",
        "data": final_results
    })
```

#### 5. **Direct Access Escape Hatches**

For cases where you NEED direct access (MCP tools, performance critical):

```python
@router.post("/api/direct/{tool_name}")
async def direct_tool_access(
    tool_name: str,
    params: Dict,
    bypass_llm: bool = True
):
    """Emergency hatch for direct tool access"""
    if tool_name not in ALLOWED_DIRECT_TOOLS:
        raise HTTPException(403, "Direct access not allowed")

    tool = get_mcp_tool(tool_name)
    return await tool.execute(**params)
```

#### 6. **Migration Execution Plan**

**Day 1-2: Setup**

- Fork repo for radical changes
- Set up new `/api/query` endpoint
- Configure multi-LLM providers

**Day 3-5: Rip and Replace**

- Delete all static endpoints
- Update all tests to use new endpoint
- Remove unused dependencies

**Day 6-7: Optimization**

- Implement caching layer
- Add request coalescing
- Set up monitoring

**Day 8-10: MCP Compatibility**

- Test all MCP tools with new system
- Add response formatters
- Ensure tool contracts maintained

### The Nuclear Option

If you want to go EVEN MORE radical:

```python
# main.py - The entire API
app = FastAPI()

@app.post("/api/{path:path}")
async def handle_everything(path: str, request: Request):
    """One endpoint to rule them all"""
    body = await request.json()

    # Give EVERYTHING to the LLM
    result = await llm.process(
        path=path,
        body=body,
        intent="Figure out what the user wants and do it"
    )

    return result

# That's it. That's the whole API.
```

### 2024-01-27 - Migration Strategy Added

- Designed radical migration with no backward compatibility
- Identified 5 critical breaking points
- Created week-by-week implementation plan
- Proposed "nuclear option" for ultimate simplification

## Simplified API Design (Full LLM-Driven)

**UPDATED**: Going fully LLM-driven, removing all static endpoints.

### Final API Structure (Only 2 endpoints!)

#### 1. **`/api/query`** - Universal LLM-Driven Endpoint

Handles EVERYTHING through LLM interpretation:

```python
class QueryRequest(BaseModel):
    # Natural language or structured input
    query: Union[str, Dict[str, Any]]

    # Optional multimodal support
    image: Optional[str] = None  # Base64 or URL

    # Context and preferences
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None
```

**Handles ALL cases through LLM:**

- Simple lookups: `{"query": "get anime with id 12345"}`
- Natural language: `{"query": "find dark psychological anime like Death Note"}`
- Complex queries: `{"query": "compare ratings of Steins;Gate across all platforms"}`
- Image search: `{"query": "find anime similar to this", "image": "..."}`
- With context: `{"query": "what about the second one?", "context": {"previous_results": [...]}}`
- Direct data: `{"query": {"anime_id": 12345, "fields": ["synopsis", "score"]}}`

#### 2. **`/api/batch`** - Bulk Operations

For efficient batch processing:

```python
class BatchRequest(BaseModel):
    queries: List[QueryRequest]
    options: Optional[Dict[str, Any]] = None
```

**Use cases:**

- Mobile app syncing multiple anime
- Bulk operations for performance
- Parallel processing of queries

### APIs to Remove (Going Full LLM)

**All static endpoints removed:**

- `/api/search` ❌
- `/api/anime/{anime_id}` ❌
- `/api/similar/{anime_id}` ❌
- `/api/workflow/search` ❌
- `/api/workflow/multimodal/search` ❌
- `/api/workflow/image-search` ❌
- `/api/workflow/conversation` ❌
- `/api/workflow/smart-conversation` ❌
- `/api/frontend/ai-search` ❌

**Keep only:**

- `/api/query` ✅ (Universal LLM endpoint)
- `/api/batch` ✅ (Bulk operations)
- `/api/admin/*` ✅ (Admin operations)
- `/health` ✅ (Health check)
- `/stats` ✅ (Statistics)

### Migration Path

```
OLD                              → NEW
/api/search?q=...                → /api/query {"query": "search for ..."}
/api/anime/12345                 → /api/query {"query": "get anime with id 12345"}
/api/similar/12345               → /api/query {"query": "find anime similar to id 12345"}
/api/workflow/*                  → /api/query {"query": "..."}
/api/frontend/ai-search          → /api/query {"query": "...", "options": {"enhance": true}}
```

### Why Full LLM Approach

1. **Radical Simplicity**: Just 2 endpoints for all operations
2. **Future Proof**: Any new query type works without API changes
3. **Cost Optimization**: LLM can recognize simple queries and optimize internally
4. **True AI-Native**: Every request benefits from LLM intelligence

### Implementation Strategy

**Phase 1**:

- Implement `/api/query` with LLM routing
- Remove all static endpoints
- Update MCP tools to use new endpoint

**Phase 2** (if needed):

- Add static endpoints back for performance optimization
- Only after LLM approach is proven and optimized

### 2024-01-28 - Updated to Full LLM Design

- Decided to go fully LLM-driven from the start
- Removed all static API endpoints
- Simplified to just 2 main endpoints: `/api/query` and `/api/batch`
- Can add static endpoints later if performance requires

### Cost Optimization

```python
class ScrapingCostOptimizer:
    """Minimize LLM costs for scraping"""

    def __init__(self):
        self.extraction_cache = TTLCache(maxsize=10000, ttl=3600)
        self.pattern_cache = {}  # Learned patterns per site

    async def extract_with_caching(self, url: str, html: str) -> Dict:
        # Check if we've extracted from this URL recently
        cache_key = hashlib.md5(url.encode()).hexdigest()
        if cached := self.extraction_cache.get(cache_key):
            return cached

        # Use smallest effective model
        if len(html) < 5000:
            model = "gpt-3.5-turbo"  # $0.001 per extraction
        else:
            # Chunk and extract
            model = "gpt-3.5-turbo-16k"

        result = await self._extract(html, model)
        self.extraction_cache[cache_key] = result
        return result
```

### 5. **Caching Strategy**

```python
# src/integrations/cache.py
class IntegrationCache:
    """Multi-layer caching for API responses"""

    def __init__(self):
        self.memory_cache = TTLCache(maxsize=10000, ttl=300)  # 5 min
        self.redis_cache = RedisCache(ttl=3600)  # 1 hour
        self.db_cache = DatabaseCache(ttl=86400)  # 24 hours

    async def get_or_fetch(
        self,
        key: str,
        fetcher: Callable,
        cache_level: str = "memory"
    ):
        # Try memory cache first
        if value := self.memory_cache.get(key):
            return value

        # Try Redis
        if cache_level in ["redis", "db"]:
            if value := await self.redis_cache.get(key):
                self.memory_cache[key] = value
                return value

        # Try database
        if cache_level == "db":
            if value := await self.db_cache.get(key):
                await self.redis_cache.set(key, value)
                self.memory_cache[key] = value
                return value

        # Fetch from API
        value = await fetcher()
        await self._cache_value(key, value, cache_level)
        return value
```

### 6. **Integration with MCP Tools**

```python
# src/mcp/tools.py
@tool("search_anime")
async def search_anime(
    query: str,
    source: Optional[str] = None,
    enhance: bool = None,
    fields: Optional[List[str]] = None
) -> List[AnimeResult]:
    """Enhanced search with third-party API support"""

    # Start with offline database
    results = await qdrant_client.search(query)

    # Determine if enhancement needed
    if enhance or _should_enhance(query, fields):
        service_manager = AnimeServiceManager()

        # Enhance results with API data
        enhanced_results = []
        for anime in results[:10]:  # Limit API calls
            enhanced = anime.dict()

            # Fetch missing fields from best sources
            if fields and "synopsis" in fields:
                synopsis = await service_manager.smart_fetch(
                    "synopsis",
                    anime.anime_id
                )
                enhanced["synopsis"] = synopsis

            if fields and "streaming" in fields:
                streaming = await service_manager.smart_fetch(
                    "streaming",
                    anime.anime_id
                )
                enhanced["streaming_links"] = streaming

            enhanced_results.append(enhanced)

        return enhanced_results

    return results
```

### 8. **Error Handling & Resilience**

```python
class ServiceHealthChecker:
    """Monitor service health and availability"""

    async def check_all_services(self):
        for name, service in self.services.items():
            try:
                await service.health_check()
                self.status[name] = "healthy"
            except Exception as e:
                self.status[name] = "unhealthy"
                await self._handle_unhealthy_service(name, e)

    async def _handle_unhealthy_service(self, name: str, error: Exception):
        # Log error
        logger.error(f"Service {name} unhealthy: {error}")

        # Update circuit breaker
        self.circuit_breakers[name].record_failure()

        # Notify monitoring
        await self.monitoring.alert(f"Service {name} is down")
```

### Implementation Priority

1. **Phase 1**: Core API Services

   - AniList (GraphQL, no auth)
   - Jikan/MAL (REST, no auth)
   - Kitsu (REST, no auth)
   - Memory caching only

2. **Phase 2**: Additional APIs & Caching

   - AniDB (limited API)
   - Notify.moe (GraphQL)
   - Redis caching layer
   - Circuit breakers

3. **Phase 3**: Scraper Services

   - AniSearch (German DB)
   - LiveChart (schedules)
   - Anime-Planet
   - AnimeNewsNetwork
   - SIMKL

4. **Phase 4**: Advanced Features
   - OAuth for MAL/Notify
   - Smart routing based on data type
   - Playwright for JS-heavy sites
   - Request coalescing

### Source Selection Strategy

```python
class SourceSelector:
    """Intelligently select best source for specific data"""

    SOURCE_STRENGTHS = {
        'anilist': ['synopsis', 'characters', 'staff', 'relations'],
        'mal': ['score', 'reviews', 'recommendations', 'popularity'],
        'kitsu': ['streaming', 'episodes', 'community'],
        'anidb': ['technical_details', 'file_hashes', 'fansubs'],
        'notify': ['watch_progress', 'social_features'],
        'anisearch': ['german_titles', 'german_synopsis'],
        'livechart': ['airing_schedule', 'countdown', 'seasonal'],
        'animenewsnetwork': ['news', 'industry_info', 'staff_details'],
        'animeplanet': ['tags', 'recommendations', 'characters'],
        'simkl': ['watch_tracking', 'ratings', 'reviews']
    }

    def select_source(self, field: str, user_preference: str = None) -> str:
        if user_preference and user_preference in self.services:
            return user_preference

        # Find best source for field
        for source, strengths in self.SOURCE_STRENGTHS.items():
            if field in strengths:
                return source

        # Default fallback order
        return 'anilist'  # Most comprehensive
```

### 2024-01-28 - Third-Party API Integration Added

- Designed service client architecture with base class
- Created service manager for intelligent routing
- Added multi-layer caching strategy
- Included authentication management
- Integrated with existing MCP tools

## Web Scraping Research & Implementation Results

### Successful Scraping Test Results (2024-01-28)

After thorough testing, we discovered that scraping non-API anime sources is **much simpler than anticipated**:

#### Key Findings:

1. **Cloudscraper Successfully Bypasses Cloudflare**

   - Tested 5 sites with Cloudflare protection
   - ALL returned 200 OK status codes
   - No need for complex browser automation or Playwright
   - Response times: 200-500ms (very fast!)

2. **Actual Test Results**:

   ```
   ✅ Anime-Planet: Full data extraction (title, synopsis, rating)
   ✅ LiveChart: JSON-LD structured data! (episodes, genres, studio, rating)
   ✅ AniSearch: Complete German synopsis (can translate)
   ✅ Anime News Network: Successfully scraped
   ✅ SIMKL: Successfully accessed
   ```

3. **Simple Implementation**:
   ```python
   # That's all we need!
   scraper = cloudscraper.create_scraper()
   response = scraper.get(url)
   soup = BeautifulSoup(response.text, 'html.parser')
   ```

#### Revised Scraping Architecture:

```python
# src/integrations/scrapers/simple_scraper.py
class SimpleAnimeScaper:
    """Lightweight scraper using cloudscraper"""

    def __init__(self):
        self.scraper = cloudscraper.create_scraper()
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache

    async def scrape_on_demand(self, source_url: str, source: str) -> Dict:
        """Only scrape when user explicitly requests a source"""

        # Check cache first
        if cached := self.cache.get(source_url):
            return cached

        # Scrape based on source type
        if source == 'livechart':
            data = await self._scrape_livechart(source_url)
        elif source == 'anime-planet':
            data = await self._scrape_anime_planet(source_url)
        # ... etc

        self.cache[source_url] = data
        return data

    async def _scrape_livechart(self, url: str) -> Dict:
        """LiveChart has JSON-LD structured data!"""
        response = self.scraper.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract JSON-LD
        json_ld = soup.find('script', {'type': 'application/ld+json'})
        if json_ld:
            return json.loads(json_ld.string)

        return self._fallback_extraction(soup)
```

#### On-Demand Scraping Flow:

1. **User Query**: "Show me Death Note from LiveChart"
2. **LLM Extracts**: `{"anime": "Death Note", "source": "livechart"}`
3. **Tool Triggers**: Only scrapes if source is specified
4. **Fast Response**: ~200-500ms additional latency
5. **Cached**: Subsequent requests are instant

#### Performance Impact:

- **Without scraping**: 50-200ms (offline DB only)
- **With scraping**: 250-700ms total (still under 1 second!)
- **Cached requests**: Back to 50-200ms

#### Why This Approach is Superior:

1. **No LLM Extraction Needed**: BeautifulSoup + selectors work perfectly
2. **No Browser Automation**: Cloudscraper handles JavaScript challenges
3. **Minimal Dependencies**: Just `cloudscraper` and `beautifulsoup4`
4. **Cost Effective**: No LLM API calls for extraction
5. **Reliable**: Structured extraction beats probabilistic LLM parsing

#### Implementation Priority:

1. **Phase 1**: Implement cloudscraper for protected sites
2. **Phase 2**: Add source-specific extractors
3. **Phase 3**: Implement caching layer
4. **No Need For**: Playwright, Selenium, or LLM-based extraction

### 2024-01-28 - Web Scraping Implementation Validated

- Tested scraping on 5 real anime sites
- Cloudscraper successfully bypassed all Cloudflare protections
- Discovered LiveChart provides JSON-LD structured data
- Confirmed simple BeautifulSoup extraction is sufficient
- Scraping adds only 200-500ms latency when triggered

---

_This document will be updated as we make decisions and iterate on the plan_
