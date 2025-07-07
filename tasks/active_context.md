# Active Context
# Anime MCP Server - Current Implementation Session

## Current Work Focus

**🔄 CURRENT SPRINT**: Universal Query Endpoint Implementation (Task #52)

**Strategy**: Consolidate existing `/api/workflow/*` endpoints into single `/api/query` endpoint
- **Foundation**: LangGraph ReactAgent + 8 MCP tools (✅ ALREADY IMPLEMENTED)
- **Current Goal**: Rename and enhance for unified LLM-driven interface
- **Files to Modify**: `src/api/workflow.py` → `src/api/query.py`

## Active Decisions and Considerations

**Implementation Approach**:
- **Phase 1**: Rename `/api/workflow/conversation` → `/api/query` with auto-detection
- **Auto-Detection Logic**: Analyze request to determine text-only vs multimodal processing
- **Unified Models**: Create `UniversalQueryRequest`/`UniversalQueryResponse` for consistent interface
- **Backward Compatibility**: Maintain existing workflow endpoints during transition

**Key Design Questions**:
- Should we keep separate multimodal and text endpoints or unify completely?
- How to handle correlation ID integration in the new unified endpoint?
- What's the optimal auto-detection strategy for query types?

## Recent Changes

**System Status**: Production-ready system with all core functionality operational
- ✅ MCP server fully functional with 8 tools and 31 total tools across implementations
- ✅ Vector database operational with 38,894+ anime entries
- ✅ Multi-platform integration working (9 anime platforms)
- ✅ All critical infrastructure implemented and tested

## Next Steps

**Immediate Actions**:
1. **Analyze current workflow endpoints** - Review existing implementation in `src/api/workflow.py`
2. **Design unified request/response models** - Create `UniversalQueryRequest`/`UniversalQueryResponse`
3. **Implement auto-detection logic** - Determine query type from request content
4. **Update endpoint routing** - Consolidate multiple endpoints into single `/api/query`
5. **Add correlation ID tracking** - Integrate with existing correlation middleware

**Success Criteria**:
- Single `/api/query` endpoint handles all LLM-driven queries
- Auto-detection correctly routes text vs multimodal requests
- Backward compatibility maintained for existing workflow endpoints
- Correlation tracking works end-to-end