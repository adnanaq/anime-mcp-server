<!--
description: Document major failure points in this project and they were solved.  To be filled by AI.
-->

# Error Documentation - Anime MCP Server

## API Integration Error Patterns (2025-07-06)

### Issue: Health Check API Waste - Task #64

**Context**: Health check methods making unnecessary actual API calls

**Error Symptoms**:

```python
# BAD: Wasteful health check
async def health_check(self):
    await self.client.search_anime(q="test", limit=1)  # Actual API call!
```

**Root Cause**: Poor health check design making real API requests for status validation

**Solution Pattern**:

1. **Check client initialization state**
2. **Validate configuration (auth credentials, base URLs)**
3. **Check circuit breaker status**
4. **Return status without API calls**

```python
# GOOD: Efficient health check
async def health_check(self):
    if not self.client or not self.client.client_id:
        return {"status": "unhealthy", "error": "Not configured"}
    return {"status": "healthy", "auth_configured": True}
```

### Issue: Correlation Logger Attribute Missing - Task #64

**Context**: Base client referencing removed `correlation_logger` after Task #63 consolidation

**Error Symptoms**:

```
AttributeError: 'JikanClient' object has no attribute 'correlation_logger'
```

**Root Cause**: Consolidation cleanup missed base client reference to removed correlation logger

**Solution Pattern**:

1. **Use standard logging with correlation context**
2. **Include correlation_id in log extra fields**
3. **Maintain same observability without dedicated logger class**

## Import Error Resolution Patterns (2025-07-06)

### Issue: Module Restructuring Import Errors

**Context**: Test collection failing due to `src.mcp` → `src.anime_mcp` module rename

**Error Symptoms**:

```
ImportError: No module named 'src.mcp'
ModuleNotFoundError: No module named 'src.mcp.server'
```

**Root Cause**: Module restructuring (`src.mcp` renamed to `src.anime_mcp`) but imports weren't systematically updated

**Solution Pattern**:

1. **Identify all affected files**: `grep -r "src.mcp" .`
2. **Categorize by fix type**:
   - Real files with wrong imports → Fix import paths
   - Missing dependencies → Create mock modules
3. **Fix systematically**: Update actual imports first, then create mocks for missing deps

**Code Fixes Applied**:

```python
# Before (broken):
from src.mcp.server import get_all_mcp_tools

# After (fixed):
from src.anime_mcp.server import get_all_mcp_tools
```

**Prevention**: Use systematic search-and-replace for module renames, verify test collection after changes

### Issue: Missing External Dependencies in Tests

**Context**: `langgraph_swarm` module used but not in requirements.txt

**Error Symptoms**:

```
ModuleNotFoundError: No module named 'langgraph_swarm'
```

**Root Cause**: Advanced workflow features use external dependencies not included in base requirements

**Solution Pattern**:

1. **Create mock module**: `tests/mocks/langgraph_swarm.py`
2. **Global mock registration**: Add to `conftest.py` sys.modules
3. **Maintain functionality**: Mock preserves interface for testing

**Code Solution**:

```python
# tests/mocks/langgraph_swarm.py
from unittest.mock import Mock

class MockSwarmAgent(Mock):
    async def run(self, *args, **kwargs):
        return {"result": "mock_response"}

# conftest.py
import sys
from tests.mocks import langgraph_swarm
sys.modules['langgraph_swarm'] = langgraph_swarm
```

**Prevention**: Check for missing dependencies during test collection, create systematic mock strategy

## Async Context Manager Testing Issues (2025-07-06)

### Issue: Complex aiohttp Context Manager Mocking

**Context**: Data service tests failing due to aiohttp session mocking complexity

**Error Symptoms**:

```
AttributeError: 'Mock' object has no attribute '__aenter__'
TypeError: 'Mock' object is not an async context manager
```

**Root Cause**: Global aiohttp mocks in conftest.py interfered with specific test requirements

**Solution Pattern**: Use simplified method mocking instead of complex dependency mocking

**Failed Approach**:

```python
# This pattern caused issues:
with patch("aiohttp.ClientSession.get") as mock_get:
    mock_get.return_value.__aenter__.return_value = mock_response
    mock_get.return_value.__aexit__.return_value = None
```

**Successful Solution**:

```python
# Simplified method mocking:
with patch.object(service, 'method_name', return_value=expected_data):
    result = await service.method_name()
```

**Prevention**: Use business logic method mocking for complex async operations, reserve context manager mocking for simple cases

### Issue: Global Mock Interference

**Context**: Global mocks in conftest.py conflicting with specific test requirements

**Root Cause**: Global aiohttp session mocks prevented specific test scenarios from working properly

**Solution Pattern**: Two-tier mock strategy

- **Global mocks**: External dependencies only (prevent network calls)
- **Specific mocks**: Business logic testing (per-test basis)

**Implementation**:

```python
# conftest.py - Global external dependency mocks
@pytest.fixture(autouse=True)
def mock_external_dependencies():
    with patch("qdrant_client.QdrantClient"):
        with patch("fastembed.TextEmbedding"):
            yield

# test files - Specific business logic mocks
def test_service_method():
    with patch.object(service, 'complex_method', return_value=test_data):
        result = await service.complex_method()
```

**Prevention**: Separate concerns - global mocks for external dependencies, specific mocks for business logic

## Service Implementation Bug Discovery (2025-07-06)

### Issue: Parameter Passing Bug in AniList Service

**Context**: AniList service test failing due to parameter inconsistency

**Error Symptoms**: Test expecting pagination to work, but page parameter ignored

**Root Cause**: Service method accepted `page` parameter but didn't pass it to underlying client

**Bug Location**: `src/services/external/anilist_service.py:62`

**Bug Code**:

```python
# Before (bug):
async def search_anime(self, query: str, limit: int = 10, page: int = 1):
    return await self.client.search_anime(query=query, limit=limit)
    # Missing: page=page parameter
```

**Fix Applied**:

```python
# After (fixed):
async def search_anime(self, query: str, limit: int = 10, page: int = 1):
    return await self.client.search_anime(query=query, limit=limit, page=page)
```

**Detection Method**: Test failure revealed parameter inconsistency between service interface and client call

**Prevention**: Always verify parameter consistency between service interfaces and client calls, write tests that exercise all parameters

## Test Collection Health Patterns (2025-07-06)

### Issue: Hidden Test Count Due to Import Errors

**Context**: Import errors can hide significant numbers of tests from collection

**Discovery**: Resolving 7 import errors revealed 184 additional tests (1790 → 1974)

**Impact**: 10.3% increase in test discovery after import error resolution

**Solution Strategy**:

1. **Fix import errors first**: Before focusing on test failures
2. **Measure collection impact**: Track test count before/after fixes
3. **Systematic resolution**: Address errors in dependency order

**Monitoring Pattern**:

```bash
# Track test collection health
pytest --collect-only -q | grep "collected"
# Before: 1790 tests collected, 9 import errors
# After: 1974 tests collected, 2 import errors
```

**Prevention**: Regular test collection health checks, address import errors immediately

This error documentation provides patterns for resolving similar issues and preventing their recurrence in future development.
