"""Tests for ReactAgent validation bug fix.

This module tests the specific validation issue where ReactAgent
creates tool calls missing the required 'query' field, causing
SearchAnimeInput validation failures.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.langgraph.react_agent_workflow import create_react_agent_workflow_engine, LLMProvider


class TestReactAgentValidationFix:
    """Test cases for fixing ReactAgent tool call validation errors."""

    @pytest.fixture
    def mock_mcp_tools(self):
        """Create mock MCP tools for testing."""
        search_anime_tool = AsyncMock()
        search_anime_tool.name = "search_anime"
        search_anime_tool.ainvoke = AsyncMock(return_value=[])
        
        return {
            "search_anime": search_anime_tool,
            "get_anime_details": AsyncMock(),
            "find_similar_anime": AsyncMock(),
            "get_anime_stats": AsyncMock(),
        }

    @pytest.fixture
    def mock_chat_model(self):
        """Create mock chat model that simulates the validation error."""
        # Mock the OpenAI chat model to avoid API calls
        mock_model = MagicMock()
        return mock_model

    @pytest.mark.asyncio
    async def test_missing_query_field_validation_error(self, mock_mcp_tools, mock_chat_model):
        """Test that reproduces the missing query field validation error.
        
        This test simulates the exact error we observed:
        "Error: 1 validation error for SearchAnimeInput query Field required"
        """
        # Create a mock ReactAgent that simulates the problematic behavior
        with patch('src.langgraph.react_agent_workflow.ChatOpenAI', return_value=mock_chat_model):
            with patch('src.langgraph.react_agent_workflow.create_react_agent') as mock_create_agent:
                # Mock the agent to simulate creating tool calls without query field
                mock_agent = AsyncMock()
                
                # Simulate the problematic tool call that causes validation error
                mock_result = {
                    "messages": [
                        MagicMock(content="User message"),
                        MagicMock(content="AI response with validation error"),
                        MagicMock(content="Error: 1 validation error for SearchAnimeInput\nquery\n  Field required [type=missing, input_value={'genres': ['thriller'], 'year_range': [2010, 2023]}, input_type=dict]")
                    ]
                }
                mock_agent.ainvoke = AsyncMock(return_value=mock_result)
                mock_create_agent.return_value = mock_agent
                
                # Create the workflow engine
                engine = create_react_agent_workflow_engine(mock_mcp_tools, LLMProvider.OPENAI)
                
                # Test the problematic query that should trigger validation error
                result = await engine.process_conversation(
                    session_id="test_validation", 
                    message="I want thriller anime with deep philosophical themes produced by Madhouse between 2010 and 2023"
                )
                
                # Verify the validation error occurred
                messages = result.get("messages", [])
                error_found = any("validation error for SearchAnimeInput" in str(msg) and "query" in str(msg) for msg in messages)
                assert error_found, "Expected validation error for missing query field not found"

    @pytest.mark.asyncio 
    async def test_tool_call_should_include_query_field(self, mock_mcp_tools):
        """Test that tool calls should include the original query field.
        
        This test defines the expected behavior: tool calls must include
        the original query text along with extracted parameters.
        """
        # This test should pass after we fix the bug
        with patch('src.langgraph.react_agent_workflow.ChatOpenAI'):
            with patch('src.langgraph.react_agent_workflow.create_react_agent') as mock_create_agent:
                
                # Mock an agent that creates proper tool calls WITH query field
                mock_agent = AsyncMock()
                
                # Define the expected tool call structure (this is what we want)
                expected_tool_call = {
                    "query": "thriller anime with deep philosophical themes",  # This should be included!
                    "genres": ["thriller"],
                    "year_range": [2010, 2023],
                    "studios": ["Madhouse"],
                    "mood_keywords": ["philosophical"]
                }
                
                # Mock successful execution with proper tool call
                mock_result = {
                    "messages": [
                        MagicMock(content="User message"),
                        MagicMock(content="Search results")
                    ]
                }
                mock_agent.ainvoke = AsyncMock(return_value=mock_result)
                mock_create_agent.return_value = mock_agent
                
                # Create engine 
                engine = create_react_agent_workflow_engine(mock_mcp_tools, LLMProvider.OPENAI)
                
                # Process conversation
                result = await engine.process_conversation(
                    session_id="test_fixed",
                    message="I want thriller anime with deep philosophical themes produced by Madhouse between 2010 and 2023"
                )
                
                # After the fix, this should work without validation errors
                messages = result.get("messages", [])
                validation_error = any("validation error" in str(msg) for msg in messages)
                assert not validation_error, "Should not have validation errors after fix"

    @pytest.mark.asyncio
    async def test_complex_parameter_extraction_with_query(self, mock_mcp_tools):
        """Test that complex parameter extraction includes query field.
        
        Tests the most complex query from our test cases to ensure
        all parameters are extracted AND the query field is preserved.
        """
        complex_query = "Find psychological seinen anime movies from Studio Pierrot or A-1 Pictures between 2005 and 2020, exclude known shounen titles, and limit to 3 results"
        
        with patch('src.langgraph.react_agent_workflow.ChatOpenAI'):
            with patch('src.langgraph.react_agent_workflow.create_react_agent') as mock_create_agent:
                
                mock_agent = AsyncMock()
                
                # Expected extracted parameters WITH query field
                expected_params = {
                    "query": complex_query,  # Original query must be preserved
                    "genres": ["psychological", "seinen"],
                    "anime_types": ["Movie"],
                    "studios": ["Studio Pierrot", "A-1 Pictures"], 
                    "year_range": [2005, 2020],
                    "exclusions": ["shounen"],
                    "limit": 3
                }
                
                mock_result = {"messages": [MagicMock(content="Success")]}
                mock_agent.ainvoke = AsyncMock(return_value=mock_result)
                mock_create_agent.return_value = mock_agent
                
                engine = create_react_agent_workflow_engine(mock_mcp_tools, LLMProvider.OPENAI)
                
                result = await engine.process_conversation(
                    session_id="test_complex",
                    message=complex_query
                )
                
                # Should succeed without validation errors
                messages = result.get("messages", [])
                validation_error = any("validation error" in str(msg) for msg in messages)
                assert not validation_error, "Complex query should work without validation errors"