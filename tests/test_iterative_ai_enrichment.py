#!/usr/bin/env python3
"""
Tests for iterative AI enrichment agent
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from services.iterative_ai_enrichment import (
    IterativeAIEnrichmentAgent,
    enrich_anime_for_vector_indexing,
    AI_CLIENTS
)


@pytest.fixture
def sample_anime_data():
    """Sample anime data for testing"""
    return {
        "sources": [
            "https://anidb.net/anime/18290",
            "https://anilist.co/anime/171018",
            "https://myanimelist.net/anime/57334"
        ],
        "title": "Test Anime",
        "type": "TV",
        "episodes": 12,
        "status": "FINISHED",
        "animeSeason": {"season": "FALL", "year": 2024},
        "tags": ["action", "comedy"]
    }


class TestIterativeAIEnrichmentAgent:
    """Test cases for IterativeAIEnrichmentAgent"""
    
    def test_ai_clients_config(self):
        """Test that AI_CLIENTS config is properly structured"""
        assert "openai" in AI_CLIENTS
        assert "anthropic" in AI_CLIENTS
        
        for provider, config in AI_CLIENTS.items():
            assert "module" in config
            assert "class" in config
            assert "key" in config
            assert "default_model" in config
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_no_api_keys(self):
        """Test initialization with no API keys"""
        agent = IterativeAIEnrichmentAgent()
        assert agent.ai_provider is None
        assert agent.ai_client is None
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('services.iterative_ai_enrichment.IterativeAIEnrichmentAgent._has_module')
    def test_detect_openai_provider(self, mock_has_module):
        """Test auto-detection of OpenAI provider"""
        mock_has_module.return_value = True
        
        agent = IterativeAIEnrichmentAgent()
        assert agent.ai_provider == "openai"
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @patch('services.iterative_ai_enrichment.IterativeAIEnrichmentAgent._has_module')
    def test_detect_anthropic_provider(self, mock_has_module):
        """Test auto-detection of Anthropic provider"""
        mock_has_module.side_effect = lambda module: module == "anthropic"
        
        agent = IterativeAIEnrichmentAgent()
        assert agent.ai_provider == "anthropic"
    
    def test_has_module_existing(self):
        """Test _has_module with existing module"""
        agent = IterativeAIEnrichmentAgent()
        assert agent._has_module("os") is True
    
    def test_has_module_non_existing(self):
        """Test _has_module with non-existing module"""
        agent = IterativeAIEnrichmentAgent()
        assert agent._has_module("non_existing_module_xyz") is False
    
    def test_create_client_unknown_provider(self):
        """Test _create_client with unknown provider"""
        agent = IterativeAIEnrichmentAgent()
        
        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            agent._create_client("unknown")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_create_client_missing_api_key(self):
        """Test _create_client with missing API key"""
        agent = IterativeAIEnrichmentAgent()
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            agent._create_client("openai")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('builtins.__import__')
    def test_create_client_import_error(self, mock_import):
        """Test _create_client with import error"""
        mock_import.side_effect = ImportError("Module not found")
        
        agent = IterativeAIEnrichmentAgent()
        
        with pytest.raises(ImportError, match="openai package not installed"):
            agent._create_client("openai")
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('builtins.__import__')
    def test_create_client_attribute_error(self, mock_import):
        """Test _create_client with attribute error"""
        mock_module = MagicMock()
        # Remove the AsyncOpenAI attribute to cause AttributeError
        del mock_module.AsyncOpenAI
        mock_import.return_value = mock_module
        
        # Create agent without auto-initialization
        agent = IterativeAIEnrichmentAgent()
        
        # Manually set environment for this test
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(AttributeError, match="Class AsyncOpenAI not found"):
                agent._create_client("openai")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('builtins.__import__')
    @patch('services.iterative_ai_enrichment.IterativeAIEnrichmentAgent._has_module')
    def test_create_client_success(self, mock_has_module, mock_import):
        """Test successful client creation"""
        mock_has_module.return_value = False  # Prevent auto-detection
        mock_client = MagicMock()
        mock_class = MagicMock(return_value=mock_client)
        mock_module = MagicMock()
        mock_module.AsyncOpenAI = mock_class
        mock_import.return_value = mock_module
        
        agent = IterativeAIEnrichmentAgent()
        result = agent._create_client("openai")
        
        assert result == mock_client
        mock_class.assert_called_once_with(api_key="test-key")
    
    @pytest.mark.asyncio
    async def test_enrich_anime_from_offline_data(self, sample_anime_data):
        """Test basic enrichment (passthrough)"""
        agent = IterativeAIEnrichmentAgent()
        
        result = await agent.enrich_anime_from_offline_data(sample_anime_data)
        
        # Should preserve all original data
        assert result == sample_anime_data
        assert len(result) == len(sample_anime_data)
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_call_ai_no_client(self):
        """Test _call_ai with no client configured"""
        agent = IterativeAIEnrichmentAgent()
        
        with pytest.raises(ValueError, match="No AI client configured"):
            await agent._call_ai("test prompt")
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    async def test_call_ai_openai(self):
        """Test _call_ai with OpenAI"""
        # Mock the client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "AI response"
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = IterativeAIEnrichmentAgent()
        agent.ai_provider = "openai"
        agent.ai_client = mock_client
        
        result = await agent._call_ai("test prompt")
        
        assert result == "AI response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0.1
        )
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    async def test_call_ai_anthropic(self):
        """Test _call_ai with Anthropic"""
        # Mock the client
        mock_response = MagicMock()
        mock_response.content[0].text = "AI response"
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_response
        
        agent = IterativeAIEnrichmentAgent()
        agent.ai_provider = "anthropic"
        agent.ai_client = mock_client
        
        result = await agent._call_ai("test prompt")
        
        assert result == "AI response"
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-haiku-20240307",
            max_tokens=4000,
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0.1
        )
    
    @pytest.mark.asyncio
    async def test_call_ai_custom_model(self):
        """Test _call_ai with custom model"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "AI response"
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = IterativeAIEnrichmentAgent()
        agent.ai_provider = "openai"
        agent.ai_client = mock_client
        
        await agent._call_ai("test prompt", model="gpt-4")
        
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0.1
        )
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_call_ai_unknown_provider(self):
        """Test _call_ai with unknown provider"""
        agent = IterativeAIEnrichmentAgent()
        agent.ai_provider = "unknown"
        agent.ai_client = MagicMock()
        
        with pytest.raises(KeyError, match="unknown"):
            await agent._call_ai("test prompt")


class TestConvenienceFunction:
    """Test cases for convenience function"""
    
    @pytest.mark.asyncio
    async def test_enrich_anime_for_vector_indexing(self, sample_anime_data):
        """Test convenience function"""
        result = await enrich_anime_for_vector_indexing(sample_anime_data)
        
        # Should preserve all original data
        assert result == sample_anime_data
    
    @pytest.mark.asyncio
    @patch('services.iterative_ai_enrichment.IterativeAIEnrichmentAgent')
    async def test_enrich_anime_for_vector_indexing_with_provider(self, mock_agent_class, sample_anime_data):
        """Test convenience function with specific provider"""
        mock_agent = AsyncMock()
        mock_agent.enrich_anime_from_offline_data.return_value = sample_anime_data
        mock_agent_class.return_value = mock_agent
        
        result = await enrich_anime_for_vector_indexing(sample_anime_data, ai_provider="openai")
        
        assert result == sample_anime_data
        mock_agent_class.assert_called_once_with(ai_provider="openai")
        mock_agent.enrich_anime_from_offline_data.assert_called_once_with(sample_anime_data)


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
    async def test_real_openai_integration(self, sample_anime_data):
        """Test with real OpenAI API (if key available)"""
        agent = IterativeAIEnrichmentAgent(ai_provider="openai")
        
        assert agent.ai_provider == "openai"
        assert agent.ai_client is not None
        
        # Test basic enrichment (should still passthrough)
        result = await agent.enrich_anime_from_offline_data(sample_anime_data)
        assert result == sample_anime_data
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available")
    async def test_real_anthropic_integration(self, sample_anime_data):
        """Test with real Anthropic API (if key available)"""
        agent = IterativeAIEnrichmentAgent(ai_provider="anthropic")
        
        assert agent.ai_provider == "anthropic"
        assert agent.ai_client is not None
        
        # Test basic enrichment (should still passthrough)
        result = await agent.enrich_anime_from_offline_data(sample_anime_data)
        assert result == sample_anime_data


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])