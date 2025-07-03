"""Tests for BaseExternalService abstract class."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.services.external.base_service import BaseExternalService


class ConcreteTestService(BaseExternalService):
    """Concrete implementation of BaseExternalService for testing."""
    
    def __init__(self, service_name: str = "test_service"):
        super().__init__(service_name)
        # Add a mock client to test client cleanup
        self.client = AsyncMock()
        self.client.close = AsyncMock()
    
    async def search_anime(self, query: str, limit: int = 10, **kwargs):
        """Concrete implementation for testing."""
        return [{"id": 1, "title": "Test Anime", "query": query, "limit": limit}]
    
    async def get_anime_details(self, anime_id):
        """Concrete implementation for testing."""
        return {"id": anime_id, "title": "Test Anime Details"}
    
    async def health_check(self):
        """Concrete implementation for testing."""
        return {"status": "healthy", "service": self.service_name}


class TestBaseExternalService:
    """Test cases for BaseExternalService base class."""
    
    def test_service_initialization(self):
        """Test BaseExternalService initialization."""
        service = ConcreteTestService("test_service")
        
        assert service.service_name == "test_service"
        assert service.cache_manager is not None
        assert service.circuit_breaker is not None
    
    def test_is_healthy(self):
        """Test is_healthy method - covers line 84."""
        service = ConcreteTestService("test_service")
        
        # Initially circuit breaker should be closed (service is healthy)
        assert service.is_healthy() == True
        
        # Mock circuit breaker as open (service is unhealthy)
        service.circuit_breaker.is_open = Mock(return_value=True)
        assert service.is_healthy() == False
        
        # Mock circuit breaker as closed (service is healthy)
        service.circuit_breaker.is_open = Mock(return_value=False)
        assert service.is_healthy() == True
    
    def test_get_service_info(self):
        """Test get_service_info method - covers lines 92-96."""
        service = ConcreteTestService("test_service")
        
        # Mock circuit breaker state
        service.circuit_breaker.is_open = Mock(return_value=False)
        
        info = service.get_service_info()
        
        assert isinstance(info, dict)
        assert info["name"] == "test_service"
        assert info["healthy"] == True
        assert info["circuit_breaker_open"] == False
        
        # Test with circuit breaker open
        service.circuit_breaker.is_open = Mock(return_value=True)
        info = service.get_service_info()
        
        assert info["name"] == "test_service"
        assert info["healthy"] == False
        assert info["circuit_breaker_open"] == True
    
    @pytest.mark.asyncio
    async def test_close_with_client(self):
        """Test close method with client cleanup - covers lines 66-68, 73."""
        service = ConcreteTestService("test_service")
        
        # Test that client.close() is called
        await service.close()
        
        service.client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_with_cache_manager(self):
        """Test close method with cache manager cleanup - covers lines 70-71, 73."""
        service = ConcreteTestService("test_service")
        
        # Mock cache manager with close method
        service.cache_manager.close = AsyncMock()
        
        await service.close()
        
        service.cache_manager.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_without_client(self):
        """Test close method without client - covers lines 66-67, 73."""
        service = ConcreteTestService("test_service")
        
        # Remove client to test hasattr check
        delattr(service, 'client')
        
        # Should not raise error
        await service.close()
    
    @pytest.mark.asyncio
    async def test_close_without_cache_manager_close(self):
        """Test close method when cache manager doesn't have close method - covers lines 70, 73."""
        service = ConcreteTestService("test_service")
        
        # Ensure cache manager doesn't have close method
        if hasattr(service.cache_manager, 'close'):
            delattr(service.cache_manager, 'close')
        
        # Should not raise error
        await service.close()
    
    @pytest.mark.asyncio
    async def test_close_with_exception(self):
        """Test close method exception handling - covers lines 75-76."""
        service = ConcreteTestService("test_service")
        
        # Mock client.close to raise exception
        service.client.close = AsyncMock(side_effect=Exception("Close error"))
        
        # Should not raise exception, but should log warning
        with patch("src.services.external.base_service.logger") as mock_logger:
            await service.close()
            mock_logger.warning.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_abstract_methods(self):
        """Test that abstract methods work in concrete implementation."""
        service = ConcreteTestService("test_service")
        
        # Test search_anime
        results = await service.search_anime("test query", limit=5)
        assert len(results) == 1
        assert results[0]["query"] == "test query"
        assert results[0]["limit"] == 5
        
        # Test get_anime_details
        details = await service.get_anime_details(123)
        assert details["id"] == 123
        assert "title" in details
        
        # Test health_check
        health = await service.health_check()
        assert health["status"] == "healthy"
        assert health["service"] == "test_service"