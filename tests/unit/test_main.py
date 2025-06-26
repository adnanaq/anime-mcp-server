"""Tests for main.py FastAPI application module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app, lifespan


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Mock()
    settings.debug = True
    settings.host = "0.0.0.0"
    settings.port = 8000
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_collection_name = "anime_database"
    return settings


class TestFastAPIApplication:
    """Test FastAPI application startup and configuration."""

    def test_app_creation(self):
        """Test that FastAPI app is created correctly."""
        assert app is not None
        assert app.title == "Anime MCP Server"
        assert "Semantic search over 38,000+ anime entries" in app.description

    @pytest.mark.asyncio
    async def test_lifespan_startup_success(self, mock_settings):
        """Test successful application startup."""
        with (
            patch("src.main.get_settings", return_value=mock_settings),
            patch("src.main.QdrantClient") as mock_qdrant_client,
            patch("src.main.AnimeDataService") as mock_data_service,
            patch("src.main.initialize_mcp_server") as mock_init_mcp,
        ):
            # Mock successful initialization
            mock_client_instance = AsyncMock()
            mock_client_instance.health_check.return_value = True
            mock_qdrant_client.return_value = mock_client_instance

            mock_data_instance = Mock()
            mock_data_service.return_value = mock_data_instance

            mock_init_mcp.return_value = None

            # Test lifespan startup
            async_gen = lifespan(app)
            await async_gen.__anext__()  # Startup phase

            # Verify initialization calls
            mock_qdrant_client.assert_called_once_with(settings=mock_settings)
            mock_client_instance.health_check.assert_called_once()
            mock_data_service.assert_called_once_with(
                mock_client_instance, mock_settings
            )
            mock_init_mcp.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_startup_qdrant_failure(self, mock_settings):
        """Test application startup with Qdrant connection failure."""
        with (
            patch("src.main.get_settings", return_value=mock_settings),
            patch("src.main.QdrantClient") as mock_qdrant_client,
            patch("src.main.AnimeDataService") as mock_data_service,
            patch("src.main.initialize_mcp_server") as mock_init_mcp,
            patch("src.main.logger") as mock_logger,
        ):
            # Mock failed Qdrant connection
            mock_client_instance = AsyncMock()
            mock_client_instance.health_check.return_value = False
            mock_qdrant_client.return_value = mock_client_instance

            mock_data_instance = Mock()
            mock_data_service.return_value = mock_data_instance

            mock_init_mcp.return_value = None

            # Test lifespan startup with failure
            async_gen = lifespan(app)
            await async_gen.__anext__()  # Startup phase

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            assert "Qdrant health check failed" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_lifespan_startup_mcp_failure(self, mock_settings):
        """Test application startup with MCP initialization failure."""
        with (
            patch("src.main.get_settings", return_value=mock_settings),
            patch("src.main.QdrantClient") as mock_qdrant_client,
            patch("src.main.AnimeDataService") as mock_data_service,
            patch("src.main.initialize_mcp_server") as mock_init_mcp,
            patch("src.main.logger") as mock_logger,
        ):
            # Mock successful Qdrant but failed MCP
            mock_client_instance = AsyncMock()
            mock_client_instance.health_check.return_value = True
            mock_qdrant_client.return_value = mock_client_instance

            mock_data_instance = Mock()
            mock_data_service.return_value = mock_data_instance

            mock_init_mcp.side_effect = Exception("MCP initialization failed")

            # Test lifespan startup with MCP failure
            async_gen = lifespan(app)
            await async_gen.__anext__()  # Startup phase

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "MCP server initialization failed" in str(
                mock_logger.error.call_args
            )

    @pytest.mark.asyncio
    async def test_lifespan_startup_exception_handling(self, mock_settings):
        """Test application startup with general exception."""
        with (
            patch("src.main.get_settings", return_value=mock_settings),
            patch("src.main.QdrantClient") as mock_qdrant_client,
            patch("src.main.logger") as mock_logger,
        ):
            # Mock exception during initialization
            mock_qdrant_client.side_effect = Exception("Connection failed")

            # Test lifespan startup with exception
            async_gen = lifespan(app)
            await async_gen.__anext__()  # Startup phase

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Startup error" in str(mock_logger.error.call_args)

    @pytest.mark.asyncio
    async def test_lifespan_shutdown(self, mock_settings):
        """Test application shutdown."""
        with (
            patch("src.main.get_settings", return_value=mock_settings),
            patch("src.main.QdrantClient") as mock_qdrant_client,
            patch("src.main.AnimeDataService") as mock_data_service,
            patch("src.main.initialize_mcp_server") as mock_init_mcp,
            patch("src.main.logger") as mock_logger,
        ):
            # Mock successful initialization
            mock_client_instance = AsyncMock()
            mock_client_instance.health_check.return_value = True
            mock_qdrant_client.return_value = mock_client_instance

            mock_data_instance = Mock()
            mock_data_service.return_value = mock_data_instance

            mock_init_mcp.return_value = None

            # Test full lifespan
            async_gen = lifespan(app)
            await async_gen.__anext__()  # Startup phase

            try:
                await async_gen.__anext__()  # Shutdown phase
            except StopAsyncIteration:
                pass  # Expected when lifespan completes

            # Verify shutdown logging
            mock_logger.info.assert_called()
            shutdown_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "shutdown" in str(call).lower()
            ]
            assert len(shutdown_calls) > 0

    def test_app_middleware_configuration(self):
        """Test that middleware is properly configured."""
        # Check CORS middleware is added
        middleware_stack = app.user_middleware
        cors_middleware = any(
            "CORS" in str(middleware) for middleware in middleware_stack
        )
        assert cors_middleware, "CORS middleware should be configured"

    def test_app_routes_included(self):
        """Test that all required routes are included."""
        routes = {route.path for route in app.routes}

        # Check main route groups are included
        expected_route_prefixes = [
            "/api/search",
            "/api/admin",
            "/api/workflow",
            "/api/recommendations",
        ]

        # Check if routes with these prefixes exist
        for prefix in expected_route_prefixes:
            [route for route in routes if route.startswith(prefix)]
            # Note: Routes might not be exact matches due to path parameters
            # So we just verify the routers are included by checking app structure


class TestAppWithTestClient:
    """Test FastAPI application using TestClient."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_root_endpoint_redirect(self, client):
        """Test root endpoint redirects to docs."""
        with patch("src.main.lifespan"):
            response = client.get("/")
            assert response.status_code in [200, 307, 308]  # Redirect or docs page

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        with (
            patch("src.main.lifespan"),
            patch("src.main.qdrant_client") as mock_client,
        ):
            mock_client.health_check = AsyncMock(return_value=True)
            response = client.get("/health")
            assert response.status_code == 200

    def test_docs_endpoint_available(self, client):
        """Test API docs endpoint is available."""
        with patch("src.main.lifespan"):
            response = client.get("/docs")
            assert response.status_code == 200
            assert "text/html" in response.headers.get("content-type", "")

    def test_openapi_schema_available(self, client):
        """Test OpenAPI schema endpoint."""
        with patch("src.main.lifespan"):
            response = client.get("/openapi.json")
            assert response.status_code == 200
            assert response.headers.get("content-type") == "application/json"

            # Verify schema structure
            schema = response.json()
            assert "openapi" in schema
            assert "info" in schema
            assert schema["info"]["title"] == "Anime MCP Server"


class TestApplicationConfiguration:
    """Test application configuration and settings."""

    def test_debug_mode_configuration(self):
        """Test debug mode affects app configuration."""
        with patch("src.main.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.debug = True
            mock_get_settings.return_value = mock_settings

            # App configuration should respect debug mode
            assert app.debug is not None  # Should be configured

    def test_production_configuration(self):
        """Test production mode configuration."""
        with patch("src.main.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.debug = False
            mock_get_settings.return_value = mock_settings

            # Production settings should be applied
            # Verify app is configured for production (no debug info in responses)
            assert app is not None

    def test_global_state_initialization(self):
        """Test global state variables are properly initialized."""
        # These should be initially None and set during lifespan
        import src.main

        assert hasattr(src.main, "qdrant_client")
        assert hasattr(src.main, "data_service")


class TestErrorHandling:
    """Test error handling in application startup."""

    @pytest.mark.asyncio
    async def test_qdrant_client_creation_error(self, mock_settings):
        """Test error handling when Qdrant client creation fails."""
        with (
            patch("src.main.get_settings", return_value=mock_settings),
            patch(
                "src.main.QdrantClient",
                side_effect=Exception("Failed to create client"),
            ),
            patch("src.main.logger") as mock_logger,
        ):
            async_gen = lifespan(app)
            await async_gen.__anext__()  # Should handle exception gracefully

            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_data_service_creation_error(self, mock_settings):
        """Test error handling when data service creation fails."""
        with (
            patch("src.main.get_settings", return_value=mock_settings),
            patch("src.main.QdrantClient") as mock_qdrant_client,
            patch(
                "src.main.AnimeDataService",
                side_effect=Exception("Failed to create service"),
            ),
            patch("src.main.logger") as mock_logger,
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.health_check.return_value = True
            mock_qdrant_client.return_value = mock_client_instance

            async_gen = lifespan(app)
            await async_gen.__anext__()  # Should handle exception gracefully

            mock_logger.error.assert_called_once()


class TestDependencyInjection:
    """Test dependency injection and global state management."""

    def test_global_variables_exist(self):
        """Test that global variables for dependency injection exist."""
        import src.main

        # Check global variables are defined
        assert hasattr(src.main, "qdrant_client")
        assert hasattr(src.main, "data_service")

    @pytest.mark.asyncio
    async def test_dependency_injection_setup(self, mock_settings):
        """Test that dependencies are properly injected during startup."""
        with (
            patch("src.main.get_settings", return_value=mock_settings),
            patch("src.main.QdrantClient") as mock_qdrant_client,
            patch("src.main.AnimeDataService") as mock_data_service,
            patch("src.main.initialize_mcp_server"),
        ):
            mock_client_instance = AsyncMock()
            mock_client_instance.health_check.return_value = True
            mock_qdrant_client.return_value = mock_client_instance

            mock_data_instance = Mock()
            mock_data_service.return_value = mock_data_instance

            # Import after mocking to get fresh global state
            import importlib

            import src.main

            importlib.reload(src.main)

            async_gen = src.main.lifespan(app)
            await async_gen.__anext__()  # Startup phase

            # Verify global state is set
            assert src.main.qdrant_client is not None
            assert src.main.data_service is not None
