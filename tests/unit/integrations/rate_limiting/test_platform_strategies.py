"""Comprehensive tests for platform-specific rate limiting strategies."""

import asyncio
import random
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.integrations.rate_limiting.platform_strategies import (
    AniListRateLimitAdapter,
    AniListRateLimitStrategy,
    GenericRateLimitAdapter,
    GenericRateLimitStrategy,
    JikanRateLimitAdapter,
    JikanRateLimitStrategy,
    MALRateLimitAdapter,
    MALRateLimitStrategy,
    PlatformRateLimitAdapter,
    RateLimitInfo,
    RateLimitStrategy,
)


class TestRateLimitInfo:
    """Test RateLimitInfo dataclass."""

    def test_rate_limit_info_defaults(self):
        """Test RateLimitInfo with default values."""
        info = RateLimitInfo()
        
        assert info.remaining is None
        assert info.limit is None
        assert info.reset_time is None
        assert info.retry_after is None
        assert info.degraded is False
        assert info.custom_data == {}

    def test_rate_limit_info_with_values(self):
        """Test RateLimitInfo with specific values."""
        custom_data = {"platform": "test", "extra": "data"}
        info = RateLimitInfo(
            remaining=50,
            limit=100,
            reset_time=1234567890,
            retry_after=60,
            degraded=True,
            custom_data=custom_data
        )
        
        assert info.remaining == 50
        assert info.limit == 100
        assert info.reset_time == 1234567890
        assert info.retry_after == 60
        assert info.degraded is True
        assert info.custom_data == custom_data

    def test_rate_limit_info_post_init(self):
        """Test RateLimitInfo __post_init__ method."""
        info = RateLimitInfo(custom_data=None)
        assert info.custom_data == {}


class TestAniListRateLimitStrategy:
    """Test AniList-specific rate limiting strategy."""

    @pytest.fixture
    def strategy(self):
        """Create AniList strategy for testing."""
        return AniListRateLimitStrategy()

    @pytest.fixture
    def rate_info(self):
        """Create rate limit info for testing."""
        return RateLimitInfo(
            remaining=10,
            limit=90,
            reset_time=int(time.time()) + 60,
            retry_after=30
        )

    @pytest.mark.asyncio
    async def test_handle_rate_limit_response_with_retry_after(self, strategy, rate_info):
        """Test handling rate limit response with retry_after."""
        with patch('asyncio.sleep') as mock_sleep, \
             patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.2):
            
            await strategy.handle_rate_limit_response(rate_info)
            
            # Should sleep for retry_after + jitter (0.2 * 30 = 6.0 jitter)
            mock_sleep.assert_called_once()
            call_args = mock_sleep.call_args[0][0]
            assert call_args == 36.0  # 30 + 6.0 jitter

    @pytest.mark.asyncio
    async def test_handle_rate_limit_response_without_retry_after(self, strategy):
        """Test handling rate limit response without retry_after."""
        rate_info = RateLimitInfo(remaining=10, limit=90)
        
        with patch('asyncio.sleep') as mock_sleep, \
             patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.25):
            
            await strategy.handle_rate_limit_response(rate_info)
            
            # Should use default 60 seconds + jitter (0.25 * 60 = 15.0 jitter)
            mock_sleep.assert_called_once()
            call_args = mock_sleep.call_args[0][0]
            assert call_args == 75.0  # 60 + 15.0 jitter

    @pytest.mark.asyncio
    async def test_calculate_backoff_delay(self, strategy, rate_info):
        """Test backoff delay calculation."""
        with patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.35):
            # Test attempt 0
            delay = await strategy.calculate_backoff_delay(rate_info, attempt=0)
            assert delay == 1.35  # 1 + (0.35 * 1) jitter
            
            # Test attempt 3
            delay = await strategy.calculate_backoff_delay(rate_info, attempt=3)
            assert delay == 10.8  # 8 + (0.35 * 8) jitter
            
            # Test attempt that would exceed cap
            delay = await strategy.calculate_backoff_delay(rate_info, attempt=10)
            assert delay == 81.0  # 60 (capped) + (0.35 * 60) jitter

    def test_should_proactive_throttle_true(self, strategy):
        """Test proactive throttling when usage is high."""
        rate_info = RateLimitInfo(remaining=10, limit=100)  # 90% used
        result = strategy.should_proactive_throttle(rate_info)
        assert result is True

    def test_should_proactive_throttle_false(self, strategy):
        """Test no proactive throttling when usage is low."""
        rate_info = RateLimitInfo(remaining=50, limit=100)  # 50% used
        result = strategy.should_proactive_throttle(rate_info)
        assert result is False

    def test_should_proactive_throttle_no_data(self, strategy):
        """Test proactive throttling with missing data."""
        rate_info = RateLimitInfo(remaining=None, limit=100)
        result = strategy.should_proactive_throttle(rate_info)
        assert result is False
        
        rate_info = RateLimitInfo(remaining=50, limit=None)
        result = strategy.should_proactive_throttle(rate_info)
        assert result is False

    def test_initialization_with_service_name(self):
        """Test strategy initialization with custom service name."""
        strategy = AniListRateLimitStrategy("custom_anilist")
        assert strategy.service_name == "custom_anilist"


class TestAniListRateLimitAdapter:
    """Test AniList-specific rate limit adapter."""

    @pytest.fixture
    def adapter(self):
        """Create AniList adapter for testing."""
        return AniListRateLimitAdapter()

    @pytest.fixture
    def mock_response(self):
        """Create mock aiohttp response."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {
            "X-RateLimit-Remaining": "45",
            "X-RateLimit-Limit": "90",
            "X-RateLimit-Reset": "1234567890",
            "Retry-After": "60"
        }
        response.status = 200
        return response

    def test_extract_rate_limit_info_full_headers(self, adapter, mock_response):
        """Test extracting rate limit info with all headers present."""
        result = adapter.extract_rate_limit_info(mock_response)
        
        assert result.remaining == 45
        assert result.limit == 90
        assert result.reset_time == 1234567890
        assert result.retry_after == 60
        assert result.degraded is False  # 90 is standard limit
        assert result.custom_data["platform"] == "anilist"
        assert result.custom_data["status_code"] == 200

    def test_extract_rate_limit_info_degraded_mode(self, adapter):
        """Test extracting rate limit info in degraded mode."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {
            "X-RateLimit-Remaining": "15",
            "X-RateLimit-Limit": "30"  # Degraded limit
        }
        response.status = 200
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining == 15
        assert result.limit == 30
        assert result.degraded is True  # Less than 90 indicates degraded

    def test_extract_rate_limit_info_missing_headers(self, adapter):
        """Test extracting rate limit info with missing headers."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {}
        response.status = 200
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining is None
        assert result.limit is None
        assert result.reset_time is None
        assert result.retry_after is None
        assert result.degraded is False

    def test_extract_rate_limit_info_invalid_values(self, adapter):
        """Test extracting rate limit info with invalid header values."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {
            "X-RateLimit-Remaining": "invalid",
            "X-RateLimit-Limit": "not_a_number",
            "X-RateLimit-Reset": "bad_timestamp",
            "Retry-After": "not_int"
        }
        response.status = 200
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining is None
        assert result.limit is None
        assert result.reset_time is None
        assert result.retry_after is None

    def test_get_strategy(self, adapter):
        """Test getting the AniList strategy."""
        strategy = adapter.get_strategy()
        assert isinstance(strategy, AniListRateLimitStrategy)


class TestGenericRateLimitStrategy:
    """Test generic rate limiting strategy."""

    @pytest.fixture
    def strategy(self):
        """Create generic strategy for testing."""
        return GenericRateLimitStrategy("test_service")

    @pytest.fixture
    def rate_info(self):
        """Create rate limit info for testing."""
        return RateLimitInfo(
            remaining=20,
            limit=100,
            retry_after=45
        )

    @pytest.mark.asyncio
    async def test_handle_rate_limit_response_with_retry_after(self, strategy, rate_info):
        """Test handling rate limit response with retry_after."""
        with patch('asyncio.sleep') as mock_sleep, \
             patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.15):
            
            await strategy.handle_rate_limit_response(rate_info)
            
            mock_sleep.assert_called_once()
            call_args = mock_sleep.call_args[0][0]
            assert call_args == 51.75  # 45 + (0.15 * 45) jitter

    @pytest.mark.asyncio
    async def test_handle_rate_limit_response_without_retry_after(self, strategy):
        """Test handling rate limit response without retry_after."""
        rate_info = RateLimitInfo(remaining=20, limit=100)
        
        with patch('asyncio.sleep') as mock_sleep, \
             patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.15):
            
            await strategy.handle_rate_limit_response(rate_info)
            
            mock_sleep.assert_called_once()
            call_args = mock_sleep.call_args[0][0]
            assert call_args == 69.0  # 60 + (0.15 * 60) jitter

    @pytest.mark.asyncio
    async def test_calculate_backoff_delay(self, strategy, rate_info):
        """Test generic backoff delay calculation."""
        with patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.2):
            # Test attempt 0
            delay = await strategy.calculate_backoff_delay(rate_info, attempt=0)
            assert delay == 1.2  # 1 + (0.2 * 1) jitter
            
            # Test attempt 2
            delay = await strategy.calculate_backoff_delay(rate_info, attempt=2)
            assert delay == 4.8  # 4 + (0.2 * 4) jitter

    def test_should_proactive_throttle_true(self, strategy):
        """Test proactive throttling at 90% threshold."""
        rate_info = RateLimitInfo(remaining=5, limit=100)  # 95% used
        result = strategy.should_proactive_throttle(rate_info)
        assert result is True

    def test_should_proactive_throttle_false(self, strategy):
        """Test no proactive throttling below threshold."""
        rate_info = RateLimitInfo(remaining=50, limit=100)  # 50% used
        result = strategy.should_proactive_throttle(rate_info)
        assert result is False

    def test_should_proactive_throttle_missing_data(self, strategy):
        """Test proactive throttling with missing data."""
        rate_info = RateLimitInfo(remaining=None, limit=100)
        result = strategy.should_proactive_throttle(rate_info)
        assert result is False


class TestMALRateLimitStrategy:
    """Test MAL-specific rate limiting strategy."""

    @pytest.fixture
    def strategy(self):
        """Create MAL strategy for testing."""
        return MALRateLimitStrategy()

    @pytest.fixture
    def rate_info_403(self):
        """Create rate limit info for 403 DoS protection."""
        return RateLimitInfo(
            custom_data={"status_code": 403, "platform": "mal"}
        )

    @pytest.fixture
    def rate_info_429(self):
        """Create rate limit info for 429 rate limit."""
        return RateLimitInfo(
            retry_after=25,
            custom_data={"status_code": 429, "platform": "mal"}
        )

    @pytest.mark.asyncio
    async def test_handle_rate_limit_response_dos_protection(self, strategy, rate_info_403):
        """Test handling 403 DoS protection response."""
        with patch('asyncio.sleep') as mock_sleep, \
             patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.2):
            
            await strategy.handle_rate_limit_response(rate_info_403)
            
            mock_sleep.assert_called_once()
            call_args = mock_sleep.call_args[0][0]
            assert call_args == 360.0  # 300 + (0.2 * 300) jitter

    @pytest.mark.asyncio
    async def test_handle_rate_limit_response_with_retry_after(self, strategy, rate_info_429):
        """Test handling rate limit response with retry_after."""
        with patch('asyncio.sleep') as mock_sleep, \
             patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.2):
            
            await strategy.handle_rate_limit_response(rate_info_429)
            
            mock_sleep.assert_called_once()
            call_args = mock_sleep.call_args[0][0]
            assert call_args == 30.0  # 25 + (0.2 * 25) jitter

    @pytest.mark.asyncio
    async def test_handle_rate_limit_response_default(self, strategy):
        """Test handling rate limit response with defaults."""
        rate_info = RateLimitInfo(custom_data={"status_code": 429})
        
        with patch('asyncio.sleep') as mock_sleep, \
             patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.2):
            
            await strategy.handle_rate_limit_response(rate_info)
            
            mock_sleep.assert_called_once()
            call_args = mock_sleep.call_args[0][0]
            assert call_args == 36.0  # 30 + (0.2 * 30) jitter

    @pytest.mark.asyncio
    async def test_calculate_backoff_delay(self, strategy):
        """Test MAL backoff delay calculation."""
        rate_info = RateLimitInfo()
        
        with patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.25):
            # Test attempt 0
            delay = await strategy.calculate_backoff_delay(rate_info, attempt=0)
            assert delay == 1.25  # 1 + (0.25 * 1) jitter
            
            # Test attempt 5 (should be capped at 30)
            delay = await strategy.calculate_backoff_delay(rate_info, attempt=10)
            assert delay == 37.5  # 30 (capped) + (0.25 * 30) jitter

    def test_should_proactive_throttle_conservative(self, strategy):
        """Test MAL's very conservative proactive throttling."""
        # Test at 95% threshold
        rate_info = RateLimitInfo(remaining=5, limit=100)  # 95% used
        result = strategy.should_proactive_throttle(rate_info)
        assert result is True
        
        # Test below threshold
        rate_info = RateLimitInfo(remaining=10, limit=100)  # 90% used
        result = strategy.should_proactive_throttle(rate_info)
        assert result is False

    def test_should_proactive_throttle_missing_data(self, strategy):
        """Test MAL proactive throttling with missing data."""
        rate_info = RateLimitInfo(remaining=None, limit=100)
        result = strategy.should_proactive_throttle(rate_info)
        assert result is False


class TestMALRateLimitAdapter:
    """Test MAL-specific rate limit adapter."""

    @pytest.fixture
    def adapter(self):
        """Create MAL adapter for testing."""
        return MALRateLimitAdapter()

    def test_extract_rate_limit_info_403(self, adapter):
        """Test extracting rate limit info for 403 response."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {"Some-Header": "value"}
        response.status = 403
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining is None
        assert result.limit is None
        assert result.reset_time is None
        assert result.retry_after is None
        assert result.degraded is True
        assert result.custom_data["platform"] == "mal"
        assert result.custom_data["status_code"] == 403
        assert result.custom_data["dos_protection"] is True
        assert result.custom_data["headers_available"] is False

    def test_extract_rate_limit_info_429(self, adapter):
        """Test extracting rate limit info for 429 response."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {"Retry-After": "45"}
        response.status = 429
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining is None
        assert result.limit is None
        assert result.reset_time is None
        assert result.retry_after == 45
        assert result.degraded is True
        assert result.custom_data["status_code"] == 429
        assert result.custom_data["dos_protection"] is False

    def test_extract_rate_limit_info_success(self, adapter):
        """Test extracting rate limit info for successful response."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {}
        response.status = 200
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining is None
        assert result.limit is None
        assert result.degraded is False
        assert result.custom_data["status_code"] == 200

    def test_extract_rate_limit_info_invalid_retry_after(self, adapter):
        """Test extracting rate limit info with invalid retry_after."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {"Retry-After": "invalid_number"}
        response.status = 429
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.retry_after is None

    def test_get_strategy(self, adapter):
        """Test getting the MAL strategy."""
        strategy = adapter.get_strategy()
        assert isinstance(strategy, MALRateLimitStrategy)


class TestJikanRateLimitStrategy:
    """Test Jikan-specific rate limiting strategy."""

    @pytest.fixture
    def strategy(self):
        """Create Jikan strategy for testing."""
        return JikanRateLimitStrategy()

    @pytest.mark.asyncio
    async def test_handle_rate_limit_response_with_retry_after(self, strategy):
        """Test handling rate limit response with retry_after."""
        rate_info = RateLimitInfo(retry_after=40)
        
        with patch('asyncio.sleep') as mock_sleep, \
             patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.15):
            
            await strategy.handle_rate_limit_response(rate_info)
            
            mock_sleep.assert_called_once()
            call_args = mock_sleep.call_args[0][0]
            assert call_args == 46.0  # 40 + (0.15 * 40) jitter

    @pytest.mark.asyncio
    async def test_handle_rate_limit_response_default(self, strategy):
        """Test handling rate limit response with default timing."""
        rate_info = RateLimitInfo()
        
        with patch('asyncio.sleep') as mock_sleep, \
             patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.15):
            
            await strategy.handle_rate_limit_response(rate_info)
            
            mock_sleep.assert_called_once()
            call_args = mock_sleep.call_args[0][0]
            assert call_args == 69.0  # 60 + (0.15 * 60) jitter

    @pytest.mark.asyncio
    async def test_calculate_backoff_delay(self, strategy):
        """Test Jikan backoff delay calculation."""
        rate_info = RateLimitInfo()
        
        with patch('src.integrations.rate_limiting.platform_strategies.random.uniform', return_value=0.3):
            # Test attempt 0
            delay = await strategy.calculate_backoff_delay(rate_info, attempt=0)
            assert delay == 1.3  # 1 + (0.3 * 1) jitter
            
            # Test attempt that would exceed cap
            delay = await strategy.calculate_backoff_delay(rate_info, attempt=10)
            assert delay == 78.0  # 60 (capped) + (0.3 * 60) jitter

    def test_should_proactive_throttle_always_false(self, strategy):
        """Test that Jikan never does proactive throttling."""
        # Even with high usage, should return False (no headers available)
        rate_info = RateLimitInfo(remaining=1, limit=100)
        result = strategy.should_proactive_throttle(rate_info)
        assert result is False
        
        # Test with no data
        rate_info = RateLimitInfo()
        result = strategy.should_proactive_throttle(rate_info)
        assert result is False


class TestJikanRateLimitAdapter:
    """Test Jikan-specific rate limit adapter."""

    @pytest.fixture
    def adapter(self):
        """Create Jikan adapter for testing."""
        return JikanRateLimitAdapter()

    def test_extract_rate_limit_info_429(self, adapter):
        """Test extracting rate limit info for 429 response."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {}
        response.status = 429
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining is None
        assert result.limit is None
        assert result.reset_time is None
        assert result.retry_after is None
        assert result.degraded is True
        assert result.custom_data["platform"] == "jikan"
        assert result.custom_data["status_code"] == 429
        assert result.custom_data["headers_available"] is False
        assert result.custom_data["response_format"] == "json_error"

    def test_extract_rate_limit_info_success(self, adapter):
        """Test extracting rate limit info for successful response."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {}
        response.status = 200
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.degraded is False
        assert result.custom_data["status_code"] == 200

    def test_extract_rate_limit_info_with_retry_after(self, adapter):
        """Test extracting rate limit info with retry_after header."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {"Retry-After": "30"}
        response.status = 429
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.retry_after == 30

    def test_extract_rate_limit_info_invalid_retry_after(self, adapter):
        """Test extracting rate limit info with invalid retry_after."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {"Retry-After": "not_a_number"}
        response.status = 429
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.retry_after is None

    def test_get_strategy(self, adapter):
        """Test getting the Jikan strategy."""
        strategy = adapter.get_strategy()
        assert isinstance(strategy, JikanRateLimitStrategy)


class TestGenericRateLimitAdapter:
    """Test generic rate limit adapter."""

    @pytest.fixture
    def adapter(self):
        """Create generic adapter for testing."""
        return GenericRateLimitAdapter("test_service")

    def test_extract_rate_limit_info_standard_headers(self, adapter):
        """Test extracting rate limit info with standard headers."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {
            "X-RateLimit-Remaining": "25",
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Reset": "1234567890",
            "Retry-After": "30"
        }
        response.status = 200
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining == 25
        assert result.limit == 100
        assert result.reset_time == 1234567890
        assert result.retry_after == 30
        assert result.custom_data["platform"] == "test_service"
        assert result.custom_data["status_code"] == 200

    def test_extract_rate_limit_info_alternative_headers(self, adapter):
        """Test extracting rate limit info with alternative header formats."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {
            "X-Rate-Limit-Remaining": "15",
            "X-Rate-Limit-Limit": "50",
            "X-Rate-Limit-Reset": "9876543210"
        }
        response.status = 200
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining == 15
        assert result.limit == 50
        assert result.reset_time == 9876543210

    def test_extract_rate_limit_info_rate_limit_prefix(self, adapter):
        """Test extracting rate limit info with RateLimit prefix headers."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {
            "RateLimit-Remaining": "35",
            "RateLimit-Limit": "75",
            "RateLimit-Reset": "5555555555"
        }
        response.status = 200
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining == 35
        assert result.limit == 75
        assert result.reset_time == 5555555555

    def test_extract_rate_limit_info_missing_headers(self, adapter):
        """Test extracting rate limit info with missing headers."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {}
        response.status = 200
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining is None
        assert result.limit is None
        assert result.reset_time is None
        assert result.retry_after is None

    def test_extract_rate_limit_info_invalid_values(self, adapter):
        """Test extracting rate limit info with invalid header values."""
        response = MagicMock(spec=aiohttp.ClientResponse)
        response.headers = {
            "X-RateLimit-Remaining": "not_a_number",
            "X-RateLimit-Limit": "invalid",
            "X-RateLimit-Reset": "bad_timestamp",
            "Retry-After": "not_int"
        }
        response.status = 200
        
        result = adapter.extract_rate_limit_info(response)
        
        assert result.remaining is None
        assert result.limit is None
        assert result.reset_time is None
        assert result.retry_after is None

    def test_get_strategy(self, adapter):
        """Test getting the generic strategy."""
        strategy = adapter.get_strategy()
        assert isinstance(strategy, GenericRateLimitStrategy)
        assert strategy.service_name == "test_service"

    def test_initialization_with_service_name(self):
        """Test adapter initialization with service name."""
        adapter = GenericRateLimitAdapter("custom_service")
        assert adapter.service_name == "custom_service"
        strategy = adapter.get_strategy()
        assert strategy.service_name == "custom_service"


class TestAbstractBaseClasses:
    """Test abstract base class behavior."""

    def test_rate_limit_strategy_abstract(self):
        """Test that RateLimitStrategy is abstract."""
        with pytest.raises(TypeError):
            RateLimitStrategy()

    def test_platform_rate_limit_adapter_abstract(self):
        """Test that PlatformRateLimitAdapter is abstract."""
        with pytest.raises(TypeError):
            PlatformRateLimitAdapter()

    def test_abstract_methods_must_be_implemented(self):
        """Test that concrete classes must implement all abstract methods."""
        
        # Test incomplete strategy implementation
        class IncompleteStrategy(RateLimitStrategy):
            async def handle_rate_limit_response(self, rate_info):
                pass
            # Missing other abstract methods
        
        with pytest.raises(TypeError):
            IncompleteStrategy()
        
        # Test incomplete adapter implementation  
        class IncompleteAdapter(PlatformRateLimitAdapter):
            def extract_rate_limit_info(self, response):
                pass
            # Missing get_strategy method
        
        with pytest.raises(TypeError):
            IncompleteAdapter()

    @pytest.mark.asyncio
    async def test_abstract_method_implementations_call_pass(self):
        """Test that trying to call abstract method implementations hits pass statements."""
        
        # Create minimal concrete implementations that expose the abstract methods
        class MinimalStrategy(RateLimitStrategy):
            async def handle_rate_limit_response(self, rate_info):
                # Call parent to hit the pass statement at line 51
                await super().handle_rate_limit_response(rate_info)
                
            async def calculate_backoff_delay(self, rate_info, attempt=0):
                # Call parent to hit the pass statement at line 64
                return await super().calculate_backoff_delay(rate_info, attempt)
                
            def should_proactive_throttle(self, rate_info):
                # Call parent to hit the pass statement at line 76
                return super().should_proactive_throttle(rate_info)
        
        class MinimalAdapter(PlatformRateLimitAdapter):
            def extract_rate_limit_info(self, response):
                # Call parent to hit the pass statement at line 96
                return super().extract_rate_limit_info(response)
                
            def get_strategy(self):
                # Call parent to hit the pass statement at line 105
                return super().get_strategy()
        
        # Test the strategy abstract methods
        strategy = MinimalStrategy()
        rate_info = RateLimitInfo()
        
        # These should return None (from pass statements)
        result = await strategy.handle_rate_limit_response(rate_info)
        assert result is None
        
        result = await strategy.calculate_backoff_delay(rate_info)
        assert result is None
        
        result = strategy.should_proactive_throttle(rate_info)
        assert result is None
        
        # Test the adapter abstract methods
        adapter = MinimalAdapter()
        mock_response = MagicMock()
        
        result = adapter.extract_rate_limit_info(mock_response)
        assert result is None
        
        result = adapter.get_strategy()
        assert result is None