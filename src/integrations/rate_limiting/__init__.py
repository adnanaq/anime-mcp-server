"""Rate limiting module with Strategy and Adapter patterns for platform-specific behavior."""

from .core import (
    RateLimitConfig,
    RateLimitedRequest,
    RateLimitManager,
    RateLimitStrategy as CoreRateLimitStrategy,
    ServiceRateLimiter,
    TokenBucket,
    rate_limit_manager,
)
from .platform_strategies import (
    RateLimitInfo,
    RateLimitStrategy,
    PlatformRateLimitAdapter,
    AniListRateLimitStrategy,
    AniListRateLimitAdapter,
    MALRateLimitStrategy,
    MALRateLimitAdapter,
    JikanRateLimitStrategy,
    JikanRateLimitAdapter,
    GenericRateLimitStrategy,
    GenericRateLimitAdapter,
)

__all__ = [
    # Core rate limiting
    "RateLimitConfig",
    "RateLimitedRequest", 
    "RateLimitManager",
    "CoreRateLimitStrategy",
    "ServiceRateLimiter",
    "TokenBucket",
    "rate_limit_manager",
    # Platform strategies
    "RateLimitInfo",
    "RateLimitStrategy", 
    "PlatformRateLimitAdapter",
    "AniListRateLimitStrategy",
    "AniListRateLimitAdapter",
    "MALRateLimitStrategy",
    "MALRateLimitAdapter",
    "JikanRateLimitStrategy",
    "JikanRateLimitAdapter",
    "GenericRateLimitStrategy",
    "GenericRateLimitAdapter",
]