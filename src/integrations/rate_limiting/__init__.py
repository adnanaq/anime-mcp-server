"""Rate limiting module with Strategy and Adapter patterns for platform-specific behavior."""

from .core import (
    RateLimitConfig,
    RateLimitedRequest,
    RateLimitManager,
)
from .core import RateLimitStrategy as CoreRateLimitStrategy
from .core import (
    ServiceRateLimiter,
    TokenBucket,
    rate_limit_manager,
)
from .platform_strategies import (
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
