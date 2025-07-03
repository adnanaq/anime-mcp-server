"""Rate limiting module with Strategy and Adapter patterns for platform-specific behavior."""

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