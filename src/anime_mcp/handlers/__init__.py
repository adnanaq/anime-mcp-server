"""
Modern MCP handlers following separation of concerns pattern.
"""

from .anime_handler import AnimeHandler
from .base_handler import BaseAnimeHandler

__all__ = ["AnimeHandler", "BaseAnimeHandler"]
