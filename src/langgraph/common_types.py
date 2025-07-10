"""
Common types and enums used across LangGraph modules.

This module provides shared types to avoid circular dependencies when 
removing legacy workflow components.
"""

from enum import Enum


class LLMProvider(Enum):
    """LLM provider options for workflow systems."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"