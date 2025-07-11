"""Mock implementation of langgraph_swarm for testing."""

from unittest.mock import MagicMock


def create_swarm(*args, **kwargs):
    """Mock create_swarm function."""
    return MagicMock()


def create_handoff_tool(*args, **kwargs):
    """Mock create_handoff_tool function."""
    return MagicMock()