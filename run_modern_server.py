#!/usr/bin/env python3
"""
Entry point for modern MCP server that can be run directly.

This script resolves import issues when running the server as a standalone script.
"""

import sys
from pathlib import Path

# Add src directory to Python path to resolve imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now import and run the modern server
from src.anime_mcp.modern_server import main

if __name__ == "__main__":
    main()