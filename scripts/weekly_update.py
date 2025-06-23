#!/usr/bin/env python3
"""
Weekly update script for anime database
Run this via cron job every Sunday at 2 AM:
0 2 * * 0 /path/to/anime-mcp-server/scripts/weekly_update.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

import logging

from services.update_service import UpdateService

# Configure logging for cron job
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(project_root / "logs" / "weekly_update.log"),
        logging.StreamHandler(),
    ],
)


async def main():
    """Run weekly update"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting weekly anime database update")

    try:
        # Create logs directory
        (project_root / "logs").mkdir(exist_ok=True)

        # Run update service
        update_service = UpdateService()
        await update_service.schedule_weekly_update()

        logger.info("✅ Weekly update completed successfully")
        return 0

    except Exception as e:
        logger.error(f"❌ Weekly update failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
