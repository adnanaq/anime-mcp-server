#!/usr/bin/env python3
"""
Smart weekly update script that checks if it's safe to update
Run this via cron job every few hours to catch updates when they're stable:
0 */4 * * * /path/to/anime-mcp-server/scripts/smart_weekly_update.py
"""
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from services.smart_scheduler import SmartScheduler
from services.update_service import UpdateService
import logging

# Configure logging for cron job
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "logs" / "smart_update.log"),
        logging.StreamHandler()
    ]
)

async def main():
    """Run smart update check"""
    logger = logging.getLogger(__name__)
    logger.info("🤖 Starting smart anime database update check")
    
    try:
        # Create logs directory
        (project_root / "logs").mkdir(exist_ok=True)
        
        # Initialize services
        scheduler = SmartScheduler()
        update_service = UpdateService()
        
        # Check if it's safe to update
        safety_check = await scheduler.is_safe_to_update(min_hours_after_release=2)
        
        if not safety_check['safe_to_update']:
            logger.info("⏸️ Update postponed - not safe to update yet")
            for reason in safety_check['reasons']:
                logger.info(f"   • {reason}")
            logger.info(f"💡 Recommendation: {safety_check['recommendation']}")
            return 0
        
        # Safe to update - check if there are actually updates
        has_updates = await update_service.check_for_updates()
        
        if not has_updates:
            logger.info("✅ No updates available - database is current")
            return 0
        
        # Perform the update
        logger.info("🚀 Safety checks passed - performing incremental update")
        success = await update_service.perform_incremental_update()
        
        if success:
            logger.info("✅ Smart update completed successfully")
            return 0
        else:
            logger.error("❌ Smart update failed")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Smart update failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)