#!/bin/bash
# Setup smart cron job that checks every few hours for safe update opportunities

# Get the absolute path to the project
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$PROJECT_DIR/scripts/smart_weekly_update.py"
PYTHON_PATH="$PROJECT_DIR/venv/bin/python"

echo "Setting up smart anime database update cron job..."
echo "Project directory: $PROJECT_DIR"
echo "Script path: $SCRIPT_PATH"

# Make sure the script is executable
chmod +x "$SCRIPT_PATH"

# Create cron job entry - Check every 4 hours for safe update opportunities
CRON_JOB="0 */4 * * * cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $PROJECT_DIR/logs/smart_cron.log 2>&1"

# Add to crontab (avoiding duplicates)
(crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH"; echo "$CRON_JOB") | crontab -

echo "✅ Smart cron job added successfully!"
echo "Update checks will run every 4 hours (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)"
echo "Updates will only proceed when:"
echo "  • New releases are at least 2 hours old"
echo "  • No recent commit activity indicates ongoing work"
echo "  • Actual updates are available"
echo ""
echo "To verify the cron job:"
echo "  crontab -l"
echo ""
echo "To view logs:"
echo "  tail -f $PROJECT_DIR/logs/smart_update.log"
echo "  tail -f $PROJECT_DIR/logs/smart_cron.log"
echo ""
echo "To remove the cron job:"
echo "  crontab -l | grep -v '$SCRIPT_PATH' | crontab -"
echo ""
echo "To test the smart update now:"
echo "  $PYTHON_PATH $SCRIPT_PATH"
