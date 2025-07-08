#!/bin/bash
# Setup cron job for weekly anime database updates

# Get the absolute path to the project
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$PROJECT_DIR/scripts/update_database_weekly.py"
PYTHON_PATH="$PROJECT_DIR/venv/bin/python"

echo "Setting up weekly anime database update cron job..."
echo "Project directory: $PROJECT_DIR"
echo "Script path: $SCRIPT_PATH"

# Make sure the script is executable
chmod +x "$SCRIPT_PATH"

# Create cron job entry - Run Friday 2 AM (day after typical Thu updates)
CRON_JOB="0 2 * * 5 cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $PROJECT_DIR/logs/cron.log 2>&1"

# Add to crontab (avoiding duplicates)
(crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH"; echo "$CRON_JOB") | crontab -

echo "✅ Cron job added successfully!"
echo "Weekly update will run every Friday at 2 AM (day after typical database updates)"
echo ""
echo "To verify the cron job:"
echo "  crontab -l"
echo ""
echo "To view logs:"
echo "  tail -f $PROJECT_DIR/logs/weekly_update.log"
echo "  tail -f $PROJECT_DIR/logs/cron.log"
echo ""
echo "To remove the cron job:"
echo "  crontab -l | grep -v '$SCRIPT_PATH' | crontab -"
