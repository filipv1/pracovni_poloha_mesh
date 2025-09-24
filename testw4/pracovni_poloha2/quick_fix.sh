#!/bin/bash
# Quick fix script for Railway deployment

echo "Applying quick fixes for Railway deployment..."

# 1. Set environment variable for production optimizations
export FLASK_ENV=production

# 2. Clear old files before starting
echo "Cleaning up old files..."
find uploads/ -type f -mtime +1 -delete 2>/dev/null
find outputs/ -type f -mtime +1 -delete 2>/dev/null
find jobs/ -type f -mtime +2 -delete 2>/dev/null

# 3. Truncate large log files
for logfile in logs/*.log; do
    if [ -f "$logfile" ]; then
        size=$(stat -f%z "$logfile" 2>/dev/null || stat -c%s "$logfile" 2>/dev/null)
        if [ "$size" -gt 104857600 ]; then  # 100MB
            echo "Truncating large log file: $logfile"
            tail -n 10000 "$logfile" > "$logfile.tmp"
            mv "$logfile.tmp" "$logfile"
        fi
    fi
done

# 4. Start the app with memory monitoring
echo "Starting application with optimizations..."
exec python web_app.py