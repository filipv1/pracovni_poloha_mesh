#!/usr/bin/env python3
"""
BEZPEČNÝ PATCH - pouze minimální změny pro snížení spotřeby
Aplikuj postupně a testuj
"""

import os
import gc
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def safe_cleanup_old_files(upload_folder='uploads', output_folder='outputs', max_age_hours=24):
    """
    Bezpečné čištění starých souborů
    """
    try:
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned = 0
        
        for folder in [upload_folder, output_folder]:
            if not os.path.exists(folder):
                continue
                
            for file_path in Path(folder).iterdir():
                try:
                    if file_path.is_file():
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff_time:
                            file_path.unlink()
                            cleaned += 1
                            logger.info(f"Removed old file: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {e}")
                    continue
        
        logger.info(f"Cleaned {cleaned} old files")
        return cleaned
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return 0

def aggressive_gc():
    """
    Agresivní garbage collection
    """
    collected = gc.collect(2)
    logger.info(f"GC collected {collected} objects")
    return collected

# Pokud je spuštěno přímo, provede čištění
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running safe cleanup...")
    cleaned = safe_cleanup_old_files()
    collected = aggressive_gc()
    print(f"Cleaned {cleaned} files, collected {collected} objects")