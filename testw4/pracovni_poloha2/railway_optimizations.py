#!/usr/bin/env python3
"""
Railway deployment optimizations for reducing memory and network usage
"""

import os
import sys
import time
import json
import shutil
import logging
import psutil
import gc
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread, Event
from flask import Flask, jsonify

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor and report system resource usage"""
    
    def __init__(self, app=None):
        self.app = app
        self.process = psutil.Process()
        self.start_time = time.time()
        self.last_gc_time = time.time()
        
    def get_stats(self):
        """Get current resource statistics"""
        try:
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            # Network stats (if available)
            net_io = psutil.net_io_counters()
            
            # Disk usage for temp directories
            disk_usage = {
                'uploads': self._get_dir_size('uploads'),
                'outputs': self._get_dir_size('outputs'),
                'jobs': self._get_dir_size('jobs'),
                'logs': self._get_dir_size('logs')
            }
            
            uptime = time.time() - self.start_time
            
            return {
                'memory': {
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'percent': self.process.memory_percent()
                },
                'cpu': {
                    'percent': cpu_percent,
                    'num_threads': self.process.num_threads()
                },
                'network': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                },
                'disk': disk_usage,
                'uptime_seconds': uptime,
                'uptime_formatted': self._format_uptime(uptime)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def _get_dir_size(self, path):
        """Get directory size in MB"""
        try:
            if not os.path.exists(path):
                return 0
            total = 0
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += self._get_dir_size(entry.path)
            return total / 1024 / 1024  # Convert to MB
        except:
            return 0
    
    def _format_uptime(self, seconds):
        """Format uptime in human readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{days}d {hours}h {minutes}m"
    
    def force_gc(self):
        """Force garbage collection"""
        collected = gc.collect()
        self.last_gc_time = time.time()
        logger.info(f"Forced GC: collected {collected} objects")
        return collected


class CleanupManager:
    """Manage cleanup of old files and jobs"""
    
    def __init__(self, upload_folder='uploads', output_folder='outputs', 
                 jobs_folder='jobs', log_folder='logs'):
        self.upload_folder = upload_folder
        self.output_folder = output_folder
        self.jobs_folder = jobs_folder
        self.log_folder = log_folder
        self.stop_event = Event()
        self.cleanup_thread = None
        
    def start(self):
        """Start cleanup thread"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.stop_event.clear()
            self.cleanup_thread = Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            logger.info("Cleanup manager started")
    
    def stop(self):
        """Stop cleanup thread"""
        self.stop_event.set()
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        logger.info("Cleanup manager stopped")
    
    def _cleanup_loop(self):
        """Main cleanup loop - runs every hour"""
        while not self.stop_event.is_set():
            try:
                self.cleanup_old_files()
                self.cleanup_old_jobs()
                self.cleanup_old_logs()
                
                # Force garbage collection after cleanup
                gc.collect()
                
                # Wait for 1 hour or until stopped
                self.stop_event.wait(3600)
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                # Wait 5 minutes before retry
                self.stop_event.wait(300)
    
    def cleanup_old_files(self, max_age_hours=24):
        """Remove files older than max_age_hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            total_removed = 0
            total_size = 0
            
            for folder in [self.upload_folder, self.output_folder]:
                if not os.path.exists(folder):
                    continue
                    
                for file_path in Path(folder).glob('*'):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_size = file_path.stat().st_size
                            try:
                                file_path.unlink()
                                total_removed += 1
                                total_size += file_size
                                logger.debug(f"Removed old file: {file_path}")
                            except Exception as e:
                                logger.error(f"Failed to remove {file_path}: {e}")
            
            if total_removed > 0:
                logger.info(f"Cleaned up {total_removed} old files, freed {total_size/1024/1024:.2f} MB")
                
        except Exception as e:
            logger.error(f"Error in cleanup_old_files: {e}")
    
    def cleanup_old_jobs(self, max_age_hours=48):
        """Remove old job files"""
        try:
            if not os.path.exists(self.jobs_folder):
                return
                
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            total_removed = 0
            
            for job_file in Path(self.jobs_folder).glob('*.json'):
                try:
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                    
                    # Check if job is old and completed/failed
                    if job_data.get('status') in ['completed', 'failed', 'error']:
                        completed_at = job_data.get('completed_at', job_data.get('created_at', 0))
                        if completed_at:
                            job_time = datetime.fromtimestamp(completed_at)
                            if job_time < cutoff_time:
                                # Remove associated files
                                if 'files' in job_data:
                                    for file_type, file_path in job_data['files'].items():
                                        if os.path.exists(file_path):
                                            try:
                                                os.remove(file_path)
                                                logger.debug(f"Removed job file: {file_path}")
                                            except:
                                                pass
                                
                                # Remove job file
                                job_file.unlink()
                                total_removed += 1
                                logger.debug(f"Removed old job: {job_file.stem}")
                                
                except Exception as e:
                    logger.error(f"Error processing job file {job_file}: {e}")
            
            if total_removed > 0:
                logger.info(f"Cleaned up {total_removed} old jobs")
                
        except Exception as e:
            logger.error(f"Error in cleanup_old_jobs: {e}")
    
    def cleanup_old_logs(self, max_size_mb=100):
        """Rotate logs if they get too large"""
        try:
            if not os.path.exists(self.log_folder):
                return
                
            for log_file in Path(self.log_folder).glob('*.log'):
                file_size_mb = log_file.stat().st_size / 1024 / 1024
                if file_size_mb > max_size_mb:
                    # Rotate the log
                    backup_name = f"{log_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                    backup_path = log_file.parent / backup_name
                    shutil.move(str(log_file), str(backup_path))
                    logger.info(f"Rotated log file: {log_file.name} -> {backup_name}")
                    
                    # Create new empty log file
                    log_file.touch()
                    
                    # Remove old backups (keep only last 3)
                    backups = sorted(Path(self.log_folder).glob(f"{log_file.stem}_*.log"))
                    if len(backups) > 3:
                        for old_backup in backups[:-3]:
                            old_backup.unlink()
                            logger.debug(f"Removed old log backup: {old_backup.name}")
                            
        except Exception as e:
            logger.error(f"Error in cleanup_old_logs: {e}")
    
    def emergency_cleanup(self):
        """Emergency cleanup when memory is critical"""
        logger.warning("Performing emergency cleanup due to high memory usage")
        
        # Aggressive cleanup - remove files older than 6 hours
        self.cleanup_old_files(max_age_hours=6)
        self.cleanup_old_jobs(max_age_hours=12)
        
        # Clear Python caches
        gc.collect(2)  # Full collection
        
        # Log memory after cleanup
        process = psutil.Process()
        logger.info(f"Memory after emergency cleanup: {process.memory_info().rss / 1024 / 1024:.2f} MB")


class OptimizedEmailWorker:
    """Optimized email worker with better resource management"""
    
    def __init__(self, email_queue, email_service, app):
        self.email_queue = email_queue
        self.email_service = email_service
        self.app = app
        self.stop_event = Event()
        self.worker_thread = None
        
    def start(self):
        """Start email worker"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_event.clear()
            self.worker_thread = Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Optimized email worker started")
    
    def stop(self):
        """Stop email worker"""
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Email worker stopped")
    
    def _worker_loop(self):
        """Main worker loop with proper app context"""
        logger.info("Email worker loop started")
        
        with self.app.app_context():  # Create app context for the entire worker
            while not self.stop_event.is_set():
                try:
                    # Use timeout to allow periodic checks
                    if self.stop_event.wait(0.1):  # Check stop event
                        break
                        
                    # Try to get task from queue (non-blocking)
                    try:
                        email_task = self.email_queue.get_nowait()
                    except:
                        # No task available, wait a bit
                        time.sleep(1)
                        continue
                    
                    # Process the email task
                    self._process_email_task(email_task)
                    
                    # Mark task as done
                    self.email_queue.task_done()
                    
                    # Force cleanup after each email
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Email worker error: {e}")
                    time.sleep(5)  # Wait before retry
    
    def _process_email_task(self, email_task):
        """Process a single email task"""
        job_data = email_task.get('job_data')
        retry_count = email_task.get('retry_count', 0)
        job_id = job_data.get('id')
        
        logger.info(f"Processing email for job {job_id}, attempt {retry_count + 1}")
        
        try:
            # Send email with timeout
            success = self.email_service.send_completion_email(
                job_data, 
                timeout=30  # 30 second timeout
            )
            
            if success:
                logger.info(f"Email sent successfully for job {job_id}")
            else:
                raise Exception("Email send returned False")
                
        except Exception as e:
            logger.error(f"Failed to send email for job {job_id}: {e}")
            
            # Retry logic with exponential backoff
            if retry_count < 2:  # Max 3 attempts
                retry_delay = 60 * (2 ** retry_count)  # 60s, 120s
                logger.info(f"Scheduling retry for job {job_id} in {retry_delay}s")
                
                # Schedule retry
                def schedule_retry():
                    time.sleep(retry_delay)
                    self.email_queue.put({
                        'job_data': job_data,
                        'retry_count': retry_count + 1
                    })
                
                Thread(target=schedule_retry, daemon=True).start()
            else:
                logger.error(f"Email permanently failed for job {job_id} after 3 attempts")


def create_health_endpoint(app, monitor, cleanup_manager):
    """Create health check endpoint for Railway"""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint with resource stats"""
        try:
            stats = monitor.get_stats()
            
            # Determine health status
            memory_percent = stats.get('memory', {}).get('percent', 0)
            status = 'healthy'
            
            if memory_percent > 90:
                status = 'critical'
                # Trigger emergency cleanup
                cleanup_manager.emergency_cleanup()
                monitor.force_gc()
            elif memory_percent > 75:
                status = 'warning'
                # Force garbage collection
                monitor.force_gc()
            
            return jsonify({
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'stats': stats
            })
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500


def optimize_app(app):
    """Apply optimizations to Flask app"""
    
    # Initialize resource monitor
    monitor = ResourceMonitor(app)
    
    # Initialize cleanup manager
    cleanup_manager = CleanupManager()
    cleanup_manager.start()
    
    # Create health endpoint
    create_health_endpoint(app, monitor, cleanup_manager)
    
    # Set aggressive garbage collection
    gc.set_threshold(500, 5, 5)
    
    # Log initial stats
    logger.info("Optimizations applied")
    logger.info(f"Initial stats: {monitor.get_stats()}")
    
    return monitor, cleanup_manager


# Export for use in web_app.py
__all__ = ['ResourceMonitor', 'CleanupManager', 'OptimizedEmailWorker', 'optimize_app']