"""
Progress Tracker for Real-time Updates
Manages Server-Sent Events (SSE) for job progress tracking
"""
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from flask import Response
import threading

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Manages real-time progress updates via SSE"""
    
    def __init__(self):
        # Store active SSE connections
        self.connections = {}
        self.lock = threading.Lock()
    
    def register_connection(self, job_id: int, connection_id: str):
        """Register a new SSE connection for a job"""
        with self.lock:
            if job_id not in self.connections:
                self.connections[job_id] = {}
            self.connections[job_id][connection_id] = {
                'connected_at': datetime.utcnow(),
                'last_update': None
            }
            logger.info(f"Registered SSE connection {connection_id} for job {job_id}")
    
    def unregister_connection(self, job_id: int, connection_id: str):
        """Remove an SSE connection"""
        with self.lock:
            if job_id in self.connections and connection_id in self.connections[job_id]:
                del self.connections[job_id][connection_id]
                if not self.connections[job_id]:
                    del self.connections[job_id]
                logger.info(f"Unregistered SSE connection {connection_id} for job {job_id}")
    
    def send_update(self, job_id: int, data: Dict[str, Any]):
        """Send progress update to all connections for a job"""
        with self.lock:
            if job_id in self.connections:
                for connection_id in self.connections[job_id]:
                    self.connections[job_id][connection_id]['last_update'] = datetime.utcnow()
    
    def create_sse_stream(self, job_id: int, app, db):
        """Create an SSE stream for a job"""
        def generate():
            from models import Job
            
            connection_id = f"{job_id}_{int(time.time() * 1000)}"
            self.register_connection(job_id, connection_id)
            
            try:
                # Send initial connection message
                yield self._format_sse_message({
                    'type': 'connected',
                    'job_id': job_id,
                    'message': 'Connected to progress stream'
                })
                
                # Keep track of last sent data to avoid duplicates
                last_data = None
                no_change_count = 0
                
                while True:
                    with app.app_context():
                        job = Job.query.get(job_id)
                        
                        if not job:
                            yield self._format_sse_message({
                                'type': 'error',
                                'message': 'Job not found'
                            })
                            break
                        
                        # Prepare progress data
                        data = {
                            'type': 'progress',
                            'job_id': job_id,
                            'status': job.status,
                            'stage': job.processing_stage or 'queued',
                            'percent': job.progress_percent or 0,
                            'frames_processed': job.processed_frames or 0,
                            'total_frames': job.total_frames or 0,
                            'time_elapsed': job.time_elapsed_seconds,
                            'time_remaining': job.time_remaining_seconds
                        }
                        
                        # Add stage-specific messages
                        if job.status == 'queued':
                            # Get queue position
                            from core.job_processor import JobProcessor
                            data['message'] = f'Waiting in queue (position: pending)'
                        elif job.status == 'processing':
                            if job.processing_stage == 'uploading':
                                data['message'] = 'Uploading video to GPU...'
                            elif job.processing_stage == 'mediapipe':
                                data['message'] = f'Running MediaPipe detection... ({job.processed_frames or 0}/{job.total_frames or 0} frames)'
                            elif job.processing_stage == 'smplx':
                                data['message'] = 'Fitting SMPL-X model to poses...'
                            elif job.processing_stage == 'angles':
                                data['message'] = 'Calculating skin-based angles...'
                            elif job.processing_stage == 'analysis':
                                data['message'] = 'Generating ergonomic analysis...'
                            elif job.processing_stage == 'downloading':
                                data['message'] = 'Preparing files for download...'
                            else:
                                data['message'] = f'Processing... ({job.progress_percent:.0f}%)'
                        elif job.status == 'completed':
                            data['message'] = 'Processing completed successfully!'
                            data['percent'] = 100
                        elif job.status == 'failed':
                            data['message'] = f'Processing failed: {job.error_message or "Unknown error"}'
                            data['percent'] = 0
                        
                        # Send update only if data changed
                        if data != last_data:
                            yield self._format_sse_message(data)
                            last_data = data
                            no_change_count = 0
                        else:
                            no_change_count += 1
                        
                        # If job is completed or failed, send final update and close
                        if job.status in ['completed', 'failed']:
                            # Send completion data with file info if successful
                            if job.status == 'completed':
                                files_data = []
                                for file in job.files:
                                    files_data.append({
                                        'type': file.file_type,
                                        'filename': file.filename,
                                        'url': file.r2_url,
                                        'size_mb': file.size_mb
                                    })
                                
                                yield self._format_sse_message({
                                    'type': 'completed',
                                    'job_id': job_id,
                                    'files': files_data,
                                    'message': 'Files ready for download'
                                })
                            
                            break
                        
                        # Send heartbeat if no changes for a while
                        if no_change_count >= 10:  # Every 10 seconds of no changes
                            yield self._format_sse_message({
                                'type': 'heartbeat',
                                'timestamp': datetime.utcnow().isoformat()
                            })
                            no_change_count = 0
                        
                        # Refresh database session
                        db.session.commit()
                        
                        # Wait before next update
                        time.sleep(1)
                        
            except GeneratorExit:
                logger.info(f"SSE stream closed for job {job_id}")
            except Exception as e:
                logger.error(f"SSE stream error for job {job_id}: {e}")
                yield self._format_sse_message({
                    'type': 'error',
                    'message': str(e)
                })
            finally:
                self.unregister_connection(job_id, connection_id)
        
        return Response(generate(), mimetype='text/event-stream')
    
    def _format_sse_message(self, data: Dict[str, Any]) -> str:
        """Format data as SSE message"""
        # Add timestamp to all messages
        data['timestamp'] = datetime.utcnow().isoformat()
        
        # Format as SSE
        json_data = json.dumps(data)
        return f"data: {json_data}\n\n"
    
    def cleanup_stale_connections(self, max_age_seconds: int = 3600):
        """Remove stale connections older than max_age"""
        with self.lock:
            current_time = datetime.utcnow()
            stale_connections = []
            
            for job_id in list(self.connections.keys()):
                for connection_id in list(self.connections[job_id].keys()):
                    conn_info = self.connections[job_id][connection_id]
                    age = (current_time - conn_info['connected_at']).total_seconds()
                    
                    if age > max_age_seconds:
                        stale_connections.append((job_id, connection_id))
            
            # Remove stale connections
            for job_id, connection_id in stale_connections:
                self.unregister_connection(job_id, connection_id)
                logger.info(f"Cleaned up stale connection {connection_id} for job {job_id}")
    
    def get_active_connections_count(self) -> int:
        """Get count of active SSE connections"""
        with self.lock:
            count = sum(len(conns) for conns in self.connections.values())
            return count
    
    def get_connection_stats(self) -> Dict:
        """Get statistics about active connections"""
        with self.lock:
            stats = {
                'total_connections': self.get_active_connections_count(),
                'jobs_with_connections': len(self.connections),
                'connections_by_job': {}
            }
            
            for job_id, conns in self.connections.items():
                stats['connections_by_job'][job_id] = len(conns)
            
            return stats