"""
RunPod Serverless Client
Manages communication with RunPod Serverless endpoints
"""
import os
import time
import json
import requests
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RunPodServerlessClient:
    """Client for RunPod Serverless endpoints"""
    
    def __init__(self, api_key: str, endpoint_id: str = None):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = "https://api.runpod.ai/v2"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def create_job(self, video_url: str, output_key: str, quality: str = "high") -> Tuple[bool, str]:
        """
        Create a new processing job
        Returns (success, job_id_or_error)
        """
        if not self.endpoint_id:
            # Use simulation mode if no endpoint
            logger.warning("No endpoint ID configured - using simulation mode")
            return self._create_simulation_job(video_url, output_key)
        
        try:
            url = f"{self.base_url}/{self.endpoint_id}/run"
            
            payload = {
                "input": {
                    "video_url": video_url,
                    "output_bucket": os.environ.get('R2_BUCKET_NAME', 'flaskrunpod'),
                    "output_key": output_key,
                    "quality": quality
                }
            }
            
            response = requests.post(url, json=payload, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                job_id = data.get('id')
                logger.info(f"Created serverless job: {job_id}")
                return True, job_id
            else:
                error = f"Failed to create job: {response.status_code} - {response.text}"
                logger.error(error)
                return False, error
                
        except Exception as e:
            error = f"Error creating job: {str(e)}"
            logger.error(error)
            return False, error
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a job"""
        if not self.endpoint_id or job_id.startswith("sim_"):
            return self._get_simulation_status(job_id)
        
        try:
            url = f"{self.base_url}/{self.endpoint_id}/status/{job_id}"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "ERROR", "error": f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        if not self.endpoint_id or job_id.startswith("sim_"):
            return True  # Simulation always succeeds
        
        try:
            url = f"{self.base_url}/{self.endpoint_id}/cancel/{job_id}"
            response = requests.post(url, headers=self.headers, timeout=30)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error canceling job: {e}")
            return False
    
    def _create_simulation_job(self, video_url: str, output_key: str) -> Tuple[bool, str]:
        """Create simulated job for testing"""
        import uuid
        job_id = f"sim_{uuid.uuid4().hex[:8]}"
        
        # Store simulation state (in production, use Redis or database)
        simulation_jobs[job_id] = {
            "created": time.time(),
            "video_url": video_url,
            "output_key": output_key,
            "status": "IN_QUEUE"
        }
        
        logger.info(f"Created simulation job: {job_id}")
        return True, job_id
    
    def _get_simulation_status(self, job_id: str) -> Dict:
        """Get simulated job status"""
        if job_id not in simulation_jobs:
            return {"status": "NOT_FOUND"}
        
        job = simulation_jobs[job_id]
        elapsed = time.time() - job["created"]
        
        # Simulate processing stages
        if elapsed < 5:
            status = "IN_QUEUE"
            progress = 0
        elif elapsed < 10:
            status = "IN_PROGRESS"
            progress = int((elapsed - 5) / 5 * 50)
        elif elapsed < 15:
            status = "IN_PROGRESS"
            progress = 50 + int((elapsed - 10) / 5 * 50)
        else:
            status = "COMPLETED"
            progress = 100
        
        return {
            "status": status,
            "progress": progress,
            "id": job_id,
            "output": {
                "results": [
                    {
                        "filename": "analysis.xlsx",
                        "url": f"simulation://results/{job_id}/analysis.xlsx"
                    }
                ] if status == "COMPLETED" else None
            }
        }

# Temporary storage for simulation
simulation_jobs = {}