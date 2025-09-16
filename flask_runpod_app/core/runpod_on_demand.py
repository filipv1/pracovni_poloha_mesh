"""
RunPod On-Demand Controller
Manages pods with Network Volume - starts only when needed
"""
import os
import json
import time
import requests
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class RunPodOnDemandController:
    """
    Controls RunPod pods on-demand with persistent Network Volume
    Network Volume contains: conda env, SMPL-X models, git repo
    """
    
    def __init__(self, api_key: str, network_volume_id: str = None, template_id: str = None):
        self.api_key = api_key
        self.network_volume_id = network_volume_id or os.environ.get('RUNPOD_NETWORK_VOLUME_ID')
        self.template_id = template_id or os.environ.get('RUNPOD_TEMPLATE_ID')
        self.graphql_url = "https://api.runpod.io/graphql"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": api_key
        }
        self.current_pod_id = None
        self._last_activity = None
        
    def process_video(self, video_path: str, output_dir: str, quality: str = "high") -> Tuple[bool, str]:
        """
        Main entry point - processes video using on-demand pod
        """
        try:
            # 1. Upload video to storage (R2/S3)
            logger.info("Uploading video to storage...")
            video_url = self._upload_to_storage(video_path)
            
            # 2. Ensure pod is running
            logger.info("Ensuring pod is available...")
            pod_id = self._ensure_pod_running()
            if not pod_id:
                return False, "Failed to start pod"
            
            # 3. Send processing job to pod
            logger.info(f"Sending job to pod {pod_id}...")
            job_id = self._send_job_to_pod(pod_id, video_url, quality)
            
            # 4. Wait for processing
            logger.info("Processing video...")
            result = self._wait_for_completion(pod_id, job_id)
            
            # 5. Download results
            if result['status'] == 'completed':
                logger.info("Downloading results...")
                self._download_results(result['output_url'], output_dir)
                
                # Update last activity
                self._last_activity = time.time()
                
                return True, f"Processing completed: {job_id}"
            else:
                return False, f"Processing failed: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Error in process_video: {e}")
            return False, str(e)
    
    def _ensure_pod_running(self) -> Optional[str]:
        """
        Ensures a pod is running, creates or resumes as needed
        Returns pod_id if successful
        """
        # First check existing pods
        existing_pod = self._find_existing_pod()
        
        if existing_pod:
            pod_id = existing_pod['id']
            status = existing_pod['desiredStatus']
            
            if status == 'RUNNING':
                logger.info(f"Pod {pod_id} already running")
                self.current_pod_id = pod_id
                return pod_id
            elif status in ['EXITED', 'STOPPED']:
                logger.info(f"Resuming pod {pod_id}...")
                if self._resume_pod(pod_id):
                    self.current_pod_id = pod_id
                    return pod_id
        
        # No existing pod, create new one with network volume
        logger.info("Creating new pod with network volume...")
        pod_id = self._create_pod_with_volume()
        if pod_id:
            self.current_pod_id = pod_id
            return pod_id
        
        return None
    
    def _find_existing_pod(self) -> Optional[Dict]:
        """Find existing pod with our network volume"""
        query = """
        query {
            myself {
                pods {
                    id
                    name
                    desiredStatus
                    runtime {
                        uptimeInSeconds
                    }
                }
            }
        }
        """
        
        try:
            response = requests.post(
                self.graphql_url,
                json={"query": query},
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                pods = data.get('data', {}).get('myself', {}).get('pods', [])
                
                # Find pod with our network volume or template
                for pod in pods:
                    # You might want to tag pods or check by name pattern
                    if 'pose' in pod.get('name', '').lower():
                        return pod
                        
        except Exception as e:
            logger.error(f"Error finding pods: {e}")
        
        return None
    
    def _resume_pod(self, pod_id: str) -> bool:
        """Resume a stopped pod"""
        mutation = f"""
        mutation {{
            podResume(input: {{ podId: "{pod_id}" }}) {{
                id
                desiredStatus
            }}
        }}
        """
        
        try:
            response = requests.post(
                self.graphql_url,
                json={"query": mutation},
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if not data.get('errors'):
                    logger.info(f"Pod {pod_id} resuming...")
                    
                    # Wait for pod to be ready (10-30 seconds typically)
                    for i in range(60):  # Max 1 minute wait
                        time.sleep(2)
                        if self._is_pod_ready(pod_id):
                            logger.info(f"Pod {pod_id} is ready!")
                            return True
                    
            logger.error(f"Failed to resume pod: {response.text}")
                    
        except Exception as e:
            logger.error(f"Error resuming pod: {e}")
        
        return False
    
    def _create_pod_with_volume(self) -> Optional[str]:
        """Create new pod with network volume attached"""
        
        if not self.network_volume_id:
            logger.error("No network volume ID configured")
            return None
        
        mutation = f"""
        mutation {{
            podRentInterruptable(input: {{
                networkVolumeId: "{self.network_volume_id}"
                gpuTypeId: "NVIDIA GeForce RTX 3070"
                cloudType: SECURE
                minVcpuCount: 4
                minMemoryInGb: 20
                minGpuCount: 1
                dockerArgs: "bash /workspace/startup.sh"
                volumeMountPath: "/workspace"
                startJupyter: false
                startSsh: true
                name: "pose-analysis-{int(time.time())}"
            }}) {{
                id
                desiredStatus
                machineId
            }}
        }}
        """
        
        try:
            response = requests.post(
                self.graphql_url,
                json={"query": mutation},
                headers=self.headers,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if not data.get('errors'):
                    pod_data = data.get('data', {}).get('podRentInterruptable', {})
                    pod_id = pod_data.get('id')
                    
                    if pod_id:
                        logger.info(f"Created pod {pod_id}, waiting for ready state...")
                        
                        # Wait for pod to be ready
                        for i in range(120):  # Max 2 minutes
                            time.sleep(2)
                            if self._is_pod_ready(pod_id):
                                logger.info(f"Pod {pod_id} is ready!")
                                return pod_id
                        
        except Exception as e:
            logger.error(f"Error creating pod: {e}")
        
        return None
    
    def _is_pod_ready(self, pod_id: str) -> bool:
        """Check if pod is ready for processing"""
        query = f"""
        query {{
            pod(input: {{ podId: "{pod_id}" }}) {{
                id
                desiredStatus
                runtime {{
                    uptimeInSeconds
                }}
            }}
        }}
        """
        
        try:
            response = requests.post(
                self.graphql_url,
                json={"query": query},
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                pod = data.get('data', {}).get('pod', {})
                return pod.get('desiredStatus') == 'RUNNING' and pod.get('runtime', {}).get('uptimeInSeconds', 0) > 5
                
        except:
            pass
        
        return False
    
    def _send_job_to_pod(self, pod_id: str, video_url: str, quality: str) -> str:
        """Send processing job to pod via SSH or API"""
        # This would SSH into pod and run processing
        # For now, return mock job ID
        job_id = f"job_{int(time.time())}"
        
        # In real implementation:
        # 1. SSH to pod
        # 2. Run: python process_video.py --url {video_url} --quality {quality}
        # 3. Return job ID
        
        return job_id
    
    def _wait_for_completion(self, pod_id: str, job_id: str, timeout: int = 600) -> Dict:
        """Wait for job completion"""
        # In real implementation, poll pod for job status
        # For now, simulate processing
        
        time.sleep(5)  # Simulate processing
        
        return {
            'status': 'completed',
            'output_url': f"s3://results/{job_id}/output.xlsx"
        }
    
    def _upload_to_storage(self, file_path: str) -> str:
        """Upload file to R2/S3"""
        # Use existing storage client
        # Return URL
        return f"s3://uploads/{os.path.basename(file_path)}"
    
    def _download_results(self, url: str, output_dir: str):
        """Download results from storage"""
        # Use existing storage client
        pass
    
    def auto_shutdown_check(self):
        """
        Check if pod should be shut down due to inactivity
        Call this periodically (e.g., every 5 minutes)
        """
        if not self.current_pod_id or not self._last_activity:
            return
        
        idle_time = time.time() - self._last_activity
        max_idle = int(os.environ.get('POD_MAX_IDLE_SECONDS', 300))  # 5 min default
        
        if idle_time > max_idle:
            logger.info(f"Pod idle for {idle_time}s, shutting down...")
            self.stop_current_pod()
    
    def stop_current_pod(self):
        """Stop the current pod to save money"""
        if not self.current_pod_id:
            return
        
        mutation = f"""
        mutation {{
            podStop(input: {{ podId: "{self.current_pod_id}" }}) {{
                id
                desiredStatus
            }}
        }}
        """
        
        try:
            response = requests.post(
                self.graphql_url,
                json={"query": mutation},
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Pod {self.current_pod_id} stopped")
                self.current_pod_id = None
                
        except Exception as e:
            logger.error(f"Error stopping pod: {e}")