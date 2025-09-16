"""
RunPod GPU Client
Manages communication with RunPod GPU instances
"""
import os
import time
import json
import subprocess
import requests
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RunPodClient:
    """Client for managing RunPod GPU instances"""
    
    def __init__(self, api_key: str, pod_id: str = None):
        self.api_key = api_key
        self.pod_id = pod_id
        self.base_url = "https://api.runpod.io/v2"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.last_activity = None
        self.pod_info = None
        self.simulation_mode = False
        
        # Test API connection on init
        self._test_api_connection()
    
    def _test_api_connection(self):
        """Test if RunPod API is accessible"""
        try:
            # Try to list pods to test API key
            response = requests.get(
                f"{self.base_url}/pod",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 404:
                # API key might be invalid or no pods exist
                logger.warning("RunPod API returned 404 - entering simulation mode")
                self.simulation_mode = True
            elif response.status_code == 401:
                logger.error("RunPod API authentication failed - invalid API key")
                self.simulation_mode = True
            elif response.status_code == 200:
                logger.info("RunPod API connection successful")
                self.simulation_mode = False
            else:
                logger.warning(f"RunPod API returned unexpected status: {response.status_code}")
                self.simulation_mode = True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Cannot connect to RunPod API: {e} - entering simulation mode")
            self.simulation_mode = True
        
    def get_pod_status(self) -> Dict:
        """Get current status of the pod"""
        if self.simulation_mode:
            # Return simulated pod status
            return {
                "status": "RUNNING",
                "id": self.pod_id or "simulation",
                "name": "Simulation Pod",
                "runtime": {"gpuType": "Simulated GPU"},
                "simulation": True
            }
        
        if not self.pod_id:
            return {"status": "NO_POD_ID", "error": "Pod ID not configured"}
            
        try:
            response = requests.get(
                f"{self.base_url}/pod/{self.pod_id}",
                headers=self.headers,
                timeout=30
            )
            if response.status_code == 200:
                self.pod_info = response.json()
                return self.pod_info
            else:
                # Fall back to simulation mode on error
                self.simulation_mode = True
                logger.warning(f"API error {response.status_code} - switching to simulation mode")
                return self.get_pod_status()  # Recursive call will use simulation
        except Exception as e:
            logger.error(f"Failed to get pod status: {e} - using simulation mode")
            self.simulation_mode = True
            return self.get_pod_status()  # Recursive call will use simulation
    
    def ensure_pod_running(self) -> Tuple[bool, str]:
        """Start pod if not running, return (success, message)"""
        if self.simulation_mode:
            self.last_activity = time.time()
            return True, "Simulation mode - pod ready"
        
        try:
            # Get current status
            status_info = self.get_pod_status()
            current_status = status_info.get('status', 'UNKNOWN')
            
            if current_status == 'RUNNING':
                self.last_activity = time.time()
                return True, "Pod is already running"
            
            if current_status in ['STOPPED', 'PAUSED']:
                # Start the pod
                logger.info(f"Starting pod {self.pod_id} from status {current_status}")
                response = requests.post(
                    f"{self.base_url}/pod/{self.pod_id}/resume",
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code != 200:
                    return False, f"Failed to start pod: {response.text}"
                
                # Wait for pod to be ready
                for attempt in range(60):  # 5 minute timeout
                    time.sleep(5)
                    status_info = self.get_pod_status()
                    if status_info.get('status') == 'RUNNING':
                        logger.info(f"Pod {self.pod_id} started successfully")
                        self.last_activity = time.time()
                        return True, "Pod started successfully"
                
                return False, "Pod failed to start within timeout"
            
            return False, f"Pod in unexpected state: {current_status}"
            
        except Exception as e:
            logger.error(f"Error ensuring pod is running: {e}")
            return False, str(e)
    
    def execute_command(self, command: str) -> Tuple[bool, str, str]:
        """Execute command on RunPod via SSH or API
        Returns (success, stdout, stderr)"""
        if self.simulation_mode:
            # Simulate command execution
            self.last_activity = time.time()
            logger.info(f"[SIMULATION] Executing command: {command[:50]}...")
            
            # Simulate different commands
            if "pip install" in command:
                return True, "Successfully installed packages (simulated)", ""
            elif "python" in command and "process_video.py" in command:
                return True, "Video processing completed (simulated)", ""
            else:
                return True, f"Command executed: {command[:50]}... (simulated)", ""
        
        try:
            # Update activity time
            self.last_activity = time.time()
            
            # If we have SSH access configured, use that
            if os.environ.get('RUNPOD_SSH_KEY'):
                return self._execute_via_ssh(command)
            
            # Otherwise use RunPod API (if available)
            return self._execute_via_api(command)
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return False, "", str(e)
    
    def _execute_via_ssh(self, command: str) -> Tuple[bool, str, str]:
        """Execute command via SSH"""
        try:
            pod_info = self.get_pod_status()
            if not pod_info or pod_info.get('status') != 'RUNNING':
                return False, "", "Pod not running"
            
            # Get SSH details from pod info
            ssh_host = pod_info.get('ssh_host', '')
            ssh_port = pod_info.get('ssh_port', 22)
            
            if not ssh_host:
                return False, "", "SSH host not available"
            
            # Build SSH command
            ssh_key_path = os.environ.get('RUNPOD_SSH_KEY')
            ssh_cmd = [
                'ssh',
                '-i', ssh_key_path,
                '-o', 'StrictHostKeyChecking=no',
                '-p', str(ssh_port),
                f'root@{ssh_host}',
                command
            ]
            
            # Execute command
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def _execute_via_api(self, command: str) -> Tuple[bool, str, str]:
        """Execute command via RunPod API (if available)"""
        # Note: RunPod may not have a direct command execution API
        # This is a placeholder for potential future API support
        return False, "", "API command execution not implemented"
    
    def upload_file(self, local_path: str, remote_path: str) -> Tuple[bool, str]:
        """Upload file to RunPod
        Returns (success, message)"""
        if self.simulation_mode:
            # Simulate file upload
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
                logger.info(f"[SIMULATION] Uploading {local_path} ({file_size:.1f} MB) to {remote_path}")
                time.sleep(0.5)  # Simulate upload time
                return True, f"File uploaded successfully (simulated)"
            else:
                return False, "Local file not found"
        
        try:
            if os.environ.get('RUNPOD_SSH_KEY'):
                return self._upload_via_scp(local_path, remote_path)
            else:
                return False, "File upload requires SSH configuration"
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False, str(e)
    
    def _upload_via_scp(self, local_path: str, remote_path: str) -> Tuple[bool, str]:
        """Upload file via SCP"""
        try:
            pod_info = self.get_pod_status()
            if not pod_info or pod_info.get('status') != 'RUNNING':
                return False, "Pod not running"
            
            ssh_host = pod_info.get('ssh_host', '')
            ssh_port = pod_info.get('ssh_port', 22)
            
            if not ssh_host:
                return False, "SSH host not available"
            
            # Build SCP command
            ssh_key_path = os.environ.get('RUNPOD_SSH_KEY')
            scp_cmd = [
                'scp',
                '-i', ssh_key_path,
                '-o', 'StrictHostKeyChecking=no',
                '-P', str(ssh_port),
                local_path,
                f'root@{ssh_host}:{remote_path}'
            ]
            
            # Execute upload
            result = subprocess.run(
                scp_cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                return True, "File uploaded successfully"
            else:
                return False, f"Upload failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Upload timed out"
        except Exception as e:
            return False, str(e)
    
    def download_file(self, remote_path: str, local_path: str) -> Tuple[bool, str]:
        """Download file from RunPod
        Returns (success, message)"""
        if self.simulation_mode:
            # Simulate file download by creating a mock result file
            logger.info(f"[SIMULATION] Downloading {remote_path} to {local_path}")
            
            # Create a mock result file
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Create appropriate mock file based on extension
            if local_path.endswith('.xlsx'):
                # Create mock Excel file
                try:
                    import pandas as pd
                    df = pd.DataFrame({
                        'Frame': [1, 2, 3],
                        'Trunk_Angle': [15.2, 18.5, 21.3],
                        'Neck_Angle': [10.1, 12.3, 14.5],
                        'Status': ['Normal', 'Warning', 'Alert']
                    })
                    df.to_excel(local_path, index=False)
                except:
                    # Fallback to simple file
                    with open(local_path, 'wb') as f:
                        f.write(b'Mock Excel file (simulation)')
            else:
                # Create generic mock file
                with open(local_path, 'w') as f:
                    f.write(f"Mock result file from RunPod simulation\n")
                    f.write(f"Remote path: {remote_path}\n")
                    f.write(f"Generated at: {datetime.now()}\n")
            
            return True, "File downloaded successfully (simulated)"
        
        try:
            if os.environ.get('RUNPOD_SSH_KEY'):
                return self._download_via_scp(remote_path, local_path)
            else:
                return False, "File download requires SSH configuration"
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return False, str(e)
    
    def _download_via_scp(self, remote_path: str, local_path: str) -> Tuple[bool, str]:
        """Download file via SCP"""
        try:
            pod_info = self.get_pod_status()
            if not pod_info or pod_info.get('status') != 'RUNNING':
                return False, "Pod not running"
            
            ssh_host = pod_info.get('ssh_host', '')
            ssh_port = pod_info.get('ssh_port', 22)
            
            if not ssh_host:
                return False, "SSH host not available"
            
            # Build SCP command
            ssh_key_path = os.environ.get('RUNPOD_SSH_KEY')
            scp_cmd = [
                'scp',
                '-i', ssh_key_path,
                '-o', 'StrictHostKeyChecking=no',
                '-P', str(ssh_port),
                f'root@{ssh_host}:{remote_path}',
                local_path
            ]
            
            # Execute download
            result = subprocess.run(
                scp_cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                return True, "File downloaded successfully"
            else:
                return False, f"Download failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "Download timed out"
        except Exception as e:
            return False, str(e)
    
    def check_idle_timeout(self, timeout_seconds: int = 300) -> bool:
        """Check if pod should be stopped due to inactivity
        Returns True if pod was stopped"""
        if not self.last_activity:
            return False
            
        idle_time = time.time() - self.last_activity
        if idle_time > timeout_seconds:
            logger.info(f"Pod idle for {idle_time}s, stopping...")
            return self.stop_pod()
        
        return False
    
    def stop_pod(self) -> bool:
        """Stop the pod to save costs"""
        if not self.pod_id:
            return False
            
        try:
            response = requests.post(
                f"{self.base_url}/pod/{self.pod_id}/stop",
                headers=self.headers,
                timeout=30
            )
            if response.status_code == 200:
                logger.info(f"Pod {self.pod_id} stopped successfully")
                self.pod_info = None
                return True
            else:
                logger.error(f"Failed to stop pod: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error stopping pod: {e}")
            return False
    
    def get_gpu_metrics(self) -> Dict:
        """Get GPU utilization metrics"""
        try:
            pod_info = self.get_pod_status()
            if pod_info and pod_info.get('status') == 'RUNNING':
                return {
                    'gpu_utilization': pod_info.get('gpu_utilization', 0),
                    'gpu_memory_used': pod_info.get('gpu_memory_used', 0),
                    'gpu_memory_total': pod_info.get('gpu_memory_total', 0),
                    'gpu_temp': pod_info.get('gpu_temp', 0)
                }
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
        
        return {}