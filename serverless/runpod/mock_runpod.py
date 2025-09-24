"""
Mock RunPod module for local testing
Simulates RunPod serverless environment without actual RunPod dependency
"""

class MockServerless:
    """Mock serverless module"""

    @staticmethod
    def progress_update(job, progress, message=None):
        """Mock progress update - just prints"""
        if message:
            print(f"[PROGRESS {progress}%] {message}")
        else:
            print(f"[PROGRESS {progress}%]")

    @staticmethod
    def start(config):
        """Mock start - runs handler once with test data"""
        print("=== MOCK RUNPOD SERVERLESS STARTED ===")

        handler = config.get("handler")
        if not handler:
            print("ERROR: No handler provided")
            return

        # Create test job
        test_job = {
            "input": {
                "video_base64": "dGVzdCB2aWRlbyBkYXRh",  # "test video data" in base64
                "video_name": "test.mp4",
                "quality": "medium",
                "user_email": "test@example.com"
            }
        }

        print("Running handler with test job...")
        try:
            result = handler(test_job)
            print("Handler result:")
            print(f"  Status: {result.get('status')}")
            if result.get('status') == 'success':
                print("  [OK] Handler executed successfully")
            else:
                print(f"  [ERROR] {result.get('error')}")
        except Exception as e:
            print(f"ERROR running handler: {e}")
            import traceback
            traceback.print_exc()

# Create mock runpod module structure
class MockRunPod:
    def __init__(self):
        self.serverless = MockServerless()

# Export as runpod
runpod = MockRunPod()