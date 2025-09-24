"""
Local testing wrapper for RunPod handler
Simulates RunPod environment for local development
"""

import json
import base64
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, '/app/code')

# Import the actual handler
from handler import handler

def test_handler():
    """Test the handler with a sample video"""

    print("=== Local Handler Test ===")

    # Check if test video exists
    test_video_path = Path('/app/test/sample_video.mp4')

    if test_video_path.exists():
        print(f"Using test video: {test_video_path}")
        with open(test_video_path, 'rb') as f:
            video_data = f.read()
        video_base64 = base64.b64encode(video_data).decode('utf-8')
    else:
        print("No test video found. Creating minimal test data...")
        # Create minimal test data
        video_base64 = base64.b64encode(b"test_video_data").decode('utf-8')

    # Create test job
    test_job = {
        "input": {
            "video_base64": video_base64,
            "video_name": "test_video.mp4",
            "quality": "medium",
            "user_email": "test@example.com"
        }
    }

    print("Processing test job...")

    # Mock RunPod progress update function
    class MockRunPod:
        @staticmethod
        def progress_update(job, progress, message):
            print(f"[{progress}%] {message}")

    # Replace runpod module
    import runpod
    runpod.serverless = MockRunPod()

    try:
        # Run handler
        result = handler(test_job)

        # Print results
        print("\n=== Handler Results ===")
        print(f"Status: {result.get('status')}")

        if result['status'] == 'success':
            print(f"PKL size: {len(result.get('pkl_base64', '')) / 1024:.2f} KB")
            print(f"XLSX size: {len(result.get('xlsx_base64', '')) / 1024:.2f} KB")
            print(f"Statistics: {json.dumps(result.get('statistics', {}), indent=2)}")

            # Save results to output
            output_dir = Path('/app/output')
            output_dir.mkdir(exist_ok=True)

            if 'pkl_base64' in result:
                pkl_path = output_dir / 'test_result.pkl'
                pkl_path.write_bytes(base64.b64decode(result['pkl_base64']))
                print(f"Saved PKL to: {pkl_path}")

            if 'xlsx_base64' in result:
                xlsx_path = output_dir / 'test_result.xlsx'
                xlsx_path.write_bytes(base64.b64decode(result['xlsx_base64']))
                print(f"Saved XLSX to: {xlsx_path}")

        else:
            print(f"Error: {result.get('error')}")
            if 'traceback' in result:
                print(f"Traceback:\n{result['traceback']}")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_handler()