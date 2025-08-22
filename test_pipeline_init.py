#!/usr/bin/env python3
"""
Test just pipeline initialization
"""

import sys
from pathlib import Path

print("TESTING PIPELINE INITIALIZATION")
print("=" * 40)

try:
    # Import the pipeline from our RunPod-safe version
    sys.path.append('.')
    from run_production_simple import MasterPipeline
    
    print("OK Imported MasterPipeline")
    
    # Initialize pipeline
    device = 'cpu'  # Force CPU for consistency
    print(f"Using device: {device}")
    
    pipeline = MasterPipeline(
        smplx_path="models/smplx",
        device=device,
        gender='neutral'
    )
    
    print("OK Pipeline initialized successfully!")
    
except Exception as e:
    print(f"ERROR Pipeline initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("TEST COMPLETED")