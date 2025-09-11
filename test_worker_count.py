#!/usr/bin/env python3
"""
Quick test script to verify worker count logic
"""
import multiprocessing

def get_safe_worker_count():
    """Test the worker count calculation logic"""
    cpu_count = multiprocessing.cpu_count()
    print(f"System CPU count: {cpu_count}")
    
    # Apply the same logic from the fixed code
    if cpu_count >= 24:  # High-end systems like RunPod RTX 4090
        max_workers = min(8, cpu_count // 4)  # Cap at 8 workers for stability
        print(f"High-end system detected: Using {max_workers} workers (capped at 8)")
    elif cpu_count >= 12:  # Mid-range systems
        max_workers = min(6, cpu_count // 2)
        print(f"Mid-range system detected: Using {max_workers} workers")
    else:  # Low-end systems
        max_workers = max(1, cpu_count - 1)
        print(f"Low-end system detected: Using {max_workers} workers")
    
    print(f"\nBefore fix: Would have used {max(1, cpu_count - 1)} workers")
    print(f"After fix: Will use {max_workers} workers")
    print(f"Reduction: {((cpu_count - 1) - max_workers) / (cpu_count - 1) * 100:.1f}% fewer workers for stability")
    
    return max_workers

if __name__ == "__main__":
    print("WORKER COUNT TEST")
    print("=" * 50)
    get_safe_worker_count()
    print("\nThis should prevent the hanging issue on RunPod systems!")