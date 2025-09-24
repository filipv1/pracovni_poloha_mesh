#!/usr/bin/env python3
"""
Local test for the extended pipeline
Tests all components without RunPod
"""

import sys
import os
from pathlib import Path
import tempfile
import json

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

def test_imports():
    """Test that all required modules can be imported"""
    print("=" * 50)
    print("TESTING IMPORTS")
    print("=" * 50)

    errors = []

    # Test each import
    try:
        import run_production_simple_p
        print("[OK] run_production_simple_p imported")
    except ImportError as e:
        errors.append(f"[FAIL] run_production_simple_p: {e}")
        print(f"[FAIL] run_production_simple_p: {e}")

    try:
        import create_combined_angles_csv_skin
        print("[OK] create_combined_angles_csv_skin imported")

        # Check function exists
        if hasattr(create_combined_angles_csv_skin, 'create_combined_angles_csv_skin'):
            print("  [OK] Function create_combined_angles_csv_skin exists")
        else:
            errors.append("  [FAIL] Function create_combined_angles_csv_skin not found")
            print("  [FAIL] Function create_combined_angles_csv_skin not found")

    except ImportError as e:
        errors.append(f"[FAIL] create_combined_angles_csv_skin: {e}")
        print(f"[FAIL] create_combined_angles_csv_skin: {e}")

    try:
        import ergonomic_time_analysis
        print("[OK] ergonomic_time_analysis imported")

        # Check class exists
        if hasattr(ergonomic_time_analysis, 'ErgonomicTimeAnalyzer'):
            print("  [OK] Class ErgonomicTimeAnalyzer exists")
        else:
            errors.append("  [FAIL] Class ErgonomicTimeAnalyzer not found")
            print("  [FAIL] Class ErgonomicTimeAnalyzer not found")

    except ImportError as e:
        errors.append(f"[FAIL] ergonomic_time_analysis: {e}")
        print(f"[FAIL] ergonomic_time_analysis: {e}")

    try:
        import generate_4videos_from_pkl
        print("[OK] generate_4videos_from_pkl imported")

        # Check class exists
        if hasattr(generate_4videos_from_pkl, 'VideoGeneratorFromPKL'):
            print("  [OK] Class VideoGeneratorFromPKL exists")
        else:
            errors.append("  [FAIL] Class VideoGeneratorFromPKL not found")
            print("  [FAIL] Class VideoGeneratorFromPKL not found")

    except ImportError as e:
        errors.append(f"[FAIL] generate_4videos_from_pkl: {e}")
        print(f"[FAIL] generate_4videos_from_pkl: {e}")

    print("\n" + "=" * 50)
    if errors:
        print(f"ERRORS FOUND: {len(errors)}")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("ALL IMPORTS SUCCESSFUL!")
        return True

def test_pkl_structure():
    """Test if we can read and understand PKL file structure"""
    print("\n" + "=" * 50)
    print("TESTING PKL FILE STRUCTURE")
    print("=" * 50)

    # Look for a test PKL file
    test_pkl = None
    for pkl_file in Path(parent_dir).glob("*.pkl"):
        print(f"Found PKL file: {pkl_file}")
        test_pkl = pkl_file
        break

    if not test_pkl:
        print("No PKL file found for testing")
        return False

    try:
        import pickle
        with open(test_pkl, 'rb') as f:
            data = pickle.load(f)

        # Check structure
        if isinstance(data, dict) and 'mesh_sequence' in data:
            print("[OK] New PKL format with metadata")
            meshes = data['mesh_sequence']
            metadata = data.get('metadata', {})
            print(f"  Frames: {len(meshes)}")
            print(f"  FPS: {metadata.get('fps', 'unknown')}")
        elif isinstance(data, list):
            print("[OK] Old PKL format (list of meshes)")
            meshes = data
            print(f"  Frames: {len(meshes)}")
        else:
            print("[FAIL] Unknown PKL format")
            return False

        # Check mesh structure
        if meshes and len(meshes) > 0:
            first_mesh = meshes[0]
            if first_mesh and 'vertices' in first_mesh and 'joints' in first_mesh:
                print(f"[OK] Mesh structure valid")
                print(f"  Vertices: {len(first_mesh['vertices'])}")
                print(f"  Joints: {len(first_mesh['joints'])}")
            else:
                print("[FAIL] Invalid mesh structure")
                return False

        return True

    except Exception as e:
        print(f"[FAIL] Error reading PKL: {e}")
        return False

def test_csv_generation():
    """Test if CSV generation would work with dummy data"""
    print("\n" + "=" * 50)
    print("TESTING CSV GENERATION")
    print("=" * 50)

    try:
        # Create dummy mesh data
        import numpy as np

        dummy_mesh = {
            'vertices': np.random.randn(10475, 3),  # SMPL-X has 10475 vertices
            'joints': np.random.randn(55, 3),  # SMPL-X has 55 joints
            'faces': []
        }

        # Save to temp PKL
        import pickle
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            pickle.dump([dummy_mesh] * 10, tmp)  # 10 frames
            temp_pkl = tmp.name

        print(f"Created temp PKL: {temp_pkl}")

        # Try to generate CSV (this will fail if dependencies are missing)
        import create_combined_angles_csv_skin

        temp_csv = temp_pkl.replace('.pkl', '.csv')
        try:
            result = create_combined_angles_csv_skin.create_combined_angles_csv_skin(
                pkl_file=temp_pkl,
                output_csv=temp_csv,
                lumbar_vertex=5614
            )

            if Path(temp_csv).exists():
                print(f"[OK] CSV generated: {temp_csv}")
                os.unlink(temp_csv)
                return True
            else:
                print("[FAIL] CSV generation failed")
                return False

        except Exception as e:
            print(f"[FAIL] CSV generation error: {e}")
            return False
        finally:
            if Path(temp_pkl).exists():
                os.unlink(temp_pkl)

    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False

def test_handler_structure():
    """Test handler_v3.py structure"""
    print("\n" + "=" * 50)
    print("TESTING HANDLER STRUCTURE")
    print("=" * 50)

    handler_path = Path(__file__).parent / "runpod" / "handler_v3.py"

    if not handler_path.exists():
        print(f"[FAIL] Handler not found: {handler_path}")
        return False

    print(f"[OK] Handler found: {handler_path}")

    # Check if handler has proper error handling
    with open(handler_path, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ('try/except around imports', 'except ImportError'),
        ('process_video_async function', 'def process_video_async'),
        ('4-step pipeline', 'Step 4/4'),
        ('results upload', 'results[\'pkl_url\']'),
        ('video upload loop', 'for idx, video_file'),
    ]

    all_good = True
    for check_name, check_string in checks:
        if check_string in content:
            print(f"  [OK] {check_name}")
        else:
            print(f"  [FAIL] {check_name} not found")
            all_good = False

    return all_good

def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("LOCAL PIPELINE TEST")
    print("=" * 50)

    results = {
        'imports': test_imports(),
        'pkl_structure': test_pkl_structure(),
        'csv_generation': test_csv_generation(),
        'handler_structure': test_handler_structure()
    }

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    for test_name, result in results.items():
        status = "[OK] PASSED" if result else "[FAIL] FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("Pipeline should work on RunPod")
    else:
        print("SOME TESTS FAILED!")
        print("Fix the issues before deploying to RunPod")
    print("=" * 50)

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())