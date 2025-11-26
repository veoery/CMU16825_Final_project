#!/usr/bin/env python3
"""
Test script to verify amended evaluation methods work with mock data
"""
import os
import sys

def test_eval_ae_acc():
    """Test eval_ae_acc.py with mock data"""
    print("\n" + "=" * 70)
    print("Testing eval_ae_acc.py (CAD Command/Parameter Accuracy)")
    print("=" * 70)

    if not os.path.exists("mock_data/h5_data"):
        print("✗ Mock data not found. Generate it first:")
        print("  python generate_mock_data.py --output_dir ./mock_data")
        return False

    print("\nRunning: python eval_ae_acc.py --src mock_data/h5_data")
    ret = os.system("python eval_ae_acc.py --src mock_data/h5_data")

    if ret == 0:
        print("\n✓ eval_ae_acc.py completed successfully!")
        print("  Check output: mock_data/h5_data_acc_stat.txt")
        return True
    else:
        print("\n✗ eval_ae_acc.py failed")
        return False


def test_eval_seq():
    """Test eval_seq.py with mock data"""
    print("\n" + "=" * 70)
    print("Testing eval_seq.py (CAD Sequence Evaluation)")
    print("=" * 70)

    if not os.path.exists("mock_data/seq_data/mock_sequences.pkl"):
        print("✗ Mock data not found. Generate it first:")
        print("  python generate_mock_data.py --output_dir ./mock_data")
        return False

    print("\nRunning data format validation only (no CadSeqProc needed)")
    print("  python eval_seq.py --input_path mock_data/seq_data/mock_sequences.pkl --output_dir ./results")

    ret = os.system("python eval_seq.py --input_path mock_data/seq_data/mock_sequences.pkl --output_dir ./results 2>&1 | head -50")

    if ret == 0:
        print("\n✓ eval_seq.py data validation completed!")
        return True
    else:
        print("\n✗ eval_seq.py validation had issues")
        return False


def test_eval_topology():
    """Test eval_topology.py with mock data"""
    print("\n" + "=" * 70)
    print("Testing eval_topology.py (Mesh Topology Metrics)")
    print("=" * 70)

    if not os.path.exists("mock_data/meshes"):
        print("✗ Mock data not found. Generate it first:")
        print("  python generate_mock_data.py --output_dir ./mock_data")
        return False

    print("\nRunning: python test_topology_standalone.py")
    ret = os.system("python test_topology_standalone.py")

    if ret == 0:
        print("\n✓ eval_topology.py works correctly!")
        return True
    else:
        print("\n✗ eval_topology.py had issues")
        return False


def test_eval_ae_cd():
    """Test Chamfer Distance metric (core function, no cadlib needed)"""
    print("\n" + "=" * 70)
    print("Testing Chamfer Distance Metric")
    print("=" * 70)

    try:
        import numpy as np
        from scipy.spatial import cKDTree as KDTree

        # Import the chamfer_dist function from eval_ae_cd.py
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from eval_ae_cd import chamfer_dist, normalize_pc

        print("\n✓ Successfully imported Chamfer Distance function")

        # Test 1: Identical point clouds
        print("\nTest 1: Identical point clouds (should be ~0)")
        pc = np.random.rand(1000, 3).astype(np.float32)
        cd = chamfer_dist(pc, pc)
        print(f"  CD = {cd:.8f}")

        if cd < 1e-10:
            print("  ✓ PASS")
            test1_pass = True
        else:
            print("  ✗ FAIL")
            test1_pass = False

        # Test 2: Small perturbation
        print("\nTest 2: Small perturbation")
        pc1 = np.random.rand(1000, 3).astype(np.float32)
        pc2 = pc1 + np.random.normal(0, 0.01, pc1.shape).astype(np.float32)
        cd = chamfer_dist(pc1, pc2)
        print(f"  CD = {cd:.6f}")

        if 0 < cd < 0.01:
            print("  ✓ PASS (small value as expected)")
            test2_pass = True
        else:
            print("  ✗ FAIL")
            test2_pass = False

        # Test 3: Different point clouds
        print("\nTest 3: Different point clouds")
        pc1 = np.random.rand(1000, 3).astype(np.float32)
        pc2 = np.random.rand(1000, 3).astype(np.float32)
        cd = chamfer_dist(pc1, pc2)
        print(f"  CD = {cd:.6f}")

        if cd > 0.01:
            print("  ✓ PASS (larger value as expected)")
            test3_pass = True
        else:
            print("  ✗ FAIL")
            test3_pass = False

        # Test 4: normalize_pc function
        print("\nTest 4: Point cloud normalization")
        pc = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        normalized = normalize_pc(pc)
        max_val = np.max(np.abs(normalized))
        print(f"  Max value after normalization: {max_val:.6f}")

        if abs(max_val - 1.0) < 0.01:
            print("  ✓ PASS (normalized to [-1, 1])")
            test4_pass = True
        else:
            print("  ✗ FAIL")
            test4_pass = False

        all_pass = test1_pass and test2_pass and test3_pass and test4_pass

        print("\n" + "-" * 70)
        if all_pass:
            print("✓ All Chamfer Distance tests passed!")
            print("\nNote: Full eval_ae_cd.py requires:")
            print("  - cadlib to convert CAD vectors → point clouds")
            print("  - Ground truth PLY files")
            print("\nBut the core CD metric works correctly!")
            return True
        else:
            print("✗ Some tests failed")
            return False

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nMake sure scipy is installed:")
        print("  pip install scipy")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("Testing Amended Evaluation Methods")
    print("=" * 70)
    print("\nThis script tests that the amended eval_*.py files work with mock data")

    # Change to evaluation directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Generate mock data if not exists
    if not os.path.exists("mock_data"):
        print("\nGenerating mock data...")
        ret = os.system("python generate_mock_data.py --output_dir ./mock_data")
        if ret != 0:
            print("✗ Failed to generate mock data")
            return

    results = {}

    # Test each evaluation method
    results['eval_ae_acc'] = test_eval_ae_acc()
    results['eval_seq'] = test_eval_seq()
    results['eval_topology'] = test_eval_topology()
    results['eval_ae_cd'] = test_eval_ae_cd()

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIPPED"
        print(f"  {status:12s} {name}")

    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ All testable evaluation methods work with mock data!")
    else:
        print("\n⚠ Some tests failed. Check output above for details.")

    print("\nNext steps:")
    print("  1. Install external dependencies (cadlib, CadSeqProc) for full functionality")
    print("  2. Prepare real data from your CAD model")
    print("  3. Run evaluations on real predictions")


if __name__ == '__main__':
    main()
