#!/usr/bin/env python3
"""
Test script for evaluation methods using mock data

This demonstrates how to use the mock data with evaluation scripts.
"""

import sys
sys.path.append('..')

# Example 1: Test topology evaluation (standalone)
print("=" * 60)
print("Testing eval_topology.py (standalone - no external dependencies)")
print("=" * 60)

try:
    import trimesh
    from eval_topology import seg_error, dangling_edge_length, self_intersection_ratio, flux_enclosure_error

    # Load mock meshes
    mesh1 = trimesh.load('mock_data/meshes/mock_mesh_000.obj')
    mesh2 = trimesh.load('mock_data/meshes/mock_mesh_001.obj')

    print(f"Segment Error: {seg_error(mesh1, mesh2):.4f}")
    print(f"Dangling Edge Length: {dangling_edge_length(mesh1):.4f}")
    print(f"Self-Intersection Ratio: {self_intersection_ratio(mesh1):.4f}")
    print(f"Flux Enclosure Error: {flux_enclosure_error(mesh1):.4f}")
    print("✓ Topology evaluation successful!\n")
except ImportError as e:
    print(f"✗ Missing dependency: {e}\n")

# Example 2: Test H5 data loading
print("=" * 60)
print("Testing H5 data format (for eval_ae_acc.py)")
print("=" * 60)

try:
    import h5py
    import glob

    h5_files = glob.glob('mock_data/h5_data/*.h5')
    if h5_files:
        with h5py.File(h5_files[0], 'r') as f:
            out_vec = f['out_vec'][:]
            gt_vec = f['gt_vec'][:]
            print(f"Loaded H5 file: {h5_files[0]}")
            print(f"  out_vec shape: {out_vec.shape}")
            print(f"  gt_vec shape: {gt_vec.shape}")
            print(f"  Command accuracy: {(out_vec[:, 0] == gt_vec[:, 0]).mean():.2%}")
            print("✓ H5 data loading successful!\n")
except Exception as e:
    print(f"✗ Error: {e}\n")

# Example 3: Test pickle data loading
print("=" * 60)
print("Testing pickle data format (for eval_seq.py)")
print("=" * 60)

try:
    import pickle

    with open('mock_data/seq_data/mock_sequences.pkl', 'rb') as f:
        data = pickle.load(f)

    uids = list(data.keys())
    print(f"Loaded pickle file with {len(uids)} UIDs")

    # Check structure
    first_uid = uids[0]
    levels = list(data[first_uid].keys())
    print(f"  UID example: {first_uid}")
    print(f"  Levels: {levels}")
    print(f"  Keys in level_1: {list(data[first_uid]['level_1'].keys())}")
    print(f"  GT CAD vec shape: {data[first_uid]['level_1']['gt_cad_vec'].shape}")
    print(f"  Number of predictions: {len(data[first_uid]['level_1']['pred_cad_vec'])}")
    print("✓ Pickle data loading successful!\n")
except Exception as e:
    print(f"✗ Error: {e}\n")

print("=" * 60)
print("Mock data generation and testing complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Install missing dependencies (cadlib, CadSeqProc) to run full evaluations")
print("2. Replace mock data with real predictions from your model")
print("3. Run evaluation scripts:")
print("   - python eval_topology.py  (works with current dependencies)")
print("   - python eval_ae_acc.py --src mock_data/h5_data  (requires cadlib)")
print("   - python eval_seq.py --input_path mock_data/seq_data/mock_sequences.pkl --output_dir ./results  (requires CadSeqProc)")
