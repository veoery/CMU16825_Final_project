#!/usr/bin/env python3
"""
Mock Data Generator for CAD Evaluation Scripts

This script generates mock data to test evaluation methods without requiring
the full dataset or external dependencies.

Usage:
    python generate_mock_data.py --output_dir ./mock_data
"""

import argparse
import h5py
import numpy as np
import pickle
import os
from pathlib import Path


def generate_mock_h5_data(output_path, num_sequences=10, seq_length=50, num_params=14):
    """
    Generate mock H5 files for eval_ae_acc.py and eval_ae_cd.py

    Args:
        output_path: Path to save H5 files
        num_sequences: Number of H5 files to generate
        seq_length: Length of each CAD command sequence
        num_params: Number of parameters per command (default 14)
    """
    os.makedirs(output_path, exist_ok=True)

    # Mock command indices (based on typical CAD command sets)
    SOL_IDX = 0  # Start of Line
    EOS_IDX = 1  # End of Sequence
    LINE_IDX = 2
    ARC_IDX = 3
    CIRCLE_IDX = 4
    EXT_IDX = 5  # Extrude

    ALL_COMMANDS = [SOL_IDX, EOS_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX, EXT_IDX]

    for i in range(num_sequences):
        filename = f"mock_cad_{i:08d}.h5"
        filepath = os.path.join(output_path, filename)

        # Generate mock CAD command sequence
        out_vec = np.zeros((seq_length, num_params + 1), dtype=np.int32)
        gt_vec = np.zeros((seq_length, num_params + 1), dtype=np.int32)

        for j in range(seq_length):
            # Command (column 0)
            if j == 0:
                cmd = SOL_IDX
            elif j == seq_length - 1:
                cmd = EOS_IDX
            else:
                cmd = np.random.choice([LINE_IDX, ARC_IDX, CIRCLE_IDX, EXT_IDX])

            gt_vec[j, 0] = cmd

            # Add some noise to output (90% accuracy)
            if np.random.random() < 0.9:
                out_vec[j, 0] = cmd
            else:
                out_vec[j, 0] = np.random.choice(ALL_COMMANDS)

            # Parameters (columns 1+)
            if cmd not in [SOL_IDX, EOS_IDX]:
                params = np.random.randint(0, 256, size=num_params)
                gt_vec[j, 1:] = params

                # Add noise to parameters (tolerance of 3)
                noise = np.random.randint(-2, 3, size=num_params)
                out_vec[j, 1:] = np.clip(params + noise, 0, 255)

        # Save to H5 file
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('out_vec', data=out_vec)
            f.create_dataset('gt_vec', data=gt_vec)

        print(f"Generated: {filepath}")

    print(f"\nGenerated {num_sequences} H5 files in {output_path}")
    return output_path


def generate_mock_pkl_data(output_path, num_uids=5, num_levels=4):
    """
    Generate mock pickle file for eval_seq.py

    Args:
        output_path: Path to save pickle file
        num_uids: Number of unique CAD designs
        num_levels: Number of hierarchy levels
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    data = {}

    for uid_idx in range(num_uids):
        uid = f"uid_{uid_idx:05d}"
        data[uid] = {}

        for level in range(1, num_levels + 1):
            level_key = f"level_{level}"

            # Generate mock CAD vector (simplified)
            seq_length = np.random.randint(20, 50)
            gt_cad_vec = np.random.randint(0, 256, size=(seq_length, 15))

            # Generate multiple predictions
            num_predictions = 3
            pred_cad_vecs = []
            cds = []

            for _ in range(num_predictions):
                # Add noise to ground truth
                pred_vec = gt_cad_vec + np.random.randint(-5, 6, size=gt_cad_vec.shape)
                pred_vec = np.clip(pred_vec, 0, 255)
                pred_cad_vecs.append(pred_vec)

                # Mock Chamfer Distance (lower is better)
                cd = np.random.uniform(0.001, 0.1)
                cds.append(cd)

            data[uid][level_key] = {
                'gt_cad_vec': gt_cad_vec,
                'pred_cad_vec': pred_cad_vecs,
                'cd': cds
            }

    # Save to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Generated: {output_path}")
    print(f"  - {num_uids} UIDs")
    print(f"  - {num_levels} levels per UID")
    return output_path


def generate_mock_mesh_data(output_path, num_meshes=5):
    """
    Generate mock mesh files for eval_topology.py

    Note: Requires trimesh library. This generates simple procedural meshes.
    """
    try:
        import trimesh
    except ImportError:
        print("Warning: trimesh not installed. Skipping mesh generation.")
        print("Install with: pip install trimesh")
        return None

    os.makedirs(output_path, exist_ok=True)

    for i in range(num_meshes):
        # Create simple geometric primitives
        if i % 3 == 0:
            mesh = trimesh.creation.box(extents=[1, 1, 1])
        elif i % 3 == 1:
            mesh = trimesh.creation.cylinder(radius=0.5, height=1.0)
        else:
            mesh = trimesh.creation.icosphere(subdivisions=2)

        # Add some random transformations
        mesh.apply_translation(np.random.uniform(-0.1, 0.1, 3))

        # Save mesh
        filepath = os.path.join(output_path, f"mock_mesh_{i:03d}.obj")
        mesh.export(filepath)
        print(f"Generated: {filepath}")

    print(f"\nGenerated {num_meshes} mesh files in {output_path}")
    return output_path


def generate_test_script(output_dir):
    """Generate a test script to demonstrate using the mock data"""
    test_script = f"""#!/usr/bin/env python3
\"\"\"
Test script for evaluation methods using mock data

This demonstrates how to use the mock data with evaluation scripts.
\"\"\"

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
    mesh1 = trimesh.load('{output_dir}/meshes/mock_mesh_000.obj')
    mesh2 = trimesh.load('{output_dir}/meshes/mock_mesh_001.obj')

    print(f"Segment Error: {{seg_error(mesh1, mesh2):.4f}}")
    print(f"Dangling Edge Length: {{dangling_edge_length(mesh1):.4f}}")
    print(f"Self-Intersection Ratio: {{self_intersection_ratio(mesh1):.4f}}")
    print(f"Flux Enclosure Error: {{flux_enclosure_error(mesh1):.4f}}")
    print("✓ Topology evaluation successful!\\n")
except ImportError as e:
    print(f"✗ Missing dependency: {{e}}\\n")

# Example 2: Test H5 data loading
print("=" * 60)
print("Testing H5 data format (for eval_ae_acc.py)")
print("=" * 60)

try:
    import h5py
    import glob

    h5_files = glob.glob('{output_dir}/h5_data/*.h5')
    if h5_files:
        with h5py.File(h5_files[0], 'r') as f:
            out_vec = f['out_vec'][:]
            gt_vec = f['gt_vec'][:]
            print(f"Loaded H5 file: {{h5_files[0]}}")
            print(f"  out_vec shape: {{out_vec.shape}}")
            print(f"  gt_vec shape: {{gt_vec.shape}}")
            print(f"  Command accuracy: {{(out_vec[:, 0] == gt_vec[:, 0]).mean():.2%}}")
            print("✓ H5 data loading successful!\\n")
except Exception as e:
    print(f"✗ Error: {{e}}\\n")

# Example 3: Test pickle data loading
print("=" * 60)
print("Testing pickle data format (for eval_seq.py)")
print("=" * 60)

try:
    import pickle

    with open('{output_dir}/seq_data/mock_sequences.pkl', 'rb') as f:
        data = pickle.load(f)

    uids = list(data.keys())
    print(f"Loaded pickle file with {{len(uids)}} UIDs")

    # Check structure
    first_uid = uids[0]
    levels = list(data[first_uid].keys())
    print(f"  UID example: {{first_uid}}")
    print(f"  Levels: {{levels}}")
    print(f"  Keys in level_1: {{list(data[first_uid]['level_1'].keys())}}")
    print(f"  GT CAD vec shape: {{data[first_uid]['level_1']['gt_cad_vec'].shape}}")
    print(f"  Number of predictions: {{len(data[first_uid]['level_1']['pred_cad_vec'])}}")
    print("✓ Pickle data loading successful!\\n")
except Exception as e:
    print(f"✗ Error: {{e}}\\n")

print("=" * 60)
print("Mock data generation and testing complete!")
print("=" * 60)
print("\\nNext steps:")
print("1. Install missing dependencies (cadlib, CadSeqProc) to run full evaluations")
print("2. Replace mock data with real predictions from your model")
print("3. Run evaluation scripts:")
print("   - python eval_topology.py  (works with current dependencies)")
print("   - python eval_ae_acc.py --src {output_dir}/h5_data  (requires cadlib)")
print("   - python eval_seq.py --input_path {output_dir}/seq_data/mock_sequences.pkl --output_dir ./results  (requires CadSeqProc)")
"""

    test_script_path = os.path.join(output_dir, 'test_mock_data.py')
    with open(test_script_path, 'w') as f:
        f.write(test_script)

    os.chmod(test_script_path, 0o755)
    print(f"\nGenerated test script: {test_script_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate mock data for CAD evaluation')
    parser.add_argument('--output_dir', type=str, default='./mock_data',
                        help='Directory to save mock data')
    parser.add_argument('--num_h5', type=int, default=10,
                        help='Number of H5 files to generate')
    parser.add_argument('--num_uids', type=int, default=5,
                        help='Number of UIDs in pickle file')
    parser.add_argument('--num_meshes', type=int, default=5,
                        help='Number of mesh files to generate')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Mock Data for CAD Evaluation")
    print("=" * 60)

    # Generate different types of mock data
    print("\n1. Generating H5 data (for eval_ae_acc.py and eval_ae_cd.py)...")
    h5_dir = output_dir / 'h5_data'
    generate_mock_h5_data(str(h5_dir), num_sequences=args.num_h5)

    print("\n2. Generating pickle data (for eval_seq.py)...")
    pkl_dir = output_dir / 'seq_data'
    pkl_path = pkl_dir / 'mock_sequences.pkl'
    generate_mock_pkl_data(str(pkl_path), num_uids=args.num_uids)

    print("\n3. Generating mesh data (for eval_topology.py)...")
    mesh_dir = output_dir / 'meshes'
    generate_mock_mesh_data(str(mesh_dir), num_meshes=args.num_meshes)

    print("\n4. Generating test script...")
    generate_test_script(str(output_dir))

    print("\n" + "=" * 60)
    print("Mock data generation complete!")
    print("=" * 60)
    print(f"\nGenerated files in: {output_dir}")
    print(f"  - {args.num_h5} H5 files in h5_data/")
    print(f"  - 1 pickle file in seq_data/")
    print(f"  - {args.num_meshes} mesh files in meshes/")
    print(f"\nTo test the mock data:")
    print(f"  cd {output_dir}")
    print(f"  python test_mock_data.py")


if __name__ == '__main__':
    main()
