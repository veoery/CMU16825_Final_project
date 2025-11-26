#!/usr/bin/env python3
"""
Quick viewer for mock data - simple text-based visualization
"""
import os
import numpy as np


def view_h5_file(h5_path):
    """Quick text view of H5 file"""
    import h5py

    CMD_NAMES = ['SOL', 'EOS', 'LINE', 'ARC', 'CIRCLE', 'EXT']

    print("\n" + "=" * 60)
    print(f"H5 File: {os.path.basename(h5_path)}")
    print("=" * 60)

    with h5py.File(h5_path, 'r') as f:
        out_vec = f['out_vec'][:]
        gt_vec = f['gt_vec'][:]

    print(f"\nShape: {gt_vec.shape} (sequence_length, num_params)")
    print(f"\nFirst 10 commands:")
    print(f"{'Index':<8} {'GT Cmd':<12} {'Pred Cmd':<12} {'Match':<8}")
    print("-" * 60)

    for i in range(min(10, len(gt_vec))):
        gt_cmd = int(gt_vec[i, 0])
        out_cmd = int(out_vec[i, 0])
        match = "✓" if gt_cmd == out_cmd else "✗"

        gt_name = CMD_NAMES[gt_cmd] if gt_cmd < len(CMD_NAMES) else f"UNK({gt_cmd})"
        out_name = CMD_NAMES[out_cmd] if out_cmd < len(CMD_NAMES) else f"UNK({out_cmd})"

        print(f"{i:<8} {gt_name:<12} {out_name:<12} {match:<8}")

        # Show parameters for non-SOL/EOS commands
        if gt_cmd not in [0, 1]:
            gt_params = gt_vec[i, 1:5]
            out_params = out_vec[i, 1:5]
            print(f"         GT params:   {gt_params}")
            print(f"         Pred params: {out_params}")

    # Summary
    accuracy = (gt_vec[:, 0] == out_vec[:, 0]).mean()
    print(f"\n✓ Command accuracy: {accuracy:.1%}")


def view_mesh(mesh_path):
    """Quick text view of mesh"""
    import trimesh

    print("\n" + "=" * 60)
    print(f"Mesh: {os.path.basename(mesh_path)}")
    print("=" * 60)

    mesh = trimesh.load(mesh_path)

    print(f"\nVertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Edges: {len(mesh.edges)}")
    print(f"\nBounds:")
    print(f"  X: [{mesh.vertices[:, 0].min():.2f}, {mesh.vertices[:, 0].max():.2f}]")
    print(f"  Y: [{mesh.vertices[:, 1].min():.2f}, {mesh.vertices[:, 1].max():.2f}]")
    print(f"  Z: [{mesh.vertices[:, 2].min():.2f}, {mesh.vertices[:, 2].max():.2f}]")

    print(f"\nProperties:")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume: {mesh.volume:.4f}")
    print(f"  Surface area: {mesh.area:.4f}")

    # Show topology metrics
    try:
        from eval_topology import dangling_edge_length, seg_error
        dang_el = dangling_edge_length(mesh)
        print(f"\n✓ Dangling Edge Length: {dang_el:.4f}")
    except:
        pass


def view_pickle(pkl_path):
    """Quick text view of pickle data"""
    import pickle

    print("\n" + "=" * 60)
    print(f"Pickle: {os.path.basename(pkl_path)}")
    print("=" * 60)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    uids = list(data.keys())
    print(f"\nTotal UIDs: {len(uids)}")

    # Show first UID
    if len(uids) > 0:
        uid = uids[0]
        levels = list(data[uid].keys())

        print(f"\nExample UID: {uid}")
        print(f"Levels: {levels}")

        for level in levels[:2]:
            print(f"\n  {level}:")
            print(f"    GT CAD vec: {data[uid][level]['gt_cad_vec'].shape}")
            print(f"    Predictions: {len(data[uid][level]['pred_cad_vec'])}")
            print(f"    Chamfer distances: {[f'{cd:.4f}' for cd in data[uid][level]['cd']]}")


def main():
    import glob

    print("Quick Data Viewer")
    print("=" * 60)

    if not os.path.exists('mock_data'):
        print("\n✗ Mock data not found. Generate it first:")
        print("  python generate_mock_data.py --output_dir ./mock_data")
        return

    # View one of each type
    print("\n1. H5 FILE (CAD Commands):")
    h5_files = glob.glob('mock_data/h5_data/*.h5')
    if h5_files:
        view_h5_file(h5_files[0])

    print("\n2. MESH FILE (3D Geometry):")
    mesh_files = glob.glob('mock_data/meshes/*.obj')
    if mesh_files:
        view_mesh(mesh_files[0])

    print("\n3. PICKLE FILE (Sequences):")
    pkl_path = 'mock_data/seq_data/mock_sequences.pkl'
    if os.path.exists(pkl_path):
        view_pickle(pkl_path)

    print("\n" + "=" * 60)
    print("For graphical visualization, run:")
    print("  python visualize_mock_data.py")


if __name__ == '__main__':
    main()
