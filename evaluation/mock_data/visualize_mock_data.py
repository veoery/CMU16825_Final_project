#!/usr/bin/env python3
"""
Visualize mock data for CAD evaluation

Shows:
1. CAD command sequences from H5 files
2. 3D meshes from OBJ files
3. Data structure from pickle files
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_h5_commands(h5_path, num_files=3):
    """Visualize CAD command sequences from H5 files"""
    import h5py
    import glob

    print("\n" + "=" * 70)
    print("1. CAD Command Sequences (H5 Files)")
    print("=" * 70)

    h5_files = sorted(glob.glob(os.path.join(h5_path, "*.h5")))[:num_files]

    if not h5_files:
        print(f"✗ No H5 files found in {h5_path}")
        return

    # Command names (mock definitions)
    CMD_NAMES = ['SOL', 'EOS', 'LINE', 'ARC', 'CIRCLE', 'EXT']

    fig, axes = plt.subplots(len(h5_files), 2, figsize=(14, 4*len(h5_files)))
    if len(h5_files) == 1:
        axes = axes.reshape(1, -1)

    for idx, h5_file in enumerate(h5_files):
        with h5py.File(h5_file, 'r') as f:
            out_vec = f['out_vec'][:]
            gt_vec = f['gt_vec'][:]

        filename = os.path.basename(h5_file)

        # Plot ground truth commands
        ax1 = axes[idx, 0]
        gt_cmds = gt_vec[:, 0]
        cmd_counts = np.bincount(gt_cmds.astype(int), minlength=len(CMD_NAMES))

        ax1.bar(range(len(CMD_NAMES)), cmd_counts, color='steelblue', alpha=0.7)
        ax1.set_xticks(range(len(CMD_NAMES)))
        ax1.set_xticklabels(CMD_NAMES, rotation=45)
        ax1.set_ylabel('Count')
        ax1.set_title(f'{filename}\nGround Truth Commands')
        ax1.grid(axis='y', alpha=0.3)

        # Plot prediction commands
        ax2 = axes[idx, 1]
        out_cmds = out_vec[:, 0]
        out_counts = np.bincount(out_cmds.astype(int), minlength=len(CMD_NAMES))

        ax2.bar(range(len(CMD_NAMES)), out_counts, color='coral', alpha=0.7)
        ax2.set_xticks(range(len(CMD_NAMES)))
        ax2.set_xticklabels(CMD_NAMES, rotation=45)
        ax2.set_ylabel('Count')
        ax2.set_title(f'{filename}\nPredicted Commands')
        ax2.grid(axis='y', alpha=0.3)

        # Print summary
        accuracy = (gt_cmds == out_cmds).mean()
        print(f"\n{filename}:")
        print(f"  Sequence length: {len(gt_cmds)}")
        print(f"  Command accuracy: {accuracy:.1%}")
        print(f"  GT commands: {dict(zip(CMD_NAMES, cmd_counts))}")
        print(f"  Pred commands: {dict(zip(CMD_NAMES, out_counts))}")

    plt.tight_layout()
    plt.savefig('mock_data_commands.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: mock_data_commands.png")
    plt.show()


def visualize_meshes(mesh_path, num_meshes=5):
    """Visualize 3D meshes"""
    try:
        import trimesh
    except ImportError:
        print("\n✗ trimesh not installed. Run: pip install trimesh")
        return

    print("\n" + "=" * 70)
    print("2. 3D Meshes")
    print("=" * 70)

    import glob
    mesh_files = sorted(glob.glob(os.path.join(mesh_path, "*.obj")))[:num_meshes]

    if not mesh_files:
        print(f"✗ No mesh files found in {mesh_path}")
        return

    fig = plt.figure(figsize=(15, 3 * ((len(mesh_files) + 2) // 3)))

    for idx, mesh_file in enumerate(mesh_files):
        mesh = trimesh.load(mesh_file)
        filename = os.path.basename(mesh_file)

        # Create 3D subplot
        ax = fig.add_subplot((len(mesh_files) + 2) // 3, 3, idx + 1, projection='3d')

        # Plot mesh
        vertices = mesh.vertices
        faces = mesh.faces

        # Plot vertices
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                  c='lightblue', s=1, alpha=0.3)

        # Plot edges
        for face in faces:
            for i in range(3):
                start = vertices[face[i]]
                end = vertices[face[(i+1)%3]]
                ax.plot([start[0], end[0]],
                       [start[1], end[1]],
                       [start[2], end[2]],
                       'b-', linewidth=0.5, alpha=0.6)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{filename}\n{len(vertices)} verts, {len(faces)} faces')

        # Equal aspect ratio
        max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                             vertices[:, 1].max()-vertices[:, 1].min(),
                             vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
        mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        print(f"\n{filename}:")
        print(f"  Vertices: {len(vertices)}")
        print(f"  Faces: {len(faces)}")
        print(f"  Bounds: [{vertices.min():.2f}, {vertices.max():.2f}]")
        print(f"  Watertight: {mesh.is_watertight}")

    plt.tight_layout()
    plt.savefig('mock_data_meshes.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: mock_data_meshes.png")
    plt.show()


def visualize_sequence_data(pkl_path):
    """Visualize sequence data structure from pickle"""
    import pickle

    print("\n" + "=" * 70)
    print("3. Sequence Data Structure (Pickle)")
    print("=" * 70)

    if not os.path.exists(pkl_path):
        print(f"✗ Pickle file not found: {pkl_path}")
        return

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    uids = list(data.keys())
    print(f"\nTotal UIDs: {len(uids)}")

    # Analyze first UID
    if len(uids) > 0:
        uid = uids[0]
        levels = list(data[uid].keys())

        print(f"\nSample UID: {uid}")
        print(f"Levels: {levels}")

        # Plot sequence lengths across levels
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Sequence lengths by level
        ax1 = axes[0, 0]
        seq_lengths = []
        level_names = []
        for level in levels:
            gt_vec = data[uid][level]['gt_cad_vec']
            seq_lengths.append(len(gt_vec))
            level_names.append(level)

        ax1.bar(level_names, seq_lengths, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Sequence Length')
        ax1.set_title('CAD Sequence Length by Level')
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Chamfer distances
        ax2 = axes[0, 1]
        for level in levels[:2]:  # First 2 levels
            cds = data[uid][level]['cd']
            ax2.plot(range(len(cds)), cds, marker='o', label=level)
        ax2.set_xlabel('Prediction Index')
        ax2.set_ylabel('Chamfer Distance')
        ax2.set_title('Chamfer Distance for Multiple Predictions')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Plot 3: Parameter distribution (first level)
        ax3 = axes[1, 0]
        level = levels[0]
        gt_vec = data[uid][level]['gt_cad_vec']
        pred_vec = data[uid][level]['pred_cad_vec'][0]

        # Plot first few parameters
        params_to_plot = min(5, gt_vec.shape[1])
        x = np.arange(params_to_plot)
        width = 0.35

        gt_means = [gt_vec[:, i].mean() for i in range(params_to_plot)]
        pred_means = [pred_vec[:, i].mean() for i in range(params_to_plot)]

        ax3.bar(x - width/2, gt_means, width, label='Ground Truth', alpha=0.7)
        ax3.bar(x + width/2, pred_means, width, label='Prediction', alpha=0.7)
        ax3.set_xlabel('Parameter Index')
        ax3.set_ylabel('Mean Value')
        ax3.set_title(f'Parameter Comparison ({level})')
        ax3.set_xticks(x)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # Plot 4: Statistics across all UIDs
        ax4 = axes[1, 1]
        all_cds = []
        for uid in uids:
            for level in data[uid].keys():
                all_cds.extend(data[uid][level]['cd'])

        ax4.hist(all_cds, bins=20, color='coral', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Chamfer Distance')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'CD Distribution (all {len(uids)} UIDs)')
        ax4.axvline(np.median(all_cds), color='red', linestyle='--',
                   label=f'Median: {np.median(all_cds):.4f}')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('mock_data_sequences.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization: mock_data_sequences.png")
        plt.show()

        # Print detailed info
        print(f"\nDetailed structure for {uid}:")
        for level in levels[:2]:  # Show first 2 levels
            print(f"\n  {level}:")
            print(f"    gt_cad_vec shape: {data[uid][level]['gt_cad_vec'].shape}")
            print(f"    pred_cad_vec: {len(data[uid][level]['pred_cad_vec'])} predictions")
            print(f"    cd values: {data[uid][level]['cd']}")


def main():
    print("=" * 70)
    print("Mock Data Visualization")
    print("=" * 70)

    # Check if mock data exists
    if not os.path.exists('mock_data'):
        print("\n✗ Mock data not found. Generate it first:")
        print("  python generate_mock_data.py --output_dir ./mock_data")
        return

    print("\nThis will create 3 visualization files:")
    print("  1. mock_data_commands.png - CAD command sequences")
    print("  2. mock_data_meshes.png - 3D mesh visualizations")
    print("  3. mock_data_sequences.png - Sequence data analysis")

    # Visualize each type of mock data
    try:
        visualize_h5_commands('mock_data/h5_data', num_files=3)
    except Exception as e:
        print(f"\n✗ Error visualizing H5 data: {e}")

    try:
        visualize_meshes('mock_data/meshes', num_meshes=5)
    except Exception as e:
        print(f"\n✗ Error visualizing meshes: {e}")

    try:
        visualize_sequence_data('mock_data/seq_data/mock_sequences.pkl')
    except Exception as e:
        print(f"\n✗ Error visualizing sequence data: {e}")

    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - mock_data_commands.png")
    print("  - mock_data_meshes.png")
    print("  - mock_data_sequences.png")
    print("\nThese show what the mock data looks like and what the")
    print("evaluation methods are measuring.")


if __name__ == '__main__':
    main()
