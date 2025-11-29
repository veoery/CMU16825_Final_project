#!/usr/bin/env python3
"""
Simple script to visualize and save screenshots of STEP files.
Uses trimesh's built-in rendering.
"""

import os
import glob
import argparse
import trimesh
import numpy as np
from pathlib import Path

# Try importing rendering libraries
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def visualize_mesh(mesh_path, output_path, resolution=(800, 600)):
    """Load a mesh and save a screenshot."""
    try:
        # Load mesh
        mesh = trimesh.load(mesh_path, process=False)

        # Handle assemblies - take first mesh or combine
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return False, "Empty scene"
            # Combine all geometries
            meshes = [m for m in mesh.geometry.values() if isinstance(m, trimesh.Trimesh)]
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
            else:
                return False, "No valid meshes in scene"

        if not isinstance(mesh, trimesh.Trimesh):
            return False, "Not a valid mesh"

        # Use matplotlib for rendering
        if HAS_MATPLOTLIB:
            fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')

            # Plot mesh
            vertices = mesh.vertices
            faces = mesh.faces

            # Create 3D polygons
            mesh_plot = Poly3DCollection(vertices[faces], alpha=0.7, facecolor='cyan', edgecolor='darkblue', linewidth=0.5)
            ax.add_collection3d(mesh_plot)

            # Auto-scale
            ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])

            # Save figure
            fig.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return True, "Success"
        else:
            return False, "matplotlib not available"

    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Visualize and screenshot STEP files")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with STEP files')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for images')
    parser.add_argument('--format', type=str, default='step', help='File format')
    parser.add_argument('--resolution', type=int, nargs=2, default=[800, 600], help='Image resolution (width height)')

    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input_dir), 'imgs')

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Find STEP files
    mesh_pattern = os.path.join(args.input_dir, f'*.{args.format}')
    mesh_files = sorted(glob.glob(mesh_pattern))

    if not mesh_files:
        print(f"❌ No {args.format} files found in {args.input_dir}")
        return

    print(f"\n🎨 STEP FILE VISUALIZATION")
    print(f"{'='*80}")
    print(f"Input Dir:    {args.input_dir}")
    print(f"Output Dir:   {args.output_dir}")
    print(f"Files Found:  {len(mesh_files)}")
    print(f"Resolution:   {args.resolution[0]}x{args.resolution[1]}")
    print(f"{'='*80}\n")

    successful = 0
    failed = 0

    for i, mesh_path in enumerate(mesh_files, 1):
        filename = os.path.basename(mesh_path)
        output_file = os.path.join(args.output_dir, filename.replace(f'.{args.format}', '.png'))

        print(f"[{i:2d}/{len(mesh_files)}] {filename:<40}", end=' ... ', flush=True)

        success, msg = visualize_mesh(mesh_path, output_file, tuple(args.resolution))

        if success:
            print(f"✅ {output_file.split('/')[-1]}")
            successful += 1
        else:
            print(f"❌ {msg[:50]}")
            failed += 1

    print(f"\n{'='*80}")
    print(f"✅ Successful: {successful}/{len(mesh_files)}")
    print(f"❌ Failed:     {failed}/{len(mesh_files)}")
    print(f"📁 Images saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
