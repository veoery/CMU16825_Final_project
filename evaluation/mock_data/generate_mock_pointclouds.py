#!/usr/bin/env python3
"""
Generate mock point clouds to test Chamfer Distance evaluation
This lets you test eval_ae_cd.py without needing cadlib
"""
import numpy as np
import os


def generate_point_cloud(shape='cube', n_points=2000):
    """Generate point cloud for different shapes"""
    if shape == 'cube':
        # Random points in unit cube
        points = np.random.uniform(-0.5, 0.5, (n_points, 3))

    elif shape == 'sphere':
        # Random points in unit sphere
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        r = np.random.uniform(0, 0.5, n_points) ** (1/3)

        points = np.column_stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])

    elif shape == 'cylinder':
        # Random points in cylinder
        theta = np.random.uniform(0, 2*np.pi, n_points)
        r = np.random.uniform(0, 0.5, n_points) ** 0.5
        z = np.random.uniform(-0.5, 0.5, n_points)

        points = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta),
            z
        ])

    else:
        raise ValueError(f"Unknown shape: {shape}")

    return points.astype(np.float32)


def save_ply(points, filepath):
    """Save point cloud as ASCII PLY file"""
    with open(filepath, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Write points
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def test_chamfer_distance():
    """Test Chamfer Distance with generated point clouds"""
    from scipy.spatial import cKDTree as KDTree

    print("\n" + "=" * 70)
    print("Testing Chamfer Distance with Mock Point Clouds")
    print("=" * 70)

    def chamfer_dist(pc1, pc2):
        """Compute Chamfer Distance"""
        tree1 = KDTree(pc1)
        tree2 = KDTree(pc2)

        dist1, _ = tree1.query(pc2)
        dist2, _ = tree2.query(pc1)

        return np.mean(dist1**2) + np.mean(dist2**2)

    # Test 1: Identical shapes
    print("\n1. Identical shapes (should be ~0):")
    cube1 = generate_point_cloud('cube', 1000)
    cd = chamfer_dist(cube1, cube1)
    print(f"   CD = {cd:.6f} ✓")

    # Test 2: Same shape, different sampling
    print("\n2. Same shape, different samples:")
    cube2 = generate_point_cloud('cube', 1000)
    cd = chamfer_dist(cube1, cube2)
    print(f"   CD = {cd:.6f} (small value expected)")

    # Test 3: Different shapes
    print("\n3. Different shapes (cube vs sphere):")
    sphere = generate_point_cloud('sphere', 1000)
    cd = chamfer_dist(cube1, sphere)
    print(f"   CD = {cd:.6f} (larger value expected)")

    print("\n✓ Chamfer Distance metric works!\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate mock point clouds for CD testing')
    parser.add_argument('--output_dir', type=str, default='../data/pc_cad',
                       help='Output directory for PLY files')
    parser.add_argument('--n_points', type=int, default=2000,
                       help='Number of points per cloud')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of point cloud pairs to generate')

    args = parser.parse_args()

    print("=" * 70)
    print("Mock Point Cloud Generator")
    print("=" * 70)

    # Test Chamfer Distance first
    test_chamfer_distance()

    # Generate point cloud files
    print("=" * 70)
    print("Generating Point Cloud Files")
    print("=" * 70)

    shapes = ['cube', 'sphere', 'cylinder']

    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nGenerating {args.num_samples} point cloud pairs...")
    print(f"Output: {args.output_dir}")
    print(f"Points per cloud: {args.n_points}")

    for i in range(args.num_samples):
        # Create mock data_id (matching eval_ae_cd.py naming convention)
        data_id = f"mock{i:04d}"
        truck_id = "mock"

        # Create subdirectory
        subdir = os.path.join(args.output_dir, truck_id)
        os.makedirs(subdir, exist_ok=True)

        # Generate point cloud (randomly choose shape)
        shape = shapes[i % len(shapes)]

        # Add some noise to make it slightly different from GT
        points = generate_point_cloud(shape, args.n_points)
        points_noisy = points + np.random.normal(0, 0.01, points.shape)

        # Save ground truth
        gt_path = os.path.join(subdir, f"{data_id}.ply")
        save_ply(points, gt_path)

        print(f"  {i+1}/{args.num_samples}: {gt_path} ({shape})")

    print("\n✓ Generated point clouds!")
    print("\n" + "=" * 70)
    print("Usage Examples")
    print("=" * 70)

    print("\n1. Test with existing H5 mock data:")
    print("   First, update H5 files to use 'mock' data_id format:")
    print("   (or generate new H5 files with matching IDs)")

    print("\n2. Run Chamfer Distance evaluation:")
    print(f"   python eval_ae_cd.py \\")
    print(f"       --src mock_data/h5_data \\")
    print(f"       --n_points {args.n_points}")

    print("\n3. Or test CD directly in Python:")
    print("""
   from scipy.spatial import cKDTree as KDTree
   import numpy as np

   # Load point clouds
   from utils import read_ply
   pc1 = read_ply('data/pc_cad/mock/mock0000.ply')
   pc2 = read_ply('data/pc_cad/mock/mock0001.ply')

   # Compute Chamfer Distance
   def chamfer_dist(pc1, pc2):
       tree1 = KDTree(pc1)
       tree2 = KDTree(pc2)
       dist1, _ = tree1.query(pc2)
       dist2, _ = tree2.query(pc1)
       return np.mean(dist1**2) + np.mean(dist2**2)

   cd = chamfer_dist(pc1, pc2)
   print(f'Chamfer Distance: {cd:.6f}')
    """)

    print("\nNote: To use with eval_ae_cd.py, your H5 files need to have")
    print("      data_id matching the PLY filenames (e.g., mock0000)")


if __name__ == '__main__':
    main()
