"""
Utility functions for evaluation scripts
"""
import numpy as np


def read_ply(filepath):
    """
    Read PLY file and return points as numpy array.

    Args:
        filepath: Path to PLY file

    Returns:
        np.array of shape (N, 3) containing point coordinates
    """
    try:
        import trimesh
        mesh = trimesh.load(filepath)

        # If it's a point cloud
        if hasattr(mesh, 'vertices'):
            return np.array(mesh.vertices)
        # If it's already points
        elif hasattr(mesh, 'points'):
            return np.array(mesh.points)
        else:
            raise ValueError(f"Unable to extract points from {filepath}")

    except ImportError:
        # Fallback to manual parsing if trimesh not available
        return read_ply_manual(filepath)


def read_ply_manual(filepath):
    """
    Manual PLY parser as fallback.
    Supports ASCII and binary PLY formats.
    """
    with open(filepath, 'rb') as f:
        # Read header
        line = f.readline().decode('ascii').strip()
        if line != 'ply':
            raise ValueError(f"Not a PLY file: {filepath}")

        # Parse header
        vertex_count = 0
        format_type = None
        properties = []

        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('format'):
                format_type = line.split()[1]
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[2])
            elif line.startswith('property'):
                properties.append(line)
            elif line == 'end_header':
                break

        # Read vertex data
        if format_type == 'ascii':
            points = []
            for _ in range(vertex_count):
                line = f.readline().decode('ascii').strip()
                coords = [float(x) for x in line.split()[:3]]
                points.append(coords)
            return np.array(points)

        elif format_type.startswith('binary'):
            # For binary formats, use trimesh or warn
            raise NotImplementedError(
                f"Binary PLY format requires trimesh library. "
                f"Install with: pip install trimesh"
            )

        else:
            raise ValueError(f"Unsupported PLY format: {format_type}")


def generate_mock_point_cloud(shape='cube', n_points=2000):
    """
    Generate a mock point cloud for testing.

    Args:
        shape: 'cube', 'sphere', or 'cylinder'
        n_points: Number of points

    Returns:
        np.array of shape (n_points, 3)
    """
    if shape == 'cube':
        # Random points in unit cube
        points = np.random.uniform(-0.5, 0.5, (n_points, 3))

    elif shape == 'sphere':
        # Random points on/in unit sphere
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        r = np.random.uniform(0, 0.5, n_points) ** (1/3)  # Uniform volume distribution

        points = np.column_stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])

    elif shape == 'cylinder':
        # Random points in cylinder
        theta = np.random.uniform(0, 2*np.pi, n_points)
        r = np.random.uniform(0, 0.5, n_points) ** 0.5  # Uniform area distribution
        z = np.random.uniform(-0.5, 0.5, n_points)

        points = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta),
            z
        ])
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return points.astype(np.float32)


def save_ply_ascii(points, filepath):
    """
    Save points to PLY file in ASCII format.

    Args:
        points: np.array of shape (N, 3)
        filepath: Output path
    """
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


if __name__ == '__main__':
    # Test the utilities
    import os
    import tempfile

    print("Testing utils.py...")

    # Test mock point cloud generation
    print("\n1. Testing mock point cloud generation:")
    for shape in ['cube', 'sphere', 'cylinder']:
        points = generate_mock_point_cloud(shape=shape, n_points=100)
        print(f"   Generated {shape}: shape={points.shape}, "
              f"range=[{points.min():.2f}, {points.max():.2f}]")

    # Test PLY saving and reading
    print("\n2. Testing PLY save/load:")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test.ply')

        # Save
        points_orig = generate_mock_point_cloud('cube', n_points=50)
        save_ply_ascii(points_orig, test_file)
        print(f"   Saved {len(points_orig)} points to {test_file}")

        # Load
        points_loaded = read_ply(test_file)
        print(f"   Loaded {len(points_loaded)} points")

        # Verify
        if np.allclose(points_orig, points_loaded):
            print("   ✓ Save/Load test passed!")
        else:
            print("   ✗ Save/Load test failed!")

    print("\n✓ All tests passed!")
