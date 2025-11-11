#!/usr/bin/env python3
"""
Standalone test for topology evaluation with mock data.

This script demonstrates eval_topology.py functionality without external dependencies.
It only requires trimesh which can be installed with: pip install trimesh

Usage:
    python test_topology_standalone.py
"""

import numpy as np
import sys
import os

def create_simple_cube_mesh():
    """Create a simple cube mesh for testing"""
    try:
        import trimesh
    except ImportError:
        print("Error: trimesh not installed")
        print("Install with: pip install trimesh")
        sys.exit(1)

    # Cube vertices
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # Top face
    ])

    # Cube faces (triangulated)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 6, 5], [4, 7, 6],  # Top
        [0, 5, 1], [0, 4, 5],  # Front
        [2, 7, 3], [2, 6, 7],  # Back
        [0, 7, 4], [0, 3, 7],  # Left
        [1, 6, 2], [1, 5, 6],  # Right
    ])

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def create_open_mesh():
    """Create a mesh with boundary edges (incomplete cube)"""
    try:
        import trimesh
    except ImportError:
        return None

    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # Top face
    ])

    # Only 3 faces - this creates boundary edges
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [0, 5, 1],             # Partial front face
    ])

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def create_self_intersecting_mesh():
    """Create a mesh with self-intersections"""
    try:
        import trimesh
    except ImportError:
        return None

    # Two triangles that intersect each other
    vertices = np.array([
        # First triangle
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0],
        # Second triangle (intersects first)
        [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [1.5, 0.5, 0],
    ])

    faces = np.array([
        [0, 1, 2],  # First triangle
        [3, 4, 5],  # Second triangle
    ])

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def test_topology_metrics():
    """Test all topology metrics"""
    try:
        import trimesh
        from eval_topology import (
            seg_error, dangling_edge_length,
            self_intersection_ratio, flux_enclosure_error
        )
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("\nRequired installations:")
        print("  pip install trimesh")
        return False

    print("=" * 70)
    print("Testing CAD Topology Evaluation Metrics")
    print("=" * 70)

    # Test 1: Closed mesh (cube)
    print("\n1. Testing with closed mesh (cube):")
    print("-" * 70)
    cube = create_simple_cube_mesh()
    print(f"   Vertices: {len(cube.vertices)}, Faces: {len(cube.faces)}")
    print(f"   Is watertight: {cube.is_watertight}")

    dang_el = dangling_edge_length(cube)
    sir = self_intersection_ratio(cube)
    flux_ee = flux_enclosure_error(cube)

    print(f"   ✓ Dangling Edge Length (DangEL): {dang_el:.4f}")
    print(f"     Expected: 0.0 (closed mesh)")
    print(f"   ✓ Self-Intersection Ratio (SIR): {sir:.4f}")
    print(f"     Expected: 0.0 (no intersections)")
    print(f"   ✓ Flux Enclosure Error (FluxEE): {flux_ee:.6f}")
    print(f"     Expected: ~0.0 (closed mesh)")

    # Test 2: Segment error
    print("\n2. Testing segment error (comparing two cubes):")
    print("-" * 70)
    cube2 = create_simple_cube_mesh()
    cube2.apply_translation([2, 0, 0])  # Move second cube
    combined = trimesh.util.concatenate([cube, cube2])

    seg_e = seg_error(combined, cube)
    print(f"   Combined mesh: {len(combined.split())} components")
    print(f"   Reference mesh: {len(cube.split())} component")
    print(f"   ✓ Segment Error (SegE): {seg_e:.4f}")
    print(f"     Expected: 1.0 (2 components vs 1 component)")

    # Test 3: Open mesh with dangling edges
    print("\n3. Testing with open mesh (has boundary edges):")
    print("-" * 70)
    open_mesh = create_open_mesh()
    print(f"   Vertices: {len(open_mesh.vertices)}, Faces: {len(open_mesh.faces)}")
    print(f"   Is watertight: {open_mesh.is_watertight}")

    dang_el = dangling_edge_length(open_mesh)
    print(f"   ✓ Dangling Edge Length (DangEL): {dang_el:.4f}")
    print(f"     Expected: >0 (has boundary edges)")

    # Test 4: Self-intersecting mesh
    print("\n4. Testing with self-intersecting mesh:")
    print("-" * 70)
    si_mesh = create_self_intersecting_mesh()
    print(f"   Vertices: {len(si_mesh.vertices)}, Faces: {len(si_mesh.faces)}")

    sir = self_intersection_ratio(si_mesh)
    print(f"   ✓ Self-Intersection Ratio (SIR): {sir:.4f}")
    print(f"     Expected: >0 (has intersections)")

    # Test 5: Built-in trimesh shapes
    print("\n5. Testing with trimesh built-in shapes:")
    print("-" * 70)

    shapes = [
        ("Sphere", trimesh.creation.icosphere(subdivisions=2)),
        ("Cylinder", trimesh.creation.cylinder(radius=0.5, height=1.0)),
        ("Box", trimesh.creation.box(extents=[1, 1, 1])),
    ]

    for name, shape in shapes:
        dang_el = dangling_edge_length(shape)
        sir = self_intersection_ratio(shape)
        flux_ee = flux_enclosure_error(shape)

        print(f"\n   {name}:")
        print(f"     DangEL: {dang_el:.4f}")
        print(f"     SIR: {sir:.4f}")
        print(f"     FluxEE: {flux_ee:.6f}")
        print(f"     Watertight: {shape.is_watertight}")

    print("\n" + "=" * 70)
    print("All topology tests completed successfully! ✓")
    print("=" * 70)

    return True


def main():
    """Main function"""
    print("\nTopology Evaluation - Standalone Test\n")

    success = test_topology_metrics()

    if success:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("1. Generate mock data:")
        print("   python generate_mock_data.py --output_dir ./mock_data")
        print("\n2. Test with mock meshes:")
        print("   import trimesh")
        print("   from eval_topology import *")
        print("   mesh = trimesh.load('mock_data/meshes/mock_mesh_000.obj')")
        print("   print('DangEL:', dangling_edge_length(mesh))")
        print("\n3. Use with your own CAD model predictions")
    else:
        print("\n✗ Tests failed. Please install missing dependencies.")
        sys.exit(1)


if __name__ == '__main__':
    main()
