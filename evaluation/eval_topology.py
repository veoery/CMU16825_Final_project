# evaluation_topology.py
# ref: CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM
# -*- coding: utf-8 -*-
"""
Topology & enclosure evaluation metrics for CAD/mesh generation.

Dependencies:
  - numpy
  - trimesh   (pip install trimesh)

Provided metrics (callable functions):
  1) seg_error(G_hat, G)
  2) dangling_edge_length(mesh)
  3) self_intersection_ratio(mesh, *, use_aabb_prefilter=True)
  4) flux_enclosure_error(mesh)

Notes
-----
- All functions assume triangle meshes (Trimesh objects). If your data isn’t
  triangular, call mesh.triangles or mesh.triangles.convert to triangulate.
- seg_error compares the number of connected components ("islands") S(·).
- dangling_edge_length uses edges that belong to exactly one face (boundary).
- self_intersection_ratio detects non-adjacent triangle-triangle intersections.
- flux_enclosure_error approximates ∮_S F·n dS with F = (1,1,1).

Usage (example)
---------------
    import trimesh
    from evaluation_topology import (
        seg_error, dangling_edge_length, self_intersection_ratio, flux_enclosure_error
    )

    G = trimesh.load('gt_mesh.obj', process=True)
    Gh = trimesh.load('gen_mesh.obj', process=True)

    print('SegE:', seg_error(Gh, G))
    print('DangEL:', dangling_edge_length(Gh))
    print('SIR:', self_intersection_ratio(Gh))
    print('FluxEE:', flux_enclosure_error(Gh))
"""
from __future__ import annotations
import numpy as np
import trimesh

# ------------------------------------------------------------
# 1) Segment Error (SegE)
# ------------------------------------------------------------
def _num_segments(mesh: trimesh.Trimesh) -> int:
    """
    Count connected components (islands) in the mesh.
    This acts as S(mesh).
    """
    # trimesh.split returns a list of Trimesh components
    # keep_vertices=True keeps original vertex indexing (not needed for counting)
    components = mesh.split(only_watertight=False)
    return len(components)

def seg_error(G_hat: trimesh.Trimesh, G: trimesh.Trimesh) -> float:
    """
    Segment Error: |S(G_hat) - S(G)| / S(G)
    """
    S_gt = _num_segments(G)
    S_gen = _num_segments(G_hat)
    if S_gt == 0:
        # Degenerate case; fall back to absolute difference
        return float(abs(S_gen - S_gt))
    return float(abs(S_gen - S_gt)) / float(S_gt)


# ------------------------------------------------------------
# 2) Dangling Edge Length (DangEL)
# ------------------------------------------------------------
def dangling_edge_length(mesh: trimesh.Trimesh) -> float:
    """
    Sum of lengths of boundary edges (edges incident to exactly one face).
    """
    # Get boundary edges - try multiple approaches for compatibility
    boundary_edges = None

    if hasattr(mesh, 'edges_boundary'):
        boundary_edges = mesh.edges_boundary
    elif hasattr(mesh, 'edges_unique'):
        # Compute boundary edges manually from edge counts
        edges = mesh.edges_unique
        edge_face_count = mesh.edges_unique_length
        # Boundary edges appear in only one face
        boundary_mask = edge_face_count == 1
        boundary_edges = edges[boundary_mask]
    else:
        # Fallback: compute from faces
        from collections import Counter
        edge_counts = Counter()
        for face in mesh.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_counts[edge] += 1
        boundary_edges = np.array([list(edge) for edge, count in edge_counts.items() if count == 1])

    if boundary_edges is None or len(boundary_edges) == 0:
        return 0.0

    V = mesh.vertices
    seg = V[boundary_edges]               # (E_b, 2, 3)
    lengths = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
    return float(lengths.sum())


# ------------------------------------------------------------
# 3) Self-Intersection Ratio (SIR)
#     Percentage (0~1) of faces that participate in at least one
#     non-adjacent triangle-triangle intersection.
# ------------------------------------------------------------
def self_intersection_ratio(mesh: trimesh.Trimesh,
                            use_aabb_prefilter: bool = True) -> float:
    """
    Compute the fraction of faces in 'mesh' that self-intersect
    with any non-adjacent face.

    Heuristic acceleration:
      - Axis-aligned bounding box (AABB) prefilter reduces candidate pairs.
    Robustness:
      - Adjacency (sharing vertex or edge) is ignored (not counted as self-intersection).
    """
    faces = mesh.faces
    tris = mesh.triangles  # (F, 3, 3)
    F = len(faces)
    if F == 0:
        return 0.0

    # Build adjacency to exclude neighbors (share a vertex or edge).
    # Map face -> set of adjacent faces.
    adj = _face_adjacency_sets(mesh)

    # AABB prefilter (vectorized)
    tri_min = tris.min(axis=1)  # (F, 3)
    tri_max = tris.max(axis=1)  # (F, 3)

    # Check intersections
    intersecting_faces = set()

    if use_aabb_prefilter:
        # Use AABB to filter candidate pairs
        for i in range(F):
            for j in range(i + 1, F):
                # Skip adjacent faces
                if j in adj[i]:
                    continue

                # AABB overlap test
                if _aabb_overlap(tri_min[i], tri_max[i], tri_min[j], tri_max[j]):
                    # Precise triangle-triangle intersection test
                    if _triangles_intersect(tris[i], tris[j]):
                        intersecting_faces.add(i)
                        intersecting_faces.add(j)
    else:
        # Brute force without AABB prefilter
        for i in range(F):
            for j in range(i + 1, F):
                if j in adj[i]:
                    continue
                if _triangles_intersect(tris[i], tris[j]):
                    intersecting_faces.add(i)
                    intersecting_faces.add(j)

    return float(len(intersecting_faces)) / float(F) if F > 0 else 0.0


def _face_adjacency_sets(mesh: trimesh.Trimesh) -> dict:
    """
    Build a dictionary mapping each face index to a set of adjacent face indices.
    Two faces are adjacent if they share at least one vertex.
    """
    from collections import defaultdict
    adj = defaultdict(set)

    # Use face_adjacency if available
    if hasattr(mesh, 'face_adjacency'):
        for i, j in mesh.face_adjacency:
            adj[i].add(j)
            adj[j].add(i)
    else:
        # Fallback: build from vertices
        faces = mesh.faces
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                # Check if faces share a vertex
                if len(set(faces[i]) & set(faces[j])) > 0:
                    adj[i].add(j)
                    adj[j].add(i)

    return adj


def _aabb_overlap(min1, max1, min2, max2):
    """Check if two axis-aligned bounding boxes overlap"""
    return (min1[0] <= max2[0] and max1[0] >= min2[0] and
            min1[1] <= max2[1] and max1[1] >= min2[1] and
            min1[2] <= max2[2] and max1[2] >= min2[2])


def _triangles_intersect(tri1, tri2):
    """
    Simple triangle-triangle intersection test.
    Returns True if the triangles intersect.
    """
    # Simple edge-triangle intersection test
    # This is a simplified version - for production use a robust library
    for i in range(3):
        edge_start = tri1[i]
        edge_end = tri1[(i + 1) % 3]
        if _segment_intersects_triangle(edge_start, edge_end, tri2):
            return True

    for i in range(3):
        edge_start = tri2[i]
        edge_end = tri2[(i + 1) % 3]
        if _segment_intersects_triangle(edge_start, edge_end, tri1):
            return True

    return False


def _segment_intersects_triangle(p1, p2, tri):
    """Check if line segment intersects triangle (simplified)"""
    # Compute plane equation of triangle
    v0, v1, v2 = tri
    normal = np.cross(v1 - v0, v2 - v0)
    d = -np.dot(normal, v0)

    # Check if segment crosses plane
    dist1 = np.dot(normal, p1) + d
    dist2 = np.dot(normal, p2) + d

    if dist1 * dist2 > 0:
        return False  # Same side of plane

    # Compute intersection point
    t = dist1 / (dist1 - dist2 + 1e-10)
    intersection = p1 + t * (p2 - p1)

    # Check if intersection point is inside triangle
    return _point_in_triangle(intersection, tri)


def _point_in_triangle(p, tri):
    """Check if point p is inside triangle tri (2D barycentric)"""
    v0, v1, v2 = tri

    # Compute barycentric coordinates
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0

    dot00 = np.dot(v0v1, v0v1)
    dot01 = np.dot(v0v1, v0v2)
    dot02 = np.dot(v0v1, v0p)
    dot11 = np.dot(v0v2, v0v2)
    dot12 = np.dot(v0v2, v0p)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) and (v >= 0) and (u + v <= 1)


# ------------------------------------------------------------
# 4) Flux Enclosure Error (FluxEE)
# ------------------------------------------------------------
def flux_enclosure_error(mesh: trimesh.Trimesh) -> float:
    """
    Compute flux enclosure error.

    For a closed surface, the flux integral ∮_S F·n dS with constant vector
    field F = (1,1,1) should be zero. Deviation from zero indicates the mesh
    is not properly closed.

    Returns:
        Absolute value of the normalized flux integral
    """
    if not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
        return 0.0

    vertices = mesh.vertices
    faces = mesh.faces

    # Compute face normals and areas
    if hasattr(mesh, 'face_normals') and hasattr(mesh, 'area_faces'):
        normals = mesh.face_normals
        areas = mesh.area_faces
    else:
        # Compute manually
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Cross product for normals
        normals = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(normals, axis=1) / 2.0
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)

    # Flux with F = (1, 1, 1)
    F = np.array([1.0, 1.0, 1.0])

    # Compute ∮ F·n dA
    flux = np.sum((normals * areas[:, np.newaxis]) @ F)

    # Normalize by total surface area
    total_area = np.sum(areas)

    if total_area > 0:
        return float(abs(flux) / total_area)
    else:
        return 0.0
