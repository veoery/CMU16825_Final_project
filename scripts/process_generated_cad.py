"""
Unified pipeline for processing generated CAD:
JSON -> STEP -> STL/Mesh -> Point Cloud + Screenshot

Combines logic from:
- export2step_progress.py: JSON/H5 -> STEP
- pointcloud_preprocess.py: STEP -> STL -> Point Cloud
"""
import os
import glob
import json
import h5py
import numpy as np
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import threading

# Add DeepCAD to path
deepcad_path = os.path.join(os.path.dirname(__file__), '..', '..', 'DeepCAD')
if os.path.exists(deepcad_path):
    sys.path.insert(0, deepcad_path)
    sys.path.insert(0, os.path.join(deepcad_path, 'utils'))
else:
    sys.path.append("..")

from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD
from file_utils import ensure_dir

# OCC imports for STEP and mesh processing
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import read_step_file, write_step_file
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE


class DevNull:
    """Suppress verbose output."""
    def write(self, msg):
        pass
    def flush(self):
        pass


def screenshot_step(step_path, output_png):
    """Take a screenshot of a STEP file using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Load and mesh the shape
        shape = read_step_file(step_path)
        BRepMesh_IncrementalMesh(shape, 0.01)

        # Extract vertices and faces
        vertices = []
        faces = []
        vertex_map = {}
        vertex_count = 0

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            triangulation = BRep_Tool.Triangulation(face, None)

            if triangulation:
                nodes = triangulation.Nodes()
                triangles = triangulation.Triangles()

                for i in range(1, nodes.Length() + 1):
                    pt = nodes.Value(i)
                    vertices.append([pt.X(), pt.Y(), pt.Z()])
                    vertex_map[i] = vertex_count
                    vertex_count += 1

                for i in range(1, triangles.Length() + 1):
                    n1, n2, n3 = triangles.Value(i)
                    faces.append([
                        vertices[vertex_map[n1]],
                        vertices[vertex_map[n2]],
                        vertices[vertex_map[n3]]
                    ])

            explorer.Next()

        # Create plot
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        if faces:
            face_collection = Poly3DCollection(faces, alpha=0.7, edgecolor='k', linewidth=0.3)
            face_collection.set_facecolor('cyan')
            ax.add_collection3d(face_collection)

        if vertices:
            vertices = np.array(vertices)
            ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
            ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
            ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(os.path.basename(step_path))

        plt.savefig(output_png, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception:
        return False


def extract_point_cloud_from_step(step_path, n_samples=4096, deflection=0.01):
    """Extract point cloud from STEP file by triangulation."""
    try:
        shape = read_step_file(step_path)
        BRepMesh_IncrementalMesh(shape, deflection)

        # Collect all vertices from triangulation
        all_pts = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)

        while explorer.More():
            face = explorer.Current()
            triangulation = BRep_Tool.Triangulation(face, None)

            if triangulation:
                nodes = triangulation.Nodes()
                for i in range(1, nodes.Length() + 1):
                    pt = nodes.Value(i)
                    all_pts.append([pt.X(), pt.Y(), pt.Z()])

            explorer.Next()

        if not all_pts:
            return None

        pts = np.array(all_pts, dtype=np.float32)

        # Downsample with FPS if needed
        if len(pts) > n_samples:
            pts = fps_downsample(pts, n_samples)

        # Normalize
        pts = normalize_pc(pts)
        return pts

    except Exception as e:
        return None


def fps_downsample(points, n_samples=4096):
    """Farthest Point Sampling."""
    N = points.shape[0]
    if N <= n_samples:
        return points

    selected = np.zeros(n_samples, dtype=int)
    selected[0] = np.random.randint(0, N)
    distances = np.full(N, np.inf)

    for i in range(1, n_samples):
        last = points[selected[i - 1]]
        dist = np.linalg.norm(points - last, axis=1)
        distances = np.minimum(distances, dist)
        selected[i] = np.argmax(distances)

    return points[selected]


def normalize_pc(pc):
    """Normalize point cloud to unit sphere at origin."""
    pc = pc - pc.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc = pc / (scale + 1e-8)
    return pc.astype(np.float32)


def process_cad_file(cad_path, step_dir, pc_dir, args):
    """Convert single CAD file (JSON/H5) to STEP and Point Cloud."""
    try:
        filename = os.path.basename(cad_path).split('.')[0]
        step_path = os.path.join(step_dir, filename + ".step")
        pc_path = os.path.join(pc_dir, filename + ".npz")
        png_path = os.path.join(step_dir, filename + ".png")

        # Skip if already processed
        if os.path.exists(step_path) and os.path.exists(pc_path):
            return "SKIP"

        # Load and convert to CAD shape
        if args.form == "h5":
            with h5py.File(cad_path, 'r') as fp:
                out_vec = fp["out_vec"][:].astype(np.float32)
                out_shape = vec2CADsolid(out_vec)
        else:  # json
            with open(cad_path, 'r') as fp:
                data = json.load(fp)
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            out_shape = create_CAD(cad_seq)

        # Validate shape
        if args.filter:
            analyzer = BRepCheck_Analyzer(out_shape)
            if not analyzer.IsValid():
                return "INVALID"

        # Write STEP file
        ensure_dir(os.path.dirname(step_path))
        devnull = DevNull()
        old_stdout = sys.stdout
        sys.stdout = devnull

        write_success = [False]
        write_error = [None]

        def write_step():
            try:
                write_step_file(out_shape, step_path)
                write_success[0] = True
            except Exception as e:
                write_error[0] = e

        write_thread = threading.Thread(target=write_step)
        write_thread.daemon = True
        write_thread.start()
        write_thread.join(timeout=240)
        sys.stdout = old_stdout

        if not write_success[0]:
            return "WRITE_ERROR"

        # Extract point cloud from STEP
        ensure_dir(os.path.dirname(pc_path))
        pc = extract_point_cloud_from_step(step_path, n_samples=args.pc_samples)

        if pc is None:
            return "PC_EXTRACT_ERROR"

        np.savez_compressed(pc_path, points=pc)

        # Generate screenshot if requested
        if args.screenshot:
            screenshot_step(step_path, png_path)

        return "SUCCESS"

    except Exception as e:
        return f"ERROR: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Process generated CAD: JSON/H5 -> STEP -> Point Cloud + Screenshot"
    )
    parser.add_argument('--src', type=str, required=True, help="source folder with CAD files")
    parser.add_argument('--form', type=str, default="json", choices=["h5", "json"],
                       help="input file format")
    parser.add_argument('--idx', type=int, default=0, help="start index")
    parser.add_argument('--num', type=int, default=-1, help="number of files to process (-1 = all)")
    parser.add_argument('--filter', action="store_true", help="validate shapes with BRepCheck")
    parser.add_argument('--screenshot', action="store_true", help="generate screenshots")
    parser.add_argument('--pc-samples', type=int, default=4096, help="point cloud sample count")
    parser.add_argument('--step-dir', type=str, default=None, help="output STEP directory")
    parser.add_argument('--pc-dir', type=str, default=None, help="output point cloud directory")
    args = parser.parse_args()

    src_dir = args.src
    step_dir = args.step_dir or (src_dir + "_step")
    pc_dir = args.pc_dir or (src_dir + "_pc")

    print(f"Source directory: {src_dir}")
    print(f"STEP output: {step_dir}")
    print(f"Point Cloud output: {pc_dir}")

    # Find CAD files
    cad_paths = sorted(glob.glob(os.path.join(src_dir, f"*.{args.form}")))
    if args.num != -1:
        cad_paths = cad_paths[args.idx:args.idx + args.num]

    print(f"Found {len(cad_paths)} files\n")

    # Create output directories
    ensure_dir(step_dir)
    ensure_dir(pc_dir)

    # Process files
    stats = {
        "SUCCESS": 0,
        "SKIP": 0,
        "INVALID": 0,
        "WRITE_ERROR": 0,
        "PC_EXTRACT_ERROR": 0,
        "ERROR": 0
    }

    with tqdm(total=len(cad_paths), desc="Processing CAD", unit="file") as pbar:
        for cad_path in cad_paths:
            filename = os.path.basename(cad_path)
            result = process_cad_file(cad_path, step_dir, pc_dir, args)

            # Track statistics
            if result == "SUCCESS":
                stats["SUCCESS"] += 1
                status_str = "[✓] SUCCESS"
            elif result == "SKIP":
                stats["SKIP"] += 1
                status_str = "[→] SKIP"
            elif result == "INVALID":
                stats["INVALID"] += 1
                status_str = "[✗] INVALID"
            else:
                stats["ERROR"] += 1
                status_str = f"[!] {result}"

            pbar.set_postfix_str(f"{filename}: {status_str}")
            pbar.update(1)

    # Print summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total files:           {len(cad_paths)}")
    print(f"Successfully processed: {stats['SUCCESS']}")
    print(f"Skipped (existing):     {stats['SKIP']}")
    print(f"Invalid models:         {stats['INVALID']}")
    print(f"Errors:                 {stats['ERROR']}")
    print("="*80)
    print(f"STEP files saved to:       {step_dir}")
    print(f"Point clouds saved to:     {pc_dir}")
    if args.screenshot:
        print(f"Screenshots saved to:      {step_dir} (*.png)")
    print("="*80)


if __name__ == "__main__":
    main()
