"""Enhanced version of export2step.py with tqdm progress bar and screenshots."""
import os
import glob
import json
import h5py
import numpy as np
from OCC.Core.BRepCheck import BRepCheck_Analyzer
np = np  # Ensure numpy is available for auto_generate_properties
from OCC.Extend.DataExchange import read_step_file, write_step_file, write_ply_file
import argparse
import sys
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
import threading

# Add DeepCAD to path to find cadlib
deepcad_path = os.path.join(os.path.dirname(__file__), '..', '..', 'DeepCAD')
if os.path.exists(deepcad_path):
    sys.path.insert(0, deepcad_path)
    sys.path.insert(0, os.path.join(deepcad_path, 'utils'))
else:
    # Fallback: try relative path from current directory
    sys.path.append("..")

from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD
from file_utils import ensure_dir

def auto_generate_sequence(entities):
    """Auto-generate sequence from entities when it's missing.

    Sequence is the execution order of CAD operations.
    Algorithm:
    1. Find all sketches referenced by extrudes
    2. Add sketches in order of extrude references
    3. Add all extrudes in order they appear
    """
    sequence = []
    sketch_added = set()

    # Find all extrudes and process them in order
    for ent_id in sorted(entities.keys()):
        ent = entities[ent_id]
        ent_type = ent.get('type', '')

        # Check if this is an extrude (Extrude, ExtrudeFeature, PadFeature, etc.)
        if 'Extrude' in ent_type or 'Pad' in ent_type:
            # Find sketch referenced by this extrude
            profiles = ent.get('profiles', [])
            if isinstance(profiles, list) and profiles:
                # Get the first profile's sketch reference
                first_profile = profiles[0] if profiles else {}
                sketch_ref = None

                if isinstance(first_profile, dict):
                    sketch_ref = first_profile.get('sketch')

                # Add sketch to sequence if it exists and not already added
                if sketch_ref and sketch_ref in entities and sketch_ref not in sketch_added:
                    sequence.append({
                        'type': 'Sketch',
                        'entity': sketch_ref
                    })
                    sketch_added.add(sketch_ref)

            # Add the extrude itself
            sequence.append({
                'type': ent_type,
                'entity': ent_id
            })

    return sequence if sequence else []


def auto_generate_properties(entities):
    """Auto-generate properties (bounding box) from entities.

    Extracts all 3D points from entities and computes bounding box.
    """
    all_points = []

    # Collect all points from sketches
    for ent_id, ent in entities.items():
        if ent.get('type') == 'Sketch':
            profiles = ent.get('profiles', {})
            if isinstance(profiles, dict):
                for profile_id, profile in profiles.items():
                    loops = profile.get('loops', [])
                    if isinstance(loops, list):
                        for loop in loops:
                            curves = loop.get('profile_curves', [])
                            if isinstance(curves, list):
                                for curve in curves:
                                    # Extract points from curve
                                    if 'center_point' in curve:
                                        pt = curve['center_point']
                                        all_points.append([pt.get('x', 0), pt.get('y', 0), pt.get('z', 0)])
                                    if 'start_point' in curve:
                                        pt = curve['start_point']
                                        all_points.append([pt.get('x', 0), pt.get('y', 0), pt.get('z', 0)])
                                    if 'end_point' in curve:
                                        pt = curve['end_point']
                                        all_points.append([pt.get('x', 0), pt.get('y', 0), pt.get('z', 0)])

    # Also add extrude extent values as z bounds
    for ent_id, ent in entities.items():
        ent_type = ent.get('type', '')
        if 'Extrude' in ent_type or 'Pad' in ent_type:
            # Get extent values to determine z height
            extent_one = ent.get('extent_one', {})
            if isinstance(extent_one, dict):
                distance_obj = extent_one.get('distance', {})
                if isinstance(distance_obj, dict):
                    z_val = distance_obj.get('value', 0.0)
                    # Add this as a z max point
                    if all_points:
                        all_points.append([all_points[0][0], all_points[0][1], z_val])

    if not all_points:
        # Default bounding box if no points found
        return {
            "bounding_box": {
                "max_point": {"x": 0.1, "y": 0.1, "z": 0.1},
                "min_point": {"x": -0.1, "y": -0.1, "z": 0.0},
                "type": "BoundingBox3D"
            }
        }

    # Compute bounding box
    all_points = np.array(all_points)
    min_point = all_points.min(axis=0)
    max_point = all_points.max(axis=0)

    return {
        "bounding_box": {
            "min_point": {
                "x": float(min_point[0]),
                "y": float(min_point[1]),
                "z": float(min_point[2])
            },
            "max_point": {
                "x": float(max_point[0]),
                "y": float(max_point[1]),
                "z": float(max_point[2])
            },
            "type": "BoundingBox3D"
        }
    }

# Suppress OpenCASCADE verbose output
class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass

def screenshot_step(step_path, output_png):
    """Take a screenshot of a STEP file using matplotlib. Returns True if successful."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use Agg backend (no GUI required)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        from OCC.Extend.DataExchange import read_step_file
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE

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

                # Map old vertices to new indices
                for i in range(1, nodes.Length() + 1):
                    pt = nodes.Value(i)
                    vertices.append([pt.X(), pt.Y(), pt.Z()])
                    vertex_map[i] = vertex_count
                    vertex_count += 1

                # Add faces
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

        # Plot the mesh
        if faces:
            face_collection = Poly3DCollection(faces, alpha=0.7, edgecolor='k', linewidth=0.3)
            face_collection.set_facecolor('cyan')
            ax.add_collection3d(face_collection)

        # Set axis limits
        if vertices:
            vertices = np.array(vertices)
            ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
            ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
            ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(os.path.basename(step_path))

        # Save the figure
        plt.savefig(output_png, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return True
    except Exception as e:
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help="source folder")
parser.add_argument('--form', type=str, default="h5", choices=["h5", "json"], help="file format")
parser.add_argument('--idx', type=int, default=0, help="export n files starting from idx.")
parser.add_argument('--num', type=int, default=10, help="number of shapes to export. -1 exports all shapes.")
parser.add_argument('--filter', action="store_true", help="use opencascade analyzer to filter invalid model")
parser.add_argument('-o', '--outputs', type=str, default=None, help="save folder")
parser.add_argument('--output_form', type=str, default="step", choices=["step", "ply"], help="output file format")
parser.add_argument('--screenshot', action="store_true", help="generate screenshots of STEP files")
args = parser.parse_args()

src_dir = args.src
print(f"Source directory: {src_dir}")

out_paths = sorted(glob.glob(os.path.join(src_dir, "**", f"*.{args.form}"), recursive=True))
print(f"Found {len(out_paths)} files")

# Filter out non-CAD JSON files (validation results, summaries, etc.)
if args.form == "json":
    filtered_paths = []
    for path in out_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Keep only files with CAD structure (entities and either sequence or properties)
                if isinstance(data, dict) and 'entities' in data and \
                   (('sequence' in data or 'properties' in data) or len(data.get('entities', {})) > 0):
                    filtered_paths.append(path)
        except (json.JSONDecodeError, IOError):
            # Skip files that can't be parsed
            pass

    out_paths = filtered_paths
    print(f"After filtering non-CAD files: {len(out_paths)} CAD JSON files")

if args.num != -1:
    out_paths = out_paths[args.idx:args.idx+args.num]
    print(f"Processing {len(out_paths)} files (from index {args.idx})")

suffix = args.output_form

save_dir = args.src + f"_{suffix}" if args.outputs is None else args.outputs
ensure_dir(save_dir)

print(f"Output directory: {save_dir}\n")

# Counters for statistics
success_count = 0
skip_count = 0
error_count = 0
invalid_count = 0
screenshot_count = 0

# Process with progress bar
with tqdm(total=len(out_paths), desc=f"Exporting to {suffix.upper()}", unit="file") as pbar:
    for path in out_paths:
        rel_path = os.path.relpath(path, src_dir)
        save_path = os.path.join(save_dir, os.path.splitext(rel_path)[0] + f".{suffix}")

        # Update progress bar description with current file
        filename = os.path.basename(path)
        pbar.set_postfix_str(f"{filename}")

        if os.path.exists(save_path):
            skip_count += 1
            pbar.update(1)
            continue

        try:
            if args.form == "h5":
                with h5py.File(path, 'r') as fp:
                    out_vec = fp["out_vec"][:].astype(np.float32)
                    out_shape = vec2CADsolid(out_vec)
            else:
                with open(path, 'r') as fp:
                    data = json.load(fp)

                # Auto-repair: generate missing fields from entities
                if 'entities' in data:
                    entities = data['entities']

                    # 1. Auto-generate sequence if missing
                    if 'sequence' not in data:
                        data['sequence'] = auto_generate_sequence(entities)

                    # 2. Auto-generate properties (bounding box) if missing
                    if 'properties' not in data:
                        data['properties'] = auto_generate_properties(entities)

                    # 3. Fix misplaced transform in sketch entities
                    for ent_id, entity in entities.items():
                        if entity.get('type') == 'Sketch' and 'profiles' in entity:
                            # Check if transform is inside profiles dict instead of at entity level
                            if 'transform' not in entity and isinstance(entity['profiles'], dict):
                                if 'transform' in entity['profiles']:
                                    # Move transform to correct location
                                    entity['transform'] = entity['profiles']['transform']
                                    del entity['profiles']['transform']

                cad_seq = CADSequence.from_dict(data)
                cad_seq.normalize()
                out_shape = create_CAD(cad_seq)
        except Exception as e:
            error_count += 1
            tqdm.write(f"[ERROR] Failed to process {filename}: {e}")
            pbar.update(1)
            continue

        if args.filter:
            analyzer = BRepCheck_Analyzer(out_shape)
            if not analyzer.IsValid():
                invalid_count += 1
                tqdm.write(f"[INVALID] {filename}")
                pbar.update(1)
                continue

        ensure_dir(os.path.dirname(save_path))
        try:
            # Suppress verbose OpenCASCADE output with timeout
            import threading
            devnull = DevNull()
            old_stdout = sys.stdout
            sys.stdout = devnull

            write_success = [False]
            write_error = [None]

            def write_file():
                try:
                    if args.output_form == "ply":
                        write_ply_file(out_shape, save_path)
                    elif args.output_form == "step":
                        write_step_file(out_shape, save_path)
                    write_success[0] = True
                except Exception as e:
                    write_error[0] = e

            # Write with 2-minute timeout
            write_thread = threading.Thread(target=write_file)
            write_thread.daemon = True
            write_thread.start()
            write_thread.join(timeout=240)  # 120 seconds = 2 minutes

            sys.stdout = old_stdout

            if write_thread.is_alive():
                # Timeout - file is hanging
                error_count += 1
                tqdm.write(f"[TIMEOUT] {filename} - skipped (> 2 min)")
            elif write_success[0]:
                success_count += 1

                # Generate screenshot if requested
                if args.screenshot and args.output_form == "step":
                    png_path = save_path.replace(".step", ".png")
                    if screenshot_step(save_path, png_path):
                        screenshot_count += 1
                        tqdm.write(f"[SCREENSHOT] {os.path.basename(png_path)}")
                    else:
                        tqdm.write(f"[WARNING] Screenshot failed for {filename}")
            else:
                error_count += 1
                if write_error[0]:
                    tqdm.write(f"[ERROR] {filename}: {write_error[0]}")
        except Exception as e:
            sys.stdout = old_stdout
            error_count += 1
            tqdm.write(f"[ERROR] Failed to write {filename}: {e}")

        pbar.update(1)

# Print summary
print("\n" + "="*80)
print("EXPORT SUMMARY")
print("="*80)
print(f"Total files:          {len(out_paths)}")
print(f"Successfully exported: {success_count}")
print(f"Skipped (existing):    {skip_count}")
print(f"Errors:               {error_count}")
if args.filter:
    print(f"Invalid models:       {invalid_count}")
if args.screenshot:
    print(f"Screenshots generated: {screenshot_count}")
print("="*80)
