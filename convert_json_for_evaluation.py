#!/usr/bin/env python3
"""
Convert generated CAD JSON to evaluation-friendly formats.

Converts to:
1. H5 - for eval_ae_acc.py (command accuracy)
2. Pickle - for eval_seq.py (sequence evaluation)
3. Preserves JSON - for eval_topology.py (after STEP export)
"""

import json
import pickle
import numpy as np
import glob
import os
import argparse
from pathlib import Path

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not installed. H5 output will be skipped.")


def extract_cad_features(cad_data):
    """
    Extract CAD features into a vector.

    Returns: numpy array of shape (15,)
    """
    features = []

    # Extract entities
    entities = cad_data.get("entities", {})

    # Count entity types
    sketch_count = sum(1 for e in entities.values() if e.get("type") == "Sketch")
    extrude_count = sum(1 for e in entities.values() if e.get("type") == "ExtrudeFeature")

    # Extract bounding box
    bbox = cad_data.get("bounding_box", {})
    max_pt = bbox.get("max_point", {})
    min_pt = bbox.get("min_point", {})

    max_x = max_pt.get("x", 0.0)
    max_y = max_pt.get("y", 0.0)
    max_z = max_pt.get("z", 0.0)
    min_x = min_pt.get("x", 0.0)
    min_y = min_pt.get("y", 0.0)
    min_z = min_pt.get("z", 0.0)

    # Calculate dimensions
    dim_x = abs(max_x - min_x)
    dim_y = abs(max_y - min_y)
    dim_z = abs(max_z - min_z)

    # Extract sequence operations
    sequence = cad_data.get("sequence", [])
    op_count = len(sequence)

    # Count loops and curves from all sketches
    total_loops = 0
    total_curves = 0
    for entity_id, entity in entities.items():
        if entity.get("type") == "Sketch":
            profiles = entity.get("profiles", {})
            for profile_id, profile in profiles.items():
                loops = profile.get("loops", [])
                total_loops += len(loops)
                for loop in loops:
                    curves = loop.get("profile_curves", [])
                    total_curves += len(curves)

    # Build feature vector
    features = [
        sketch_count,      # 0: number of sketches
        extrude_count,     # 1: number of extrudes
        op_count,          # 2: number of operations
        dim_x,             # 3: bounding box width
        dim_y,             # 4: bounding box height
        dim_z,             # 5: bounding box depth
        max_x,             # 6: max x
        max_y,             # 7: max y
        max_z,             # 8: max z
        min_x,             # 9: min x
        min_y,             # 10: min y
        min_z,             # 11: min z
        total_loops,       # 12: total loops
        total_curves,      # 13: total curves
        (dim_x * dim_y * dim_z) / 1000  # 14: volume (normalized)
    ]

    return np.array(features, dtype=np.float32)


def json_to_h5(json_dir, output_h5):
    """Convert JSON files to H5 format for eval_ae_acc.py"""
    if not HAS_H5PY:
        print("❌ h5py not installed. Skipping H5 conversion.")
        return False

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))

    if not json_files:
        print(f"❌ No JSON files found in {json_dir}")
        return False

    print(f"\n📦 Converting {len(json_files)} JSON files to H5...")

    predictions = []

    for json_file in json_files:
        try:
            with open(json_file) as f:
                cad_data = json.load(f)

            vec = extract_cad_features(cad_data)
            predictions.append(vec)
            print(f"  ✓ {os.path.basename(json_file)}")

        except Exception as e:
            print(f"  ✗ {os.path.basename(json_file)}: {e}")

    if predictions:
        # Save as H5
        os.makedirs(os.path.dirname(output_h5) or '.', exist_ok=True)
        with h5py.File(output_h5, 'w') as f:
            f.create_dataset('out_vec', data=np.array(predictions))

        print(f"\n✅ Saved {len(predictions)} vectors to: {output_h5}")
        print(f"   Shape: {np.array(predictions).shape}")
        return True
    else:
        print("❌ No vectors created")
        return False


def json_to_pickle(json_dir, output_pkl):
    """Convert JSON files to pickle format for eval_seq.py"""

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))

    if not json_files:
        print(f"❌ No JSON files found in {json_dir}")
        return False

    print(f"\n📦 Converting {len(json_files)} JSON files to pickle...")

    data = {}

    for idx, json_file in enumerate(json_files):
        try:
            with open(json_file) as f:
                cad_data = json.load(f)

            # Create UID (matches expected format)
            uid = f"uid_{idx:05d}"

            # Structure expected by eval_seq.py
            data[uid] = {
                "level_1": {
                    "pred_cad_vec": [cad_data],  # List of predictions
                    "cd": [0.0]  # Placeholder Chamfer distance
                }
            }

            print(f"  ✓ {os.path.basename(json_file)} → {uid}")

        except Exception as e:
            print(f"  ✗ {os.path.basename(json_file)}: {e}")

    if data:
        # Save pickle
        os.makedirs(os.path.dirname(output_pkl) or '.', exist_ok=True)
        with open(output_pkl, 'wb') as f:
            pickle.dump(data, f)

        print(f"\n✅ Saved {len(data)} sequences to: {output_pkl}")
        return True
    else:
        print("❌ No sequences created")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert generated CAD JSON to evaluation formats"
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source directory containing JSON files"
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        default=None,
        help="Output H5 file (for eval_ae_acc.py)"
    )
    parser.add_argument(
        "--output-pkl",
        type=str,
        default=None,
        help="Output pickle file (for eval_seq.py)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all output formats"
    )

    args = parser.parse_args()

    src_path = Path(args.src)

    if not src_path.is_dir():
        print(f"❌ Source directory not found: {args.src}")
        return

    print(f"📂 Source directory: {args.src}")

    # Determine output files
    if args.all:
        output_h5 = args.output_h5 or "predictions.h5"
        output_pkl = args.output_pkl or "sequences.pkl"
    else:
        output_h5 = args.output_h5
        output_pkl = args.output_pkl

    if not output_h5 and not output_pkl and not args.all:
        print("❌ Specify --output-h5 or --output-pkl or use --all")
        return

    print("\n" + "="*80)

    # Convert to H5
    if output_h5:
        json_to_h5(args.src, output_h5)

    # Convert to pickle
    if output_pkl:
        json_to_pickle(args.src, output_pkl)

    print("\n" + "="*80)
    print("✅ Conversion complete!")
    print("\n📋 Next steps:")

    if output_h5:
        print(f"\n1. Evaluate command accuracy:")
        print(f"   python evaluation/eval_ae_acc.py --src {output_h5}")

    if output_pkl:
        print(f"\n2. Evaluate sequence generation:")
        print(f"   python evaluation/eval_seq.py \\")
        print(f"       --input_path {output_pkl} \\")
        print(f"       --output_dir ./eval_results \\")
        print(f"       --validate_only")

    print(f"\n3. Evaluate topology (after STEP export):")
    print(f"   python evaluation/eval_topology.py --src gen_cad_all/step")


if __name__ == "__main__":
    main()