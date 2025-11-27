#!/usr/bin/env python3
"""
Validate and repair generated CAD JSON files.

Fixes common issues:
1. Profile references that don't exist (removes invalid profiles from extrude)
2. Degenerate bounding boxes (recalculates from geometry)
3. Missing fields
"""

import json
import os
import glob
import argparse
import numpy as np
from pathlib import Path


def collect_all_points(entities):
    """
    Collect all 3D points from all geometries to calculate a proper bounding box.

    Args:
        entities (dict): All entities in the CAD model

    Returns:
        np.ndarray: Array of all points (N, 3), or None if no points found
    """
    points = []

    for entity_id, entity in entities.items():
        if entity.get("type") == "Sketch":
            profiles = entity.get("profiles", {})
            for profile_id, profile in profiles.items():
                loops = profile.get("loops", [])
                for loop in loops:
                    curves = loop.get("profile_curves", [])
                    for curve in curves:
                        # Extract points from different curve types
                        if "start_point" in curve:
                            pt = curve["start_point"]
                            points.append([pt["x"], pt["y"], pt["z"]])
                        if "end_point" in curve:
                            pt = curve["end_point"]
                            points.append([pt["x"], pt["y"], pt["z"]])
                        if "center_point" in curve:
                            pt = curve["center_point"]
                            points.append([pt["x"], pt["y"], pt["z"]])
                        if "mid_point" in curve:
                            pt = curve["mid_point"]
                            points.append([pt["x"], pt["y"], pt["z"]])

    if not points:
        return None

    return np.array(points)


def recalculate_bbox(entities):
    """
    Recalculate bounding box from actual geometry.

    Args:
        entities (dict): All entities

    Returns:
        dict: New bounding box with max_point and min_point
    """
    points = collect_all_points(entities)

    if points is None or len(points) == 0:
        # Default bbox if no points found
        return {
            "max_point": {"x": 0.1, "y": 0.1, "z": 0.1},
            "min_point": {"x": -0.1, "y": -0.1, "z": -0.1}
        }

    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)

    # Avoid degenerate boxes (all same point)
    for i in range(3):
        if min_pt[i] == max_pt[i]:
            max_pt[i] += 0.1
            min_pt[i] -= 0.1

    return {
        "max_point": {"x": float(max_pt[0]), "y": float(max_pt[1]), "z": float(max_pt[2])},
        "min_point": {"x": float(min_pt[0]), "y": float(min_pt[1]), "z": float(min_pt[2])}
    }


def validate_and_repair(input_json):
    """
    Validate and repair CAD JSON data.

    Returns:
        tuple: (repaired_json, list_of_repairs)
    """
    repairs = []
    data = input_json.copy()

    entities = data.get("entities", {})
    sequence = data.get("sequence", [])
    properties = data.get("properties", {})

    # Repair 1: Validate profile references
    for seq_item in sequence:
        if seq_item.get("type") == "ExtrudeFeature":
            extrude_id = seq_item.get("entity")
            if extrude_id not in entities:
                continue

            extrude = entities[extrude_id]
            profiles = extrude.get("profiles", [])
            valid_profiles = []

            for profile_ref in profiles:
                sketch_id = profile_ref.get("sketch")
                profile_id = profile_ref.get("profile")

                if sketch_id not in entities:
                    repairs.append(f"Removed profile {profile_id}: sketch {sketch_id} not found")
                    continue

                sketch = entities[sketch_id]
                if profile_id not in sketch.get("profiles", {}):
                    repairs.append(f"Removed profile {profile_id}: not found in sketch {sketch_id}")
                    continue

                valid_profiles.append(profile_ref)

            if len(valid_profiles) == 0:
                repairs.append(f"WARNING: Extrude {extrude_id} has no valid profiles!")
            else:
                extrude["profiles"] = valid_profiles

    # Repair 2: Fix bounding box
    bbox = data.get("bounding_box", {})
    max_pt = bbox.get("max_point", {})
    min_pt = bbox.get("min_point", {})

    # Check if bbox is degenerate (all dimensions are 0 or same)
    is_degenerate = (
        max_pt.get("x") == min_pt.get("x") or
        max_pt.get("y") == min_pt.get("y") or
        max_pt.get("z") == min_pt.get("z") or
        (max_pt.get("x") == 0 and max_pt.get("y") == 0 and max_pt.get("z") == 0)
    )

    if is_degenerate:
        repairs.append("Recalculated degenerate bounding box")
        data["bounding_box"] = recalculate_bbox(entities)

    # Repair 3: Ensure properties.bounding_box is synced
    if "bounding_box" in data:
        if "properties" not in data:
            data["properties"] = {}
        data["properties"]["bounding_box"] = data["bounding_box"]

    return data, repairs


def process_file(input_path, output_path=None):
    """
    Process and repair a single JSON file.

    Returns:
        tuple: (success: bool, repairs: list)
    """
    try:
        with open(input_path, 'r') as f:
            input_data = json.load(f)

        repaired_data, repairs = validate_and_repair(input_data)

        # Determine output path
        if output_path is None:
            output_path = input_path

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(repaired_data, f, indent=2)

        status = "✓" if len(repairs) == 0 else "⚠"
        print(f"{status} {os.path.basename(input_path)}")
        for repair in repairs:
            print(f"    - {repair}")

        return True, repairs

    except Exception as e:
        print(f"✗ {os.path.basename(input_path)}: {e}")
        return False, []


def main():
    parser = argparse.ArgumentParser(description="Validate and repair CAD JSON files")
    parser.add_argument("--src", type=str, required=True, help="Source directory or file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--pattern", type=str, default="*.json", help="File pattern")
    args = parser.parse_args()

    src_path = Path(args.src)

    if src_path.is_file():
        files = [str(src_path)]
    else:
        files = sorted(glob.glob(os.path.join(args.src, f"**/{args.pattern}"), recursive=True))

    if not files:
        print(f"No files found in {args.src}")
        return

    print(f"Validating and repairing {len(files)} file(s)\n")

    success = 0
    total_repairs = 0

    for input_file in files:
        if args.output and len(files) > 1:
            rel_path = os.path.relpath(input_file, args.src)
            output_file = os.path.join(args.output, rel_path)
        else:
            output_file = args.output if args.output else None

        ok, repairs = process_file(input_file, output_file)
        if ok:
            success += 1
            total_repairs += len(repairs)

    print(f"\nResults: {success}/{len(files)} successful, {total_repairs} repairs made")


if __name__ == "__main__":
    main()
