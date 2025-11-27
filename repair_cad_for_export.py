#!/usr/bin/env python3
"""
Complete pipeline: Convert generated CAD JSON to CADSequence format and repair.

Usage:
    python repair_cad_for_export.py --src generated_cad_all/ver3_cad -o output_dir
"""

import json
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from cad_validation import CADValidator


def collect_all_points(entities):
    """Collect all 3D points from geometry."""
    points = []
    for entity_id, entity in entities.items():
        if entity.get("type") == "Sketch":
            profiles = entity.get("profiles", {})
            for profile_id, profile in profiles.items():
                loops = profile.get("loops", [])
                for loop in loops:
                    curves = loop.get("profile_curves", [])
                    for curve in curves:
                        for key in ["start_point", "end_point", "center_point", "mid_point"]:
                            if key in curve:
                                pt = curve[key]
                                points.append([pt["x"], pt["y"], pt["z"]])
    return np.array(points) if points else None


def recalculate_bbox(entities):
    """Recalculate bounding box from actual geometry."""
    points = collect_all_points(entities)

    if points is None or len(points) == 0:
        return {
            "max_point": {"x": 0.1, "y": 0.1, "z": 0.1},
            "min_point": {"x": -0.1, "y": -0.1, "z": -0.1}
        }

    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)

    # Avoid degenerate boxes
    for i in range(3):
        if min_pt[i] == max_pt[i]:
            max_pt[i] += 0.1
            min_pt[i] -= 0.1

    return {
        "max_point": {"x": float(max_pt[0]), "y": float(max_pt[1]), "z": float(max_pt[2])},
        "min_point": {"x": float(min_pt[0]), "y": float(min_pt[1]), "z": float(min_pt[2])}
    }


def get_default_transform():
    """Get identity coordinate system (standard XY plane)."""
    return {
        "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
        "x_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
        "y_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
        "z_axis": {"x": 0.0, "y": 0.0, "z": 1.0}
    }


def prepare_cad_json(input_json):
    """
    Convert generated format to CADSequence and repair issues.

    Returns:
        tuple: (prepared_json, list_of_operations)
    """
    ops = []
    data = input_json.copy()
    entities = data.get("entities", {})

    # Validate entities is a dict, not a list
    if isinstance(entities, list):
        ops.append("[ERROR] entities is a list, not a dict - malformed JSON from generation")
        return None, ops

    if not isinstance(entities, dict):
        ops.append(f"[ERROR] entities is {type(entities).__name__}, not a dict")
        return None, ops

    # Step 0a: Detect fundamentally malformed structures (Fusion 360 format vs Omni-CAD)
    malformed_indicators = []
    for ent_id, ent in entities.items():
        etype = ent.get("type", "")

        # Check for Fusion 360 format entity types
        if etype in ["ExtrudedAreaFeatureWithProfiles", "SketchExtrudedAreaFeature", "ExtrudedAreaFeature"]:
            malformed_indicators.append(f"Fusion 360 entity type: {etype}")

        # Check for nested orphaned entities (embedded in extrude)
        for key in ent.keys():
            if key.startswith('F') and len(key) > 15 and isinstance(ent[key], dict):
                if 'type' in ent[key]:
                    malformed_indicators.append(f"Nested orphaned entity: {key}")

        # Check for malformed profiles field (nested lists)
        if ent.get("type") == "Sketch":
            profiles = ent.get("profiles")
            if isinstance(profiles, list) and profiles and isinstance(profiles[0], list):
                malformed_indicators.append(f"Sketch {ent_id}: profiles is nested list (Fusion format)")

        # Check for extrude with nested list profiles
        if "ExtrudedArea" in etype or "ExtrudeFeature" in etype:
            profiles = ent.get("profiles")
            if isinstance(profiles, list) and profiles and isinstance(profiles[0], list):
                malformed_indicators.append(f"{ent_id}: profiles is nested list of lists")

    if malformed_indicators:
        ops.append("[FATAL] Fundamentally malformed JSON structure detected:")
        for indicator in malformed_indicators:
            ops.append(f"  • {indicator}")
        ops.append("  → This JSON uses Fusion 360 format, not Omni-CAD format")
        ops.append("  → Regenerate with corrected generation prompt")
        return None, ops

    # Step 0: Add missing transforms to sketches
    for entity_id, entity in entities.items():
        if entity.get("type") == "Sketch" and "transform" not in entity:
            entity["transform"] = get_default_transform()
            ops.append(f"[REPAIR] Added missing transform to sketch {entity_id} (using identity frame)")

    # Step 0b: Check for extrude features in entities
    extrude_count = sum(1 for e in entities.values() if e.get("type") == "ExtrudeFeature")
    if extrude_count == 0:
        ops.append(f"[WARNING] No ExtrudeFeature entities found - only sketches present")

    # Step 1: Consolidate orphaned entities FIRST
    # This must happen before sequence cleaning so all entities are in the right place
    orphaned_entities = []
    keys_to_remove = []

    for key in list(data.keys()):
        # Skip known top-level keys
        if key in ["entities", "sequence", "properties", "bounding_box", "type", "transform", "objects", "name"]:
            continue

        value = data[key]
        # Check if this looks like an entity (has "type" field)
        if isinstance(value, dict) and "type" in value and value["type"] in ["Sketch", "ExtrudeFeature"]:
            # This is an orphaned entity - move it to entities dict
            if key not in entities:
                entities[key] = value
                orphaned_entities.append(key)
                keys_to_remove.append(key)
                ops.append(f"[REPAIR] Moved orphaned entity from root to entities dict: {key}")

    # Remove orphaned entities from root level
    for key in keys_to_remove:
        del data[key]

    if orphaned_entities:
        ops.append(f"[REPAIR] Consolidated {len(orphaned_entities)} orphaned entities")

    # Step 2: Convert to CADSequence format if needed
    if "sequence" not in data:
        ops.append("Converted to CADSequence format (created 'sequence' array)")
        sequence = []
        for entity_id, entity_data in entities.items():
            if entity_data.get("type") == "ExtrudeFeature":
                sequence.append({
                    "type": "ExtrudeFeature",
                    "entity": entity_id
                })
        data["sequence"] = sequence

    # Step 2b: Clean up sequence - remove references to non-existent entities
    # This now runs AFTER orphaned entities are consolidated
    valid_sequence = []
    invalid_count = 0
    for seq_item in data.get("sequence", []):
        entity_id = seq_item.get("entity")
        entity_type = seq_item.get("type")

        # Check if referenced entity exists
        if entity_id not in entities:
            ops.append(f"[REPAIR] Removed sequence entry: {entity_type} -> {entity_id} (entity not found)")
            invalid_count += 1
            continue

        valid_sequence.append(seq_item)

    if invalid_count > 0:
        data["sequence"] = valid_sequence
        ops.append(f"[REPAIR] Cleaned sequence: removed {invalid_count} invalid references")

    # Step 2c: Filter out degenerate extrudes (zero or negative height)
    # These cannot be exported to STEP format
    degenerate_count = 0
    filtered_sequence = []

    for seq_item in data.get("sequence", []):
        if seq_item.get("type") == "ExtrudeFeature":
            extrude_id = seq_item.get("entity")
            extrude = entities.get(extrude_id, {})

            extent_one = extrude.get("extent_one", {})
            distance_def = extent_one.get("distance", {})
            value = distance_def.get("value", 0)

            if value is None or value <= 0:
                ops.append(f"[SKIP] Removed degenerate extrude {extrude_id} (height={value})")
                degenerate_count += 1
                continue

        filtered_sequence.append(seq_item)

    if degenerate_count > 0:
        data["sequence"] = filtered_sequence
        ops.append(f"[REPAIR] Filtered out {degenerate_count} degenerate extrude(s)")

    # Step 3: Validate and repair extrude features
    for seq_item in data.get("sequence", []):
        if seq_item.get("type") == "ExtrudeFeature":
            extrude_id = seq_item.get("entity")
            if extrude_id not in entities:
                continue

            extrude = entities[extrude_id]

            # Add default values for missing required extrude fields
            if "operation" not in extrude:
                extrude["operation"] = "NewBodyFeatureOperation"
                ops.append(f"[REPAIR] Added default operation to extrude {extrude_id}")

            if "start_extent" not in extrude:
                extrude["start_extent"] = {"type": "ProfilePlaneStartDefinition"}
                ops.append(f"[REPAIR] Added default start_extent to extrude {extrude_id}")

            if "extent_type" not in extrude:
                extrude["extent_type"] = "OneSideFeatureExtentType"
                ops.append(f"[REPAIR] Added default extent_type to extrude {extrude_id}")

            # Check for other missing fields (non-critical)
            optional_fields = ["extent_one", "extent_two", "profiles"]
            for field in optional_fields:
                if field not in extrude:
                    ops.append(f"⚠ Extrude {extrude_id} missing field '{field}' (may cause export errors)")

            profiles = extrude.get("profiles", [])
            valid_profiles = []

            for profile_ref in profiles:
                sketch_id = profile_ref.get("sketch")
                profile_id = profile_ref.get("profile")

                if sketch_id not in entities or profile_id not in entities[sketch_id].get("profiles", {}):
                    ops.append(f"Removed invalid profile reference: {profile_id} (sketch: {sketch_id})")
                    continue

                valid_profiles.append(profile_ref)

            extrude["profiles"] = valid_profiles

    # Step 4: Fix bounding box if degenerate
    bbox = data.get("bounding_box", {})
    max_pt = bbox.get("max_point", {})
    min_pt = bbox.get("min_point", {})

    is_degenerate = (
        max_pt.get("x") == min_pt.get("x") or
        max_pt.get("y") == min_pt.get("y") or
        max_pt.get("z") == min_pt.get("z") or
        (max_pt.get("x") == 0 and max_pt.get("y") == 0 and max_pt.get("z") == 0)
    )

    if is_degenerate:
        ops.append("Recalculated degenerate bounding box from geometry")
        data["bounding_box"] = recalculate_bbox(entities)

    # Step 5: Ensure properties.bounding_box is synced
    if "bounding_box" in data:
        if "properties" not in data:
            data["properties"] = {}
        data["properties"]["bounding_box"] = data["bounding_box"]

    # Step 6: Validate and fix curve definitions
    invalid_curves = 0
    for entity_id, entity in entities.items():
        if entity.get("type") == "Sketch":
            profiles = entity.get("profiles", {})
            for profile_id, profile in profiles.items():
                loops = profile.get("loops", [])
                for loop_idx, loop in enumerate(loops):
                    curves = loop.get("profile_curves", [])
                    valid_curves = []

                    for curve_idx, curve in enumerate(curves):
                        # Check for incomplete curves
                        has_start = "start_point" in curve
                        has_end = "end_point" in curve
                        has_type = "type" in curve

                        if not has_type:
                            ops.append(f"[REPAIR] Removed curve missing 'type' in {entity_id}/{profile_id}")
                            invalid_curves += 1
                            continue

                        if not has_start or not has_end:
                            ops.append(f"[REPAIR] Removed incomplete curve in {entity_id}/{profile_id} (missing endpoints)")
                            invalid_curves += 1
                            continue

                        # Validate coordinates are complete (x, y, z)
                        start_pt = curve.get("start_point", {})
                        end_pt = curve.get("end_point", {})

                        start_ok = all(k in start_pt for k in ["x", "y", "z"])
                        end_ok = all(k in end_pt for k in ["x", "y", "z"])

                        if not start_ok:
                            ops.append(f"[REPAIR] Removed curve with incomplete start_point in {entity_id}/{profile_id}")
                            invalid_curves += 1
                            continue

                        if not end_ok:
                            ops.append(f"[REPAIR] Removed curve with incomplete end_point in {entity_id}/{profile_id}")
                            invalid_curves += 1
                            continue

                        # Curve is valid
                        valid_curves.append(curve)

                    # Update with valid curves only
                    loop["profile_curves"] = valid_curves

    if invalid_curves > 0:
        ops.append(f"[REPAIR] Removed {invalid_curves} invalid curves from profiles")

    # Step 7: Final validation
    final_sequence = data.get("sequence", [])
    final_extrude_count = sum(1 for s in final_sequence if s.get("type") == "ExtrudeFeature")

    if final_extrude_count == 0:
        ops.append("[WARNING] No extrudes in final sequence - model may not export correctly")
    else:
        ops.append(f"[INFO] Final model has {final_extrude_count} extrude(s) ready for export")

    return data, ops


def process_file(input_path, output_path):
    """Process a single file. Returns (success, operations_list)."""
    try:
        with open(input_path, 'r') as f:
            input_data = json.load(f)

        prepared_data, ops = prepare_cad_json(input_data)

        # Check for fatal errors
        if prepared_data is None:
            return False, ops

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(prepared_data, f, indent=2)

        return True, ops

    except Exception as e:
        return False, [f"Error: {e}"]


def main():
    parser = argparse.ArgumentParser(
        description="Convert and prepare generated CAD JSON for STEP export"
    )
    parser.add_argument("--src", type=str, required=True, help="Source directory or file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    parser.add_argument("--pattern", type=str, default="*.json", help="File pattern to match")
    args = parser.parse_args()

    src_path = Path(args.src)

    if src_path.is_file():
        files = [str(src_path)]
    else:
        files = sorted(glob.glob(os.path.join(args.src, f"**/{args.pattern}"), recursive=True))

    if not files:
        print(f"❌ No files found in {args.src}")
        return

    print(f"📦 Preparing {len(files)} CAD file(s) for export\n")
    print("=" * 80)

    success = 0
    for input_file in files:
        rel_path = os.path.relpath(input_file, args.src)
        output_file = os.path.join(args.output, rel_path)

        ok, ops = process_file(input_file, output_file)

        filename = os.path.basename(input_file)
        if ok:
            print(f"✓ {filename}")
            for op in ops:
                print(f"  • {op}")
            success += 1
        else:
            print(f"✗ {filename}")
            for op in ops:
                print(f"  • {op}")

    print("=" * 80)
    print(f"\n✅ Results: {success}/{len(files)} files prepared successfully")
    print(f"📁 Output saved to: {args.output}")


if __name__ == "__main__":
    main()
