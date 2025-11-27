#!/usr/bin/env python3
"""
Detailed diagnostic script to identify issues in CAD JSON files.
Explains exactly what's wrong and why export fails.
"""

import json
import os
import glob
import argparse
from pathlib import Path


def diagnose_file(input_path):
    """
    Diagnose all issues in a CAD JSON file.

    Returns:
        dict with detailed findings
    """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load JSON: {e}"}

    issues = []
    warnings = []

    # Check 1: Structure
    print(f"\n{'='*80}")
    print(f"FILE: {os.path.basename(input_path)}")
    print(f"{'='*80}")

    # Structure check
    print("\n📋 STRUCTURE CHECK:")
    keys = set(data.keys())
    expected_keys = {"entities", "bounding_box", "sequence", "properties"}
    missing_keys = expected_keys - keys
    extra_keys = keys - expected_keys

    if not missing_keys:
        print("  ✓ Has all expected top-level keys")
    else:
        print(f"  ✗ Missing keys: {missing_keys}")
        issues.append(f"Missing top-level keys: {missing_keys}")

    if extra_keys:
        print(f"  ℹ Extra keys (may be OK): {extra_keys}")

    # Check entities
    print("\n🏗️  ENTITIES CHECK:")
    entities = data.get("entities", {})
    print(f"  Found {len(entities)} entities")

    for entity_id, entity in entities.items():
        entity_type = entity.get("type", "UNKNOWN")
        print(f"\n  [{entity_id}] type={entity_type}")

        if entity_type == "Sketch":
            # Check sketch fields
            has_transform = "transform" in entity
            has_profiles = "profiles" in entity

            if not has_transform:
                print(f"    ✗ MISSING: 'transform' field")
                issues.append(f"Sketch {entity_id}: Missing 'transform'")
            else:
                print(f"    ✓ Has 'transform'")

            if not has_profiles:
                print(f"    ✗ MISSING: 'profiles' field")
                issues.append(f"Sketch {entity_id}: Missing 'profiles'")
            else:
                profiles = entity.get("profiles", {})
                print(f"    ✓ Has {len(profiles)} profile(s): {list(profiles.keys())}")

        elif entity_type == "ExtrudeFeature":
            # Check extrude fields
            required = ["extent_one", "extent_two", "extent_type", "operation", "profiles", "start_extent"]
            missing = [f for f in required if f not in entity]

            if missing:
                print(f"    ✗ Missing fields: {missing}")
                issues.append(f"Extrude {entity_id}: Missing {missing}")
            else:
                print(f"    ✓ Has all required fields")

            # Check profile references
            profiles = entity.get("profiles", [])
            print(f"    References {len(profiles)} profile(s):")
            for profile_ref in profiles:
                sketch_id = profile_ref.get("sketch")
                profile_id = profile_ref.get("profile")

                if sketch_id not in entities:
                    print(f"      ✗ Sketch {sketch_id} NOT FOUND")
                    issues.append(f"Extrude {entity_id}: References non-existent sketch {sketch_id}")
                elif profile_id not in entities[sketch_id].get("profiles", {}):
                    print(f"      ✗ Profile {profile_id} NOT in sketch {sketch_id}")
                    issues.append(f"Extrude {entity_id}: Profile {profile_id} not in sketch {sketch_id}")
                else:
                    print(f"      ✓ Profile {profile_id} in sketch {sketch_id}")

    # Check sequence
    print("\n📍 SEQUENCE CHECK:")
    sequence = data.get("sequence", [])
    if not sequence:
        print("  ✗ Sequence is empty!")
        issues.append("Sequence array is empty - no extrude operations to export")

        # Check if there are any extrudes at all
        extrude_count = sum(1 for e in entities.values() if e.get("type") == "ExtrudeFeature")
        if extrude_count == 0:
            print("  ✗ No ExtrudeFeature entities found - only sketches present")
            print("  → This file cannot be exported (sketches alone don't create a 3D model)")
        else:
            print(f"  ℹ Note: Found {extrude_count} ExtrudeFeature(s) but sequence is empty")
            print("  → Sequence should have been auto-populated during repair")
    else:
        invalid_refs = 0
        print(f"  Found {len(sequence)} operation(s):")
        for i, item in enumerate(sequence):
            op_type = item.get("type")
            entity_id = item.get("entity")
            if entity_id not in entities:
                print(f"    {i+1}. ✗ {op_type} references non-existent entity {entity_id}")
                issues.append(f"Sequence item {i}: References non-existent entity {entity_id}")
                invalid_refs += 1
            else:
                print(f"    {i+1}. ✓ {op_type} -> {entity_id}")

        if invalid_refs > 0:
            print(f"\n  ⚠ CRITICAL: {invalid_refs}/{len(sequence)} sequence entries are invalid!")
            print(f"  → Cause: JSON was truncated during generation or repair")
            print(f"  → Fix: Use repair_cad_for_export.py to remove invalid references")

    # Check bounding box
    print("\n📏 BOUNDING BOX CHECK:")
    bbox = data.get("bounding_box", {})
    max_pt = bbox.get("max_point", {})
    min_pt = bbox.get("min_point", {})

    is_degenerate = (
        max_pt.get("x") == min_pt.get("x") or
        max_pt.get("y") == min_pt.get("y") or
        max_pt.get("z") == min_pt.get("z")
    )

    if is_degenerate:
        print(f"  ✗ DEGENERATE: max={max_pt}, min={min_pt}")
        issues.append("Bounding box is degenerate (zero-size in some dimension)")
    else:
        print(f"  ✓ Valid: max={max_pt}")
        print(f"          min={min_pt}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")

    if not issues:
        print("✅ NO ISSUES FOUND - File should export successfully!")
    else:
        print(f"❌ FOUND {len(issues)} ISSUE(S):\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    return {"issues": issues, "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose issues in CAD JSON files"
    )
    parser.add_argument("--src", type=str, required=True, help="Source directory or file")
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

    print(f"\n🔍 Diagnosing {len(files)} file(s)...\n")

    total_issues = 0
    for input_file in files:
        result = diagnose_file(input_file)
        total_issues += len(result.get("issues", []))

    print(f"\n{'='*80}")
    print(f"TOTAL: {total_issues} issue(s) across all files")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
