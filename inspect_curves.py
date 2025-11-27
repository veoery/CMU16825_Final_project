#!/usr/bin/env python3
"""
Deep inspection of curve definitions to find issues causing export failures.
"""

import json
import os
import glob
import argparse
from pathlib import Path


def inspect_curves(json_data):
    """
    Deeply inspect all curves for issues that cause export to fail.
    """
    entities = json_data.get("entities", {})
    issues = []

    print("\n" + "="*80)
    print("DEEP CURVE INSPECTION")
    print("="*80)

    for entity_id, entity in entities.items():
        if entity.get("type") != "Sketch":
            continue

        print(f"\n📍 Sketch: {entity_id}")
        profiles = entity.get("profiles", {})

        for profile_id, profile in profiles.items():
            print(f"  └─ Profile: {profile_id}")
            loops = profile.get("loops", [])

            if not loops:
                print(f"     ✗ NO LOOPS FOUND!")
                issues.append(f"Profile {profile_id} has no loops")
                continue

            for loop_idx, loop in enumerate(loops):
                print(f"     └─ Loop {loop_idx}:")
                curves = loop.get("profile_curves", [])

                if not curves:
                    print(f"        ✗ NO CURVES IN LOOP!")
                    issues.append(f"Loop {loop_idx} in {profile_id} has no curves")
                    continue

                print(f"        Curves: {len(curves)}")

                for curve_idx, curve in enumerate(curves):
                    curve_type = curve.get("type", "UNKNOWN")
                    print(f"        ├─ Curve {curve_idx}: {curve_type}")

                    # Check all required fields
                    checks = {
                        "type": "type" in curve,
                        "start_point": "start_point" in curve,
                        "end_point": "end_point" in curve,
                    }

                    # Check for Arc-specific fields
                    if curve_type == "Arc3D":
                        checks["center_point"] = "center_point" in curve
                        checks["radius"] = "radius" in curve

                    for check_name, has_field in checks.items():
                        if not has_field:
                            print(f"        │  ✗ MISSING: {check_name}")
                            issues.append(f"Curve {curve_idx} in {profile_id}/{profile_id}: missing '{check_name}'")
                        else:
                            print(f"        │  ✓ {check_name}")

                    # Check coordinates
                    if "start_point" in curve:
                        start = curve["start_point"]
                        coords = ["x", "y", "z"]
                        missing = [c for c in coords if c not in start]
                        if missing:
                            print(f"        │  ✗ start_point missing: {missing}")
                            issues.append(f"Curve {curve_idx} start_point missing: {missing}")
                        else:
                            print(f"        │  ✓ start_point: x={start.get('x')}, y={start.get('y')}, z={start.get('z')}")

                    if "end_point" in curve:
                        end = curve["end_point"]
                        coords = ["x", "y", "z"]
                        missing = [c for c in coords if c not in end]
                        if missing:
                            print(f"        │  ✗ end_point missing: {missing}")
                            issues.append(f"Curve {curve_idx} end_point missing: {missing}")
                        else:
                            print(f"        │  ✓ end_point: x={end.get('x')}, y={end.get('y')}, z={end.get('z')}")

                    # Check arc geometry
                    if curve_type == "Arc3D":
                        center = curve.get("center_point", {})
                        radius = curve.get("radius", 0)
                        start = curve.get("start_point", {})

                        if center and start and radius:
                            dist = ((center.get("x", 0) - start.get("x", 0))**2 +
                                   (center.get("y", 0) - start.get("y", 0))**2 +
                                   (center.get("z", 0) - start.get("z", 0))**2) ** 0.5

                            if abs(dist - radius) > 0.001:
                                print(f"        │  ✗ Invalid arc: start_point distance {dist:.6f} != radius {radius}")
                                issues.append(f"Arc {curve_idx}: Invalid geometry (dist={dist}, radius={radius})")
                            else:
                                print(f"        │  ✓ Arc geometry valid")

    print("\n" + "="*80)
    if issues:
        print(f"❌ FOUND {len(issues)} ISSUE(S):\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ NO ISSUES FOUND IN CURVES")

    print("="*80 + "\n")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Deep inspect curve definitions")
    parser.add_argument("--src", type=str, required=True, help="Source directory or file")
    args = parser.parse_args()

    src_path = Path(args.src)

    if src_path.is_file():
        files = [str(src_path)]
    else:
        files = sorted(glob.glob(os.path.join(args.src, "*.json")))

    if not files:
        print(f"No JSON files found in {args.src}")
        return

    print(f"🔍 Inspecting {len(files)} file(s)...")

    total_issues = 0
    for json_file in files:
        print(f"\n📄 File: {os.path.basename(json_file)}")

        try:
            with open(json_file) as f:
                data = json.load(f)

            issues = inspect_curves(data)
            total_issues += len(issues)

        except Exception as e:
            print(f"❌ Error reading file: {e}")

    print(f"\n{'='*80}")
    print(f"TOTAL ISSUES: {total_issues}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
