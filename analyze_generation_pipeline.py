#!/usr/bin/env python3
"""
Compare raw text vs repaired JSON to identify where pipeline breaks.
"""

import json
import os
import glob
from pathlib import Path


def analyze_file_pair(txt_path, json_path):
    """
    Compare raw text generation with final JSON to understand what happened.
    """
    filename = os.path.basename(txt_path).replace(".txt", "")

    print(f"\n{'='*80}")
    print(f"📊 ANALYSIS: {filename}")
    print(f"{'='*80}")

    # Read raw text
    with open(txt_path, 'r') as f:
        raw_text = f.read()

    # Parse raw text to see structure
    print(f"\n📝 RAW TEXT ANALYSIS:")
    print(f"  Length: {len(raw_text)} characters")

    # Try to find entities in raw text
    import re
    entity_ids = set(re.findall(r'"entity":\s*"([^"]+)"', raw_text))
    sketch_ids = set(re.findall(r'"F[^"]*_0"', raw_text))

    print(f"  Entity IDs found: {len(entity_ids)}")
    print(f"  Sketch-like IDs: {len(sketch_ids)}")

    # Check if JSON exists
    if not os.path.exists(json_path):
        print(f"\n❌ JSON FILE NOT FOUND: {json_path}")
        return

    # Load JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"\n❌ JSON PARSE ERROR: {e}")
        return

    # Analyze JSON structure
    print(f"\n📄 JSON STRUCTURE:")
    entities = data.get("entities", {})
    sequence = data.get("sequence", [])

    print(f"  Entities in dict: {len(entities)}")
    for eid, entity in entities.items():
        print(f"    - {eid}: {entity.get('type')}")

    print(f"\n  Sequence operations: {len(sequence)}")
    for i, op in enumerate(sequence):
        status = "✓" if op.get("entity") in entities else "✗"
        print(f"    {status} {i+1}. {op.get('type')} -> {op.get('entity')}")

    # Count root-level entities (orphans)
    root_keys = set(data.keys()) - {"entities", "sequence", "properties", "bounding_box", "type", "transform", "objects", "name"}
    orphan_count = sum(1 for k in root_keys if isinstance(data[k], dict) and "type" in data[k])

    if orphan_count > 0:
        print(f"\n⚠️ ORPHANED ENTITIES AT ROOT LEVEL: {orphan_count}")
        for key in root_keys:
            if isinstance(data[key], dict) and "type" in data[key]:
                print(f"    - {key}: {data[key].get('type')}")

    # Identify the core issue
    print(f"\n🔍 ROOT CAUSE ANALYSIS:")

    # Issue 1: Sequence missing operations
    missing_extrudes = sum(1 for e in entities.values() if e.get("type") == "ExtrudeFeature")
    extrudes_in_seq = sum(1 for s in sequence if s.get("type") == "ExtrudeFeature")

    if missing_extrudes > extrudes_in_seq:
        print(f"  ❌ ISSUE 1: Sequence missing extrudes")
        print(f"     - Found {missing_extrudes} extrudes in entities")
        print(f"     - But only {extrudes_in_seq} in sequence")
        print(f"     - Cause: Sequence cleaned BEFORE entities consolidated")
        print(f"     - Export fails: Can't build model without extrudes")

    # Issue 2: Missing extrude fields
    for eid, entity in entities.items():
        if entity.get("type") == "ExtrudeFeature":
            required = ["extent_one", "extent_two", "extent_type", "operation", "start_extent"]
            missing = [f for f in required if f not in entity]
            if missing:
                print(f"  ❌ ISSUE 2: Extrude {eid} missing fields: {missing}")
                print(f"     - Export fails: Code tries to access missing fields")

    # Issue 3: Incomplete curves
    invalid_curves = 0
    for eid, entity in entities.items():
        if entity.get("type") == "Sketch":
            profiles = entity.get("profiles", {})
            for pid, profile in profiles.items():
                loops = profile.get("loops", [])
                for loop in loops:
                    curves = loop.get("profile_curves", [])
                    for curve in curves:
                        if "start_point" not in curve or "end_point" not in curve:
                            invalid_curves += 1

    if invalid_curves > 0:
        print(f"  ❌ ISSUE 3: {invalid_curves} invalid curves found")
        print(f"     - Export fails: Code crashes on incomplete curve data")

    # Summary
    print(f"\n📋 EXPORT PREDICTION:")
    can_export = (
        extrudes_in_seq > 0 and
        all(not any(f for f in ["extent_one", "extent_two", "extent_type", "operation", "start_extent"]
                    if f not in e) for e in entities.values() if e.get("type") == "ExtrudeFeature") and
        invalid_curves == 0
    )

    if can_export:
        print("  ✅ Should export successfully")
    else:
        print("  ❌ Will fail on export")
        if missing_extrudes > extrudes_in_seq:
            print(f"     Reason: No extrudes in sequence (only {extrudes_in_seq}/{missing_extrudes})")
        if invalid_curves > 0:
            print(f"     Reason: Invalid curve data")


def main():
    src_dir = "/root/cmu/16825_l43d/CMU16825_Final_project/gen_cad_all/v4_cube_try"

    # Find all JSON files
    json_files = sorted(glob.glob(os.path.join(src_dir, "*.json")))

    print(f"\n🔍 ANALYSIS OF GENERATION PIPELINE")
    print(f"{'='*80}")
    print(f"Directory: {src_dir}")
    print(f"Files found: {len(json_files)}\n")

    for json_file in json_files:
        txt_file = json_file.replace(".json", ".txt")

        if os.path.exists(txt_file):
            analyze_file_pair(txt_file, json_file)
        else:
            print(f"\n⚠️ No corresponding .txt file for {os.path.basename(json_file)}")

    print(f"\n{'='*80}")
    print("KEY FINDINGS:")
    print("="*80)
    print("""
1. REPAIR SCRIPT ISSUE: Steps execute in wrong order
   - Step 1b (clean sequence) runs BEFORE Step 5 (consolidate entities)
   - Result: Valid extrudes get removed from sequence
   - Fix: Reorder so consolidation happens first

2. JSON REPAIR ISSUE: json_repair corrupts structure
   - Moves entities to root level instead of keeping in entities dict
   - Splits JSON across fragments
   - Result: Orphaned entities that confuse the repair script

3. MODEL GENERATION ISSUE: Incomplete extrude definitions
   - Missing required fields (extent_type, operation, start_extent)
   - Likely due to model truncation or incomplete generation

SOLUTION: Fix prepare_cad_for_export.py step order
""")


if __name__ == "__main__":
    main()
