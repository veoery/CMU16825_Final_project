#!/usr/bin/env python3
"""
Convert generated CAD JSON (entity-based format) to CADSequence format (sequence-based).

Generated format has:
  - entities: {sketch_id, extrude_id, ...}
  - bounding_box: at top level

Expected CADSequence format needs:
  - sequence: array of operations in order
  - properties: {bounding_box: {max_point, min_point}}
"""

import json
import os
import glob
import argparse
from pathlib import Path


def convert_generated_to_cadseq(input_json):
    """
    Convert generated CAD JSON to CADSequence format.

    Args:
        input_json (dict): The generated CAD JSON data

    Returns:
        dict: Converted JSON in CADSequence format
    """
    entities = input_json.get("entities", {})
    bbox = input_json.get("bounding_box", {})

    # Find all ExtrudeFeature entities and build sequence
    sequence = []
    for entity_id, entity_data in entities.items():
        if entity_data.get("type") == "ExtrudeFeature":
            sequence.append({
                "type": "ExtrudeFeature",
                "entity": entity_id
            })

    # Build the converted structure
    converted = {
        "entities": entities,
        "sequence": sequence,
        "properties": {
            "bounding_box": {
                "max_point": bbox.get("max_point", {"x": 0, "y": 0, "z": 0}),
                "min_point": bbox.get("min_point", {"x": 0, "y": 0, "z": 0})
            }
        }
    }

    return converted


def process_file(input_path, output_path=None):
    """
    Process a single JSON file.

    Args:
        input_path (str): Path to input JSON file
        output_path (str, optional): Path to output JSON file.
                                     If None, overwrites input or creates .cadseq.json

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read input
        with open(input_path, 'r') as f:
            input_data = json.load(f)

        # Check if already in CADSequence format
        if "sequence" in input_data:
            print(f"✓ {input_path}: Already in CADSequence format (has 'sequence')")
            return True

        if "entities" not in input_data:
            print(f"✗ {input_path}: Missing 'entities' key - cannot convert")
            return False

        # Convert
        converted_data = convert_generated_to_cadseq(input_data)

        # Determine output path
        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_path = base + ".cadseq.json"

        # Write output
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(converted_data, f, indent=2)

        # Count extrudes
        n_extrudes = len(converted_data.get("sequence", []))
        print(f"✓ {os.path.basename(input_path)}: Converted with {n_extrudes} extrude(s) → {os.path.basename(output_path)}")
        return True

    except json.JSONDecodeError as e:
        print(f"✗ {input_path}: JSON decode error: {e}")
        return False
    except Exception as e:
        print(f"✗ {input_path}: Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert generated CAD JSON to CADSequence format"
    )
    parser.add_argument("--src", type=str, required=True, help="Source directory or file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    parser.add_argument("--pattern", type=str, default="*.json", help="File pattern to match")
    args = parser.parse_args()

    src_path = Path(args.src)

    # Handle single file vs directory
    if src_path.is_file():
        files = [str(src_path)]
    else:
        files = sorted(glob.glob(os.path.join(args.src, f"**/{args.pattern}"), recursive=True))

    if not files:
        print(f"No files found in {args.src}")
        return

    print(f"Found {len(files)} file(s)")
    print("-" * 80)

    success = 0
    failed = 0

    for input_file in files:
        # Determine output path
        if args.output and len(files) > 1:
            rel_path = os.path.relpath(input_file, args.src)
            output_file = os.path.join(args.output, os.path.splitext(rel_path)[0] + ".json")
        else:
            output_file = args.output if args.output else None

        if process_file(input_file, output_file):
            success += 1
        else:
            failed += 1

    print("-" * 80)
    print(f"\nResults: {success} successful, {failed} failed")


if __name__ == "__main__":
    main()
