#!/usr/bin/env python3
"""
Convert generated JSON CAD files to sequence evaluation format (pickle).

Expected pickle format for eval_seq.py:
{
    "uid_xxxx": {
        "level_1": {
            "gt_cad_vec": ground_truth_array,
            "pred_cad_vec": [pred1, pred2, ...],
            "cd": [cd1, cd2, ...]
        }
    }
}

Usage:
    python convert_json_to_sequence.py \
        --generated_dir ../output_ckpt_2/output_checkpoint-.../  \
        --gt_dir /path/to/gt/test/json \
        --output_pkl ./sequence_data.pkl
"""

import os
import json
import pickle
import glob
import numpy as np
import argparse
from collections import defaultdict
from pathlib import Path

def json_to_cad_vec(json_data):
    """Convert JSON structure to CAD vector representation.

    Extracts key parameters from entities and sequence to create
    a simplified CAD vector for comparison.
    """
    vec = []

    entities = json_data.get('entities', {})

    # Count entity types
    type_counts = defaultdict(int)
    for entity in entities.values():
        etype = entity.get('type', 'Unknown')
        type_counts[etype] += 1

    # Create feature vector: [num_sketches, num_extrudes, num_total_entities, ...]
    # This is a simplified representation - can be extended
    vec = [
        type_counts.get('Sketch', 0),
        type_counts.get('ExtrudeFeature', 0),
        len(entities),
    ]

    return np.array(vec, dtype=np.float32)


def load_ground_truth(gt_file):
    """Load ground truth JSON and convert to CAD vector."""
    try:
        with open(gt_file, 'r') as f:
            data = json.load(f)
        return json_to_cad_vec(data)
    except Exception as e:
        return None


def convert_generated_to_sequence(generated_dir, gt_dir, output_pkl):
    """Convert generated JSON files to sequence evaluation pickle format."""

    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)

    # Find all generated JSON files
    gen_files = sorted(glob.glob(os.path.join(generated_dir, '*_repaired.json')))

    print(f"\n📊 CONVERTING JSON TO SEQUENCE FORMAT")
    print(f"{'='*80}")
    print(f"Generated files: {len(gen_files)}")
    print(f"Output pickle:  {output_pkl}")
    print(f"{'='*80}\n")

    sequence_data = {}
    successful = 0
    failed = 0
    no_gt = 0

    for i, gen_file in enumerate(gen_files, 1):
        filename = os.path.basename(gen_file)
        basename = filename.replace('_repaired.json', '')

        # Extract sample ID and variant
        parts = basename.split('_')
        sample_id = parts[0]  # e.g., "00900284"
        variant = parts[1] if len(parts) > 1 else "00001"

        # Create UID (combination of sample_id and variant)
        uid = f"{sample_id}_{variant}"

        print(f"[{i:2d}/{len(gen_files)}] {filename:<50}", end=' ... ', flush=True)

        try:
            # Load generated JSON
            with open(gen_file, 'r') as f:
                gen_data = json.load(f)

            # Convert to CAD vector
            gen_vec = json_to_cad_vec(gen_data)

            # Find ground truth
            folder = sample_id[:4]  # e.g., "0090"
            gt_pattern = os.path.join(gt_dir, folder, f"{sample_id}*.json")
            gt_files = glob.glob(gt_pattern)

            if not gt_files:
                print(f"⚠️  No GT found")
                no_gt += 1
                continue

            # Load ground truth
            gt_file = gt_files[0]
            gt_vec = load_ground_truth(gt_file)

            if gt_vec is None:
                print(f"❌ Failed to load GT")
                failed += 1
                continue

            # Create sequence data entry
            if uid not in sequence_data:
                sequence_data[uid] = {}

            sequence_data[uid]["level_1"] = {
                "gt_cad_vec": gt_vec,
                "pred_cad_vec": [gen_vec],  # Wrap in list for multi-prediction support
                "cd": [0.0]  # Placeholder chamfer distance
            }

            print(f"✅ Success")
            successful += 1

        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            failed += 1

    # Save pickle file
    print(f"\n{'='*80}")
    print(f"💾 SAVING PICKLE FILE")
    print(f"{'='*80}\n")

    with open(output_pkl, 'wb') as f:
        pickle.dump(sequence_data, f)

    print(f"✅ Saved {len(sequence_data)} UIDs to {output_pkl}")
    print(f"\nStatistics:")
    print(f"  ✅ Successful:  {successful}")
    print(f"  ❌ Failed:      {failed}")
    print(f"  ⚠️  No GT found: {no_gt}")
    print(f"\n{'='*80}\n")

    return output_pkl


def main():
    parser = argparse.ArgumentParser(description="Convert JSON to sequence evaluation format")
    parser.add_argument('--generated_dir', type=str, required=True,
                       help='Directory with generated JSON files')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Ground truth JSON directory')
    parser.add_argument('--output_pkl', type=str, required=True,
                       help='Output pickle file path')

    args = parser.parse_args()

    # Convert
    pkl_path = convert_generated_to_sequence(
        args.generated_dir,
        args.gt_dir,
        args.output_pkl
    )

    print(f"✨ Ready for eval_seq.py evaluation!")
    print(f"   Run: python eval_seq.py --input_path {pkl_path} --output_dir ./seq_results --validate_only")


if __name__ == '__main__':
    main()
