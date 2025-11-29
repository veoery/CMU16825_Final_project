#!/usr/bin/env python3
"""
Comprehensive JSON structure validation for CAD models.

Evaluates:
1. JSON validity (well-formed, loadable)
2. Schema validation (required keys, entity types)
3. Structure comparison with ground truth (entity counts, types, parameters)
4. Statistical summary

Usage:
    python validate_json_structure.py \
        --generated_dir ../output_ckpt_2/output_checkpoint-.../  \
        --gt_dir /path/to/gt/test/json \
        --output_dir ./json_validation_results
"""

import os
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

# Expected entity types in CAD models
VALID_ENTITY_TYPES = {
    'Sketch', 'ExtrudeFeature', 'RevolveFeature', 'PadFeature',
    'PocketFeature', 'FilletFeature', 'ChamferFeature', 'HoleFeature',
    'PatternFeature', 'MirrorFeature', 'DraftFeature', 'ShellFeature',
    'SweepFeature', 'LoftFeature', 'WireFeature', 'PlaneFeature'
}

# Parameter value ranges for features
PARAM_RANGES = {
    'distance': (0.0, 10000.0),  # mm
    'angle': (-360.0, 360.0),    # degrees
    'radius': (0.0, 10000.0),    # mm
    'scale': (0.1, 100.0)        # dimensionless
}


class JSONValidator:
    """Validates CAD JSON structure."""

    def __init__(self, gt_root=None):
        self.gt_root = gt_root
        self.results = []

    def validate_json_file(self, json_path):
        """Validate a single JSON file."""
        result = {
            'file': os.path.basename(json_path),
            'path': json_path,
            'status': 'unknown',
            'errors': [],
            'warnings': [],
            'metrics': {}
        }

        # 1. Check if loadable
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            result['status'] = 'error_invalid_json'
            result['errors'].append(f"JSON load failed: {str(e)[:80]}")
            return result

        # 2. Check top-level structure
        if not isinstance(data, dict):
            result['errors'].append("Root is not a dictionary")
            result['status'] = 'error_structure'
            return result

        required_keys = {'entities'}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            result['errors'].append(f"Missing required keys: {missing}")
            result['status'] = 'error_structure'
            return result

        # 3. Validate entities
        entities = data.get('entities', {})
        if not isinstance(entities, dict):
            result['errors'].append("'entities' is not a dictionary")
            result['status'] = 'error_structure'
            return result

        entity_types = []
        param_counts = defaultdict(int)

        for entity_id, entity in entities.items():
            if not isinstance(entity, dict):
                result['warnings'].append(f"Entity {entity_id} is not a dict")
                continue

            entity_type = entity.get('type', 'Unknown')
            entity_types.append(entity_type)

            # Count parameters
            for key, value in entity.items():
                if isinstance(value, dict) and 'value' in value:
                    param_counts[key] += 1

        # 4. Gather metrics
        result['metrics'] = {
            'entity_count': len(entities),
            'entity_types': entity_types,
            'type_distribution': dict(Counter(entity_types)),
            'parameters': dict(param_counts),
            'has_properties': 'properties' in data,
            'has_sequence': 'sequence' in data
        }

        # 5. Compare with ground truth if available
        if self.gt_root:
            gt_result = self._compare_with_groundtruth(json_path, data)
            result['metrics']['gt_comparison'] = gt_result

        # 6. Determine status
        if result['errors']:
            result['status'] = 'error'
        elif result['warnings']:
            result['status'] = 'warning'
        else:
            result['status'] = 'valid'

        return result

    def _compare_with_groundtruth(self, gen_path, gen_data):
        """Compare generated file with ground truth."""
        try:
            # Extract ID from filename
            basename = os.path.basename(gen_path)
            sample_id = basename.split('_')[0]  # e.g., "00900284"

            # Find ground truth file
            folder = sample_id[:4]  # e.g., "0090"
            gt_pattern = os.path.join(self.gt_root, folder, f"{sample_id}*.json")
            gt_files = glob.glob(gt_pattern)

            if not gt_files:
                return {'status': 'no_gt_found'}

            gt_path = gt_files[0]
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)

            # Compare metrics
            gen_entities = gen_data.get('entities', {})
            gt_entities = gt_data.get('entities', {})

            gen_types = [e.get('type') for e in gen_entities.values()]
            gt_types = [e.get('type') for e in gt_entities.values()]

            comparison = {
                'status': 'compared',
                'gt_file': os.path.basename(gt_path),
                'entity_count': {
                    'generated': len(gen_entities),
                    'groundtruth': len(gt_entities),
                    'difference': len(gen_entities) - len(gt_entities),
                    'match': len(gen_entities) == len(gt_entities)
                },
                'entity_types': {
                    'generated': dict(Counter(gen_types)),
                    'groundtruth': dict(Counter(gt_types))
                }
            }

            return comparison

        except Exception as e:
            return {'status': 'comparison_error', 'error': str(e)[:80]}

    def validate_directory(self, gen_dir, pattern='*.json'):
        """Validate all JSON files in a directory."""
        json_files = sorted(glob.glob(os.path.join(gen_dir, pattern)))

        if not json_files:
            print(f"❌ No JSON files found in {gen_dir}")
            return []

        print(f"📊 VALIDATING {len(json_files)} JSON FILES")
        print(f"{'='*80}")

        for i, json_path in enumerate(json_files, 1):
            filename = os.path.basename(json_path)
            print(f"[{i:2d}/{len(json_files)}] {filename:<50}", end=' ... ', flush=True)

            result = self.validate_json_file(json_path)
            self.results.append(result)

            status_symbol = '✅' if result['status'] == 'valid' else '⚠️' if result['status'] == 'warning' else '❌'
            print(f"{status_symbol} {result['status']}")

        return self.results

    def print_summary(self):
        """Print summary statistics."""
        if not self.results:
            return

        valid = [r for r in self.results if r['status'] == 'valid']
        warnings = [r for r in self.results if r['status'] == 'warning']
        errors = [r for r in self.results if r['status'] in ['error', 'error_invalid_json', 'error_structure']]

        print(f"\n{'='*80}")
        print(f"📈 SUMMARY")
        print(f"{'='*80}")
        print(f"Total Files:    {len(self.results)}")
        print(f"✅ Valid:       {len(valid)}")
        print(f"⚠️  Warnings:    {len(warnings)}")
        print(f"❌ Errors:      {len(errors)}")

        if valid:
            # Entity statistics
            entity_counts = [r['metrics']['entity_count'] for r in valid]
            print(f"\n📋 ENTITY STATISTICS (Valid files):")
            print(f"  Entity Count:")
            print(f"    Mean:  {np.mean(entity_counts):.1f}")
            print(f"    Min:   {np.min(entity_counts):.0f}")
            print(f"    Max:   {np.max(entity_counts):.0f}")
            print(f"    Std:   {np.std(entity_counts):.2f}")

            # Type distribution
            all_types = []
            for r in valid:
                all_types.extend(r['metrics']['entity_types'])
            type_dist = Counter(all_types)
            print(f"\n🔧 ENTITY TYPE DISTRIBUTION:")
            for etype, count in type_dist.most_common():
                pct = (count / len(all_types)) * 100
                print(f"  {etype:20s}: {count:3d} ({pct:5.1f}%)")

            # Ground truth comparison
            gt_comparisons = [r['metrics'].get('gt_comparison', {})
                              for r in valid if 'gt_comparison' in r['metrics']]
            if gt_comparisons:
                matching = sum(1 for c in gt_comparisons if c.get('entity_count', {}).get('match'))
                print(f"\n🎯 GROUND TRUTH COMPARISON:")
                print(f"  Files with matching entity count: {matching}/{len(gt_comparisons)}")

        print(f"{'='*80}\n")

    def save_results(self, output_dir):
        """Save detailed results to JSON."""
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        results_file = os.path.join(output_dir, 'json_validation_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save summary
        summary = {
            'total': len(self.results),
            'valid': sum(1 for r in self.results if r['status'] == 'valid'),
            'warnings': sum(1 for r in self.results if r['status'] == 'warning'),
            'errors': sum(1 for r in self.results if r['status'] in ['error', 'error_invalid_json', 'error_structure'])
        }

        summary_file = os.path.join(output_dir, 'json_validation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📁 Results saved to:")
        print(f"  - {results_file}")
        print(f"  - {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Validate CAD JSON structure")
    parser.add_argument('--generated_dir', type=str, required=True,
                       help='Directory with generated JSON files')
    parser.add_argument('--gt_dir', type=str, default=None,
                       help='Ground truth JSON directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for validation results')
    parser.add_argument('--pattern', type=str, default='*_repaired.json',
                       help='File pattern to match')

    args = parser.parse_args()

    # Create validator
    validator = JSONValidator(gt_root=args.gt_dir)

    # Validate directory
    validator.validate_directory(args.generated_dir, pattern=args.pattern)

    # Print summary
    validator.print_summary()

    # Save results
    validator.save_results(args.output_dir)


if __name__ == '__main__':
    main()
