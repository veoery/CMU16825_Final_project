#!/usr/bin/env python3
"""
Evaluate generated CAD JSON against ground truth.

Supports multiple evaluation modes:
1. Direct JSON structure comparison
2. Parameter accuracy (sequence and feature parameters)
3. Geometric metrics (if STEP files available)
"""

import json
import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import argparse

sys.path.insert(0, '/root/cmu/16825_l43d/CMU16825_Final_project')
from cad_validation import CADValidator


class GroundTruthEvaluator:
    """Evaluate generated CAD against ground truth."""

    def __init__(self, generated_dir: str, groundtruth_dir: str):
        """
        Args:
            generated_dir: Directory containing generated JSON files
            groundtruth_dir: Directory containing ground truth JSON files
        """
        self.generated_dir = generated_dir
        self.groundtruth_dir = groundtruth_dir
        self.results = {}

    def find_matching_files(self) -> List[Tuple[str, str]]:
        """
        Find matching generated and ground truth files by basename.

        Returns:
            List of (generated_path, groundtruth_path) tuples
        """
        matches = []

        # Extract IDs from generated files
        gen_files = glob.glob(os.path.join(self.generated_dir, "*.json"))
        gt_files = {
            extract_id(f): f for f in glob.glob(os.path.join(self.groundtruth_dir, "*.json"))
        }

        for gen_file in gen_files:
            gen_id = extract_id(gen_file)
            if gen_id in gt_files:
                matches.append((gen_file, gt_files[gen_id]))

        return matches

    def evaluate_pair(self, gen_path: str, gt_path: str) -> Dict[str, Any]:
        """Evaluate a generated file against its ground truth."""
        results = {
            'file': os.path.basename(gen_path),
            'match_status': 'unknown',
            'metrics': {},
        }

        try:
            with open(gen_path) as f:
                gen_data = json.load(f)
            with open(gt_path) as f:
                gt_data = json.load(f)
        except Exception as e:
            results['match_status'] = f'error_loading: {str(e)[:80]}'
            return results

        # Validation
        gen_valid, gen_issues = CADValidator.validate_all(gen_data.get('entities', {}), strict=False)
        gt_valid, gt_issues = CADValidator.validate_all(gt_data.get('entities', {}), strict=False)

        results['metrics']['generated_valid'] = gen_valid
        results['metrics']['groundtruth_valid'] = gt_valid

        if not gen_valid:
            results['match_status'] = 'generated_invalid'
            gen_issues_count = sum(len(issues) for issues in gen_issues.values())
            results['metrics']['generated_issues'] = gen_issues_count
            return results

        # Structure comparison
        gen_entities = gen_data.get('entities', {})
        gt_entities = gt_data.get('entities', {})
        gen_seq = gen_data.get('sequence', [])
        gt_seq = gt_data.get('sequence', [])

        # Entity count
        gen_entity_count = len(gen_entities)
        gt_entity_count = len(gt_entities)
        entity_count_match = gen_entity_count == gt_entity_count

        # Feature counts
        gen_sketches = sum(1 for e in gen_entities.values() if e.get('type') == 'Sketch')
        gt_sketches = sum(1 for e in gt_entities.values() if e.get('type') == 'Sketch')
        gen_extrudes = sum(1 for e in gen_entities.values() if 'Extrude' in e.get('type', ''))
        gt_extrudes = sum(1 for e in gt_entities.values() if 'Extrude' in e.get('type', ''))

        # Sequence comparison
        seq_match = len(gen_seq) == len(gt_seq)

        results['metrics']['entity_count'] = {
            'generated': gen_entity_count,
            'groundtruth': gt_entity_count,
            'match': entity_count_match,
        }

        results['metrics']['sketch_count'] = {
            'generated': gen_sketches,
            'groundtruth': gt_sketches,
            'match': gen_sketches == gt_sketches,
        }

        results['metrics']['extrude_count'] = {
            'generated': gen_extrudes,
            'groundtruth': gt_extrudes,
            'match': gen_extrudes == gt_extrudes,
        }

        results['metrics']['sequence'] = {
            'generated_length': len(gen_seq),
            'groundtruth_length': len(gt_seq),
            'match': seq_match,
        }

        # Parameter accuracy (if extrudes match in count)
        if gen_extrudes == gt_extrudes and gen_extrudes > 0:
            param_accuracy = self._evaluate_parameter_accuracy(
                gen_entities, gt_entities, gen_seq, gt_seq
            )
            results['metrics']['parameter_accuracy'] = param_accuracy

        # Overall status
        if (entity_count_match and gen_sketches == gt_sketches and
                gen_extrudes == gt_extrudes and seq_match):
            results['match_status'] = 'structure_match'
        else:
            results['match_status'] = 'structure_mismatch'

        return results

    def _evaluate_parameter_accuracy(self, gen_entities: Dict, gt_entities: Dict,
                                     gen_seq: List, gt_seq: List) -> Dict:
        """Evaluate accuracy of extrude parameters."""
        metrics = {
            'extrude_height_rmse': None,
            'profile_reference_match': 0,
            'total_profiles_compared': 0,
        }

        heights_gen = []
        heights_gt = []

        # Collect extrude heights
        for seq_item in gen_seq:
            if seq_item.get('type') == 'ExtrudeFeature':
                entity_id = seq_item.get('entity')
                entity = gen_entities.get(entity_id, {})
                extent_one = entity.get('extent_one', {})
                distance_def = extent_one.get('distance', {})
                value = distance_def.get('value', 0)
                if value:
                    heights_gen.append(float(value))

        for seq_item in gt_seq:
            if seq_item.get('type') == 'ExtrudeFeature':
                entity_id = seq_item.get('entity')
                entity = gt_entities.get(entity_id, {})
                extent_one = entity.get('extent_one', {})
                distance_def = extent_one.get('distance', {})
                value = distance_def.get('value', 0)
                if value:
                    heights_gt.append(float(value))

        # Calculate RMSE
        if heights_gen and heights_gt and len(heights_gen) == len(heights_gt):
            import math
            mse = sum((g - t) ** 2 for g, t in zip(heights_gen, heights_gt)) / len(heights_gen)
            metrics['extrude_height_rmse'] = math.sqrt(mse)
            metrics['extrude_heights_generated'] = heights_gen
            metrics['extrude_heights_groundtruth'] = heights_gt

        return metrics

    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation."""
        matches = self.find_matching_files()

        if not matches:
            print(f"❌ No matching files found!")
            print(f"   Generated dir: {self.generated_dir}")
            print(f"   Ground truth dir: {self.groundtruth_dir}")
            return {'total_pairs': 0, 'summary': {}, 'results': []}

        print(f"✓ Found {len(matches)} matching file pair(s)\n")

        all_results = []
        summary = defaultdict(int)

        for gen_path, gt_path in matches:
            result = self.evaluate_pair(gen_path, gt_path)
            all_results.append(result)

            status = result['match_status']
            summary[status] += 1

        return {
            'total_pairs': len(matches),
            'summary': dict(summary),
            'results': all_results,
        }

    def print_report(self, evaluation_results: Dict):
        """Print evaluation report."""
        print("=" * 80)
        print("EVALUATION REPORT")
        print("=" * 80)
        print(f"Total pairs evaluated: {evaluation_results['total_pairs']}\n")

        if evaluation_results['total_pairs'] == 0:
            print("No files to evaluate")
            return

        # Summary
        print("Summary:")
        for status, count in evaluation_results['summary'].items():
            print(f"  {status}: {count}")

        print("\n" + "=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80)

        for result in evaluation_results['results']:
            print(f"\n{result['file']}")
            print(f"  Status: {result['match_status']}")

            if 'metrics' in result:
                metrics = result['metrics']

                if 'generated_valid' in metrics:
                    print(f"  Generated valid: {metrics['generated_valid']}")
                    if 'generated_issues' in metrics:
                        print(f"  Generated issues: {metrics['generated_issues']}")

                if 'entity_count' in metrics:
                    ec = metrics['entity_count']
                    print(f"  Entity count: {ec['generated']} vs {ec['groundtruth']} "
                          f"({'✓' if ec['match'] else '✗'})")

                if 'sketch_count' in metrics:
                    sc = metrics['sketch_count']
                    print(f"  Sketches: {sc['generated']} vs {sc['groundtruth']} "
                          f"({'✓' if sc['match'] else '✗'})")

                if 'extrude_count' in metrics:
                    ec = metrics['extrude_count']
                    print(f"  Extrudes: {ec['generated']} vs {ec['groundtruth']} "
                          f"({'✓' if ec['match'] else '✗'})")

                if 'sequence' in metrics:
                    seq = metrics['sequence']
                    print(f"  Sequence length: {seq['generated_length']} vs {seq['groundtruth_length']} "
                          f"({'✓' if seq['match'] else '✗'})")

                if 'parameter_accuracy' in metrics:
                    pa = metrics['parameter_accuracy']
                    if pa.get('extrude_height_rmse') is not None:
                        print(f"  Height RMSE: {pa['extrude_height_rmse']:.6f}")
                        print(f"  Generated heights: {pa.get('extrude_heights_generated')}")
                        print(f"  Ground truth heights: {pa.get('extrude_heights_groundtruth')}")

        print("\n" + "=" * 80)


def extract_id(filepath: str) -> str:
    """
    Extract ID from filename for matching.

    Examples:
        'generated_cad_00003816_00001_v6.json' → '00003816_00001'
        'reference_cad_00003816_00001.json' → '00003816_00001'
    """
    basename = os.path.basename(filepath)
    # Remove extension
    name = os.path.splitext(basename)[0]

    # Extract numeric ID (format: XXXXXXXX_XXXXX)
    import re
    match = re.search(r'(\d{8}_\d{5})', name)
    if match:
        return match.group(1)

    return name  # Fallback to full name


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated CAD against ground truth"
    )
    parser.add_argument("--generated", type=str, required=True,
                        help="Directory or file with generated JSON")
    parser.add_argument("--groundtruth", type=str, required=True,
                        help="Directory or file with ground truth JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    # Handle single file comparison
    if os.path.isfile(args.generated) and os.path.isfile(args.groundtruth):
        # Single file mode
        print(f"Single file comparison mode\n")
        evaluator = GroundTruthEvaluator('.', '.')
        result = evaluator.evaluate_pair(args.generated, args.groundtruth)
        results = {
            'total_pairs': 1,
            'summary': {result['match_status']: 1},
            'results': [result]
        }
    else:
        # Directory mode
        evaluator = GroundTruthEvaluator(args.generated, args.groundtruth)
        results = evaluator.run_evaluation()

    evaluator.print_report(results)

    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
