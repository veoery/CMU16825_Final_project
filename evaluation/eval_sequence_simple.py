#!/usr/bin/env python3
"""
Simple sequence/structure accuracy evaluation comparing generated JSON with ground truth.

Evaluates:
1. Entity type accuracy (Sketch, ExtrudeFeature, etc.)
2. Entity count accuracy
3. Feature parameter presence
4. Overall model structure similarity

Usage:
    python eval_sequence_simple.py \
        --generated_dir ../output_ckpt_2/output_checkpoint-.../  \
        --gt_dir /path/to/gt/test/json \
        --output_dir ./sequence_eval_results
"""

import os
import json
import glob
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

class SequenceEvaluator:
    """Evaluate CAD sequence structure accuracy."""

    def __init__(self, gt_root):
        self.gt_root = gt_root
        self.results = []

    def extract_entity_types(self, json_data):
        """Extract entity type sequence from JSON."""
        entities = json_data.get('entities', {})
        types = []
        for entity in entities.values():
            types.append(entity.get('type', 'Unknown'))
        return types

    def extract_parameters(self, json_data):
        """Extract parameter keys from features."""
        entities = json_data.get('entities', {})
        params = defaultdict(list)

        for entity_id, entity in entities.items():
            etype = entity.get('type', 'Unknown')
            for key, value in entity.items():
                if isinstance(value, dict) and 'value' in value:
                    params[etype].append(key)

        return dict(params)

    def calculate_type_accuracy(self, gen_types, gt_types):
        """Calculate entity type sequence accuracy."""
        if len(gt_types) == 0:
            return 0.0

        # Exact match
        if gen_types == gt_types:
            return 1.0

        # Partial match - count matching consecutive types
        matches = 0
        for i in range(min(len(gen_types), len(gt_types))):
            if gen_types[i] == gt_types[i]:
                matches += 1
            else:
                break

        return matches / len(gt_types)

    def calculate_entity_count_accuracy(self, gen_count, gt_count):
        """Calculate entity count accuracy."""
        if gt_count == 0:
            return 1.0 if gen_count == 0 else 0.0

        return 1.0 - abs(gen_count - gt_count) / gt_count

    def calculate_type_distribution_similarity(self, gen_types, gt_types):
        """Calculate similarity of entity type distribution."""
        if not gen_types or not gt_types:
            return 0.0

        gen_dist = Counter(gen_types)
        gt_dist = Counter(gt_types)

        all_types = set(gen_dist.keys()) | set(gt_dist.keys())

        if not all_types:
            return 0.0

        # Jaccard similarity for type distributions
        intersect = sum(min(gen_dist.get(t, 0), gt_dist.get(t, 0)) for t in all_types)
        union = sum(max(gen_dist.get(t, 0), gt_dist.get(t, 0)) for t in all_types)

        return intersect / union if union > 0 else 0.0

    def evaluate_pair(self, gen_path):
        """Evaluate a generated file against ground truth."""
        result = {
            'file': os.path.basename(gen_path),
            'status': 'unknown',
            'metrics': {}
        }

        try:
            with open(gen_path, 'r') as f:
                gen_data = json.load(f)
        except Exception as e:
            result['status'] = 'error_loading'
            result['error'] = str(e)[:80]
            return result

        # Extract generated sequence
        gen_types = self.extract_entity_types(gen_data)
        gen_params = self.extract_parameters(gen_data)
        gen_count = len(gen_data.get('entities', {}))

        # Find ground truth
        basename = os.path.basename(gen_path)
        sample_id = basename.split('_')[0]
        folder = sample_id[:4]

        gt_pattern = os.path.join(self.gt_root, folder, f"{sample_id}*.json")
        gt_files = glob.glob(gt_pattern)

        if not gt_files:
            result['status'] = 'no_gt'
            result['metrics'] = {
                'gen_entity_count': gen_count,
                'gen_entity_types': dict(Counter(gen_types))
            }
            return result

        try:
            with open(gt_files[0], 'r') as f:
                gt_data = json.load(f)
        except Exception as e:
            result['status'] = 'error_loading_gt'
            return result

        # Extract ground truth sequence
        gt_types = self.extract_entity_types(gt_data)
        gt_params = self.extract_parameters(gt_data)
        gt_count = len(gt_data.get('entities', {}))

        # Calculate metrics
        result['metrics'] = {
            'entity_count_acc': self.calculate_entity_count_accuracy(gen_count, gt_count),
            'entity_type_sequence_acc': self.calculate_type_accuracy(gen_types, gt_types),
            'type_distribution_sim': self.calculate_type_distribution_similarity(gen_types, gt_types),
            'gen_entity_count': gen_count,
            'gt_entity_count': gt_count,
            'gen_entity_types': dict(Counter(gen_types)),
            'gt_entity_types': dict(Counter(gt_types))
        }

        result['status'] = 'success'
        return result

    def evaluate_directory(self, gen_dir, pattern='*_repaired.json'):
        """Evaluate all files in directory."""
        json_files = sorted(glob.glob(os.path.join(gen_dir, pattern)))

        if not json_files:
            print(f"❌ No JSON files found in {gen_dir}")
            return []

        print(f"\n📊 SEQUENCE ACCURACY EVALUATION")
        print(f"{'='*80}")
        print(f"Files to evaluate: {len(json_files)}")
        print(f"{'='*80}\n")

        for i, json_path in enumerate(json_files, 1):
            filename = os.path.basename(json_path)
            print(f"[{i:2d}/{len(json_files)}] {filename:<50}", end=' ... ', flush=True)

            result = self.evaluate_pair(json_path)
            self.results.append(result)

            if result['status'] == 'success':
                print(f"✅")
            elif result['status'] == 'no_gt':
                print(f"⚠️  No GT")
            else:
                print(f"❌ {result['status'][:20]}")

        return self.results

    def print_summary(self):
        """Print evaluation summary."""
        successful = [r for r in self.results if r['status'] == 'success']

        if not successful:
            print(f"\n❌ No successful evaluations")
            return

        print(f"\n{'='*80}")
        print(f"📈 EVALUATION SUMMARY")
        print(f"{'='*80}\n")

        # Calculate statistics
        entity_count_accs = [r['metrics']['entity_count_acc'] for r in successful]
        type_seq_accs = [r['metrics']['entity_type_sequence_acc'] for r in successful]
        type_dist_sims = [r['metrics']['type_distribution_sim'] for r in successful]

        print(f"Total files evaluated: {len(self.results)}")
        print(f"✅ Successful:         {len(successful)}")
        print(f"⚠️  No GT:             {sum(1 for r in self.results if r['status'] == 'no_gt')}")
        print(f"❌ Errors:             {sum(1 for r in self.results if r['status'] not in ['success', 'no_gt'])}")

        print(f"\n🎯 ACCURACY METRICS:")
        print(f"\n  Entity Count Accuracy (lower = more entities in generated):")
        print(f"    Mean:   {np.mean(entity_count_accs):.3f}")
        print(f"    Median: {np.median(entity_count_accs):.3f}")
        print(f"    Min:    {np.min(entity_count_accs):.3f}")
        print(f"    Max:    {np.max(entity_count_accs):.3f}")

        print(f"\n  Entity Type Sequence Accuracy (0-1, 1 = perfect match):")
        print(f"    Mean:   {np.mean(type_seq_accs):.3f}")
        print(f"    Median: {np.median(type_seq_accs):.3f}")
        print(f"    Min:    {np.min(type_seq_accs):.3f}")
        print(f"    Max:    {np.max(type_seq_accs):.3f}")

        print(f"\n  Type Distribution Similarity (0-1, 1 = identical distribution):")
        print(f"    Mean:   {np.mean(type_dist_sims):.3f}")
        print(f"    Median: {np.median(type_dist_sims):.3f}")
        print(f"    Min:    {np.min(type_dist_sims):.3f}")
        print(f"    Max:    {np.max(type_dist_sims):.3f}")

        print(f"\n{'='*80}\n")

    def save_results(self, output_dir):
        """Save detailed results."""
        os.makedirs(output_dir, exist_ok=True)

        # Save JSON results
        results_file = os.path.join(output_dir, 'sequence_eval_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save CSV for easier analysis
        csv_data = []
        for r in self.results:
            row = {
                'file': r['file'],
                'status': r['status'],
                'entity_count_acc': r['metrics'].get('entity_count_acc', np.nan),
                'type_seq_acc': r['metrics'].get('entity_type_sequence_acc', np.nan),
                'type_dist_sim': r['metrics'].get('type_distribution_sim', np.nan),
                'gen_count': r['metrics'].get('gen_entity_count', 0),
                'gt_count': r['metrics'].get('gt_entity_count', 0)
            }
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(output_dir, 'sequence_eval_results.csv')
        df.to_csv(csv_file, index=False)

        print(f"📁 Results saved:")
        print(f"  - {results_file}")
        print(f"  - {csv_file}\n")

        return results_file, csv_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAD sequence structure accuracy")
    parser.add_argument('--generated_dir', type=str, required=True,
                       help='Directory with generated JSON files')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Ground truth JSON directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--pattern', type=str, default='*_repaired.json',
                       help='File pattern to match')

    args = parser.parse_args()

    # Create evaluator
    evaluator = SequenceEvaluator(args.gt_dir)

    # Evaluate
    evaluator.evaluate_directory(args.generated_dir, pattern=args.pattern)

    # Print summary
    evaluator.print_summary()

    # Save results
    evaluator.save_results(args.output_dir)


if __name__ == '__main__':
    main()
