#!/usr/bin/env python3
"""
Wrapper script to evaluate topology metrics on STEP files.
Runs all 4 topology metrics: SegE, DangEL, SIR, FluxEE

Usage:
    python run_topology_eval.py \
        --input_dir ../output_ckpt_2/output_checkpoint-epoch0-step140-20251128_072200_eval_B1/step \
        --output_dir ./topology_results
"""

import os
import glob
import json
import argparse
import trimesh
import numpy as np
from pathlib import Path
from eval_topology import (
    seg_error,
    dangling_edge_length,
    self_intersection_ratio,
    flux_enclosure_error
)

def evaluate_mesh(mesh_path):
    """Evaluate a single mesh file."""
    try:
        # Load mesh
        mesh = trimesh.load(mesh_path, process=False)

        # Handle assemblies - take first mesh
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return None
            mesh = list(mesh.geometry.values())[0]

        # Ensure it's a valid mesh
        if not isinstance(mesh, trimesh.Trimesh):
            return None

        # Compute metrics (no ground truth needed for single-mesh metrics)
        results = {
            'file': os.path.basename(mesh_path),
            'status': 'success',
            'metrics': {
                'DangEL': float(dangling_edge_length(mesh)),  # Boundary edge length
                'SIR': float(self_intersection_ratio(mesh)),  # Self-intersection ratio
                'FluxEE': float(flux_enclosure_error(mesh)),  # Flux enclosure error
            },
            'mesh_info': {
                'vertices': int(len(mesh.vertices)),
                'faces': int(len(mesh.faces)),
                'is_watertight': bool(mesh.is_watertight)
            }
        }
        return results

    except Exception as e:
        return {
            'file': os.path.basename(mesh_path),
            'status': 'error',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate topology metrics on STEP/OBJ files")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with STEP/OBJ files')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results (default: input_dir/topology_results)')
    parser.add_argument('--format', type=str, default='step', help='File format (step, obj, stl)')

    args = parser.parse_args()

    # Use default output dir if not specified
    if args.output_dir is None:
        # Save to: CMU16825_Final_project/output_ckpt_2/output_checkpoint.../topology_results
        args.output_dir = os.path.join(args.input_dir, '..', 'topology_results')

    args.output_dir = os.path.abspath(args.output_dir)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all mesh files
    mesh_pattern = os.path.join(args.input_dir, f'*.{args.format}')
    mesh_files = sorted(glob.glob(mesh_pattern))

    if not mesh_files:
        print(f"❌ No {args.format} files found in {args.input_dir}")
        return

    print(f"\n📊 TOPOLOGY EVALUATION")
    print(f"{'='*80}")
    print(f"Input Dir:    {args.input_dir}")
    print(f"Output Dir:   {args.output_dir}")
    print(f"Files Found:  {len(mesh_files)}")
    print(f"{'='*80}\n")

    # Evaluate all files
    all_results = []
    successful = 0
    failed = 0

    for i, mesh_path in enumerate(mesh_files, 1):
        filename = os.path.basename(mesh_path)
        print(f"[{i:2d}/{len(mesh_files)}] {filename:<40}", end=' ... ', flush=True)

        result = evaluate_mesh(mesh_path)

        if result is None:
            print("⚠️  SKIPPED (not a valid mesh)")
            continue

        all_results.append(result)

        if result['status'] == 'success':
            print("✅ SUCCESS")
            successful += 1
        else:
            print(f"❌ ERROR: {result['error'][:50]}")
            failed += 1

    # Save detailed results
    results_json = os.path.join(args.output_dir, 'topology_results.json')
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"📁 Results saved to: {results_json}")

    # Summary statistics
    successful_results = [r for r in all_results if r['status'] == 'success']

    if successful_results:
        metrics_summary = {
            'total_files': len(mesh_files),
            'successful': len(successful_results),
            'failed': failed,
            'metrics': {
                'DangEL': {
                    'mean': float(np.mean([r['metrics']['DangEL'] for r in successful_results])),
                    'min': float(np.min([r['metrics']['DangEL'] for r in successful_results])),
                    'max': float(np.max([r['metrics']['DangEL'] for r in successful_results])),
                    'std': float(np.std([r['metrics']['DangEL'] for r in successful_results])),
                },
                'SIR': {
                    'mean': float(np.mean([r['metrics']['SIR'] for r in successful_results])),
                    'min': float(np.min([r['metrics']['SIR'] for r in successful_results])),
                    'max': float(np.max([r['metrics']['SIR'] for r in successful_results])),
                    'std': float(np.std([r['metrics']['SIR'] for r in successful_results])),
                },
                'FluxEE': {
                    'mean': float(np.mean([r['metrics']['FluxEE'] for r in successful_results])),
                    'min': float(np.min([r['metrics']['FluxEE'] for r in successful_results])),
                    'max': float(np.max([r['metrics']['FluxEE'] for r in successful_results])),
                    'std': float(np.std([r['metrics']['FluxEE'] for r in successful_results])),
                }
            }
        }

        # Save summary
        summary_json = os.path.join(args.output_dir, 'topology_summary.json')
        with open(summary_json, 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        # Print summary
        print(f"\n📈 SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Total Files:    {metrics_summary['total_files']}")
        print(f"Successful:     {metrics_summary['successful']} ✅")
        print(f"Failed:         {metrics_summary['failed']} ❌")
        print(f"\nMetrics (Successful files only):")
        print(f"\n  DangEL (Boundary Edge Length - lower is better):")
        print(f"    Mean: {metrics_summary['metrics']['DangEL']['mean']:.6f}")
        print(f"    Min:  {metrics_summary['metrics']['DangEL']['min']:.6f}")
        print(f"    Max:  {metrics_summary['metrics']['DangEL']['max']:.6f}")
        print(f"\n  SIR (Self-Intersection Ratio - lower is better, 0-1):")
        print(f"    Mean: {metrics_summary['metrics']['SIR']['mean']:.6f}")
        print(f"    Min:  {metrics_summary['metrics']['SIR']['min']:.6f}")
        print(f"    Max:  {metrics_summary['metrics']['SIR']['max']:.6f}")
        print(f"\n  FluxEE (Flux Enclosure Error - lower is better, ~0 = closed):")
        print(f"    Mean: {metrics_summary['metrics']['FluxEE']['mean']:.6f}")
        print(f"    Min:  {metrics_summary['metrics']['FluxEE']['min']:.6f}")
        print(f"    Max:  {metrics_summary['metrics']['FluxEE']['max']:.6f}")
        print(f"\n📄 Summary saved to: {summary_json}")
        print(f"{'='*80}\n")
    else:
        print("⚠️  No successful evaluations to summarize")


if __name__ == '__main__':
    main()
