"""
Evaluate CAD Autocomplete Model

Simple evaluation script using existing inference wrapper.

Usage:
    python scripts/evaluate_autocomplete.py \
        --checkpoint_path /content/gdrive/MyDrive/CAD-MLLM-checkpoints/stage3_all/checkpoint-best \
        --truncated_json_root /path/to/json_truncated \
        --omnicad_txt_path /path/to/txt \
        --omnicad_json_root /path/to/json \
        --omnicad_img_root /path/to/img \
        --omnicad_pc_root /path/to/pointcloud \
        --num_samples 50 \
        --output_results results/eval.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from cad_mllm import CADAutocomplete


def load_test_samples(
    truncated_json_root: str,
    txt_root: str,
    img_root: str,
    pc_root: str,
    json_root: str,
    num_samples: int = None,
) -> List[Dict]:
    """Load test samples with all modalities."""
    truncated_root = Path(truncated_json_root)
    samples = []

    print(f"Loading test samples from {truncated_root}...")

    # Find all truncated JSON files
    truncated_files = sorted(truncated_root.rglob("*.json"))

    if num_samples:
        truncated_files = truncated_files[:num_samples]

    for trunc_path in tqdm(truncated_files, desc="Loading samples"):
        # Parse filename: 00000071_00005_tr_02.json
        filename = trunc_path.stem  # e.g., 00000071_00005_tr_02
        parts = filename.split("_")

        if len(parts) < 4 or parts[2] != "tr":
            continue

        base_id = f"{parts[0]}_{parts[1]}"  # e.g., 00000071_00005
        subfolder = parts[0]  # e.g., 0000

        # Construct paths
        txt_path = Path(txt_root) / f"{base_id}.txt"
        img_path = Path(img_root) / subfolder / f"{base_id}.png"
        pc_path = Path(pc_root) / subfolder / f"{base_id}.npy"
        gt_json_path = Path(json_root) / f"{base_id}.json"

        # Load caption
        if txt_path.exists():
            caption = txt_path.read_text().strip()
        else:
            caption = "CAD model"

        # Load ground truth
        if not gt_json_path.exists():
            print(f"Warning: Ground truth not found for {base_id}, skipping")
            continue

        with open(gt_json_path, 'r') as f:
            gt_data = json.load(f)

        # Load truncated data to get kept_operations
        with open(trunc_path, 'r') as f:
            trunc_data = json.load(f)

        sample = {
            "truncated_json": str(trunc_path),
            "caption": caption,
            "image": str(img_path) if img_path.exists() else None,
            "point_cloud": str(pc_path) if pc_path.exists() else None,
            "ground_truth": gt_data["sequence"],
            "kept_operations": trunc_data.get("kept_operations", 0),
            "sample_id": base_id,
            "truncation_id": filename,
        }

        samples.append(sample)

    print(f"Loaded {len(samples)} test samples")
    return samples


def compute_metrics(results: List[Dict], test_samples: List[Dict]) -> Dict:
    """Compute evaluation metrics from inference results."""
    total_generated = 0
    total_expected = 0
    correct_types = 0
    total_types_compared = 0

    for result, sample in zip(results, test_samples):
        # Get generated operations (after partial)
        kept_ops = sample["kept_operations"]
        generated_ops = result["sequence"][kept_ops:]
        expected_ops = sample["ground_truth"][kept_ops:]

        total_generated += len(generated_ops)
        total_expected += len(expected_ops)

        # Count matching operation types
        for i in range(min(len(generated_ops), len(expected_ops))):
            if generated_ops[i].get("type") == expected_ops[i].get("type"):
                correct_types += 1
            total_types_compared += 1

    metrics = {
        "num_samples": len(results),
        "avg_operations_generated": total_generated / len(results) if results else 0,
        "avg_operations_expected": total_expected / len(test_samples) if test_samples else 0,
        "completion_rate": total_generated / total_expected if total_expected > 0 else 0,
        "operation_type_accuracy": correct_types / total_types_compared if total_types_compared > 0 else 0,
    }

    return metrics


def evaluate(
    checkpoint_path: str,
    test_samples: List[Dict],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 3000,
) -> Dict:
    """Run evaluation using existing inference wrapper."""

    print(f"\n{'='*80}")
    print(f"Loading model from {checkpoint_path}...")
    print(f"{'='*80}\n")

    # Initialize model once
    autocomplete = CADAutocomplete(
        checkpoint_path=checkpoint_path,
        device="cuda",
        dtype="bfloat16",
    )

    # Prepare samples for batch_complete (use existing method!)
    inference_samples = [
        {
            "truncated_json": s["truncated_json"],
            "caption": s["caption"],
            "image": s.get("image"),
            "point_cloud": s.get("point_cloud"),
        }
        for s in test_samples
    ]

    print(f"\n{'='*80}")
    print(f"Running inference on {len(inference_samples)} samples...")
    print(f"{'='*80}\n")

    # Use batch_complete from existing inference wrapper!
    results = autocomplete.batch_complete(
        inference_samples,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    print(f"\n{'='*80}")
    print(f"Computing metrics...")
    print(f"{'='*80}\n")

    # Compute metrics
    metrics = compute_metrics(results, test_samples)

    return {
        "metrics": metrics,
        "predictions": [
            {
                "sample_id": s["sample_id"],
                "truncation_id": s["truncation_id"],
                "caption": s["caption"],
                "kept_operations": s["kept_operations"],
                "generated_operations": r["metadata"]["generated_operations"],
                "total_operations": r["metadata"]["total_operations"],
            }
            for s, r in zip(test_samples, results)
        ],
        "config": {
            "checkpoint_path": checkpoint_path,
            "num_samples": len(test_samples),
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAD Autocomplete Model")

    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to trained checkpoint")

    # Data
    parser.add_argument("--truncated_json_root", type=str, required=True,
                       help="Root directory containing truncated JSON files")
    parser.add_argument("--omnicad_json_root", type=str, required=True,
                       help="Root directory containing full JSON files (ground truth)")
    parser.add_argument("--omnicad_txt_path", type=str, required=True,
                       help="Directory containing text descriptions")
    parser.add_argument("--omnicad_img_root", type=str, required=True,
                       help="Root directory containing images")
    parser.add_argument("--omnicad_pc_root", type=str, required=True,
                       help="Root directory containing point clouds")

    # Evaluation settings
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to evaluate (default: all)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus sampling parameter")
    parser.add_argument("--max_new_tokens", type=int, default=3000,
                       help="Maximum tokens to generate")

    # Output
    parser.add_argument("--output_results", type=str, default="results/eval_results.json",
                       help="Path to save evaluation results")

    args = parser.parse_args()

    # Load test samples
    test_samples = load_test_samples(
        truncated_json_root=args.truncated_json_root,
        txt_root=args.omnicad_txt_path,
        img_root=args.omnicad_img_root,
        pc_root=args.omnicad_pc_root,
        json_root=args.omnicad_json_root,
        num_samples=args.num_samples,
    )

    if not test_samples:
        print("No test samples found!")
        return

    # Run evaluation
    eval_results = evaluate(
        checkpoint_path=args.checkpoint_path,
        test_samples=test_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    # Print metrics
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Samples evaluated: {eval_results['metrics']['num_samples']}")
    print(f"Avg operations generated: {eval_results['metrics']['avg_operations_generated']:.1f}")
    print(f"Avg operations expected: {eval_results['metrics']['avg_operations_expected']:.1f}")
    print(f"Completion rate: {eval_results['metrics']['completion_rate']:.2%}")
    print(f"Operation type accuracy: {eval_results['metrics']['operation_type_accuracy']:.2%}")
    print(f"{'='*80}\n")

    # Save results
    output_path = Path(args.output_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
