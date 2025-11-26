#!/usr/bin/env python
"""Analyze wandb sweep results and generate A100 config recommendations."""

import argparse
import wandb
import pandas as pd
import numpy as np
from pathlib import Path
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", required=True, help="Wandb sweep ID")
    parser.add_argument("--entity", default=None, help="Wandb entity/username")
    parser.add_argument("--project", default="CAD-MLLM-Hyperparam-Sweep", help="Wandb project")
    parser.add_argument("--output_dir", default="./sweep_analysis", help="Output directory")
    args = parser.parse_args()

    # Initialize wandb API
    api = wandb.Api()

    # Get sweep
    if args.entity:
        sweep_path = f"{args.entity}/{args.project}/{args.sweep_id}"
    else:
        sweep_path = f"{args.project}/{args.sweep_id}"

    print(f"Fetching sweep: {sweep_path}")
    sweep = api.sweep(sweep_path)

    # Extract run data
    runs = []
    for run in sweep.runs:
        if run.state == "finished":
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            summary = {k: v for k, v in run.summary.items() if isinstance(v, (int, float))}

            runs.append({
                'run_id': run.id,
                'run_name': run.name,
                'state': run.state,
                'final_loss': summary.get('train/loss', float('inf')),
                'seq_length': config.get('max_seq_length', 0),
                'learning_rate': config.get('stage3_lr', 0),
                'lora_r': config.get('lora_r', 0),
                'lora_alpha': config.get('lora_alpha', 0),
                'warmup_steps': config.get('warmup_steps', 0),
                'samples': config.get('max_train_samples', 0),
            })

    df = pd.DataFrame(runs)

    if len(df) == 0:
        print("No finished runs found!")
        return

    print(f"\nAnalyzed {len(df)} completed runs")
    print(f"Sequence lengths tested: {sorted(df['seq_length'].unique())}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save raw data
    df.to_csv(output_dir / "sweep_results.csv", index=False)
    print(f"\nSaved results to {output_dir / 'sweep_results.csv'}")

    # Analyze by sequence length
    print("\n" + "="*60)
    print("ANALYSIS BY SEQUENCE LENGTH")
    print("="*60)

    a100_configs = []

    for seq_len in sorted(df['seq_length'].unique()):
        subset = df[df['seq_length'] == seq_len].sort_values('final_loss')

        print(f"\n--- Sequence Length: {seq_len} ---")
        print(f"Runs: {len(subset)}")
        print(f"Best loss: {subset.iloc[0]['final_loss']:.4f}")

        # Top 3 configs
        print("\nTop 3 Configurations:")
        for i, row in subset.head(3).iterrows():
            print(f"{i+1}. Loss: {row['final_loss']:.4f} | "
                  f"LR: {row['learning_rate']:.2e} | "
                  f"LoRA r={row['lora_r']} α={row['lora_alpha']} | "
                  f"Warmup: {row['warmup_steps']}")

        # Best config for this seq_length
        best = subset.iloc[0]
        a100_configs.append({
            'seq_length_tested': int(seq_len),
            'recommended_a100_seq_length': int(seq_len * 8),  # Scale up for A100
            'learning_rate': float(best['learning_rate']),
            'lora_r': int(best['lora_r']),
            'lora_alpha': int(best['lora_alpha']),
            'warmup_steps': int(best['warmup_steps'] * 4),  # Scale warmup
            'final_loss': float(best['final_loss']),
        })

    # Generate A100 recommendations
    print("\n" + "="*60)
    print("RECOMMENDED A100 CONFIGURATIONS")
    print("="*60)

    for config in a100_configs:
        print(f"\nBased on seq_len={config['seq_length_tested']} testing:")
        print(f"  Recommended for A100 (seq_len={config['recommended_a100_seq_length']}):")
        print(f"    --stage3_lr {config['learning_rate']:.2e}")
        print(f"    --lora_r {config['lora_r']}")
        print(f"    --lora_alpha {config['lora_alpha']}")
        print(f"    --warmup_steps {config['warmup_steps']}")
        print(f"    --max_seq_length {config['recommended_a100_seq_length']}")
        print(f"  (Expected loss baseline: ~{config['final_loss']:.4f})")

    # Save A100 configs
    with open(output_dir / "a100_recommended_configs.json", 'w') as f:
        json.dump(a100_configs, f, indent=2)

    print(f"\n✓ Saved A100 configs to {output_dir / 'a100_recommended_configs.json'}")

    # Scaling analysis
    if len(a100_configs) >= 2:
        print("\n" + "="*60)
        print("SCALING TRENDS (for A100 extrapolation)")
        print("="*60)

        # LR vs seq_length
        lrs = [c['learning_rate'] for c in a100_configs]
        seqs = [c['seq_length_tested'] for c in a100_configs]

        if len(set(lrs)) > 1:
            lr_ratio = lrs[1] / lrs[0] if lrs[0] > 0 else 1
            seq_ratio = seqs[1] / seqs[0] if seqs[0] > 0 else 1
            print(f"\nLearning Rate scaling:")
            print(f"  seq_len {seqs[0]} → {seqs[1]} ({seq_ratio:.1f}x)")
            print(f"  Best LR {lrs[0]:.2e} → {lrs[1]:.2e} ({lr_ratio:.2f}x)")

            # Extrapolate to A100's typical seq_len
            target_seq = 16384
            extrapolated_lr = lrs[0] * (target_seq / seqs[0]) ** np.log(lr_ratio) / np.log(seq_ratio)
            print(f"\n  Extrapolated for seq_len={target_seq} on A100:")
            print(f"    → LR ≈ {extrapolated_lr:.2e}")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review full results on wandb dashboard")
    print(f"2. Use recommended A100 configs from {output_dir / 'a100_recommended_configs.json'}")
    print("3. Start A100 training with best config")
    print("\nGood luck! 🚀")

if __name__ == "__main__":
    main()
