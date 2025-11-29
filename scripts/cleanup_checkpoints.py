"""
Cleanup old checkpoints to free up Google Drive space.

Run this script AFTER training completes to clean up intermediate checkpoints.
"""

import os
import shutil
from pathlib import Path
import json


def get_checkpoint_info(checkpoint_dir):
    """Get info about a checkpoint."""
    metadata_file = checkpoint_dir / "checkpoint_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None


def cleanup_checkpoints(base_dir, keep_best_only=True):
    """Clean up checkpoint directories.

    Args:
        base_dir: Root checkpoint directory (e.g., /content/gdrive/MyDrive/CAD-MLLM-checkpoints)
        keep_best_only: If True, delete all checkpoints except best
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return

    total_freed = 0

    for stage_dir in base_path.iterdir():
        if not stage_dir.is_dir():
            continue

        print(f"\n{'='*80}")
        print(f"Stage: {stage_dir.name}")
        print(f"{'='*80}")

        checkpoints = []
        best_checkpoint = None

        # Find all checkpoints
        for ckpt_dir in stage_dir.iterdir():
            if not ckpt_dir.is_dir():
                continue

            # Get size
            size_bytes = sum(f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file())
            size_gb = size_bytes / (1024**3)

            # Get metadata
            info = get_checkpoint_info(ckpt_dir)
            loss = info.get('loss', float('inf')) if info else float('inf')

            checkpoint_info = {
                'path': ckpt_dir,
                'name': ckpt_dir.name,
                'size_gb': size_gb,
                'loss': loss,
            }

            if 'best' in ckpt_dir.name.lower():
                best_checkpoint = checkpoint_info
            else:
                checkpoints.append(checkpoint_info)

        # Display all checkpoints
        print(f"\nFound {len(checkpoints)} interval checkpoints:")
        for ckpt in sorted(checkpoints, key=lambda x: x['loss']):
            print(f"  - {ckpt['name']}: {ckpt['size_gb']:.2f} GB, loss={ckpt['loss']:.4f}")

        if best_checkpoint:
            print(f"\nBest checkpoint: {best_checkpoint['name']}: {best_checkpoint['size_gb']:.2f} GB, loss={best_checkpoint['loss']:.4f}")

        # Cleanup decision
        if keep_best_only:
            print(f"\n🗑️  Deleting {len(checkpoints)} interval checkpoints (keeping only best)...")

            for ckpt in checkpoints:
                try:
                    shutil.rmtree(ckpt['path'])
                    total_freed += ckpt['size_gb']
                    print(f"  ✓ Deleted: {ckpt['name']} ({ckpt['size_gb']:.2f} GB freed)")
                except Exception as e:
                    print(f"  ✗ Failed to delete {ckpt['name']}: {e}")
        else:
            print(f"\n✓ Keeping all checkpoints (use --keep_best_only to clean up)")

    print(f"\n{'='*80}")
    print(f"CLEANUP SUMMARY")
    print(f"{'='*80}")
    print(f"Total space freed: {total_freed:.2f} GB")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cleanup old checkpoints")
    parser.add_argument("--checkpoint_dir", type=str,
                       default="/content/gdrive/MyDrive/CAD-MLLM-checkpoints",
                       help="Root checkpoint directory")
    parser.add_argument("--keep_best_only", action="store_true",
                       help="Delete all checkpoints except best")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be deleted without actually deleting")

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - No files will be deleted\n")

    cleanup_checkpoints(args.checkpoint_dir, keep_best_only=args.keep_best_only)
