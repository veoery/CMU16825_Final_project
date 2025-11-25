#!/usr/bin/env python
"""Quick script to check dataset sizes and estimate total storage needed."""

import os
from pathlib import Path

def get_dir_size(directory):
    """Calculate total size of a directory in bytes."""
    total = 0
    try:
        for entry in Path(directory).rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        print(f"Error reading {directory}: {e}")
    return total

def format_bytes(bytes_size):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def count_files(directory):
    """Count total files in directory."""
    try:
        return sum(1 for _ in Path(directory).rglob('*') if _.is_file())
    except:
        return 0

def main():
    data_root = Path(__file__).parent / "data" / "Omni-CAD"

    print("=" * 80)
    print("OMNI-CAD DATASET SIZE ANALYSIS")
    print("=" * 80)
    print()

    directories = {
        "json (CAD sequences)": data_root / "json",
        "txt (text captions)": data_root / "txt",
        "step (mesh files)": data_root / "step",
        "json_step (temp)": data_root / "json_step",
        "pcd (point clouds)": data_root / "pcd",
        "img (rendered images)": data_root / "img"
    }

    total_size = 0

    for name, path in directories.items():
        if path.exists():
            size = get_dir_size(path)
            files = count_files(path)
            total_size += size
            print(f"{name:25} {format_bytes(size):>12}  ({files:,} files)")
        else:
            print(f"{name:25} {'NOT CREATED':>12}")

    print()
    print("-" * 80)
    print(f"{'TOTAL CURRENT SIZE':25} {format_bytes(total_size):>12}")
    print("=" * 80)
    print()

    # Estimate final sizes based on samples
    print("ESTIMATED FINAL SIZES (if all steps completed):")
    print("-" * 80)

    json_dir = data_root / "json"
    if json_dir.exists():
        total_json_files = count_files(json_dir)
        print(f"Total JSON files: {total_json_files:,}")
        print()

        # Estimates based on typical CAD data sizes
        estimates = {
            "JSON (existing)": get_dir_size(json_dir) if json_dir.exists() else 0,
            "TXT (existing)": get_dir_size(data_root / "txt") if (data_root / "txt").exists() else 0,
            "STEP files": total_json_files * 50 * 1024,  # ~50 KB per STEP file
            "Point clouds (.npy)": total_json_files * 800 * 1024,  # ~800 KB per point cloud
            "Images (optional)": total_json_files * 100 * 1024,  # ~100 KB per rendered image
        }

        estimated_total = sum(estimates.values())

        for name, size in estimates.items():
            print(f"{name:30} {format_bytes(size):>12}")

        print()
        print("-" * 80)
        print(f"{'ESTIMATED TOTAL':30} {format_bytes(estimated_total):>12}")
        print("=" * 80)
        print()

        print("BREAKDOWN BY NECESSITY:")
        print("-" * 80)
        print(f"Required (JSON + TXT):          {format_bytes(estimates['JSON (existing)'] + estimates['TXT (existing)']):>12}")
        print(f"Processing output (STEP + PCD): {format_bytes(estimates['STEP files'] + estimates['Point clouds (.npy)']):>12}")
        print(f"Optional (Images):              {format_bytes(estimates['Images (optional)']):>12}")
        print("=" * 80)

if __name__ == "__main__":
    main()
