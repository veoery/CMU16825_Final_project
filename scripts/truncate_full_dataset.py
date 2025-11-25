"""
Truncate Full Dataset
Processes all JSON files in data/Omni-CAD-subset/json and creates truncated versions.
No visualization - just truncation.
"""

import sys
from pathlib import Path
import time
from datetime import datetime

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

from truncate_dataset import CADTruncator


def main():
    print("="*80)
    print("FULL DATASET TRUNCATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "Omni-CAD-subset" / "json"
    output_dir = project_root / "data" / "Omni-CAD-subset" / "json_truncated"

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Count total files
    print("Scanning for JSON files...")
    all_json_files = list(input_dir.rglob("*.json"))
    total_files = len(all_json_files)

    print(f"Found {total_files:,} JSON files")
    print()

    # Initialize truncator
    truncator = CADTruncator(min_operations=1)
    max_versions = 5

    # Statistics
    stats = {
        'processed': 0,
        'skipped': 0,
        'truncations_created': 0,
        'failed': 0,
        'start_time': time.time()
    }

    print("="*80)
    print("PROCESSING FILES")
    print("="*80)
    print(f"Strategy: Up to {max_versions} versions per file, consecutively numbered (tr_01 to tr_05)")
    print()

    # Process files
    for i, json_file in enumerate(all_json_files, 1):
        stats['processed'] += 1

        # Show progress every 100 files or for first 10
        if i <= 10 or i % 100 == 0 or i == total_files:
            elapsed = time.time() - stats['start_time']
            rate = stats['processed'] / elapsed if elapsed > 0 else 0
            remaining = (total_files - stats['processed']) / rate if rate > 0 else 0
            eta_hours = remaining / 3600

            print(f"[{i}/{total_files}] {json_file.name}")
            print(f"  Progress: {i/total_files*100:.1f}% | Rate: {rate:.1f} files/sec | ETA: {eta_hours:.2f}h")

        # Preserve folder structure
        rel_path = json_file.relative_to(input_dir)
        output_subdir = output_dir / rel_path.parent

        # Check if already processed (skip if output exists)
        expected_output = output_subdir / f"{json_file.stem}_tr_01.json"
        if expected_output.exists():
            stats['skipped'] += 1
            if i <= 10 or i % 100 == 0:
                print(f"  Skipped (already exists)")
            continue

        # Generate truncations
        try:
            results = truncator.generate_truncations(
                json_file,
                output_subdir,
                max_versions=max_versions
            )

            if results:
                stats['truncations_created'] += len(results)
                if i <= 10 or i % 100 == 0:
                    print(f"  Created {len(results)} truncations")
            else:
                if i <= 10 or i % 100 == 0:
                    print(f"  No truncations (file too small or only 1 viable level)")
        except Exception as e:
            stats['failed'] += 1
            print(f"  [ERROR] {e}")

        # Print separator for readability
        if i <= 10 or i % 100 == 0 or i == total_files:
            print()

    # Final statistics
    elapsed = time.time() - stats['start_time']
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print()
    print("Statistics:")
    print(f"  Total files scanned:       {total_files:,}")
    print(f"  Files processed:           {stats['processed']:,}")
    print(f"  Files skipped (existing):  {stats['skipped']:,}")
    print(f"  Truncations created:       {stats['truncations_created']:,}")
    print(f"  Failed:                    {stats['failed']:,}")
    print()
    print(f"Average rate: {stats['processed']/elapsed:.2f} files/sec")
    print()
    print(f"Output saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Processing stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
