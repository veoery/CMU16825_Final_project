"""
Test CAD Truncation on Mixed File Sizes
Tests on a mix of small, medium, and large sequence files.
"""

import sys
from pathlib import Path
import json

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

from truncate_dataset import CADTruncator

# Try to import visualizer (optional if OCC not installed)
try:
    from visualize_truncation import CADVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization not available: {e}")
    print("Proceeding with truncation only.")
    print()
    VISUALIZATION_AVAILABLE = False
    CADVisualizer = None


def get_mixed_test_files(json_dir: Path, count_per_size: int = 3):
    """
    Get a mix of files with different operation counts.

    Args:
        json_dir: Base JSON directory
        count_per_size: Number of files per size category

    Returns:
        List of (file_path, operation_count) tuples
    """
    files_by_ops = {'small': [], 'medium': [], 'large': []}

    # Scan multiple folders
    for folder in sorted(json_dir.iterdir())[:10]:  # Check first 10 folders
        if not folder.is_dir():
            continue

        for json_file in sorted(folder.glob("*.json"))[:20]:  # Sample 20 files per folder
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    ops = len(data.get('sequence', []))

                    # Categorize by size
                    if ops >= 12:
                        files_by_ops['large'].append((json_file, ops))
                    elif ops >= 6:
                        files_by_ops['medium'].append((json_file, ops))
                    elif ops >= 2:
                        files_by_ops['small'].append((json_file, ops))
            except:
                continue

    # Select samples
    selected = []

    # Get largest files
    large_files = sorted(files_by_ops['large'], key=lambda x: x[1], reverse=True)[:count_per_size]
    selected.extend(large_files)

    # Get medium files
    medium_files = sorted(files_by_ops['medium'], key=lambda x: x[1], reverse=True)[:count_per_size]
    selected.extend(medium_files)

    # Get small files
    small_files = sorted(files_by_ops['small'], key=lambda x: x[1], reverse=True)[:count_per_size]
    selected.extend(small_files)

    return selected


def main():
    print("="*80)
    print("CAD TRUNCATION TEST - MIXED FILE SIZES")
    print("="*80)
    print()

    # Setup paths
    project_root = Path(__file__).parent.parent
    json_dir = project_root / "data" / "Omni-CAD-subset" / "json"
    output_dir = project_root / "data" / "Omni-CAD-subset" / "json_truncated_mixed"

    print(f"Input directory:  {json_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Get mixed test files
    print("Scanning for files with varied operation counts...")
    test_files = get_mixed_test_files(json_dir, count_per_size=3)

    if not test_files:
        print("No test files found. Exiting.")
        return

    print(f"Selected {len(test_files)} files:")
    print()

    # Group by size for display
    large = [f for f in test_files if f[1] >= 12]
    medium = [f for f in test_files if 6 <= f[1] < 12]
    small = [f for f in test_files if f[1] < 6]

    if large:
        print(f"  LARGE files ({len(large)}):")
        for file_path, ops in large:
            print(f"    {file_path.name}: {ops} operations")

    if medium:
        print(f"  MEDIUM files ({len(medium)}):")
        for file_path, ops in medium:
            print(f"    {file_path.name}: {ops} operations")

    if small:
        print(f"  SMALL files ({len(small)}):")
        for file_path, ops in small:
            print(f"    {file_path.name}: {ops} operations")

    print()

    # Initialize truncator
    truncator = CADTruncator(min_operations=1)
    max_versions = 5

    print("="*80)
    print("STEP 1: TRUNCATING FILES")
    print("="*80)
    print()

    total_truncated = 0
    truncation_results = {}  # Store for visualization

    for i, (json_file, ops) in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] {json_file.name} ({ops} operations)")

        # Preserve folder structure
        rel_path = json_file.relative_to(json_dir)
        output_subdir = output_dir / rel_path.parent

        # Generate truncations
        results = truncator.generate_truncations(
            json_file,
            output_subdir,
            max_versions=max_versions
        )

        if results:
            total_truncated += len(results)
            truncation_results[json_file] = results
            print(f"  Generated {len(results)} truncated versions")
            for trunc_path, data in results:
                meta = data['truncation_metadata']
                print(f"    {trunc_path.name}: {meta['kept_operations']}/{meta['original_operations']} ops ({meta['truncation_percentage']:.1f}%)")
        else:
            print(f"  No truncations generated")

        print()

    # Visualization step
    print("="*80)
    print("STEP 2: VISUALIZING (Side-by-Side Comparisons)")
    print("="*80)
    print()

    if VISUALIZATION_AVAILABLE:
        try:
            visualizer = CADVisualizer()
            vis_dir = output_dir.parent / "visualizations_mixed"

            print("Creating side-by-side comparison images...")
            print("Note: This may take several minutes per file.")
            print()

            visualized = 0
            for i, (original_json, truncated_versions) in enumerate(truncation_results.items(), 1):
                if not truncated_versions:
                    continue

                print(f"[{i}/{len(truncation_results)}] Visualizing {original_json.name}...")

                # Visualize only the first truncated version
                truncated_json_path, _ = truncated_versions[0]

                # Create output path
                vis_subdir = vis_dir / original_json.parent.name
                comparison_img = vis_subdir / f"{original_json.stem}_comparison.png"

                try:
                    success = visualizer.create_side_by_side_comparison(
                        truncated_json_path,
                        original_json,
                        comparison_img,
                        temp_dir=vis_dir / "temp_step"
                    )

                    if success:
                        visualized += 1
                        print(f"  [OK] Saved to {comparison_img.name}")
                    else:
                        print(f"  [FAILED] Could not create visualization")
                except Exception as e:
                    print(f"  [ERROR] {e}")

                print()

            print(f"Created {visualized}/{len(truncation_results)} visualizations")
            print(f"Saved to: {vis_dir}")
            print()

        except Exception as e:
            print(f"Visualization failed: {e}")
            print()
    else:
        print("Visualization not available (pythonocc-core not installed).")
        print("To enable: conda activate DeepCAD && python scripts/test_truncation_mixed.py")
        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files processed:           {len(test_files)}")
    print(f"  Large (12+ ops):         {len(large)}")
    print(f"  Medium (6-11 ops):       {len(medium)}")
    print(f"  Small (2-5 ops):         {len(small)}")
    print(f"Total truncations created: {total_truncated}")
    print(f"Truncation strategy:       Up to {max_versions} versions, consecutively numbered (tr_01 to tr_05)")

    if VISUALIZATION_AVAILABLE and 'visualized' in locals():
        print(f"Visualizations created:    {visualized}")

    print()
    print(f"Truncated JSONs saved to: {output_dir}")

    if VISUALIZATION_AVAILABLE and 'visualized' in locals() and visualized > 0:
        print(f"Visualizations saved to:  {vis_dir}")

    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
