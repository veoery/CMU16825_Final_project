"""
Test CAD Truncation Pipeline
Tests truncation and visualization on 10 sample files from folder 0000.
"""

import sys
from pathlib import Path
import json
from typing import List

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

from truncate_dataset import CADTruncator

# Try to import visualizer (optional if OCC not installed)
try:
    from visualize_truncation import CADVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization not available: {e}")
    print("Proceeding with truncation only. Install pythonocc-core for visualization.")
    print()
    VISUALIZATION_AVAILABLE = False
    CADVisualizer = None


def get_test_files(json_dir: Path, folder: str = "0000", count: int = 10) -> List[Path]:
    """
    Get test files from specified folder.

    Args:
        json_dir: Base JSON directory
        folder: Subfolder name (e.g., "0000")
        count: Number of files to get

    Returns:
        List of JSON file paths
    """
    folder_path = json_dir / folder
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return []

    json_files = sorted(folder_path.glob("*.json"))

    if len(json_files) == 0:
        print(f"Error: No JSON files found in {folder_path}")
        return []

    # Take first 'count' files
    test_files = json_files[:count]
    print(f"Selected {len(test_files)} test files from {folder}/")

    return test_files


def print_truncation_stats(truncated_json: Path):
    """Print statistics for a truncated file."""
    with open(truncated_json, 'r') as f:
        data = json.load(f)
        metadata = data.get('truncation_metadata', {})

        orig_ops = metadata.get('original_operations', '?')
        kept_ops = metadata.get('kept_operations', '?')
        pct = metadata.get('truncation_percentage', '?')

        print(f"    {truncated_json.name}: {kept_ops}/{orig_ops} ops ({pct}%)")


def main():
    print("="*80)
    print("CAD TRUNCATION PIPELINE TEST")
    print("="*80)
    print()

    # Setup paths
    project_root = Path(__file__).parent.parent
    json_dir = project_root / "data" / "Omni-CAD-subset" / "json"
    output_dir = project_root / "data" / "Omni-CAD-subset" / "json_truncated_test"
    vis_dir = project_root / "data" / "Omni-CAD-subset" / "visualizations"

    print(f"Input directory:  {json_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Visualization directory: {vis_dir}")
    print()

    # Get test files
    test_files = get_test_files(json_dir, folder="0000", count=10)

    if not test_files:
        print("No test files found. Exiting.")
        return

    # Initialize truncator
    truncator = CADTruncator(min_operations=1)

    # Number of truncation steps to generate
    num_steps = 6

    print("="*80)
    print("STEP 1: TRUNCATING JSON FILES")
    print("="*80)
    print()

    truncation_results = {}  # Store results for visualization step

    for i, json_file in enumerate(test_files, 1):
        print(f"[{i}/{len(test_files)}] Processing {json_file.name}...")

        # Preserve folder structure (0000/)
        rel_path = json_file.relative_to(json_dir)
        output_subdir = output_dir / rel_path.parent

        # Generate truncations
        results = truncator.generate_truncations(
            json_file,
            output_subdir,
            num_steps=num_steps
        )

        if results:
            print(f"  [OK] Generated {len(results)} truncated versions:")
            for trunc_path, _ in results:
                print_truncation_stats(trunc_path)

            # Store for visualization
            truncation_results[json_file] = results
        else:
            print(f"  [FAILED] Failed to generate truncations")

        print()

    print("="*80)
    print("STEP 2: GENERATING VISUALIZATIONS")
    print("="*80)
    print()
    print("Note: This step may take several minutes per file as it converts")
    print("      JSON to STEP and renders with OpenCASCADE.")
    print()

    # Initialize visualizer
    visualize = False
    if VISUALIZATION_AVAILABLE:
        try:
            visualizer = CADVisualizer()
            visualize = True
        except Exception as e:
            print(f"Warning: Could not initialize visualizer: {e}")
            print("Skipping visualization step.")
            visualize = False
    else:
        print("Visualization not available (pythonocc-core not installed).")
        print("Skipping visualization step.")
        print()

    if visualize:
        for i, (original_json, truncated_versions) in enumerate(truncation_results.items(), 1):
            print(f"[{i}/{len(truncation_results)}] Visualizing {original_json.name}...")

            # Visualize only first truncated version (20%) to save time in testing
            if truncated_versions:
                truncated_json_path, _ = truncated_versions[0]

                # Create output path for comparison image
                vis_subdir = vis_dir / original_json.parent.name
                vis_subdir.mkdir(parents=True, exist_ok=True)
                comparison_img = vis_subdir / f"{original_json.stem}_comparison.png"

                try:
                    success = visualizer.create_side_by_side_comparison(
                        truncated_json_path,
                        original_json,
                        comparison_img,
                        temp_dir=vis_dir / "temp_step"
                    )

                    if success:
                        print(f"  [OK] Comparison saved to {comparison_img}")
                    else:
                        print(f"  [FAILED] Failed to create comparison")

                except Exception as e:
                    print(f"  [ERROR] Error during visualization: {e}")

            print()

    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    total_truncated = sum(len(results) for results in truncation_results.values())
    print(f"Original files processed:  {len(test_files)}")
    print(f"Truncated files generated: {total_truncated}")
    print(f"Truncation strategy:       Up to {num_steps} evenly-spaced steps (excluding 100%)")

    if visualize:
        vis_count = len(truncation_results)
        print(f"Visualizations created:    {vis_count}")

    print()
    print(f"Output saved to: {output_dir}")
    if visualize:
        print(f"Visualizations saved to: {vis_dir}")
    print()
    print("="*80)
    print("TEST COMPLETE!")
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
