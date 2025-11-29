"""
Step-by-Step Inference Validation Pipeline

This script validates the full CAD generation pipeline:
1. Check if inference output is valid (partial + complete JSON)
2. Convert complete JSON → STEP file (using DeepCAD)
3. Render STEP → image (using OpenCASCADE)

Usage:
    # Test on a single sample
    python scripts/validate_inference_pipeline.py \
        --checkpoint_path /content/gdrive/MyDrive/CAD-MLLM-checkpoints/stage3_all/checkpoint-best \
        --truncated_json /path/to/partial.json \
        --caption "Modern chair" \
        --image /path/to/image.png \
        --point_cloud /path/to/pc.npy \
        --output_dir validation_results/sample1

Conda environment: DeepCAD (for JSON→STEP conversion and rendering)
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cad_mllm import autocomplete_cad


# ============================================================================
# STEP 1: Run Inference and Validate JSON Structure
# ============================================================================

def step1_run_inference(
    checkpoint_path: str,
    truncated_json: str,
    caption: str,
    image: str = None,
    point_cloud: str = None,
    output_dir: Path = None,
) -> Dict:
    """
    Step 1: Run inference to generate complete CAD sequence.

    Returns:
        Dict with 'sequence' (complete CAD operations) and 'metadata'
    """
    print("\n" + "="*80)
    print("STEP 1: Running Inference")
    print("="*80)

    output_json = output_dir / "complete_sequence.json"

    result = autocomplete_cad(
        checkpoint_path=checkpoint_path,
        truncated_json=truncated_json,
        caption=caption,
        image=image,
        point_cloud=point_cloud,
        output_path=str(output_json),
        temperature=0.7,
        top_p=0.9,
    )

    # Validate JSON structure
    print("\n📝 Validating JSON structure...")
    is_valid, msg = validate_json_structure(result)

    if is_valid:
        print(f"✅ Valid JSON structure!")
        print(f"   - Total operations: {result['metadata']['total_operations']}")
        print(f"   - Partial operations: {result['metadata']['partial_operations']}")
        print(f"   - Generated operations: {result['metadata']['generated_operations']}")
        print(f"   - Saved to: {output_json}")
    else:
        print(f"❌ Invalid JSON: {msg}")
        return None

    return result


def validate_json_structure(result: Dict) -> tuple[bool, str]:
    """Validate that generated JSON has correct structure for CAD engine."""

    # Check required keys
    if "sequence" not in result:
        return False, "Missing 'sequence' key"

    if "metadata" not in result:
        return False, "Missing 'metadata' key"

    sequence = result["sequence"]

    # Check sequence is a list
    if not isinstance(sequence, list):
        return False, f"'sequence' must be list, got {type(sequence)}"

    # Check sequence is not empty
    if len(sequence) == 0:
        return False, "'sequence' is empty"

    # Check each operation has required fields
    required_fields = ["type"]  # At minimum, each operation needs a type

    for i, op in enumerate(sequence):
        if not isinstance(op, dict):
            return False, f"Operation {i} is not a dict: {type(op)}"

        for field in required_fields:
            if field not in op:
                return False, f"Operation {i} missing '{field}' field"

    return True, "OK"


# ============================================================================
# STEP 2: Convert JSON to STEP (using DeepCAD)
# ============================================================================

def step2_json_to_step(
    json_path: Path,
    output_dir: Path,
    deepcad_root: Path,
) -> Path:
    """
    Step 2: Convert complete JSON to STEP file using DeepCAD.

    Args:
        json_path: Path to complete JSON file
        output_dir: Directory to save STEP file
        deepcad_root: Path to DeepCAD repository root

    Returns:
        Path to generated STEP file, or None if conversion failed
    """
    print("\n" + "="*80)
    print("STEP 2: Converting JSON → STEP (using DeepCAD)")
    print("="*80)

    step_dir = output_dir / "step"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Prepare single-file conversion command
    # DeepCAD's cadlib/extrude.py can convert single JSON → STEP
    step_output = step_dir / f"{json_path.stem}.step"

    deepcad_extrude = deepcad_root / "cadlib" / "extrude.py"

    if not deepcad_extrude.exists():
        print(f"❌ DeepCAD extrude.py not found at: {deepcad_extrude}")
        print("   Please ensure DeepCAD is installed at: {deepcad_root}")
        return None

    print(f"📦 Converting: {json_path.name}")
    print(f"   Output: {step_output}")

    # Run conversion
    cmd = [
        sys.executable,
        str(deepcad_extrude),
        "-i", str(json_path),
        "-o", str(step_output),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
        )

        if result.returncode == 0 and step_output.exists():
            print(f"✅ STEP file generated: {step_output}")
            return step_output
        else:
            print(f"❌ Conversion failed!")
            print(f"   Return code: {result.returncode}")
            if result.stdout:
                print(f"   stdout: {result.stdout[:500]}")
            if result.stderr:
                print(f"   stderr: {result.stderr[:500]}")
            return None

    except subprocess.TimeoutExpired:
        print(f"❌ Conversion timeout (60s)")
        return None
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return None


# ============================================================================
# STEP 3: Render STEP to Image (using OpenCASCADE)
# ============================================================================

def step3_render_step(
    step_path: Path,
    output_dir: Path,
    num_views: int = 4,
) -> List[Path]:
    """
    Step 3: Render STEP file to images using OpenCASCADE.

    Args:
        step_path: Path to STEP file
        output_dir: Directory to save rendered images
        num_views: Number of viewpoints to render

    Returns:
        List of paths to generated image files
    """
    print("\n" + "="*80)
    print("STEP 3: Rendering STEP → Images (using OpenCASCADE)")
    print("="*80)

    render_dir = output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎨 Rendering: {step_path.name}")
    print(f"   Views: {num_views}")
    print(f"   Output: {render_dir}")

    # Use the render_cad.py script from pipeline/
    render_script = Path(__file__).parent.parent / "pipeline" / "render_cad.py"

    if not render_script.exists():
        print(f"❌ render_cad.py not found at: {render_script}")
        return []

    # Run rendering
    cmd = [
        sys.executable,
        str(render_script),
        "--src", str(step_path.parent),
        "--output", str(render_dir),
        "--num_views", str(num_views),
        "--idx", "0",
        "--num", "1",  # Process only this one file
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        # Find generated images
        rendered_images = sorted(render_dir.glob(f"{step_path.stem}*.png"))

        if rendered_images:
            print(f"✅ Rendered {len(rendered_images)} views:")
            for img in rendered_images:
                print(f"   - {img.name}")
            return rendered_images
        else:
            print(f"❌ No images generated!")
            if result.stdout:
                print(f"   stdout: {result.stdout[:500]}")
            if result.stderr:
                print(f"   stderr: {result.stderr[:500]}")
            return []

    except subprocess.TimeoutExpired:
        print(f"❌ Rendering timeout (120s)")
        return []
    except Exception as e:
        print(f"❌ Rendering error: {e}")
        return []


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate Inference Pipeline")

    # Inference inputs
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--truncated_json", type=str, required=True,
                       help="Path to truncated JSON file")
    parser.add_argument("--caption", type=str, required=True,
                       help="Text description of the CAD model")
    parser.add_argument("--image", type=str, default=None,
                       help="Optional path to image")
    parser.add_argument("--point_cloud", type=str, default=None,
                       help="Optional path to point cloud")

    # Pipeline settings
    parser.add_argument("--output_dir", type=str, default="validation_results/sample",
                       help="Output directory for all results")
    parser.add_argument("--deepcad_root", type=str, default="3rd_party/DeepCAD",
                       help="Path to DeepCAD repository")
    parser.add_argument("--num_views", type=int, default=4,
                       help="Number of viewpoints to render")

    # Control which steps to run
    parser.add_argument("--skip_step2", action="store_true",
                       help="Skip JSON→STEP conversion")
    parser.add_argument("--skip_step3", action="store_true",
                       help="Skip STEP→Image rendering")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1: Run inference
    result = step1_run_inference(
        checkpoint_path=args.checkpoint_path,
        truncated_json=args.truncated_json,
        caption=args.caption,
        image=args.image,
        point_cloud=args.point_cloud,
        output_dir=output_dir,
    )

    if result is None:
        print("\n❌ Pipeline stopped: Invalid inference output")
        return 1

    # STEP 2: Convert JSON → STEP
    if not args.skip_step2:
        json_path = output_dir / "complete_sequence.json"
        step_path = step2_json_to_step(
            json_path=json_path,
            output_dir=output_dir,
            deepcad_root=Path(args.deepcad_root),
        )

        if step_path is None:
            print("\n❌ Pipeline stopped: JSON→STEP conversion failed")
            return 2

        # STEP 3: Render STEP → Image
        if not args.skip_step3:
            rendered_images = step3_render_step(
                step_path=step_path,
                output_dir=output_dir,
                num_views=args.num_views,
            )

            if not rendered_images:
                print("\n❌ Pipeline stopped: Rendering failed")
                return 3

    # Success!
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print(f"All results saved to: {output_dir}")
    print(f"  - JSON: {output_dir / 'complete_sequence.json'}")
    if not args.skip_step2:
        print(f"  - STEP: {output_dir / 'step' / '*.step'}")
    if not args.skip_step3:
        print(f"  - Renders: {output_dir / 'renders' / '*.png'}")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
