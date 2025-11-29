"""
Colab-Friendly Inference Validation

Copy and paste these cells into Google Colab to validate your trained model.
Run cells step-by-step to see where the pipeline might fail.
"""

# ============================================================================
# COLAB CELL 1: Setup and Test Single Inference
# ============================================================================

from cad_mllm import autocomplete_cad
import json
from pathlib import Path

# Configure paths
checkpoint_path = "/content/gdrive/MyDrive/CAD-MLLM-checkpoints/stage3_all/checkpoint-best"
data_root = "/content/data/Omni-CAD-subset"

# Test on a single sample
truncated_json = f"{data_root}/json_truncated/0000/00000071_00005_tr_02.json"
caption = "Modern minimalist chair"
image = f"{data_root}/img/0000/00000071_00005.png"
point_cloud = f"{data_root}/pointcloud/0000/00000071_00005.npy"

output_dir = Path("/content/validation_test")
output_dir.mkdir(parents=True, exist_ok=True)

# Run inference
print("=" * 80)
print("STEP 1: Testing Inference")
print("=" * 80)

result = autocomplete_cad(
    checkpoint_path=checkpoint_path,
    truncated_json=truncated_json,
    caption=caption,
    image=image,
    point_cloud=point_cloud,
    output_path=str(output_dir / "complete_sequence.json"),
    temperature=0.7,
)

# Display results
print(f"\n✅ Inference complete!")
print(f"  Caption: {result['metadata']['caption']}")
print(f"  Partial operations: {result['metadata']['partial_operations']}")
print(f"  Generated operations: {result['metadata']['generated_operations']}")
print(f"  Total operations: {result['metadata']['total_operations']}")

# Check JSON structure
print(f"\n📝 Validating JSON structure...")
sequence = result["sequence"]
print(f"  - Sequence is a list: {isinstance(sequence, list)}")
print(f"  - Sequence length: {len(sequence)}")
print(f"  - First operation type: {sequence[0].get('type', 'N/A')}")
print(f"  - Last operation type: {sequence[-1].get('type', 'N/A')}")

# Show a few operations
print(f"\n📋 Sample operations:")
for i, op in enumerate(sequence[:3]):
    print(f"  Operation {i}: {json.dumps(op, indent=2)[:100]}...")


# ============================================================================
# COLAB CELL 2: Inspect Generated vs Ground Truth
# ============================================================================

import json

# Load truncated JSON to see what was given as input
with open(truncated_json, 'r') as f:
    truncated_data = json.load(f)

kept_operations = truncated_data.get("kept_operations", 0)

# Load ground truth
gt_json = f"{data_root}/json/00000071_00005.json"
with open(gt_json, 'r') as f:
    gt_data = json.load(f)

gt_sequence = gt_data["sequence"]

print("=" * 80)
print("STEP 2: Compare Generated vs Ground Truth")
print("=" * 80)

print(f"\nInput (partial):")
print(f"  - Kept operations: {kept_operations}")
print(f"  - Total GT operations: {len(gt_sequence)}")
print(f"  - Operations to generate: {len(gt_sequence) - kept_operations}")

print(f"\nModel output:")
print(f"  - Total generated operations: {len(result['sequence'])}")
print(f"  - Generated portion: {result['metadata']['generated_operations']}")

# Compare operation types
generated_ops = result['sequence'][kept_operations:]
expected_ops = gt_sequence[kept_operations:]

print(f"\nOperation type comparison (first 10):")
print(f"{'Index':<8} {'Expected':<15} {'Generated':<15} {'Match':<10}")
print("-" * 60)
for i in range(min(10, len(expected_ops))):
    expected_type = expected_ops[i].get('type', 'N/A') if i < len(expected_ops) else 'N/A'
    generated_type = generated_ops[i].get('type', 'N/A') if i < len(generated_ops) else 'N/A'
    match = "✓" if expected_type == generated_type else "✗"
    print(f"{i:<8} {expected_type:<15} {generated_type:<15} {match:<10}")

# Calculate accuracy
if expected_ops and generated_ops:
    correct = sum(1 for i in range(min(len(expected_ops), len(generated_ops)))
                  if expected_ops[i].get('type') == generated_ops[i].get('type'))
    accuracy = correct / min(len(expected_ops), len(generated_ops)) * 100
    print(f"\nOperation type accuracy: {accuracy:.1f}%")


# ============================================================================
# COLAB CELL 3: Convert JSON → STEP (Requires DeepCAD environment)
# ============================================================================

# NOTE: This requires conda activate DeepCAD
# If not in DeepCAD conda env, this will fail

import subprocess
import sys

print("=" * 80)
print("STEP 3: Converting JSON → STEP")
print("=" * 80)

json_file = output_dir / "complete_sequence.json"
step_output_dir = output_dir / "step"
step_output_dir.mkdir(parents=True, exist_ok=True)
step_file = step_output_dir / "complete_sequence.step"

# Path to DeepCAD extrude script
deepcad_extrude = "/content/3rd_party/DeepCAD/cadlib/extrude.py"

print(f"\nConverting: {json_file}")
print(f"Output: {step_file}")

cmd = [
    sys.executable,
    deepcad_extrude,
    "-i", str(json_file),
    "-o", str(step_file),
]

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode == 0 and step_file.exists():
        print(f"\n✅ STEP file generated successfully!")
        print(f"  File: {step_file}")
        print(f"  Size: {step_file.stat().st_size / 1024:.1f} KB")
    else:
        print(f"\n❌ Conversion failed!")
        print(f"  Return code: {result.returncode}")
        print(f"  stdout: {result.stdout[:500]}")
        print(f"  stderr: {result.stderr[:500]}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nMake sure you're in the DeepCAD conda environment:")
    print("  !conda activate DeepCAD")


# ============================================================================
# COLAB CELL 4: Render STEP → Images (Requires OpenCASCADE)
# ============================================================================

# NOTE: This requires OpenCASCADE and pythonocc-core installed

print("=" * 80)
print("STEP 4: Rendering STEP → Images")
print("=" * 80)

render_output_dir = output_dir / "renders"
render_output_dir.mkdir(parents=True, exist_ok=True)

# Use the render_cad.py script
render_script = "/content/CMU16825_Final_project/pipeline/render_cad.py"

print(f"\nRendering: {step_file}")
print(f"Output dir: {render_output_dir}")

cmd = [
    sys.executable,
    render_script,
    "--src", str(step_output_dir),
    "--output", str(render_output_dir),
    "--num_views", "4",
    "--idx", "0",
    "--num", "1",
]

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Check for rendered images
    rendered_images = sorted(render_output_dir.glob("*.png"))

    if rendered_images:
        print(f"\n✅ Rendered {len(rendered_images)} images!")
        for img in rendered_images:
            print(f"  - {img.name} ({img.stat().st_size / 1024:.1f} KB)")

        # Display images in Colab
        from IPython.display import Image, display
        print(f"\n🖼️ Preview:")
        for img in rendered_images[:2]:  # Show first 2 views
            display(Image(filename=str(img)))
    else:
        print(f"\n❌ No images generated!")
        print(f"  stdout: {result.stdout[:500]}")
        print(f"  stderr: {result.stderr[:500]}")

except Exception as e:
    print(f"\n❌ Error: {e}")


# ============================================================================
# COLAB CELL 5: Full Pipeline (All Steps)
# ============================================================================

# Run the full validation pipeline script
!python /content/CMU16825_Final_project/scripts/validate_inference_pipeline.py \
    --checkpoint_path /content/gdrive/MyDrive/CAD-MLLM-checkpoints/stage3_all/checkpoint-best \
    --truncated_json /content/data/Omni-CAD-subset/json_truncated/0000/00000071_00005_tr_02.json \
    --caption "Modern minimalist chair" \
    --image /content/data/Omni-CAD-subset/img/0000/00000071_00005.png \
    --point_cloud /content/data/Omni-CAD-subset/pointcloud/0000/00000071_00005.npy \
    --output_dir /content/validation_full_pipeline \
    --deepcad_root /content/3rd_party/DeepCAD \
    --num_views 4
