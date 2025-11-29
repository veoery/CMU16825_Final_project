"""
Quick Evaluation in Google Colab

Copy and paste this code into a Colab cell to evaluate your trained model.
"""

# ============================================================================
# COLAB CELL: Quick Evaluation
# ============================================================================

from cad_mllm import CADAutocomplete
from pathlib import Path
import json

# 1. Configure paths
checkpoint_path = "/content/gdrive/MyDrive/CAD-MLLM-checkpoints/stage3_all/checkpoint-best"
data_root = "/content/data/Omni-CAD-subset"

# 2. Prepare test samples (just a few for quick eval)
test_samples = [
    {
        "truncated_json": f"{data_root}/json_truncated/0000/00000071_00005_tr_02.json",
        "caption": "Modern chair",
        "image": f"{data_root}/img/0000/00000071_00005.png",
        "point_cloud": f"{data_root}/pointcloud/0000/00000071_00005.npy",
    },
    # Add more samples here...
]

# 3. Load model
print("Loading model...")
autocomplete = CADAutocomplete(
    checkpoint_path=checkpoint_path,
    device="cuda",
    dtype="bfloat16",
)

# 4. Run batch inference
print(f"Evaluating {len(test_samples)} samples...")
results = autocomplete.batch_complete(
    test_samples,
    temperature=0.7,
    top_p=0.9,
)

# 5. Display results
print("\n" + "="*80)
print("RESULTS")
print("="*80)
for i, (sample, result) in enumerate(zip(test_samples, results), 1):
    meta = result['metadata']
    print(f"\nSample {i}: {meta['caption']}")
    print(f"  Partial ops: {meta['partial_operations']}")
    print(f"  Generated ops: {meta['generated_operations']}")
    print(f"  Total ops: {meta['total_operations']}")

# 6. Save results
output_path = "/content/gdrive/MyDrive/eval_results.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_path}")


# ============================================================================
# COLAB CELL: Full Evaluation with Script
# ============================================================================

# Or use the full evaluation script:
!python scripts/evaluate_autocomplete.py \
    --checkpoint_path /content/gdrive/MyDrive/CAD-MLLM-checkpoints/stage3_all/checkpoint-best \
    --truncated_json_root /content/data/Omni-CAD-subset/json_truncated \
    --omnicad_txt_path /content/data/Omni-CAD-subset/txt \
    --omnicad_json_root /content/data/Omni-CAD-subset/json \
    --omnicad_img_root /content/data/Omni-CAD-subset/img \
    --omnicad_pc_root /content/data/Omni-CAD-subset/pointcloud \
    --num_samples 50 \
    --output_results /content/gdrive/MyDrive/eval_results.json


# ============================================================================
# COLAB CELL: Single Sample Inference (for debugging)
# ============================================================================

from cad_mllm import autocomplete_cad

# Quick test on a single sample
result = autocomplete_cad(
    checkpoint_path="/content/gdrive/MyDrive/CAD-MLLM-checkpoints/stage3_all/checkpoint-best",
    truncated_json="/content/data/Omni-CAD-subset/json_truncated/0000/00000071_00005_tr_02.json",
    caption="Modern minimalist chair",
    image="/content/data/Omni-CAD-subset/img/0000/00000071_00005.png",
    point_cloud="/content/data/Omni-CAD-subset/pointcloud/0000/00000071_00005.npy",
    output_path="/content/output_chair.json",
    temperature=0.7,
)

print(f"Generated {result['metadata']['total_operations']} total operations!")
print(f"  - Partial: {result['metadata']['partial_operations']}")
print(f"  - Generated: {result['metadata']['generated_operations']}")
