"""Quick evaluation test - just load model and run on 1 sample."""

import sys
from pathlib import Path

# Setup paths
project_root = Path("/content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/CMU16825_Final_project")
venv_site = project_root / ".venv/lib/python3.12/site-packages"
michelangelo_path = project_root / "Michelangelo"

sys.path.insert(0, str(venv_site))
sys.path.insert(0, str(michelangelo_path))
sys.path.insert(0, str(project_root))

# Clear cache to ensure latest code is loaded
for mod in list(sys.modules.keys()):
    if 'cad_mllm' in mod:
        del sys.modules[mod]

from cad_mllm import CADAutocomplete

print("=" * 80)
print("QUICK EVAL TEST - Loading Model")
print("=" * 80)

# Data paths
data_root = "/content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete"
checkpoint_path = "/content/gdrive/MyDrive/CAD-MLLM-checkpoints/stage3_all/checkpoint-best"

try:
    autocomplete = CADAutocomplete(
        checkpoint_path=checkpoint_path,
        device="cuda",
        dtype="bfloat16",
    )
    print("\n✅ Model loaded successfully!")

    # Find first available truncated file
    import os
    truncated_root = Path(data_root) / "json_truncated"
    truncated_files = list(truncated_root.rglob("*_tr_*.json"))

    if not truncated_files:
        print(f"\n❌ No truncated files found in {truncated_root}")
        sys.exit(1)

    # Use first file
    trunc_file = truncated_files[0]
    print(f"\nUsing test file: {trunc_file.name}")

    # Parse filename to get base paths
    filename = trunc_file.stem  # e.g., 00000071_00005_tr_02
    parts = filename.split("_")
    base_id = f"{parts[0]}_{parts[1]}"  # e.g., 00000071_00005
    subfolder = parts[0]  # e.g., 0000

    # Construct paths
    txt_file = Path(data_root) / "txt" / f"{base_id}.txt"
    img_file = Path(data_root) / "img" / subfolder / f"{base_id}.png"
    pc_file = Path(data_root) / "pointcloud" / subfolder / f"{base_id}.npy"

    # Load caption
    caption = "CAD model"
    if txt_file.exists():
        caption = txt_file.read_text().strip()

    print(f"Caption: {caption}")
    print(f"Image: {img_file.exists()}")
    print(f"Point cloud: {pc_file.exists()}")

    # Quick inference test
    print("\n" + "=" * 80)
    print("Testing inference...")
    print("=" * 80)

    result = autocomplete.complete(
        truncated_json=str(trunc_file),
        caption=caption,
        image=str(img_file) if img_file.exists() else None,
        point_cloud=str(pc_file) if pc_file.exists() else None,
        temperature=0.7,
        max_new_tokens=500,  # Short test
    )

    print("\n✅ Inference complete!")
    print(f"  Partial ops: {result['metadata']['partial_operations']}")
    print(f"  Generated ops: {result['metadata']['generated_operations']}")
    print(f"  Total ops: {result['metadata']['total_operations']}")

except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
