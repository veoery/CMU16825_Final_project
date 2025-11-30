"""Test text-only inference with aggressive decoding to force JSON generation."""

import json
from pathlib import Path
from cad_mllm.inference import CADAutocomplete

# Configuration
CHECKPOINT_PATH = "outputs_curriculum/stage1_text/checkpoint-best"
TEST_DATA_DIR = Path("data/Omni-CAD-subset")

def find_test_sample():
    """Find a test sample with both truncated and full JSON."""
    truncated_root = TEST_DATA_DIR / "json_truncated"
    full_root = TEST_DATA_DIR / "json"
    txt_root = TEST_DATA_DIR / "txt"

    # Find first truncated JSON
    truncated_files = sorted(truncated_root.rglob("*_tr_01.json"))

    for trunc_path in truncated_files[:5]:  # Try first 5
        # Extract CAD ID
        rel_path = trunc_path.relative_to(truncated_root)
        stem = rel_path.stem  # e.g., "00000071_00005_tr_01"
        base_name = "_".join(stem.split("_")[:-2])  # "00000071_00005"
        parent = rel_path.parent  # "0000"
        cad_id = str(parent / base_name)

        # Check if full JSON exists
        full_path = full_root / f"{cad_id}.json"
        if full_path.exists():
            # Load caption
            caption = ""
            for txt_file in txt_root.glob("*.json"):
                with open(txt_file) as f:
                    data = json.load(f)
                    for entry in data:
                        if entry["id"] == cad_id:
                            caption = entry["text caption"]
                            break
                if caption:
                    break

            return {
                "truncated_path": trunc_path,
                "full_path": full_path,
                "caption": caption or "A CAD model",
                "cad_id": cad_id,
            }

    return None

def main():
    print("="*80)
    print("TEXT-ONLY INFERENCE TEST")
    print("Testing if checkpoint can generate CAD JSON with aggressive decoding")
    print("="*80 + "\n")

    # Find test sample
    print("Finding test sample...")
    sample = find_test_sample()

    if not sample:
        print("❌ No test samples found!")
        print("Make sure data is in: data/Omni-CAD-subset/")
        return

    print(f"✓ Found sample: {sample['cad_id']}")
    print(f"  Caption: {sample['caption'][:100]}...")
    print(f"  Truncated JSON: {sample['truncated_path']}")
    print(f"  Full JSON: {sample['full_path']}\n")

    # Load JSONs to compare
    with open(sample['truncated_path']) as f:
        truncated_data = json.load(f)
    with open(sample['full_path']) as f:
        full_data = json.load(f)

    kept_ops = truncated_data.get("truncation_metadata", {}).get("kept_operations", 0)
    total_ops = len(full_data.get("sequence", []))
    expected_new_ops = total_ops - kept_ops

    print(f"Ground Truth:")
    print(f"  Kept operations: {kept_ops}")
    print(f"  Total operations: {total_ops}")
    print(f"  Expected new operations: {expected_new_ops}\n")

    # Load model
    print("Loading model...")
    autocomplete = CADAutocomplete(
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
        dtype="bfloat16",
        max_seq_length=13000,
    )

    print("\n" + "="*80)
    print("TEST 1: Very Low Temperature (0.01) - Almost Deterministic")
    print("="*80)

    result1 = autocomplete.complete(
        truncated_json=sample['truncated_path'],
        caption=sample['caption'],
        output_path="test_output_temp001.json",
        max_new_tokens=2000,
        temperature=0.01,  # Almost deterministic
        top_p=0.5,         # Very restricted
        do_sample=True,
    )

    print(f"\n📊 RESULT 1:")
    print(f"  Generated operations: {result1['metadata']['generated_operations']}")
    print(f"  Total operations: {result1['metadata']['total_operations']}")
    print(f"  Expected new ops: {expected_new_ops}")

    if result1['metadata']['generated_operations'] > 0:
        print(f"\n✓ Generated some operations! First 3:")
        for i, op in enumerate(result1['sequence'][-3:]):
            print(f"    {i+1}. {op}")
    else:
        print(f"\n❌ Generated ZERO operations")

    print("\n" + "="*80)
    print("TEST 2: Greedy Decoding (temperature=0)")
    print("="*80)

    result2 = autocomplete.complete(
        truncated_json=sample['truncated_path'],
        caption=sample['caption'],
        output_path="test_output_greedy.json",
        max_new_tokens=2000,
        temperature=1.0,  # Ignored when do_sample=False
        do_sample=False,  # Greedy decoding
    )

    print(f"\n📊 RESULT 2:")
    print(f"  Generated operations: {result2['metadata']['generated_operations']}")
    print(f"  Total operations: {result2['metadata']['total_operations']}")
    print(f"  Expected new ops: {expected_new_ops}")

    if result2['metadata']['generated_operations'] > 0:
        print(f"\n✓ Generated some operations! First 3:")
        for i, op in enumerate(result2['sequence'][-3:]):
            print(f"    {i+1}. {op}")
    else:
        print(f"\n❌ Generated ZERO operations")

    print("\n" + "="*80)
    print("TEST 3: Higher Temperature (1.0) - More Diverse")
    print("="*80)

    result3 = autocomplete.complete(
        truncated_json=sample['truncated_path'],
        caption=sample['caption'],
        output_path="test_output_temp10.json",
        max_new_tokens=2000,
        temperature=1.0,
        top_p=0.9,
        do_sample=True,
    )

    print(f"\n📊 RESULT 3:")
    print(f"  Generated operations: {result3['metadata']['generated_operations']}")
    print(f"  Total operations: {result3['metadata']['total_operations']}")
    print(f"  Expected new ops: {expected_new_ops}")

    if result3['metadata']['generated_operations'] > 0:
        print(f"\n✓ Generated some operations! First 3:")
        for i, op in enumerate(result3['sequence'][-3:]):
            print(f"    {i+1}. {op}")
    else:
        print(f"\n❌ Generated ZERO operations")

    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    max_generated = max(
        result1['metadata']['generated_operations'],
        result2['metadata']['generated_operations'],
        result3['metadata']['generated_operations'],
    )

    if max_generated == 0:
        print("❌ CHECKPOINT FAILED - Cannot generate CAD JSON at all")
        print("\nRecommended Action: Retrain from scratch with curriculum fix")
    elif max_generated < expected_new_ops * 0.1:
        print(f"⚠️  CHECKPOINT WEAK - Generated only {max_generated}/{expected_new_ops} operations")
        print("\nRecommended Action: Retrain from scratch for better performance")
    else:
        print(f"✓ CHECKPOINT WORKS - Generated {max_generated}/{expected_new_ops} operations")
        print("\nThe checkpoint can generate CAD JSON!")
        print("You may want to tune decoding parameters or retrain for better quality")

    print("\nRaw outputs saved to:")
    print("  - test_output_temp001.json / test_output_temp001_raw.txt")
    print("  - test_output_greedy.json / test_output_greedy_raw.txt")
    print("  - test_output_temp10.json / test_output_temp10_raw.txt")

if __name__ == "__main__":
    main()
