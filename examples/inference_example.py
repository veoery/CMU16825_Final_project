"""
Example: Using CAD Autocomplete for Inference

This script demonstrates how to use the trained model to complete partial CAD sequences.
"""

from cad_mllm import CADAutocomplete, autocomplete_cad


def example_1_quick_inference():
    """Example 1: Quick one-line inference"""
    print("="*80)
    print("Example 1: Quick One-Line Inference")
    print("="*80)

    result = autocomplete_cad(
        checkpoint_path="outputs/stage3_all/checkpoint-best",
        truncated_json="data/json_truncated/0000/00000071_00005_tr_02.json",
        caption="Modern minimalist chair with wooden legs",
        image="data/img/0000/00000071_00005.png",
        point_cloud="data/pointcloud/0000/00000071_00005.npy",
        output_path="output_complete_chair.json",
        temperature=0.7,
        top_p=0.9,
    )

    print(f"\n✅ Complete CAD sequence generated!")
    print(f"   Partial operations: {result['metadata']['partial_operations']}")
    print(f"   Generated operations: {result['metadata']['generated_operations']}")
    print(f"   Total operations: {result['metadata']['total_operations']}")
    print(f"   Saved to: output_complete_chair.json")


def example_2_evaluation_pipeline():
    """Example 2: Process multiple samples for evaluation"""
    print("\n" + "="*80)
    print("Example 2: Batch Processing for Evaluation")
    print("="*80)

    # Initialize model once
    autocomplete = CADAutocomplete(
        checkpoint_path="outputs/checkpoint-best",
        device="cuda",
        dtype="bfloat16",
        max_seq_length=13000,
    )

    # Prepare evaluation samples
    samples = [
        {
            "truncated_json": "data/json_truncated/0000/00000071_00005_tr_01.json",
            "caption": "Modern minimalist chair",
            "image": "data/img/0000/00000071_00005.png",
            "point_cloud": "data/pointcloud/0000/00000071_00005.npy",
        },
        {
            "truncated_json": "data/json_truncated/0000/00000071_00006_tr_02.json",
            "caption": "Wooden table with metal legs",
            "image": "data/img/0000/00000071_00006.png",
            "point_cloud": "data/pointcloud/0000/00000071_00006.npy",
        },
        {
            "truncated_json": "data/json_truncated/0001/00000072_00001_tr_03.json",
            "caption": "Cabinet with drawers",
            "image": "data/img/0001/00000072_00001.png",
            "point_cloud": "data/pointcloud/0001/00000072_00001.npy",
        },
    ]

    # Batch process
    print(f"\nProcessing {len(samples)} samples...")
    results = autocomplete.batch_complete(samples, temperature=0.7, top_p=0.9)

    # Display results
    print("\nResults:")
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        print(f"\nSample {i}: {meta['caption']}")
        print(f"  Partial ops: {meta['partial_operations']}")
        print(f"  Generated ops: {meta['generated_operations']}")
        print(f"  Total ops: {meta['total_operations']}")

        # Save individual results
        autocomplete.save_result(result, f"output_sample_{i}.json")

    print(f"\n✅ All {len(samples)} samples processed!")


def example_3_single_sample_control():
    """Example 3: Fine-grained control over generation"""
    print("\n" + "="*80)
    print("Example 3: Fine-Grained Generation Control")
    print("="*80)

    autocomplete = CADAutocomplete(
        checkpoint_path="outputs/checkpoint-best",
        device="cuda",
    )

    # Generate with different strategies
    strategies = [
        {"temperature": 0.3, "top_p": 0.9, "do_sample": True, "name": "Conservative"},
        {"temperature": 0.7, "top_p": 0.9, "do_sample": True, "name": "Balanced"},
        {"temperature": 1.0, "top_p": 0.95, "do_sample": True, "name": "Creative"},
        {"temperature": 0.0, "do_sample": False, "name": "Greedy (deterministic)"},
    ]

    for strategy in strategies:
        name = strategy.pop("name")
        print(f"\n{name} generation:")

        result = autocomplete.complete(
            truncated_json="data/json_truncated/0000/00000071_00005_tr_02.json",
            caption="Modern minimalist chair",
            image="data/img/0000/00000071_00005.png",
            point_cloud="data/pointcloud/0000/00000071_00005.npy",
            max_new_tokens=3000,
            **strategy
        )

        print(f"  Generated {result['metadata']['generated_operations']} operations")
        autocomplete.save_result(result, f"output_{name.lower().replace(' ', '_')}.json")


def example_4_text_only():
    """Example 4: Text-only inference (no image/point cloud)"""
    print("\n" + "="*80)
    print("Example 4: Text-Only Inference")
    print("="*80)

    autocomplete = CADAutocomplete(checkpoint_path="outputs/checkpoint-best")

    result = autocomplete.complete(
        truncated_json="data/json_truncated/0000/00000071_00005_tr_02.json",
        caption="Modern minimalist chair with wooden legs",
        # No image or point cloud!
        temperature=0.7,
    )

    print(f"\n✅ Text-only generation complete!")
    print(f"   Generated {result['metadata']['generated_operations']} operations")


if __name__ == "__main__":
    # Run examples
    try:
        example_1_quick_inference()
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        example_2_evaluation_pipeline()
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        example_3_single_sample_control()
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        example_4_text_only()
    except Exception as e:
        print(f"Example 4 failed: {e}")

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)
