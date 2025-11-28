"""
Memory Optimization Script: Find Optimal max_seq_length + batch_size
Tests multiple configurations and recommends best setup for 80GB A100

Upload this to Google Colab and run it to find your optimal training configuration.
"""

import torch
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Setup paths (UPDATE THESE FOR YOUR GOOGLE DRIVE PATHS)
project_root = Path("/content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete/CMU16825_Final_project")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "Michelangelo"))

from transformers import AutoTokenizer, AutoImageProcessor
from cad_mllm import CADMLLMModel, CADMLLMConfig
from cad_mllm.data.multimodal_autocomplete import MultimodalAutocompleteDataset, MultimodalAutocompleteCollator
from torch.utils.data import DataLoader

print("=" * 80)
print("MEMORY OPTIMIZATION TEST")
print("=" * 80)

# Initialize tokenizer and image processor once
print("\n[1/4] Loading tokenizer and image processor...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

# Create base dataset once
print("[2/4] Loading dataset...")
dataset = MultimodalAutocompleteDataset(
    data_path="/content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete/txt",
    truncated_json_root="/content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete/json_truncated",
    full_json_root="/content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete/json",
    image_root="/content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete/img",
    pc_root="/content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete/pointcloud",
)
print(f"Dataset size: {len(dataset)} samples")

# Configurations to test: (max_seq_length, batch_size, description)
configs_to_test = [
    (8192, 2, "Small seq, larger batch"),
    (10240, 1, "Balanced approach"),
    (12288, 1, "Medium seq"),
    (13000, 1, "Current user config"),
    (16384, 1, "Large seq (low NaN target)"),
]

print(f"\n[3/4] Testing {len(configs_to_test)} configurations...")
print("Each config will test 20 batches to measure:")
print("  - Peak memory usage")
print("  - NaN percentage (all-masked + NaN loss)")
print("  - Average valid loss")
print("  - Throughput (batches/sec)")
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
results: List[Dict] = []

for idx, (max_seq, batch_size, description) in enumerate(configs_to_test, 1):
    print(f"\n{'='*80}")
    print(f"CONFIG {idx}/{len(configs_to_test)}: max_seq_length={max_seq}, batch_size={batch_size}")
    print(f"Description: {description}")
    print(f"{'='*80}")

    try:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create collator with this config
        collator = MultimodalAutocompleteCollator(
            tokenizer=tokenizer,
            max_seq_length=max_seq,
            image_processor=image_processor,
        )

        # Create dataloader with this batch size
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False,
            num_workers=0,
        )

        # Create model
        print("  Loading model...")
        config = CADMLLMConfig(
            llm_model_name="Qwen/Qwen3-8B",
            use_lora=True,
            lora_r=8,
            lora_alpha=16,
        )
        model = CADMLLMModel(config)
        model = model.to(device).to(torch.bfloat16)
        model.train()

        # Test 20 batches
        num_test_batches = 20
        all_masked_count = 0
        nan_loss_count = 0
        valid_count = 0
        loss_values = []
        error_count = 0

        print(f"  Testing {num_test_batches} batches...")
        start_time = time.time()

        for i, batch in enumerate(dataloader):
            if i >= num_test_batches:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Check if all labels masked
            labels = batch["labels"]
            unmasked = (labels != -100).sum().item()

            if unmasked == 0:
                all_masked_count += 1
                print(f"    Batch {i+1}/{num_test_batches}: All masked")
                continue

            # Forward pass
            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_loss_count += 1
                    print(f"    Batch {i+1}/{num_test_batches}: NaN/Inf loss")
                else:
                    valid_count += 1
                    loss_values.append(loss.item())
                    print(f"    Batch {i+1}/{num_test_batches}: Loss = {loss.item():.4f}")

            except torch.cuda.OutOfMemoryError as e:
                print(f"    Batch {i+1}/{num_test_batches}: ❌ OOM ERROR")
                error_count += 1
                # Stop testing this config if we hit OOM
                break
            except Exception as e:
                print(f"    Batch {i+1}/{num_test_batches}: Error - {e}")
                error_count += 1

        elapsed_time = time.time() - start_time

        # Get peak memory
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

        # Calculate metrics
        total_batches = all_masked_count + nan_loss_count + valid_count + error_count
        nan_percentage = ((all_masked_count + nan_loss_count) / total_batches * 100) if total_batches > 0 else 0
        avg_loss = sum(loss_values) / len(loss_values) if loss_values else float('nan')
        throughput = total_batches / elapsed_time if elapsed_time > 0 else 0

        # Store results
        result = {
            'max_seq_length': max_seq,
            'batch_size': batch_size,
            'description': description,
            'peak_memory_gb': peak_memory_gb,
            'all_masked': all_masked_count,
            'nan_loss': nan_loss_count,
            'valid': valid_count,
            'errors': error_count,
            'total': total_batches,
            'nan_percentage': nan_percentage,
            'avg_loss': avg_loss,
            'throughput': throughput,
            'time_elapsed': elapsed_time,
        }
        results.append(result)

        # Print summary for this config
        print(f"\n  SUMMARY:")
        print(f"    Peak Memory: {peak_memory_gb:.2f} GB / 80 GB")
        print(f"    All-masked batches: {all_masked_count}/{total_batches}")
        print(f"    NaN loss batches: {nan_loss_count}/{total_batches}")
        print(f"    Valid batches: {valid_count}/{total_batches}")
        print(f"    Errors (OOM, etc.): {error_count}/{total_batches}")
        print(f"    NaN percentage: {nan_percentage:.1f}%")
        print(f"    Average valid loss: {avg_loss:.4f}" if not torch.isnan(torch.tensor(avg_loss)) else "    Average valid loss: N/A")
        print(f"    Throughput: {throughput:.2f} batches/sec")
        print(f"    Time elapsed: {elapsed_time:.1f}s")

        # Assess this config
        if error_count > 0:
            assessment = "❌ FAILED (OOM)"
        elif peak_memory_gb > 75:
            assessment = "⚠️ RISKY (>75 GB)"
        elif nan_percentage > 20:
            assessment = "⚠️ INEFFICIENT (>20% NaN)"
        elif nan_percentage > 10:
            assessment = "✓ MARGINAL (10-20% NaN)"
        else:
            assessment = "✅ EXCELLENT (<10% NaN)"

        print(f"    Assessment: {assessment}")

        # Cleanup
        del model
        del dataloader
        del collator
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n  ❌ FAILED TO TEST CONFIG: {e}")
        results.append({
            'max_seq_length': max_seq,
            'batch_size': batch_size,
            'description': description,
            'error': str(e),
        })

# Final recommendations
print(f"\n\n{'='*80}")
print("FINAL RESULTS & RECOMMENDATIONS")
print(f"{'='*80}\n")

# Print table
print(f"{'Config':<35} {'Memory':<12} {'NaN%':<8} {'Avg Loss':<10} {'Speed':<12} {'Status':<20}")
print("-" * 110)

for result in results:
    if 'error' in result:
        config_str = f"{result['max_seq_length']}/{result['batch_size']}"
        print(f"{config_str:<35} {'N/A':<12} {'N/A':<8} {'N/A':<10} {'N/A':<12} {'❌ Failed':<20}")
    else:
        config_str = f"{result['max_seq_length']}/{result['batch_size']}"
        memory_str = f"{result['peak_memory_gb']:.1f} GB"
        nan_str = f"{result['nan_percentage']:.1f}%"
        loss_str = f"{result['avg_loss']:.4f}" if not torch.isnan(torch.tensor(result['avg_loss'])) else "N/A"
        speed_str = f"{result['throughput']:.2f} b/s"

        # Status
        if result['errors'] > 0:
            status = "❌ OOM"
        elif result['peak_memory_gb'] > 75:
            status = "⚠️ Risky memory"
        elif result['nan_percentage'] > 20:
            status = "⚠️ Too many NaN"
        elif result['nan_percentage'] > 10:
            status = "✓ Marginal"
        else:
            status = "✅ Excellent"

        print(f"{config_str:<35} {memory_str:<12} {nan_str:<8} {loss_str:<10} {speed_str:<12} {status:<20}")

# Find best config
viable_results = [r for r in results if 'error' not in r and r['errors'] == 0 and r['peak_memory_gb'] <= 75]

if viable_results:
    print(f"\n{'='*80}")
    print("RECOMMENDATION:")
    print(f"{'='*80}\n")

    # Prioritize: NaN% < 10%, then maximize seq_length for better context
    excellent_results = [r for r in viable_results if r['nan_percentage'] < 10]

    if excellent_results:
        # Among excellent results, choose largest seq_length
        best = max(excellent_results, key=lambda r: r['max_seq_length'])
        print("✅ RECOMMENDED CONFIG (Excellent NaN%, maximum context):")
    else:
        # Fall back to lowest NaN%
        best = min(viable_results, key=lambda r: r['nan_percentage'])
        print("✓ RECOMMENDED CONFIG (Best available, but not ideal):")

    print(f"  max_seq_length = {best['max_seq_length']}")
    print(f"  batch_size = {best['batch_size']}")
    print(f"\n  Expected performance:")
    print(f"    - Peak memory: {best['peak_memory_gb']:.1f} GB / 80 GB ({best['peak_memory_gb']/80*100:.1f}% utilization)")
    print(f"    - NaN percentage: {best['nan_percentage']:.1f}% (waste {best['nan_percentage']:.1f}% of compute)")
    print(f"    - Average loss: {best['avg_loss']:.4f}" if not torch.isnan(torch.tensor(best['avg_loss'])) else "    - Average loss: N/A")
    print(f"    - Throughput: {best['throughput']:.2f} batches/sec")

    print(f"\n  For overnight 10-hour run, use:")
    print(f"    python train_autocomplete.py \\")
    print(f"      --max_seq_length {best['max_seq_length']} \\")
    print(f"      --per_device_train_batch_size {best['batch_size']} \\")
    print(f"      --gradient_accumulation_steps 32 \\")
    print(f"      --num_train_epochs 3 \\")
    print(f"      --max_samples 5000")

else:
    print("\n❌ NO VIABLE CONFIGURATIONS FOUND")
    print("All tested configs either OOM'd or exceeded 75GB memory limit.")
    print("Recommendations:")
    print("  1. Enable gradient checkpointing (saves 30-50% memory)")
    print("  2. Reduce LoRA rank to r=4")
    print("  3. Try even smaller max_seq_length (e.g., 6144)")

print(f"\n{'='*80}")
print("TEST COMPLETE")
print(f"{'='*80}\n")
