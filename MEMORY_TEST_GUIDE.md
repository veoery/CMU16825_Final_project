# Memory Optimization Test Guide for RTX 5070 Ti

## Problem
Getting OOM (Out of Memory) errors when training with original settings (seq_len=8192, lora_r=32).

## Solution Strategy
Test with progressively less conservative settings until we find what works.

## ✨ NEW: Gradient Checkpointing Enabled
**All test scripts now enable gradient checkpointing**, which trades compute for memory by not storing all intermediate activations. This should reduce memory usage by 20-30% with only a ~10-15% increase in training time.

**What this means:**
- Memory usage should be **2-4 GB lower** than before
- Training will be slightly slower (~10-15%)
- You may be able to use longer sequence lengths than expected

---

## Test Scripts (Run in Order)

### 1. **MOST CONSERVATIVE: seq_len=1024** (Start here)

```bash
bash scripts/test_autocomplete_5070ti_1024.sh
```

**Settings:**
- Sequence length: 1024
- Batch size: 1
- Gradient accumulation: 64
- LoRA rank: 8
- Test samples: 50
- **Gradient checkpointing: ENABLED**

**Expected memory:** ~6-8 GB (with checkpointing)
**This should definitely work** - if this OOMs, there's another issue.

---

### 2. **MODERATE: seq_len=2048** (Try after 1024 works)

```bash
bash scripts/test_autocomplete_5070ti_2048.sh
```

**Settings:**
- Sequence length: 2048
- Batch size: 1
- Gradient accumulation: 32
- LoRA rank: 8
- Test samples: 100
- **Gradient checkpointing: ENABLED**

**Expected memory:** ~8-10 GB (with checkpointing)
**This is a reasonable balance** for 5070 Ti.

---

### 3. **LESS CONSERVATIVE: seq_len=4096** (Try after 2048 works)

```bash
bash scripts/test_autocomplete_5070ti.sh
```

**Settings:**
- Sequence length: 4096
- Batch size: 1
- Gradient accumulation: 16
- LoRA rank: 16
- Test samples: 1000
- **Gradient checkpointing: ENABLED**

**Expected memory:** ~11-13 GB (with checkpointing)
**This should work** with gradient checkpointing!

---

## Recommended Approach

### Step 1: Find Working Configuration
Start with the **MOST CONSERVATIVE** (1024) and work your way up:

```bash
# In WSL terminal
cd /mnt/w/CMU_Academics/2025\ Fall/Learning\ for\ 3D\ Vision/CMU16825_Final_project
conda activate p3d_5070ti

# Test 1: Should definitely work
bash scripts/test_autocomplete_5070ti_1024.sh

# If that works, try test 2
bash scripts/test_autocomplete_5070ti_2048.sh

# If that works, try test 3
bash scripts/test_autocomplete_5070ti.sh
```

### Step 2: Monitor Memory Usage

In a separate terminal, watch GPU memory:
```bash
watch -n 1 nvidia-smi
```

Look for:
- Memory usage should be < 15 GB
- Should see "stage3_all - Epoch 0" progress bar
- No OOM errors

### Step 3: Choose Configuration for Full Training

Once you find a working configuration:
- **If seq_len=1024 works but 2048 OOMs**: Use seq_len=1024 for local testing, seq_len=4096+ for A100
- **If seq_len=2048 works but 4096 OOMs**: Use seq_len=2048 for local testing, seq_len=8192+ for A100
- **If seq_len=4096 works**: Great! Use seq_len=4096 locally, seq_len=16384+ for A100

---

## Understanding the Trade-offs

### Sequence Length
- **Lower (1024-2048)**: Less memory, but may truncate long CAD sequences
- **Higher (4096-8192)**: More memory, but handles complex CAD models better
- **For autocompletion**: We need reasonable length to see context + completion

### LoRA Rank
- **Lower (8-16)**: Less memory, less model capacity
- **Higher (32-64)**: More memory, more model capacity
- **For testing**: Start with 8, increase if training works

### Effective Batch Size
All scripts maintain **effective batch size of 64**:
- seq_len=1024: batch=1, grad_accum=64 → effective=64
- seq_len=2048: batch=1, grad_accum=32 → effective=32
- seq_len=4096: batch=1, grad_accum=16 → effective=16

Lower is ok for testing, but for full training you want effective batch ≥ 32.

---

## If ALL Tests OOM

If even seq_len=1024 with lora_r=8 OOMs, try:

1. **Check GPU isn't being used by other processes:**
   ```bash
   nvidia-smi  # Should show minimal usage before training
   fuser -v /dev/nvidia*  # Check what's using GPU
   ```

2. **Clear GPU memory:**
   ```bash
   # Kill any Python processes
   pkill -9 python

   # Clear CUDA cache
   python -c "import torch; torch.cuda.empty_cache()"
   ```

3. **Try text-only (no image/PC encoders):**
   Modify script to use `--start_from_stage 1` instead of stage 3

4. **Use A100 only:**
   Skip local testing, go straight to Google Colab A100

---

## Memory Breakdown (Approximate)

For RTX 5070 Ti (16 GB VRAM):

### Without Gradient Checkpointing (OLD)
| Component | Memory Usage (seq_len=2048) |
|-----------|----------------------------|
| Qwen3-8B model (bfloat16) | ~8 GB |
| LoRA adapters (r=8) | ~0.5 GB |
| Image encoder (DINOv2) | ~1 GB |
| Point cloud encoder | ~1 GB |
| Activations (batch=1) | ~3-4 GB |
| Optimizer states | ~1 GB |
| **Total** | **~14-15 GB** |

### With Gradient Checkpointing (NEW) ✨
| Component | Memory Usage (seq_len=2048) |
|-----------|----------------------------|
| Qwen3-8B model (bfloat16) | ~8 GB |
| LoRA adapters (r=8) | ~0.5 GB |
| Image encoder (DINOv2) | ~1 GB |
| Point cloud encoder | ~1 GB |
| **Activations (batch=1)** | **~1-2 GB** (saved!) |
| Optimizer states | ~1 GB |
| **Total** | **~12-13 GB** |

### Comparison by Sequence Length (with checkpointing):
- seq_len=1024: ~8-9 GB → **Should work easily**
- seq_len=2048: ~10-11 GB → **Should work**
- seq_len=4096: ~12-13 GB → **Should work**
- seq_len=8192: ~16-18 GB → **Might OOM** (too close to limit)

---

## Next Steps After Finding Working Config

1. **Update RUN_GUIDE.md** with working configuration
2. **Test full pipeline** with 1000 samples
3. **Verify wandb logging** works
4. **Plan A100 training** with larger seq_len (16384-32768)

---

## Quick Reference Commands

```bash
# Activate environment
conda activate p3d_5070ti

# Run most conservative test (START HERE)
bash scripts/test_autocomplete_5070ti_1024.sh

# Monitor GPU
watch -n 1 nvidia-smi

# Check wandb
# Go to: https://wandb.ai

# Kill training if needed
pkill -9 python
```

---

## Expected Behavior When Working

You should see:
```
Loading samples from ./data/Omni-CAD-subset/json_truncated...
Loaded 50 samples  # (or 100, or 1000 depending on test)

Loading LLM model Qwen/Qwen3-8B...
Loading image encoder...
Loading point cloud encoder...

Starting Stage 3: multimodal (text+img+pc)
stage3_all - Epoch 0:   0%|          | 0/50 [00:00<?, ?it/s]
stage3_all - Epoch 0:   2%|▏         | 1/50 [00:15<12:30, 15.31s/it]
...
```

If you see this, **training is working!** 🎉

---

## Good luck! 🚀

Start with `test_autocomplete_5070ti_1024.sh` and work your way up.
