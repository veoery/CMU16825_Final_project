# OOM Fix Summary

## What I Did to Fix the Memory Issues

### 1. ✅ Added Gradient Checkpointing Support
**What it does:** Reduces memory usage by 20-30% by not storing all intermediate activations during forward pass. Trades a bit of compute (recomputing activations during backward pass) for significant memory savings.

**Files modified:**
- [cad_mllm/model.py](cad_mllm/model.py#L199-L215): Added `enable_gradient_checkpointing()` method
- [scripts/train_curriculum.py](scripts/train_curriculum.py#L50-L51): Added `--use_gradient_checkpointing` argument
- [scripts/train_curriculum.py](scripts/train_curriculum.py#L487-L489): Calls `model.enable_gradient_checkpointing()` if flag is set

### 2. ✅ Created 3 Progressive Test Scripts
**Why:** Test with increasingly aggressive memory settings to find what works on your 5070 Ti.

**Test scripts created:**
1. **Most conservative** - [scripts/test_autocomplete_5070ti_1024.sh](scripts/test_autocomplete_5070ti_1024.sh)
   - seq_len=1024, lora_r=8, 50 samples
   - Expected memory: ~6-8 GB
   - **START HERE** - this should definitely work

2. **Moderate** - [scripts/test_autocomplete_5070ti_2048.sh](scripts/test_autocomplete_5070ti_2048.sh)
   - seq_len=2048, lora_r=8, 100 samples
   - Expected memory: ~8-10 GB
   - Good balance for 5070 Ti

3. **Less conservative** - [scripts/test_autocomplete_5070ti.sh](scripts/test_autocomplete_5070ti.sh)
   - seq_len=4096, lora_r=16, 1000 samples
   - Expected memory: ~11-13 GB
   - Should work with gradient checkpointing!

### 3. ✅ Created Memory Test Guide
**File:** [MEMORY_TEST_GUIDE.md](MEMORY_TEST_GUIDE.md)

**Contents:**
- Step-by-step testing procedure
- Expected memory usage for each configuration
- Troubleshooting tips
- Memory breakdown comparison (with/without gradient checkpointing)
- What to do if tests still OOM

---

## How to Test (Quick Start)

### Step 1: Navigate to project and activate conda env
```bash
cd /mnt/w/CMU_Academics/2025\ Fall/Learning\ for\ 3D\ Vision/CMU16825_Final_project
conda activate p3d_5070ti
```

### Step 2: Start with MOST CONSERVATIVE test
```bash
bash scripts/test_autocomplete_5070ti_1024.sh
```

**Watch for:**
- "Enabling gradient checkpointing for PEFT model" message
- "stage3_all - Epoch 0" progress bar starting
- Memory usage in `nvidia-smi` (should be ~6-8 GB)

### Step 3: If that works, try MODERATE
```bash
bash scripts/test_autocomplete_5070ti_2048.sh
```

### Step 4: If that works, try LESS CONSERVATIVE
```bash
bash scripts/test_autocomplete_5070ti.sh
```

---

## Expected Behavior When Working

You should see output like:
```
Loading samples from ./data/Omni-CAD-subset/json_truncated...
Loaded 50 samples

Loading LLM: Qwen/Qwen3-8B
...
Setting up LoRA with r=8, alpha=16
trainable params: X,XXX,XXX || all params: XXX,XXX,XXX

Enabling gradient checkpointing for PEFT model  ← NEW! This saves memory

Loading datasets...
...

Starting Stage 3: multimodal (text+img+pc)
stage3_all - Epoch 0:   0%|          | 0/50 [00:00<?, ?it/s]
stage3_all - Epoch 0:   2%|▏         | 1/50 [00:15<12:30, 15.31s/it]
...
```

---

## Memory Savings Comparison

| Configuration | Without Checkpointing | With Checkpointing | Savings |
|--------------|---------------------|-------------------|---------|
| seq_len=1024 | ~10-11 GB | **~6-8 GB** | **~3-4 GB** |
| seq_len=2048 | ~14-15 GB (OOM risk) | **~10-11 GB** | **~4 GB** |
| seq_len=4096 | ~18-19 GB (OOM) | **~12-13 GB** | **~6 GB** |

**With gradient checkpointing, seq_len=4096 should now work on your 5070 Ti!**

---

## What Changed vs. Previous Run

Your previous run (that OOMed) had:
- seq_len=8192, lora_r=32, NO gradient checkpointing
- Memory usage: ~26 GB → **OOM on 16 GB GPU**

New recommended config:
- seq_len=4096, lora_r=16, WITH gradient checkpointing
- Memory usage: ~12-13 GB → **Should fit comfortably**

**Tradeoff:** Slightly slower training (~10-15% slower), but it will actually run!

---

## If All Tests Still OOM

1. **Check GPU is free:**
   ```bash
   nvidia-smi  # Should show minimal usage
   pkill -9 python  # Kill any lingering processes
   ```

2. **Try text-only mode:**
   Change `--start_from_stage 3` to `--start_from_stage 1` in test script
   This disables image/PC encoders, saving ~2 GB

3. **Use A100 only:**
   Skip local testing, train directly on Google Colab A100

---

## For Full Training (After Tests Work)

Once you find a working configuration (e.g., seq_len=2048 works):

### On 5070 Ti:
```bash
python scripts/train_curriculum.py \
    --use_autocomplete_dataset \
    --truncated_json_root ./data/Omni-CAD-subset/json_truncated \
    --omnicad_txt_path ./data/Omni-CAD-subset/txt \
    --omnicad_json_root ./data/Omni-CAD-subset/json \
    --omnicad_img_root ./data/Omni-CAD-subset/img \
    --omnicad_pc_root ./data/Omni-CAD-subset/pointcloud \
    --llm_model_name "Qwen/Qwen3-8B" \
    --start_from_stage 3 \
    --stage3_epochs 1 \
    --max_seq_length 2048 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lora_r 8 \
    --lora_alpha 16 \
    --use_gradient_checkpointing \
    --device cuda \
    --dtype bfloat16 \
    --use_wandb \
    --wandb_project "CAD-MLLM-Autocompletion" \
    --output_dir "./outputs_autocomplete"
```

### On A100 (Google Colab):
```bash
# Use LONGER sequences, HIGHER LoRA rank
python scripts/train_curriculum.py \
    --use_autocomplete_dataset \
    --truncated_json_root ./data/Omni-CAD-subset/json_truncated \
    --omnicad_txt_path ./data/Omni-CAD-subset/txt \
    --omnicad_json_root ./data/Omni-CAD-subset/json \
    --omnicad_img_root ./data/Omni-CAD-subset/img \
    --omnicad_pc_root ./data/Omni-CAD-subset/pointcloud \
    --llm_model_name "Qwen/Qwen3-8B" \
    --start_from_stage 3 \
    --stage3_epochs 2 \
    --max_seq_length 16384 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lora_r 32 \
    --lora_alpha 64 \
    --use_gradient_checkpointing \
    --device cuda \
    --dtype bfloat16 \
    --use_wandb
```

---

## Summary

✅ **Added gradient checkpointing** - saves ~4-6 GB memory
✅ **Created 3 progressive test scripts** - find what works
✅ **Updated all scripts** to use gradient checkpointing
✅ **Created comprehensive guide** - [MEMORY_TEST_GUIDE.md](MEMORY_TEST_GUIDE.md)

**Next action:** Run `bash scripts/test_autocomplete_5070ti_1024.sh` and work your way up!

Good luck! 🚀
