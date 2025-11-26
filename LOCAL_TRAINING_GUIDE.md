# Local Training Guide: Stages 2-3 with RTX 5070 Ti

## Overview

This guide covers training CAD-MLLM stages 2-3 on your local machine using the Jupyter notebook, adapted for:
- **Local Data Path**: `W:\CMU_Academics\2025 Fall\Learning for 3D Vision\CMU16825_Final_project\data\Omni-CAD-subset`
- **GPU**: RTX 5070 Ti (16GB VRAM)
- **Environment**: WSL conda environment (p3d_5070ti)
- **Training**: Stages 2-3 (Stage 2: Text+PointCloud, Stage 3: Text+PointCloud+Image)

---

## Prerequisites Checklist

### Data Requirements
- [x] Text data: `./data/Omni-CAD-subset/txt`
- [x] JSON data: `./data/Omni-CAD-subset/json`
- [x] Image data: `./data/Omni-CAD-subset/img`
- [ ] **Point cloud data**: `./data/Omni-CAD-subset/pointcloud` ⚠️ **DOWNLOADING**

### Model Requirements
- [ ] **Stage 1 checkpoint**: Required to start Stage 2/3 training
  - Default location: `./checkpoints/stage1_text_model/`
  - If you don't have this, you must train Stage 1 first

### Environment Requirements
- [ ] WSL conda environment `p3d_5070ti` activated
- [ ] Project dependencies installed (torch, transformers, etc.)
- [ ] Michelangelo submodule installed
- [ ] Michelangelo checkpoint downloaded (for point cloud encoder)

---

## Part 1: What You Can Do NOW (While Point Cloud Data Downloads)

### Option A: Train Stage 1 (Text Only)
Since Stage 1 only uses text data (which you have), you can train Stage 1 now to generate the checkpoint needed for Stages 2-3.

**Benefits**:
- No waiting for point cloud data
- Generates the required checkpoint for Stage 2/3
- Utilizes GPU time productively

**Estimated Time**: 4-8 hours depending on dataset size and hyperparameters

### Option B: Test Training with Dummy Data
Test the full training pipeline with synthetic data to ensure everything works.

**Benefits**:
- Validates environment setup
- Catches configuration issues early
- Quick (minutes, not hours)

---

## Part 2: Notebook Modifications for Local Training

### Cells to Skip (Colab-specific)
```python
# SKIP Cell 0: Google Drive mounting
#@title mount your Google Drive
# ...

# SKIP Cell 2: Git clone
!git clone https://github.com/veoery/CMU16825_Final_project.git

# SKIP Cell 3: Change directory
%cd CMU16825_Final_project
```

### Cells to Keep (Environment Setup)

**Cell 5: Create Virtual Environment**
```bash
# If using WSL conda environment, skip this
# Otherwise:
!uv venv
```

**Cell 6-7: Install Dependencies**
```bash
# Activate your conda environment first in WSL terminal:
# conda activate p3d_5070ti

# Then install project dependencies:
!uv pip install -e .
!uv pip install -e ./Michelangelo --no-build-isolation
```

### Critical Path Updates

All training commands need updated paths. Here's the mapping:

| Colab Path | Your Local Path |
|------------|-----------------|
| `./data/Omni-CAD-subset-complete/txt` | `./data/Omni-CAD-subset/txt` |
| `./data/Omni-CAD-subset-complete/json` | `./data/Omni-CAD-subset/json` |
| `./data/Omni-CAD-subset-complete/img` | `./data/Omni-CAD-subset/img` |
| `./data/Omni-CAD-subset-complete/pointcloud` | `./data/Omni-CAD-subset/pointcloud` |

---

## Part 3: Memory Optimization for RTX 5070 Ti (16GB)

### Problem
Default config from notebook uses:
- `batch_size=4`
- `max_seq_length=32768`
- `gradient_accumulation_steps=16`

**Estimated memory**: 35-44GB → **Won't fit in 16GB**

### Solution: Optimized Configuration

```python
# Memory-optimized settings for RTX 5070 Ti
--batch_size 2                        # Reduced from 4
--gradient_accumulation_steps 32      # Increased from 16 (maintains effective batch size)
--max_seq_length 16384                # Reduced from 32768 (if needed)
--dtype bfloat16                      # Keep this
```

**Trade-offs**:
- ✅ Fits in 16GB VRAM
- ✅ Same effective batch size (2 × 32 = 64, same as 4 × 16)
- ⚠️ Slightly slower training (more gradient accumulation steps)
- ⚠️ Shorter context window if you reduce max_seq_length

### Additional Optimizations

Add these to training command:
```bash
--gradient_checkpointing true         # Trade computation for memory
```

---

## Part 4: Modified Training Commands

### Stage 1: Text Only (Run NOW if you don't have checkpoint)

**Cell to Add/Modify**:
```bash
# Stage 1: Text-only training (no point cloud needed)
!python scripts/train_curriculum.py \
    --use_wandb \
    --omnicad_txt_path ./data/Omni-CAD-subset/txt \
    --omnicad_json_root ./data/Omni-CAD-subset/json \
    --llm_model_name "Qwen/Qwen3-8B" \
    --start_from_stage 1 \
    --stage1_epochs 1 \
    --stage2_epochs 0 \
    --stage3_epochs 0 \
    --stage1_lr 2e-4 \
    --max_seq_length 16384 \
    --batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 100 \
    --device cuda \
    --dtype bfloat16
```

**Output**: Checkpoint saved to `./checkpoints/stage1_text_model/`

---

### Stage 2: Text + Point Cloud (Run AFTER point cloud data downloads)

**Modified Cell 18/19**:
```bash
# Stage 2: Text + Point Cloud training
!python scripts/train_curriculum.py \
    --use_wandb \
    --omnicad_txt_path ./data/Omni-CAD-subset/txt \
    --omnicad_json_root ./data/Omni-CAD-subset/json \
    --omnicad_pc_root ./data/Omni-CAD-subset/pointcloud \
    --llm_model_name "Qwen/Qwen3-8B" \
    --resume_from_ckpt ./checkpoints/stage1_text_model \
    --start_from_stage 2 \
    --stage2_epochs 1 \
    --stage3_epochs 0 \
    --stage2_lr 2e-4 \
    --max_seq_length 16384 \
    --batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 100 \
    --device cuda \
    --dtype bfloat16
```

**Requirements**:
- ✅ Stage 1 checkpoint at `./checkpoints/stage1_text_model/`
- ✅ Point cloud data at `./data/Omni-CAD-subset/pointcloud/`
- ✅ Michelangelo checkpoint at `./checkpoints/michelangelo_point_encoder_state_dict.pt`

**Output**: Checkpoint saved to `./checkpoints/stage2_text_pc_model/`

---

### Stage 3: Text + Point Cloud + Image

**New Cell to Add**:
```bash
# Stage 3: Text + Point Cloud + Image training
!python scripts/train_curriculum.py \
    --use_wandb \
    --omnicad_txt_path ./data/Omni-CAD-subset/txt \
    --omnicad_json_root ./data/Omni-CAD-subset/json \
    --omnicad_img_root ./data/Omni-CAD-subset/img \
    --omnicad_pc_root ./data/Omni-CAD-subset/pointcloud \
    --llm_model_name "Qwen/Qwen3-8B" \
    --resume_from_ckpt ./checkpoints/stage2_text_pc_model \
    --start_from_stage 3 \
    --stage3_epochs 1 \
    --stage3_lr 2e-4 \
    --max_seq_length 16384 \
    --batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 100 \
    --device cuda \
    --dtype bfloat16
```

**Requirements**:
- ✅ Stage 2 checkpoint at `./checkpoints/stage2_text_pc_model/`
- ✅ All data (text, json, img, pointcloud)

**Output**: Final checkpoint saved to `./checkpoints/stage3_all_model/`

---

## Part 5: Pre-Training Verification

### Step 1: Check Data Completeness
```bash
# In WSL terminal
cd /mnt/w/CMU_Academics/2025\ Fall/Learning\ for\ 3D\ Vision/CMU16825_Final_project

# Check data directories exist
ls -la data/Omni-CAD-subset/txt | wc -l      # Should show text files
ls -la data/Omni-CAD-subset/json | wc -l     # Should show json directories
ls -la data/Omni-CAD-subset/img | wc -l      # Should show image files
ls -la data/Omni-CAD-subset/pointcloud | wc -l  # Should show pointcloud files (after download)
```

### Step 2: Check Michelangelo Checkpoint
```bash
# Verify Michelangelo checkpoint exists
ls -lh checkpoints/michelangelo_point_encoder_state_dict.pt

# Should show file size ~150-200MB
```

### Step 3: Test GPU Availability
```python
# Add this test cell to notebook
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Expected output:
```
CUDA available: True
CUDA device: NVIDIA GeForce RTX 5070 Ti
CUDA memory: 16.00 GB
```

### Step 4: Test with Dummy Data (Optional but Recommended)
```bash
# Quick test run with synthetic data
!python scripts/train_curriculum.py \
    --start_from_stage 2 \
    --create_dummy_data \
    --stage2_epochs 1 \
    --stage3_epochs 0 \
    --batch_size 2 \
    --max_seq_length 16384 \
    --device cuda
```

If this works, your environment is correctly configured.

---

## Part 6: Monitoring Training

### GPU Memory Monitoring
Open a separate WSL terminal and run:
```bash
watch -n 1 nvidia-smi
```

**What to watch for**:
- Memory usage should stay < 15GB (leave 1GB headroom)
- GPU utilization should be 80-100%
- Temperature should be < 85°C

**If memory exceeds 15GB**:
- Reduce `batch_size` to 1
- Reduce `max_seq_length` to 12288 or 8192
- Kill the process and restart with adjusted settings

### Weights & Biases (wandb) Monitoring
If `--use_wandb` is enabled, you can monitor:
- Training loss curve
- Learning rate schedule
- GPU memory usage
- Throughput (samples/sec)

Access at: https://wandb.ai/your-username/CAD-MLLM

---

## Part 7: Expected Training Times

### Assumptions
- Dataset size: ~10,000 samples (adjust based on your actual dataset)
- GPU: RTX 5070 Ti (16GB)
- Batch size: 2, Gradient accumulation: 32

### Estimates

| Stage | Epochs | Time per Epoch | Total Time |
|-------|--------|----------------|------------|
| Stage 1 (Text) | 1 | 3-5 hours | 3-5 hours |
| Stage 2 (Text+PC) | 1 | 4-6 hours | 4-6 hours |
| Stage 3 (Text+PC+Img) | 1 | 5-7 hours | 5-7 hours |

**Total for all stages**: 12-18 hours

**Note**: These are rough estimates. Actual time depends on:
- Exact dataset size
- max_seq_length (shorter = faster)
- GPU thermal throttling
- Disk I/O speed

---

## Part 8: Recommended Workflow

### Option A: Sequential Training (Safest)
```
Day 1: Train Stage 1 (no point cloud needed)
        ↓ generates checkpoint
Day 2: Wait for point cloud download to complete
        ↓
Day 3: Train Stage 2 (with point cloud)
        ↓ generates checkpoint
Day 4: Train Stage 3 (with all modalities)
```

### Option B: Combined Stage 2+3 (After Point Cloud Downloads)
```bash
# Train stages 2 and 3 together
!python scripts/train_curriculum.py \
    --use_wandb \
    --omnicad_txt_path ./data/Omni-CAD-subset/txt \
    --omnicad_json_root ./data/Omni-CAD-subset/json \
    --omnicad_img_root ./data/Omni-CAD-subset/img \
    --omnicad_pc_root ./data/Omni-CAD-subset/pointcloud \
    --llm_model_name "Qwen/Qwen3-8B" \
    --resume_from_ckpt ./checkpoints/stage1_text_model \
    --start_from_stage 2 \
    --stage2_epochs 1 \
    --stage3_epochs 1 \
    --stage2_lr 2e-4 \
    --stage3_lr 2e-4 \
    --max_seq_length 16384 \
    --batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 100 \
    --device cuda \
    --dtype bfloat16
```

This runs both stages sequentially in one command (saves time on setup/teardown).

---

## Part 9: Troubleshooting

### Issue: Out of Memory (OOM)
**Error**: `CUDA out of memory`

**Solutions** (try in order):
1. Reduce `batch_size` from 2 to 1
2. Reduce `max_seq_length` from 16384 to 12288
3. Add `--gradient_checkpointing true`
4. Reduce `max_seq_length` to 8192
5. Close other GPU processes: `nvidia-smi` → kill unnecessary processes

### Issue: Point Cloud Data Not Ready
**Error**: `FileNotFoundError: [Errno 2] No such file or directory: './data/Omni-CAD-subset/pointcloud/...'`

**Solution**:
- Wait for point cloud download to complete
- Run Stage 1 in the meantime (doesn't need point cloud)
- Verify files exist: `ls data/Omni-CAD-subset/pointcloud | head`

### Issue: Stage 1 Checkpoint Not Found
**Error**: `FileNotFoundError: Checkpoint not found at ./checkpoints/stage1_text_model`

**Solution**:
- Train Stage 1 first (see Part 4)
- Or specify correct checkpoint path with `--resume_from_ckpt`
- Verify checkpoint exists: `ls -la checkpoints/stage1_text_model/`

### Issue: Michelangelo Checkpoint Missing
**Error**: `FileNotFoundError: ./checkpoints/michelangelo_point_encoder_state_dict.pt`

**Solution**:
- Download from: https://drive.google.com/file/d/1wzfa4EoijmyfTpLPfma9r03wvQ_mbjFD/view?usp=drive_link
- Place at: `./checkpoints/michelangelo_point_encoder_state_dict.pt`
- Verify: `ls -lh checkpoints/michelangelo_point_encoder_state_dict.pt`

### Issue: WSL Path Issues
**Error**: `FileNotFoundError` with Windows paths

**Solution**:
In WSL, Windows paths are mounted at `/mnt/`:
```bash
# Windows path
W:\CMU_Academics\2025 Fall\Learning for 3D Vision\CMU16825_Final_project

# WSL equivalent
/mnt/w/CMU_Academics/2025\ Fall/Learning\ for\ 3D\ Vision/CMU16825_Final_project

# Or use relative paths from project root
cd /mnt/w/CMU_Academics/2025\ Fall/Learning\ for\ 3D\ Vision/CMU16825_Final_project
# Then use ./data/...
```

---

## Part 10: Next Steps After Training

### Model Evaluation
```bash
# Test inference with trained model
python scripts/inference.py \
    --prompt "Generate a CAD model of a simple cube." \
    --checkpoint_path ./checkpoints/stage3_all_model \
    --device cuda \
    --dtype bfloat16
```

### Export for Deployment
The final checkpoint at `./checkpoints/stage3_all_model/` contains:
- LoRA weights for the LLM
- Trained projector weights (text, image, point cloud)
- Training metadata

---

## Quick Reference: Command Comparison

| What | Original Notebook | Your Local Setup |
|------|-------------------|------------------|
| Data path | `./data/Omni-CAD-subset-complete/` | `./data/Omni-CAD-subset/` |
| Batch size | 4 | 2 (for 16GB GPU) |
| Max seq length | 32768 | 16384 (for 16GB GPU) |
| Gradient accum | 16 | 32 (to maintain effective batch size) |
| Device | cuda | cuda |
| Checkpoint dir | Not specified | `./checkpoints/` |

---

## Summary Checklist

Before starting training, ensure:

- [ ] WSL conda environment `p3d_5070ti` is activated
- [ ] Project installed: `uv pip install -e .`
- [ ] Michelangelo installed: `uv pip install -e ./Michelangelo --no-build-isolation`
- [ ] Michelangelo checkpoint downloaded and placed correctly
- [ ] Data paths verified (text, json, img exist; pointcloud downloading)
- [ ] GPU accessible from WSL: `nvidia-smi` works
- [ ] Memory-optimized settings configured (batch_size=2, max_seq_length=16384)
- [ ] Stage 1 trained or checkpoint available (required for Stage 2/3)
- [ ] wandb configured (optional): `wandb login`

---

**You're ready to train!** Start with Stage 1 while waiting for point cloud data to download.
