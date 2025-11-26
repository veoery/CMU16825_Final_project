# Complete Running Guide: Autocompletion Training on 5070 Ti

## ✅ What's Been Done

1. ✅ Created `MultimodalAutocompleteDataset` - combines truncated/full JSON pairing with multimodal support
2. ✅ Created `MultimodalAutocompleteCollator` - proper loss masking (only completion tokens)
3. ✅ Updated `train_curriculum.py` with autocompletion arguments
4. ✅ Added robustness checks for missing files (images, point clouds, etc.)
5. ✅ Created test script for 5070 Ti

---

## 🚀 Step-by-Step Running Instructions

### **Step 1: Activate Environment (5 minutes)**

Open WSL terminal and navigate to project:

```bash
cd /mnt/w/CMU_Academics/2025\ Fall/Learning\ for\ 3D\ Vision/CMU16825_Final_project
conda activate p3d_5070ti
```

Verify GPU:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 5070 Ti
```

---

### **Step 2: Setup Wandb (2 minutes)**

First-time setup only:
```bash
pip install wandb
wandb login
```

Enter your API key from https://wandb.ai/authorize

---

### **Step 3: Verify Data Structure (2 minutes)**

Check that your data is organized correctly:

```bash
ls data/Omni-CAD-subset/
```

Expected output:
```
txt/             # Text captions
json/            # Full JSON sequences
json_truncated/  # Truncated JSON sequences
img/             # Images
pointcloud/      # Point clouds (can be empty if still downloading)
```

Count files:
```bash
find data/Omni-CAD-subset/json_truncated -name "*.json" | wc -l
# Should show ~138,893
```

---

### **Step 4: Test Run with 1000 Samples (30 minutes)**

Run the test script:

```bash
bash scripts/test_autocomplete_5070ti.sh
```

**What this does:**
- Loads 1000 truncated samples
- Tests data loading, loss masking, multimodal input
- Trains for 1 epoch (~15-30 minutes)
- Logs to wandb project "CAD-MLLM-Autocomplete-Test"
- Saves checkpoints to `./outputs_test/`

**Monitor progress:**
1. Watch terminal output for errors
2. Open wandb dashboard: https://wandb.ai
3. Check GPU usage: `watch -n 1 nvidia-smi`

**Expected memory usage:**
- Should stay **< 15GB** on 5070 Ti
- If OOM: reduce `--batch_size` to 1 or `--max_seq_length` to 12288

---

### **Step 5: Verify Test Results (5 minutes)**

After test run completes, check:

1. **Loss is decreasing** (wandb chart or terminal output)
2. **No file loading errors** (check terminal for warnings)
3. **Checkpoints saved** (`ls outputs_test/`)
4. **Loss masking works** - check first batch logs for `-100` in labels

If you see missing file warnings:
```
[WARNING] Failed to load image ...
[WARNING] Failed to load point cloud ...
```
This is OK! The dataset will skip missing modalities.

---

### **Step 6: Full Training on 5070 Ti (Overnight, 8-12 hours)**

If test run succeeds, run full training:

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
    --stage1_epochs 0 \
    --stage2_epochs 0 \
    --stage3_epochs 1 \
    --stage3_lr 2e-4 \
    --max_seq_length 16384 \
    --batch_size 2 \
    --gradient_accumulation_steps 32 \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 200 \
    --logging_steps 50 \
    --save_steps 500 \
    --device cuda \
    --dtype bfloat16 \
    --use_wandb \
    --wandb_project "CAD-MLLM-Autocompletion" \
    --output_dir "./outputs_autocomplete"
```

**Run overnight** - this will take 8-12 hours on 5070 Ti.

---

## 🔧 Troubleshooting

### Issue 1: Out of Memory (OOM)
**Error**: `CUDA out of memory`

**Fix**:
```bash
# Reduce batch size
--batch_size 1 \
--gradient_accumulation_steps 64  # Keep effective batch = 64

# OR reduce sequence length
--max_seq_length 12288  # Down from 16384
```

### Issue 2: Missing Point Cloud Files
**Warning**: `[WARNING] Failed to load point cloud ...`

**This is OK!** The dataset will:
- Skip point cloud modality for those samples
- Use text + image instead
- Continue training normally

Once point clouds download, they'll be automatically used.

### Issue 3: Truncated JSON Not Found
**Error**: `No truncated JSON files found`

**Check**:
```bash
ls data/Omni-CAD-subset/json_truncated/0000/ | head
# Should show files like: 00000071_00005_tr_01.json
```

If empty, verify truncation scripts ran successfully.

### Issue 4: ImportError
**Error**: `ModuleNotFoundError: No module named 'cad_mllm'`

**Fix**:
```bash
# Install package in editable mode
conda activate p3d_5070ti
cd /mnt/w/CMU_Academics/2025\ Fall/Learning\ for\ 3D\ Vision/CMU16825_Final_project
pip install -e .
pip install -e ./Michelangelo --no-build-isolation
```

### Issue 5: wandb Not Logging
**Fix**:
```bash
wandb login  # Re-login
# OR disable wandb
# Remove --use_wandb flag from command
```

---

## 📊 Expected Training Metrics

### Test Run (1000 samples, 1 epoch)
- **Time**: 15-30 minutes
- **Steps**: ~31 steps (1000 / 32 effective batch)
- **Memory**: 12-15GB
- **Initial loss**: ~8-10
- **Final loss**: ~6-8 (should decrease)

### Full Training (138,893 samples, 1 epoch)
- **Time**: 8-12 hours
- **Steps**: ~4,340 steps
- **Memory**: 13-15GB
- **Expected loss curve**: 8 → 4 → 2 → 1.5

---

## 🎯 Success Criteria

After training, verify:
- [ ] Loss decreased steadily
- [ ] Final loss < 2.0
- [ ] Checkpoints saved every 500 steps
- [ ] No crashes or OOM errors
- [ ] Wandb charts show smooth training

---

## 🔄 Next Steps for Google Colab A100

Once 5070 Ti training works, you can use the **exact same code** on Colab!

Just update the command in `scripts/curriculum_training.ipynb`:

```bash
# In Colab notebook cell
!python scripts/train_curriculum.py \
    --use_autocomplete_dataset \
    --truncated_json_root ./data/Omni-CAD-subset/json_truncated \
    --omnicad_txt_path ./data/Omni-CAD-subset/txt \
    --omnicad_json_root ./data/Omni-CAD-subset/json \
    --omnicad_img_root ./data/Omni-CAD-subset/img \
    --omnicad_pc_root ./data/Omni-CAD-subset/pointcloud \
    --llm_model_name "Qwen/Qwen3-8B" \
    --start_from_stage 3 \
    --stage3_epochs 2 \
    --max_seq_length 32768 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --device cuda \
    --use_wandb
```

**Key differences for A100:**
- `--batch_size 8` (up from 2)
- `--max_seq_length 32768` (up from 16384)
- `--gradient_accumulation_steps 4` (down from 32)
- **Effective batch size stays 32** (same quality)

---

## 📁 Output Structure

After training, you'll have:

```
outputs_autocomplete/
├── stage3_all_model/           # Final checkpoint
│   ├── pytorch_model.bin
│   ├── config.json
│   └── ...
├── stage3_all/
│   ├── checkpoint-epoch0-step500/
│   ├── checkpoint-epoch0-step1000/
│   └── ...
└── training_log.txt
```

---

## 🎓 Tips

1. **Start small**: Always test with 1000 samples first
2. **Monitor wandb**: Check loss curves remotely on your phone
3. **Save frequently**: `--save_steps 500` ensures you can recover
4. **Use overnight**: Let 5070 Ti train while you sleep
5. **Check memory**: `watch -n 1 nvidia-smi` in separate terminal

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Environment setup | 5 min |
| Wandb setup | 2 min |
| Test run (1000 samples) | 30 min |
| Full training (138K samples, 1 epoch) | 8-12 hours |
| **Total Day 1** | **~9-13 hours** |

You can start the test run now, then leave full training overnight!

---

## 🔗 Useful Commands

**Check GPU:**
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Live monitoring
```

**Check training progress:**
```bash
tail -f outputs_autocomplete/stage3_all/training_log.txt
```

**Kill training:**
```bash
# Ctrl+C in terminal
# OR find process
ps aux | grep train_curriculum.py
kill -9 <PID>
```

**Resume from checkpoint:**
```bash
# Add to training command:
--resume_from_ckpt ./outputs_autocomplete/stage3_all_model
```

---

### **Run Autocompletion Inference (after training)**

```bash
python scripts/inference_autocomplete.py \
    --model_path ./outputs_autocomplete/stage3_all_model \
    --cad_id 0000/00000071_00005 \
    --txt_json_root ./data/Omni-CAD-subset/txt \
    --truncated_json_root ./data/Omni-CAD-subset/json_truncated \
    --truncation_index 1 \
    --image_path ./data/Omni-CAD-subset/img/0000/00000071_00005.png \
    --pointcloud_path ./data/Omni-CAD-subset/pointcloud/0000/00000071_00005.npz \
    --max_new_tokens 1024 \
    --device cuda
```

- If you already know the caption or truncated JSON file, pass `--text_caption` or `--truncated_json_path` directly.
- The script prints the formatted prompt and the generated completion so you can diff with the ground truth JSON.

---

## 📞 Quick Reference

| What | Command |
|------|---------|
| Activate env | `conda activate p3d_5070ti` |
| Test run | `bash scripts/test_autocomplete_5070ti.sh` |
| Full training | See "Step 6" above |
| Monitor GPU | `nvidia-smi` |
| View wandb | https://wandb.ai |

---

## ✅ Final Checklist Before Starting

- [ ] Conda environment activated (`p3d_5070ti`)
- [ ] GPU working (`nvidia-smi` shows 5070 Ti)
- [ ] Wandb logged in (`wandb login`)
- [ ] Data verified (`ls data/Omni-CAD-subset/json_truncated`)
- [ ] Test script ready (`scripts/test_autocomplete_5070ti.sh`)

**Ready to go? Run:**
```bash
bash scripts/test_autocomplete_5070ti.sh
```

Good luck! 🚀
