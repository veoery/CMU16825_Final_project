# Quick Start: DeepSpeed ZeRO-3 Distributed Training

## Current Status

✅ **Completed:**
1. DeepSpeed config files created (`configs/deepspeed/`)
2. Training script partially modified with distributed support
3. Launch script created (`scripts/train_distributed.sh`)
4. Helper functions added for distributed training

⚠️ **Remaining:** Complete code modifications in `train_curriculum.py` (see `DEEPSPEED_SETUP.md`)

## Installation

```bash
# Install DeepSpeed
uv pip install deepspeed

# Verify installation
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
```

## Quick Test (Recommended First Step)

Test with dummy data on 4 GPUs:

```bash
deepspeed --num_gpus=4 scripts/train_curriculum.py \
    --deepspeed \
    --deepspeed_config configs/deepspeed/ds_zero3_config.json \
    --create_dummy_data \
    --num_dummy_samples 100 \
    --batch_size 2 \
    --stage1_epochs 1 \
    --stage2_epochs 1 \
    --stage3_epochs 1 \
    --use_gradient_checkpointing \
    --attn_implementation flash_attention_2
```

## Full Training

After testing, run full training:

```bash
bash scripts/train_distributed.sh
```

## Configuration Files

### ZeRO-3 (Recommended - Maximum Memory Efficiency)
```bash
--deepspeed_config configs/deepspeed/ds_zero3_config.json
```

### ZeRO-2 (Faster, Less Memory Efficient)
```bash
--deepspeed_config configs/deepspeed/ds_zero2_config.json
```

### ZeRO-3 with CPU Offload (For Larger Models)
```bash
--deepspeed_config configs/deepspeed/ds_zero3_offload.json
```

## Key Parameters

### Batch Size Scaling
With 4 GPUs, you can increase batch sizes:
- Single GPU: `batch_size=2, grad_accum=4` → effective batch = 8
- 4 GPUs: `batch_size=4, grad_accum=4` → effective batch = 64

### Memory vs Speed Trade-offs
- **Flash Attention**: 2-4x faster, 30-50% less memory
- **Gradient Checkpointing**: 20-30% slower, 30-50% less memory
- **ZeRO-3**: Distributes model across GPUs

Recommended for 4x L40 (48GB):
```bash
--use_gradient_checkpointing \
--attn_implementation flash_attention_2 \
--deepspeed_config configs/deepspeed/ds_zero3_config.json \
--batch_size 8  # Can go higher with 48GB GPUs!
```

## Monitoring

### Check GPU Usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### WandB Logging
Only rank 0 logs to WandB (automatically handled)

### DeepSpeed Logs
Check for communication stats:
```bash
tail -f outputs_curriculum_distributed/deepspeed_*.log
```

## Common Issues & Solutions

### 1. OOM (Out of Memory)
**Solution:**
```bash
# Option A: Use CPU offloading
--deepspeed_config configs/deepspeed/ds_zero3_offload.json

# Option B: Reduce batch size
--batch_size 2 --gradient_accumulation_steps 8

# Option C: Disable Flash Attention temporarily
# Remove --attn_implementation flag
```

### 2. Slow Startup
**Normal**: First epoch is slow due to JIT compilation and setup
**Wait**: 1-2 minutes for initialization

### 3. Hanging at Initialization
**Check**: All GPUs are visible
```bash
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
```

### 4. NCCL Errors
**Solution:**
```bash
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # If P2P issues
```

## Expected Performance

### Memory Usage (per GPU)
- **Without ZeRO**: ~20GB
- **With ZeRO-3**: ~5-8GB
- **With ZeRO-3 + Offload**: ~3-5GB

### Training Speed (relative to single GPU)
- **2 GPUs**: ~1.8x faster
- **4 GPUs**: ~3.5x faster
- **8 GPUs**: ~6.5x faster

### Throughput (4x L40 with ZeRO-3)
- **Batch size**: 32-64 effective
- **Tokens/sec**: ~4x single GPU
- **Epoch time**: ~1/3.5 of single GPU

## Next Steps

1. **Install DeepSpeed**: `uv pip install deepspeed`
2. **Complete code modifications**: Follow `DEEPSPEED_SETUP.md`
3. **Test with dummy data**: Run quick test above
4. **Full training**: `bash scripts/train_distributed.sh`
5. **Monitor & tune**: Adjust batch sizes and hyperparameters

## File Structure

```
CAD-MLLM/
├── configs/
│   └── deepspeed/
│       ├── ds_zero3_config.json          # ZeRO-3 (recommended)
│       ├── ds_zero2_config.json          # ZeRO-2 (faster)
│       └── ds_zero3_offload.json         # ZeRO-3 + offload
├── scripts/
│   ├── train_curriculum.py               # Training script (needs completion)
│   └── train_distributed.sh              # Launch script
├── DEEPSPEED_SETUP.md                    # Detailed setup guide
└── QUICK_START_DEEPSPEED.md             # This file
```

## Resources

- [DeepSpeed Documentation](https://www.deepspeed.ai/getting-started/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [DeepSpeed Configuration Guide](https://www.deepspeed.ai/docs/config-json/)
