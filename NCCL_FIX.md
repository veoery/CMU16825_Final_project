# NCCL Error Fix Summary

## Problem
```
[rank1]: torch.distributed.DistBackendError: NCCL error
[rank1]: nvmlDeviceGetHandleByIndex(0) failed: Unknown Error
```

## What Was Fixed

### 1. Updated `setup_distributed()` in `scripts/train_curriculum.py`
Added robustness checks:
- Check `torch.cuda.is_available()` before calling `torch.cuda.set_device()`
- Check `torch.distributed.is_initialized()` before calling `init_process_group()`
- Made barrier conditional on successful initialization

### 2. Added NCCL Environment Variables
Updated `scripts/train_distributed.sh` with:
```bash
export NCCL_DEBUG=INFO          # Verbose logging
export NCCL_TIMEOUT=1800        # 30 min timeout
```

### 3. Created Diagnostic Script
New file: `scripts/test_distributed.py`
- Tests distributed setup independently
- Verifies CUDA, NCCL, and collective operations
- Helps isolate the problem

## How to Debug

### Step 1: Test Distributed Setup
```bash
deepspeed --num_gpus=4 scripts/test_distributed.py
```

This will show exactly where the failure occurs.

### Step 2: Check GPU Visibility
```bash
nvidia-smi
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Step 3: Enable Debug Logging
Already added to `train_distributed.sh`, but you can also run:
```bash
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
deepspeed --num_gpus=4 scripts/test_distributed.py
```

### Step 4: If Still Failing - Disable P2P/IB
Edit `scripts/train_distributed.sh` and uncomment:
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

This disables peer-to-peer GPU communication (slower but more compatible).

### Step 5: Test with Fewer GPUs
```bash
# Try 2 GPUs instead of 4
deepspeed --num_gpus=2 scripts/test_distributed.py
```

## Common Causes

1. **Driver/NVML Issue**: NVIDIA driver not properly initialized
   - Solution: Restart node or reload nvidia drivers

2. **Permission Issue**: Process doesn't have GPU access
   - Solution: Check `nvidia-smi` works without sudo

3. **Environment Conflict**: Conflicting CUDA/NCCL environment variables
   - Solution: Unset all CUDA_* variables and try again

4. **Network Configuration**: NCCL trying to use wrong network interface
   - Solution: Set `export NCCL_SOCKET_IFNAME=lo` for single-node

5. **DeepSpeed Version**: Incompatible DeepSpeed/NCCL versions
   - Solution: `uv pip install --upgrade deepspeed`

## Next Steps

1. Run the test script first: `deepspeed --num_gpus=4 scripts/test_distributed.py`
2. Check the output to see which specific operation fails
3. Apply the appropriate fix from above
4. Once test script passes, run full training

## Files Modified

- [scripts/train_curriculum.py](scripts/train_curriculum.py) - Fixed `setup_distributed()`
- [scripts/train_distributed.sh](scripts/train_distributed.sh) - Added NCCL env vars
- [scripts/test_distributed.py](scripts/test_distributed.py) - New diagnostic tool (created)
- [DEEPSPEED.md](DEEPSPEED.md) - Updated troubleshooting section
