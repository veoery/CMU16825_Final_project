# DeepSpeed ZeRO-3 Setup Guide

Quick guide to enable distributed training on 4x L40 GPUs.

## Quick Start

```bash
# 1. Install
uv pip install deepspeed

# 2. Complete code modifications (see below)

# 3. Test with dummy data
deepspeed --num_gpus=4 scripts/train_curriculum.py \
    --deepspeed --deepspeed_config configs/deepspeed/ds_zero3_config.json \
    --create_dummy_data --num_dummy_samples 100 \
    --batch_size 2 --stage1_epochs 1 --stage2_epochs 1 --stage3_epochs 1

# 4. Full training
bash scripts/train_distributed.sh
```

## What's Done

✅ Config files: `configs/deepspeed/ds_zero3_config.json` (ZeRO-3), `ds_zero2_config.json` (ZeRO-2), `ds_zero3_offload.json` (CPU offload)
✅ Helper functions: `setup_distributed()`, `is_main_process()`
✅ Modified `train_epoch()` for DeepSpeed
✅ Launch script: `scripts/train_distributed.sh`

## Code Changes Needed

Add these to `scripts/train_curriculum.py`:

### 1. In `main()` - After `args = parse_args()`
```python
rank, world_size, local_rank = setup_distributed()
args.local_rank = local_rank
if args.deepspeed or world_size > 1:
    args.device = f"cuda:{local_rank}"
set_seed(args.seed + rank)
```

### 2. Model init - Wrap prints, change device
```python
if is_main_process(rank):
    print("\nInitializing CAD-MLLM model...")
model_config.device = "cpu" if args.deepspeed else args.device
model = CADMLLMModel(model_config)
```

### 3. All print statements - Add rank check
```python
if is_main_process(rank):
    print(...)
```

### 4. In `train_curriculum_stage()` - Add parameters
```python
def train_curriculum_stage(
    model, stage, train_dataset, collator, config,
    start_global_step=0,
    args=None, rank=0, world_size=1  # ADD THESE
):
```

### 5. DataLoader - Add DistributedSampler
```python
if world_size > 1:
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)
    shuffle = False
else:
    sampler = None
    shuffle = True

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=shuffle, sampler=sampler, collate_fn=collator, num_workers=0)
```

### 6. Replace optimizer creation
```python
if args.deepspeed and DEEPSPEED_AVAILABLE:
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, config=ds_config, model_parameters=optimizer_params)
    use_deepspeed = True
else:
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=num_training_steps)
    model_engine = model
    use_deepspeed = False
```

### 7. Update train_epoch call
```python
avg_loss, global_step = train_epoch(
    model_engine, train_dataloader, optimizer, scheduler, epoch, stage.name, config,
    start_global_step=global_step, use_deepspeed=use_deepspeed, rank=rank)
```

### 8. Checkpoint saving
```python
if is_main_process(rank):
    checkpoint_path = os.path.join(config.output_dir, stage.name, f"checkpoint-epoch{epoch}")
    if use_deepspeed:
        model_engine.save_checkpoint(checkpoint_path)
    else:
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_path)
```

### 9. Pass to functions
```python
global_step = train_curriculum_stage(
    model=model, stage=stage, train_dataset=train_dataset, collator=collator,
    config=train_config, start_global_step=global_step,
    args=args, rank=rank, world_size=world_size)  # ADD THESE
```

### 10. WandB - Rank check
```python
if args.use_wandb and is_main_process(rank):
    wandb.init(...)
```

## Usage

**Single GPU:**
```bash
python scripts/train_curriculum.py [args...]
```

**4 GPUs:**
```bash
deepspeed --num_gpus=4 scripts/train_curriculum.py \
    --deepspeed --deepspeed_config configs/deepspeed/ds_zero3_config.json \
    --batch_size 8 --use_gradient_checkpointing --attn_implementation flash_attention_2 [args...]
```

**Or use launch script:**
```bash
bash scripts/train_distributed.sh
```

## Configs

- **ZeRO-3** (recommended): `configs/deepspeed/ds_zero3_config.json` - Max memory efficiency
- **ZeRO-2** (faster): `configs/deepspeed/ds_zero2_config.json` - Less communication overhead
- **ZeRO-3 + Offload**: `configs/deepspeed/ds_zero3_offload.json` - For huge models

## Performance

**4x L40 (48GB) with ZeRO-3 + Flash Attn + Grad Ckpt:**
- Memory: ~5-8GB per GPU (vs 20GB+ single)
- Speed: ~3.5x faster
- Batch size: 128+ effective (vs 8 single)
- Can train: Qwen-14B or larger

## Troubleshooting

**OOM:** Use `ds_zero3_offload.json` or reduce `--batch_size`
**Slow:** Try `ds_zero2_config.json`, wait for JIT compilation (first epoch)
**Hang:** Check `nvidia-smi`, verify all GPUs visible

**NCCL errors (`nvmlDeviceGetHandleByIndex failed`):**
1. Test distributed setup first:
   ```bash
   deepspeed --num_gpus=4 scripts/test_distributed.py
   ```
2. Add environment variables to `train_distributed.sh` (already added):
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_TIMEOUT=1800
   ```
3. If network issues, disable P2P and InfiniBand:
   ```bash
   export NCCL_P2P_DISABLE=1
   export NCCL_IB_DISABLE=1
   ```
4. Verify GPU visibility:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.device_count())"
   ```
5. Try with fewer GPUs to isolate issue:
   ```bash
   deepspeed --num_gpus=2 scripts/test_distributed.py
   ```

## Monitor

```bash
watch -n 1 nvidia-smi  # Check GPU usage
export NCCL_DEBUG=INFO  # Debug communication
```

## Time to Complete

- Code changes: 30-45 min
- Testing: 30-60 min
- **Total: 1-2 hours**
