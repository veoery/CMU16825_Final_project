# DeepSpeed ZeRO-3 Distributed Training Setup

## Overview

This guide explains how to complete the DeepSpeed ZeRO-3 integration for distributed training on 4x L40 GPUs.

## What's Already Done

✅ DeepSpeed configuration files created in `configs/deepspeed/`
- `ds_zero3_config.json` - ZeRO-3 (recommended)
- `ds_zero2_config.json` - ZeRO-2 (fallback)
- `ds_zero3_offload.json` - ZeRO-3 with CPU offload

✅ Distributed training arguments added to `train_curriculum.py`
✅ Helper functions for distributed setup added
✅ Training loop modified to support DeepSpeed

## What Still Needs to Be Done

### 1. Install DeepSpeed

```bash
uv pip install deepspeed
```

### 2. Remaining Code Changes in `train_curriculum.py`

You need to complete these modifications in the `main()` function:

#### A. Setup distributed environment at the start of main():

```python
def main():
    """Main training function."""
    args = parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    args.local_rank = local_rank

    # Update device for this process
    if args.deepspeed or world_size > 1:
        args.device = f"cuda:{local_rank}"

    # Set seed
    set_seed(args.seed + rank)  # Different seed per rank
```

#### B. Modify model initialization (around line 462):

```python
    # Initialize model
    if is_main_process(rank):
        print("\nInitializing CAD-MLLM model...")

    # For DeepSpeed, don't move model to device yet
    if not args.deepspeed:
        model_config.device = args.device
    else:
        model_config.device = "cpu"  # DeepSpeed will handle device placement

    model = CADMLLMModel(model_config)
```

#### C. Handle gradient checkpointing before DeepSpeed init:

```python
    # Enable gradient checkpointing BEFORE DeepSpeed initialization
    if args.use_gradient_checkpointing:
        if is_main_process(rank):
            print("Enabling gradient checkpointing for memory efficiency...")
        if hasattr(model.llm, 'enable_input_require_grads'):
            model.llm.enable_input_require_grads()
        if hasattr(model.llm, 'gradient_checkpointing_enable'):
            model.llm.gradient_checkpointing_enable()
            if is_main_process(rank):
                print("  ✓ Gradient checkpointing enabled on LLM")
```

#### D. Modify data loading to use DistributedSampler:

```python
    # Create dataloader for this stage (in train_curriculum_stage function)
    if world_size > 1:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.seed
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
    )
```

#### E. Initialize DeepSpeed (in train_curriculum_stage function, replace optimizer creation):

```python
    if args.deepspeed and DEEPSPEED_AVAILABLE:
        # Load DeepSpeed config
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)

        # DeepSpeed initialization
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=optimizer_params,
        )

        use_deepspeed = True
    else:
        # Regular optimizer setup
        optimizer = torch.optim.AdamW(
            optimizer_params,
            weight_decay=config.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        model_engine = model
        use_deepspeed = False
```

#### F. Update train_epoch calls to pass DeepSpeed flag:

```python
    avg_loss, global_step = train_epoch(
        model_engine,  # Use model_engine instead of model
        train_dataloader,
        optimizer,
        scheduler,
        epoch,
        stage.name,
        config,
        start_global_step=global_step,
        use_deepspeed=use_deepspeed,
        rank=rank,
    )
```

#### G. Update checkpoint saving for DeepSpeed:

```python
    # Save checkpoint at end of epoch
    if is_main_process(rank):
        checkpoint_path = os.path.join(config.output_dir, stage.name, f"checkpoint-epoch{epoch}")
        if use_deepspeed:
            model_engine.save_checkpoint(checkpoint_path)
        else:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_path)
```

#### H. Only log from main process:

```python
    # Wrap all print statements and wandb logs with:
    if is_main_process(rank):
        # ... logging code
```

### 3. Create Launch Script

See `scripts/train_distributed.sh` (created separately)

## Usage

### Single GPU (existing behavior):
```bash
python scripts/train_curriculum.py [args...]
```

### Multi-GPU with DeepSpeed:
```bash
bash scripts/train_distributed.sh
```

Or manually:
```bash
deepspeed --num_gpus=4 scripts/train_curriculum.py \
    --deepspeed \
    --deepspeed_config configs/deepspeed/ds_zero3_config.json \
    [other args...]
```

## Expected Performance

With 4x L40 (48GB each) and ZeRO-3:
- **Memory per GPU**: 3-5GB (vs 12-20GB single GPU)
- **Effective batch size**: Can increase from 8 to 32+
- **Training speed**: ~3.5-3.8x faster
- **Can train larger models**: Potentially Qwen-14B or larger

## Troubleshooting

### OOM Errors
- Use `ds_zero3_offload.json` for CPU offloading
- Reduce `batch_size` or increase `gradient_accumulation_steps`
- Disable Flash Attention temporarily

### Slow Training
- Check if using ZeRO-2 instead (faster but less memory efficient)
- Ensure NCCL is properly configured
- Check network bandwidth between GPUs

### Checkpoint Issues
- DeepSpeed checkpoints are partitioned across GPUs
- To convert back to regular checkpoint: use `deepspeed.utils.zero_to_fp32`

## Notes

- DeepSpeed handles its own mixed precision training (bf16)
- Gradient clipping is configured in DeepSpeed config
- Learning rate scheduling is handled by DeepSpeed
- Flash Attention and gradient checkpointing are compatible with ZeRO-3
