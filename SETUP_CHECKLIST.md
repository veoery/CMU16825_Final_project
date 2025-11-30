# DeepSpeed ZeRO-3 Setup Checklist

## Pre-flight Checks

- [ ] 4x L40 GPUs available and visible
  ```bash
  nvidia-smi
  # Should show 4 GPUs
  ```

- [ ] CUDA and drivers working
  ```bash
  nvcc --version
  nvidia-smi
  ```

- [ ] Flash Attention installed
  ```bash
  python -c "import flash_attn; print(flash_attn.__version__)"
  ```

## Installation

- [ ] Install DeepSpeed
  ```bash
  uv pip install deepspeed
  ```

- [ ] Verify DeepSpeed installation
  ```bash
  python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"
  ds_report  # Check DeepSpeed environment
  ```

## Code Modifications

Open `scripts/train_curriculum.py` and make these changes:

### Section A: Main function setup (around line 410)
- [ ] Add distributed setup at start of `main()`
  ```python
  rank, world_size, local_rank = setup_distributed()
  args.local_rank = local_rank
  if args.deepspeed or world_size > 1:
      args.device = f"cuda:{local_rank}"
  ```

- [ ] Modify seed setting
  ```python
  set_seed(args.seed + rank)  # Different seed per rank
  ```

### Section B: Model initialization (around line 462)
- [ ] Wrap print statements with `if is_main_process(rank):`

- [ ] Modify device placement for DeepSpeed
  ```python
  if not args.deepspeed:
      model_config.device = args.device
  else:
      model_config.device = "cpu"  # DeepSpeed handles placement
  ```

### Section C: Gradient checkpointing (around line 492)
- [ ] Move gradient checkpointing BEFORE DeepSpeed init
- [ ] Add rank check to print statements

### Section D: Data loading in `train_curriculum_stage()` (around line 330)
- [ ] Add DistributedSampler support
  ```python
  if world_size > 1:
      sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
      shuffle = False
  else:
      sampler = None
      shuffle = True
  ```

- [ ] Add sampler to DataLoader
  ```python
  train_dataloader = DataLoader(..., sampler=sampler, shuffle=shuffle, ...)
  ```

### Section E: DeepSpeed initialization in `train_curriculum_stage()` (around line 345)
- [ ] Load DeepSpeed config
  ```python
  if args.deepspeed and DEEPSPEED_AVAILABLE:
      with open(args.deepspeed_config, 'r') as f:
          ds_config = json.load(f)
  ```

- [ ] Initialize DeepSpeed
  ```python
  model_engine, optimizer, _, scheduler = deepspeed.initialize(
      model=model,
      config=ds_config,
      model_parameters=optimizer_params,
  )
  use_deepspeed = True
  ```

### Section F: Training loop calls (around line 385)
- [ ] Update train_epoch calls with DeepSpeed params
  ```python
  avg_loss, global_step = train_epoch(
      model_engine,  # Changed from model
      train_dataloader,
      optimizer,
      scheduler,
      epoch,
      stage.name,
      config,
      start_global_step=global_step,
      use_deepspeed=use_deepspeed,  # Added
      rank=rank,  # Added
  )
  ```

### Section G: Checkpoint saving (around line 415)
- [ ] Add rank check and DeepSpeed-aware saving
  ```python
  if is_main_process(rank):
      checkpoint_path = os.path.join(config.output_dir, stage.name, f"checkpoint-epoch{epoch}")
      if use_deepspeed:
          model_engine.save_checkpoint(checkpoint_path)
      else:
          save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_path)
  ```

### Section H: WandB initialization (around line 560)
- [ ] Add rank check
  ```python
  if args.use_wandb and is_main_process(rank):
      # ... wandb init code
  ```

### Section I: All print statements
- [ ] Wrap all remaining print statements with `if is_main_process(rank):`

### Section J: Pass rank and world_size through functions
- [ ] Add `rank` parameter to `train_curriculum_stage()` signature
- [ ] Pass `rank` and `world_size` from `main()` to all stage functions

## Testing

### Test 1: Single GPU (Sanity Check)
- [ ] Run without DeepSpeed flag
  ```bash
  python scripts/train_curriculum.py --create_dummy_data --num_dummy_samples 10 --stage1_epochs 1
  ```

### Test 2: Multi-GPU with Dummy Data
- [ ] Test DeepSpeed with small dummy dataset
  ```bash
  deepspeed --num_gpus=4 scripts/train_curriculum.py \
      --deepspeed \
      --deepspeed_config configs/deepspeed/ds_zero3_config.json \
      --create_dummy_data \
      --num_dummy_samples 100 \
      --batch_size 2 \
      --stage1_epochs 1 \
      --stage2_epochs 1 \
      --stage3_epochs 1
  ```

- [ ] Check all 4 GPUs show activity in `nvidia-smi`
- [ ] Verify no errors in console output
- [ ] Confirm checkpoints are created

### Test 3: Full Training with Real Data
- [ ] Run full distributed training
  ```bash
  bash scripts/train_distributed.sh
  ```

- [ ] Monitor GPU memory usage
  ```bash
  watch -n 1 nvidia-smi
  ```

- [ ] Check WandB for logging (if enabled)

## Validation

- [ ] Training starts without errors
- [ ] All 4 GPUs show ~equal memory usage
- [ ] Throughput is 3-4x faster than single GPU
- [ ] Memory per GPU is <15GB for Qwen-8B
- [ ] Checkpoints are being saved
- [ ] WandB logs appear (if enabled)
- [ ] No NCCL errors or hangs

## Performance Tuning

After successful run, optimize:

- [ ] Increase batch size from 4 to 8 or 16
- [ ] Test ZeRO-2 vs ZeRO-3 performance
- [ ] Enable/disable Flash Attention to compare
- [ ] Adjust gradient accumulation steps
- [ ] Profile with different configurations

## Troubleshooting

If issues occur:

- [ ] Check `DEEPSPEED_SETUP.md` troubleshooting section
- [ ] Enable debug logging:
  ```bash
  export DEEPSPEED_LOG_LEVEL=DEBUG
  export NCCL_DEBUG=INFO
  ```

- [ ] Try ZeRO-2 instead of ZeRO-3
  ```bash
  --deepspeed_config configs/deepspeed/ds_zero2_config.json
  ```

- [ ] Test with smaller model or CPU offload
  ```bash
  --deepspeed_config configs/deepspeed/ds_zero3_offload.json
  ```

## Success Criteria

✅ Training completes at least 1 epoch on all 3 curriculum stages
✅ GPU memory usage is balanced across all 4 GPUs
✅ Training speed is 3-4x faster than single GPU
✅ Checkpoints can be saved and loaded
✅ WandB logging works (if enabled)
✅ No CUDA OOM errors
✅ No NCCL communication errors

## Estimated Time

- Code modifications: 30-45 minutes
- Testing and debugging: 30-60 minutes
- First successful run: 15-30 minutes
- **Total: 1.5-2.5 hours**

## Resources

- Detailed guide: `DEEPSPEED_SETUP.md`
- Quick reference: `QUICK_START_DEEPSPEED.md`
- Summary: `IMPLEMENTATION_SUMMARY.md`

## Notes

- Make changes incrementally and test after each section
- Keep a backup of `train_curriculum.py` before modifying
- Start with small dummy datasets for quick iteration
- Monitor GPU memory during first runs
- Don't worry about perfect optimization initially - get it working first!

Good luck! 🚀
