"""Curriculum-based training script for CAD-MLLM with progressive modality introduction."""

import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: uv add wandb")

try:
    import deepspeed
    from deepspeed import DeepSpeedConfig
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("Warning: deepspeed not installed. Install with: uv pip install deepspeed")

from cad_mllm import (
    CADMLLMConfig,
    CurriculumTrainingConfig,
    CurriculumStage,
    CADMLLMModel,
    set_seed,
    print_model_info,
    verify_lora_training,
    get_optimizer,
    get_cosine_schedule_with_warmup,
    AverageMeter,
    save_checkpoint,
    MultimodalCADDataset,
    MultimodalCADCollator,
    DummyCADDataset,
    CADCollator,
    create_dummy_dataset,
)


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'LOCAL_RANK' in os.environ:
        # DeepSpeed launcher
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = local_rank
        world_size = torch.cuda.device_count()
    else:
        # Single GPU
        rank = 0
        world_size = 1
        local_rank = 0

    # Set device before initializing process group
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        # Initialize process group if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )

        # Synchronize all processes
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    return rank, world_size, local_rank


def is_main_process(rank=None):
    """Check if this is the main process."""
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    return rank == 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CAD-MLLM with Curriculum Learning")

    # Model arguments
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen3-8B", help="Name of the LLM model from HuggingFace")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--attn_implementation", type=str, default=None, choices=[None, "flash_attention_2", "sdpa", "eager"],
                       help="Attention implementation (flash_attention_2 for Flash Attention 2, sdpa for PyTorch SDPA, eager for standard, None for default)")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs_curriculum", help="Output directory")
    parser.add_argument("--train_data_path", type=str, default=None, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--projector_lr_multiplier", type=float, default=5.0, help="LR multiplier for projectors")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Enable gradient checkpointing for memory efficiency")

    # Curriculum arguments
    parser.add_argument("--enable_curriculum", action="store_true", default=True, help="Enable curriculum training")
    parser.add_argument("--stage1_epochs", type=int, default=3, help="Epochs for stage 1 (text only)")
    parser.add_argument("--stage2_epochs", type=int, default=3, help="Epochs for stage 2 (text + PC)")
    parser.add_argument("--stage3_epochs", type=int, default=5, help="Epochs for stage 3 (all modalities)")
    parser.add_argument("--stage1_lr", type=float, default=2e-5, help="Learning rate for stage 1")
    parser.add_argument("--stage2_lr", type=float, default=2e-5, help="Learning rate for stage 2")
    parser.add_argument("--stage3_lr", type=float, default=1e-5, help="Learning rate for stage 3")

    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="CAD-MLLM-Curriculum", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name (defaults to auto-generated)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/team name")

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu/mps)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")

    # Distributed training arguments
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed training")
    parser.add_argument("--deepspeed_config", type=str, default="configs/deepspeed/ds_zero3_config.json",
                       help="Path to DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (set by launcher)")

    # Data arguments
    parser.add_argument("--create_dummy_data", action="store_true", help="Create dummy dataset")
    parser.add_argument("--num_dummy_samples", type=int, default=100, help="Number of dummy samples")
    parser.add_argument("--omnicad_txt_path", type=str, default="data/Omni-CAD/txt/", help="Path to Omni-CAD text descriptions")
    parser.add_argument("--omnicad_json_root", type=str, default="data/Omni-CAD/json", help="Root directory for Omni-CAD JSON files")
    parser.add_argument("--omnicad_img_root", type=str, default="data/Omni-CAD/img", help="Root directory for images")
    parser.add_argument("--omnicad_pc_root", type=str, default="data/Omni-CAD/pcd", help="Root directory for point clouds")

    # Checkpoint resumption arguments
    parser.add_argument("--resume_from_ckpt", type=str, default=None,
                       help="Path to checkpoint to resume from (e.g., 'stage1_text_model', 'outputs/stage2_text_pc_model', 'stage2_text_pc/checkpoint-epoch2'). "
                            "Can be any checkpoint - stage model or epoch checkpoint. Works independently of --start_from_stage.")
    parser.add_argument("--start_from_stage", type=int, default=1,
                       help="Stage number to start training from (1=Stage 1, 2=Stage 2, 3=Stage 3). "
                            "Stages before this will be skipped. Can start from any stage with or without a checkpoint.")

    return parser.parse_args()


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    epoch,
    stage_name,
    config,
    start_global_step=0,
    use_deepspeed=False,
    rank=0,
):
    """Train for one epoch."""
    if hasattr(model, 'train'):
        model.train()
    elif use_deepspeed:
        # DeepSpeed engine
        model.train()

    loss_meter = AverageMeter()

    # Only show progress bar on main process
    if is_main_process(rank):
        progress_bar = tqdm(dataloader, desc=f"{stage_name} - Epoch {epoch}")
    else:
        progress_bar = dataloader

    global_step = start_global_step
    saved = False

    for step, batch in enumerate(progress_bar):
        # Move batch to device (DeepSpeed handles this internally, but doesn't hurt)
        if not use_deepspeed:
            batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if use_deepspeed:
            # DeepSpeed training step
            loss = model(**batch).loss
            model.backward(loss)
            model.step()

            if model.is_gradient_accumulation_boundary():
                global_step += 1
                grad_norm = 0.0  # DeepSpeed handles gradient clipping internally
        else:
            # Regular training step
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

            # Log to wandb after optimizer step
            if config.use_wandb and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "train/loss": loss.item() * config.gradient_accumulation_steps,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm.item(),
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/stage": stage_name,
                    },
                    step=global_step,
                )

        # Update metrics
        loss_meter.update(loss.item() * config.gradient_accumulation_steps)

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

        # Logging
        if global_step % config.logging_steps == 0 and global_step > start_global_step:
            print(f"\n{stage_name} - Step {global_step} | Loss: {loss_meter.avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if global_step % config.save_steps == 0 and global_step > start_global_step:
            if saved == False:
                checkpoint_path = os.path.join(
                    config.output_dir,
                    stage_name,
                    f"checkpoint-epoch{epoch}-step{global_step}"
                )
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_path)
                saved = True
        else:
            saved = False

    return loss_meter.avg, global_step


def load_stage_checkpoint(model, checkpoint_path: str):
    """Load a stage checkpoint into the model.

    Args:
        model: CADMLLMModel instance
        checkpoint_path: Path to stage checkpoint directory

    Returns:
        True if checkpoint was loaded successfully, False otherwise
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint path does not exist: {checkpoint_path}")
        return False

    print(f"\nLoading checkpoint from: {checkpoint_path}")

    try:
        # Load the model using the from_pretrained method
        loaded_model = CADMLLMModel.from_pretrained(str(checkpoint_path))

        # Transfer the loaded state to our model
        # LLM (with LoRA)
        model.llm.load_state_dict(loaded_model.llm.state_dict())
        print("  ✓ Loaded LLM with LoRA adapters")

        # Projectors (if they exist in checkpoint)
        if loaded_model.image_projector is not None:
            if model.image_projector is None:
                model.enable_image_encoder()
                model.enable_image_projector()
            model.image_projector.load_state_dict(loaded_model.image_projector.state_dict())
            print("  ✓ Loaded image module")

        if loaded_model.point_projector is not None:
            if model.point_projector is None:
                model.enable_point_encoder()
                model.enable_point_projector()
            model.point_projector.load_state_dict(loaded_model.point_projector.state_dict())
            print("  ✓ Loaded point cloud module")

        del loaded_model
        torch.cuda.empty_cache()
        print(f"Checkpoint loaded successfully!\n")
        return True

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


def train_curriculum_stage(
    model,
    stage: CurriculumStage,
    train_dataset,
    collator,
    config: CurriculumTrainingConfig,
    start_global_step: int = 0,
    args=None,
    rank=0,
    world_size=1,
):
    """Train a single curriculum stage."""
    if is_main_process(rank):
        print(f"\n{'='*80}")
        print(f"Starting {stage.name}")
        print(f"Modalities: {stage.modalities}")
        print(f"Epochs: {stage.num_epochs}")
        print(f"Learning Rate: {stage.learning_rate}")
        print(f"Train Projectors: {stage.train_projectors}")
        print(f"{'='*80}\n")

    # Enable required modalities
    if "point_cloud" in stage.modalities:
        if not model.has_point_encoder:
            if is_main_process(rank):
                print("Enabling point cloud encoder and projector...")
            model.enable_point_encoder()
            model.enable_point_projector()

    if "image" in stage.modalities:
        if not model.has_image_encoder:
            if is_main_process(rank):
                print("Enabling image encoder and projector...")
            model.enable_image_encoder()
            model.enable_image_projector()

    # Set trainable parameters for this stage
    model.set_trainable_params(
        train_llm=True,  # Always train LLM (LoRA)
        train_projectors=stage.train_projectors,
        train_encoders=stage.train_encoders,
    )

    # Update dataset modality sampling probabilities
    if hasattr(train_dataset, 'modality_sample_probs'):
        train_dataset.modality_sample_probs = stage.modality_sample_probs
        train_dataset.available_modalities = stage.modalities
        if is_main_process(rank):
            print(f"Updated dataset sampling: {stage.modality_sample_probs}")

    # Create distributed sampler if using multiple GPUs
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

    # Create dataloader for this stage
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
    )

    # Get parameter groups with different learning rates
    param_groups = model.get_trainable_parameters()

    optimizer_params = []
    if param_groups['llm']:
        optimizer_params.append({
            'params': param_groups['llm'],
            'lr': stage.learning_rate,
        })
    if param_groups['projectors']:
        optimizer_params.append({
            'params': param_groups['projectors'],
            'lr': stage.learning_rate * config.projector_lr_multiplier,
        })
    if param_groups['encoders']:
        optimizer_params.append({
            'params': param_groups['encoders'],
            'lr': stage.learning_rate,
        })

    # Setup optimizer and DeepSpeed
    if args and args.deepspeed and DEEPSPEED_AVAILABLE:
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

        if is_main_process(rank):
            print(f"DeepSpeed initialized with ZeRO stage {ds_config.get('zero_optimization', {}).get('stage', 'N/A')}")
    else:
        # Regular optimizer setup
        optimizer = torch.optim.AdamW(
            optimizer_params,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        num_training_steps = len(train_dataloader) * stage.num_epochs // config.gradient_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        model_engine = model
        use_deepspeed = False

    if is_main_process(rank):
        print(f"Trainable LLM params: {len(param_groups['llm'])}")
        print(f"Trainable projector params: {len(param_groups['projectors'])}")
        print(f"Trainable encoder params: {len(param_groups['encoders'])}")
        if not use_deepspeed:
            print(f"Total training steps: {num_training_steps}\n")
        print_model_info(model)

    # Training loop for this stage
    global_step = start_global_step
    for epoch in range(stage.num_epochs):
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

        if is_main_process(rank):
            print(f"\n{stage.name} - Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")

        # Log epoch metrics to wandb
        if config.use_wandb and WANDB_AVAILABLE and is_main_process(rank):
            wandb.log(
                {
                    f"epoch/{stage.name}_loss": avg_loss,
                    "epoch/epoch": epoch,
                    "epoch/stage": stage.name,
                },
                step=global_step,
            )

        # Save checkpoint at end of epoch
        if is_main_process(rank):
            checkpoint_path = os.path.join(config.output_dir, stage.name, f"checkpoint-epoch{epoch}")
            if use_deepspeed:
                model_engine.save_checkpoint(checkpoint_path)
            else:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_path)

    # Save stage model
    if is_main_process(rank):
        stage_model_path = os.path.join(config.output_dir, f"{stage.name}_model")
        if use_deepspeed:
            model_engine.save_checkpoint(stage_model_path)
        else:
            model.save_pretrained(stage_model_path)
        print(f"\n{stage.name} completed! Model saved to {stage_model_path}")

    return global_step


def main():
    """Main training function."""
    args = parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    args.local_rank = local_rank

    # Update device for this process
    if args.deepspeed or world_size > 1:
        args.device = f"cuda:{local_rank}"

    # Set seed (different per rank for data shuffling)
    set_seed(args.seed + rank)

    # Setup data
    use_dummy = args.create_dummy_data

    if use_dummy:
        if is_main_process(rank):
            print("Creating dummy dataset...")
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)

        train_path = data_dir / "train_dummy.json"
        val_path = data_dir / "val_dummy.json"

        if is_main_process(rank):
            create_dummy_dataset(str(train_path), num_samples=args.num_dummy_samples)
            create_dummy_dataset(str(val_path), num_samples=20)

        # Wait for main process to create files
        if world_size > 1:
            torch.distributed.barrier()

        args.train_data_path = str(train_path)
        args.val_data_path = str(val_path)
    elif args.train_data_path is None:
        args.train_data_path = args.omnicad_txt_path

    # Create configurations
    model_config = CADMLLMConfig(
        llm_model_name=args.llm_model_name,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    # Create curriculum stages
    curriculum_stages = [
        CurriculumStage(
            name="stage1_text",
            modalities=["text"],
            num_epochs=args.stage1_epochs,
            learning_rate=args.stage1_lr,
            train_projectors=False,
            modality_sample_probs={"text": 1.0},
        ),
        CurriculumStage(
            name="stage2_text_pc",
            modalities=["text", "point_cloud"],
            num_epochs=args.stage2_epochs,
            learning_rate=args.stage2_lr,
            train_projectors=True,
            modality_sample_probs={
                "text": 0.3,
                "text+point_cloud": 0.7,
            },
        ),
        CurriculumStage(
            name="stage3_all",
            modalities=["text", "point_cloud", "image"],
            num_epochs=args.stage3_epochs,
            learning_rate=args.stage3_lr,
            train_projectors=True,
            modality_sample_probs={
                "text": 0.2,
                "text+point_cloud": 0.3,
                "text+image": 0.2,
                "text+point_cloud+image": 0.3,
            },
        ),
    ]

    train_config = CurriculumTrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        use_wandb=args.use_wandb,
        seed=args.seed,
        device=args.device,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        enable_curriculum=args.enable_curriculum,
        curriculum_stages=curriculum_stages,
        projector_lr_multiplier=args.projector_lr_multiplier,
    )

    # Initialize model
    if is_main_process(rank):
        print("\nInitializing CAD-MLLM model...")

    # For DeepSpeed, don't move model to device yet
    if not args.deepspeed:
        model_config.device = args.device
    else:
        model_config.device = "cpu"  # DeepSpeed handles device placement

    model = CADMLLMModel(model_config)

    # Load checkpoint if provided
    if args.resume_from_ckpt:
        checkpoint_path = Path(args.resume_from_ckpt)

        if is_main_process(rank):
            print(f"\n{'='*80}")
            print(f"RESUMING FROM CHECKPOINT")
            print(f"{'='*80}")

        success = load_stage_checkpoint(model, str(checkpoint_path))

        if is_main_process(rank):
            if not success:
                print("Warning: Failed to load checkpoint. Starting from scratch.")
            else:
                print(f"Successfully loaded checkpoint from {checkpoint_path}")
                print(f"Will start training from Stage {args.start_from_stage}")
            print(f"{'='*80}\n")

    # Apply training configurations (after checkpoint load to ensure they override checkpoint settings)

    # Verify attention implementation
    if is_main_process(rank):
        if hasattr(model.llm.config, '_attn_implementation'):
            actual_attn = model.llm.config._attn_implementation
            print(f"Current attention implementation: {actual_attn}")
            if args.attn_implementation and actual_attn != args.attn_implementation:
                print(f"  ⚠ Warning: Requested {args.attn_implementation} but model is using {actual_attn}")
        elif hasattr(model.llm.config, 'attn_implementation'):
            actual_attn = model.llm.config.attn_implementation
            print(f"Current attention implementation: {actual_attn}")

    # Enable gradient checkpointing if requested
    if args.use_gradient_checkpointing:
        if is_main_process(rank):
            print("Enabling gradient checkpointing for memory efficiency...")
        try:
            # For PEFT/LoRA models, enable on the base model
            if hasattr(model.llm, 'enable_input_require_grads'):
                model.llm.enable_input_require_grads()
            if hasattr(model.llm, 'gradient_checkpointing_enable'):
                model.llm.gradient_checkpointing_enable()
                if is_main_process(rank):
                    print("  ✓ Gradient checkpointing enabled on LLM")
            else:
                if is_main_process(rank):
                    print("  ⚠ Warning: Model does not support gradient_checkpointing_enable")
        except Exception as e:
            if is_main_process(rank):
                print(f"  ⚠ Warning: Could not enable gradient checkpointing: {e}")

    if is_main_process(rank):
        print("\nBase model:")
        print_model_info(model)
        verify_lora_training(model)

    # Create dataset and collator
    if is_main_process(rank):
        print("\nLoading datasets...")
    if use_dummy:
        # Use simple dummy dataset for text-only initially
        train_dataset = DummyCADDataset(
            data_path=train_config.train_data_path,
            tokenizer=model.tokenizer,
            max_seq_length=model_config.max_seq_length,
        )
        collator = CADCollator(model.tokenizer)
    else:
        # Use multimodal dataset
        train_dataset = MultimodalCADDataset(
            data_path=train_config.train_data_path,
            json_root=args.omnicad_json_root,
            image_root=args.omnicad_img_root,
            pc_root=args.omnicad_pc_root,
            tokenizer=model.tokenizer,
            max_seq_length=model_config.max_seq_length,
            available_modalities=["text"],  # Start with text only
            modality_sample_probs={"text": 1.0},
        )
        from transformers import AutoImageProcessor
        img_processor = AutoImageProcessor.from_pretrained(model_config.image_encoder_name)
        collator = MultimodalCADCollator(
            model.tokenizer,
            max_seq_length=model_config.max_seq_length,
            image_processor=img_processor
        )

    if is_main_process(rank):
        print(f"Training samples: {len(train_dataset)}")

    # Initialize wandb
    if args.use_wandb and is_main_process(rank):
        if not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Skipping wandb logging.")
            train_config.use_wandb = False
        else:
            # Auto-generate run name if not provided
            model_name = args.llm_model_name.split("/")[-1] if "/" in args.llm_model_name else args.llm_model_name
            run_name = f"{model_name}-curriculum-{args.stage1_epochs}+{args.stage2_epochs}+{args.stage3_epochs}ep-{args.max_seq_length}"
            run_name = args.wandb_run_name + run_name

            wandb.init(
                project=args.wandb_project,
                name=run_name,
                entity=args.wandb_entity,
                config={
                    # Model config
                    "llm_model_name": args.llm_model_name,
                    "use_lora": args.use_lora,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "max_seq_length": args.max_seq_length,
                    # Training config
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
                    "warmup_steps": args.warmup_steps,
                    "seed": args.seed,
                    "device": args.device,
                    "dtype": args.dtype,
                    # Curriculum config
                    "enable_curriculum": args.enable_curriculum,
                    "stage1_epochs": args.stage1_epochs,
                    "stage2_epochs": args.stage2_epochs,
                    "stage3_epochs": args.stage3_epochs,
                    "stage1_lr": args.stage1_lr,
                    "stage2_lr": args.stage2_lr,
                    "stage3_lr": args.stage3_lr,
                    "projector_lr_multiplier": args.projector_lr_multiplier,
                    # Data config
                    "train_data_path": args.train_data_path,
                    "num_train_samples": len(train_dataset),
                    "use_dummy_data": use_dummy,
                },
            )
            wandb.watch(model, log="all", log_freq=args.logging_steps)
            print(f"Wandb run name: {run_name}")
            print(f"Wandb run URL: {wandb.run.url}")

    # Curriculum training
    if is_main_process(rank):
        print("\n" + "="*80)
        print("STARTING CURRICULUM TRAINING")
        if args.start_from_stage > 1:
            print(f"Resuming from Stage {args.start_from_stage} (skipping Stages 1-{args.start_from_stage-1})")
        print("="*80)

    global_step = 0
    for stage_idx, stage in enumerate(train_config.curriculum_stages):
        stage_number = stage_idx + 1  # Convert to 1-indexed for display

        # Skip stages if resuming from a checkpoint (convert user input to 0-indexed)
        if stage_number < args.start_from_stage:
            if is_main_process(rank):
                print(f"\n{'#'*80}")
                print(f"# SKIPPING STAGE {stage_number}/{len(train_config.curriculum_stages)}: {stage.name}")
                print(f"# (Already completed in previous run)")
                print(f"{'#'*80}\n")
            continue

        if is_main_process(rank):
            print(f"\n\n{'#'*80}")
            print(f"# CURRICULUM STAGE {stage_number}/{len(train_config.curriculum_stages)}")
            print(f"{'#'*80}\n")

        global_step = train_curriculum_stage(
            model=model,
            stage=stage,
            train_dataset=train_dataset,
            collator=collator,
            config=train_config,
            start_global_step=global_step,
            args=args,
            rank=rank,
            world_size=world_size,
        )

    # Save final model
    if is_main_process(rank):
        final_path = os.path.join(train_config.output_dir, "final_model")
        model.save_pretrained(final_path)
        print(f"\n\n{'='*80}")
        print(f"CURRICULUM TRAINING COMPLETED!")
        print(f"Final model saved to {final_path}")
        print(f"{'='*80}\n")

    # Finish wandb run
    if train_config.use_wandb and WANDB_AVAILABLE and is_main_process(rank):
        wandb.finish()


if __name__ == "__main__":
    main()
