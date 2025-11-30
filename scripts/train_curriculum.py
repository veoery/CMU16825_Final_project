"""Curriculum-based training script for CAD-MLLM with progressive modality introduction."""

import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: uv add wandb")

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
from cad_mllm.data.multimodal_autocomplete import (
    MultimodalAutocompleteDataset,
    MultimodalAutocompleteCollator,
)
from transformers import AutoImageProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CAD-MLLM with Curriculum Learning")

    # Model arguments
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen3-8B", help="Name of the LLM model from HuggingFace")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce memory usage")

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
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=3, help="Keep only last N interval checkpoints (0 = keep all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--projector_lr_multiplier", type=float, default=5.0, help="LR multiplier for projectors")

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

    # Data arguments
    parser.add_argument("--create_dummy_data", action="store_true", help="Create dummy dataset")
    parser.add_argument("--num_dummy_samples", type=int, default=100, help="Number of dummy samples")
    parser.add_argument("--omnicad_txt_path", type=str, default="data/Omni-CAD/txt/", help="Path to Omni-CAD text descriptions")
    parser.add_argument("--omnicad_json_root", type=str, default="data/Omni-CAD/json", help="Root directory for Omni-CAD JSON files")
    parser.add_argument("--omnicad_img_root", type=str, default="data/Omni-CAD/img", help="Root directory for images")
    parser.add_argument("--omnicad_pc_root", type=str, default="data/Omni-CAD/pcd", help="Root directory for point clouds")

    # Autocompletion arguments
    parser.add_argument("--use_autocomplete_dataset", action="store_true", help="Use autocompletion dataset (truncated/full JSON pairs)")
    parser.add_argument("--truncated_json_root", type=str, default=None, help="Root directory for truncated JSON files (for autocompletion)")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples to use (for testing)")

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
    best_loss=float('inf'),
    interval_checkpoints=None,
):
    """Train for one epoch.

    Args:
        best_loss: Best loss seen so far (for saving best checkpoint)
        interval_checkpoints: List to track interval checkpoint paths (for cleanup)
    """
    model.train()

    loss_meter = AverageMeter()
    progress_bar = tqdm(dataloader, desc=f"{stage_name} - Epoch {epoch}")

    global_step = start_global_step

    # Track NaN batches
    total_batches = 0
    nan_batches = 0

    # Initialize interval checkpoints list if not provided
    if interval_checkpoints is None:
        interval_checkpoints = []

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Track NaN/Inf batches
        total_batches += 1
        is_nan = torch.isnan(loss) or torch.isinf(loss)
        if is_nan:
            nan_batches += 1

        # Calculate NaN percentage
        nan_pct = (nan_batches / total_batches * 100) if total_batches > 0 else 0

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
                        "train/nan_percentage": nan_pct,
                        "train/nan_count": nan_batches,
                        "train/total_batches": total_batches,
                    },
                    step=global_step,
                )

        # Update metrics - skip NaN/Inf losses to keep progress bar accurate
        if not is_nan:
            loss_meter.update(loss.item() * config.gradient_accumulation_steps)

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            "nan%": f"{nan_pct:.1f}",
        })

        # Logging
        if global_step % config.logging_steps == 0 and global_step > start_global_step:
            print(f"\n{stage_name} - Step {global_step} | Loss: {loss_meter.avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | NaN: {nan_batches}/{total_batches} ({nan_pct:.1f}%)")

        # Check if this is the best loss so far (only check on valid batches)
        if not is_nan and loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            best_checkpoint_path = os.path.join(
                config.output_dir,
                stage_name,
                "checkpoint-best"
            )
            print(f"\n🏆 New best loss: {best_loss:.4f} - Saving best checkpoint to {best_checkpoint_path}")
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_checkpoint_path, is_best=True, loss=best_loss, config=vars(config))

        # Save interval checkpoint
        if global_step % config.save_steps == 0 and global_step > start_global_step:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint-step{global_step}-loss{loss_meter.avg:.4f}-{timestamp}"
            checkpoint_path = os.path.join(
                config.output_dir,
                stage_name,
                checkpoint_name
            )
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_path, loss=loss_meter.avg, config=vars(config))

            # Track this checkpoint for cleanup
            interval_checkpoints.append(checkpoint_path)

            # Remove old checkpoints if we exceed the limit
            if config.keep_last_n_checkpoints > 0 and len(interval_checkpoints) > config.keep_last_n_checkpoints:
                old_checkpoint = interval_checkpoints.pop(0)  # Remove oldest
                if os.path.exists(old_checkpoint):
                    import shutil
                    shutil.rmtree(old_checkpoint)
                    print(f"🗑️  Removed old checkpoint: {os.path.basename(old_checkpoint)}")

    return loss_meter.avg, global_step, best_loss, interval_checkpoints


def load_stage_checkpoint(model, checkpoint_path: str, optimizer=None, scheduler=None):
    """Load a stage checkpoint into the model, optimizer, and scheduler.

    Args:
        model: CADMLLMModel instance
        checkpoint_path: Path to stage checkpoint directory
        optimizer: Optimizer instance to load state into (optional)
        scheduler: Scheduler instance to load state into (optional)

    Returns:
        Dictionary with loaded state info: {
            'success': bool,
            'epoch': int,
            'step': int,
            'loss': float,
        }
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint path does not exist: {checkpoint_path}")
        return {'success': False}

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

        # Load trainer state (optimizer, scheduler, metadata)
        trainer_state_path = checkpoint_path / "trainer_state.pt"
        result = {'success': True, 'epoch': 0, 'step': 0, 'loss': None}
        
        if trainer_state_path.exists():
            trainer_state = torch.load(trainer_state_path, map_location='cpu')
            
            # Load optimizer state
            if optimizer is not None and 'optimizer' in trainer_state:
                optimizer.load_state_dict(trainer_state['optimizer'])
                print("  ✓ Loaded optimizer state")
            
            # Load scheduler state
            if scheduler is not None and 'scheduler' in trainer_state and trainer_state['scheduler'] is not None:
                scheduler.load_state_dict(trainer_state['scheduler'])
                print("  ✓ Loaded scheduler state")
            
            # Extract metadata
            result['epoch'] = trainer_state.get('epoch', 0)
            result['step'] = trainer_state.get('step', 0)
            result['loss'] = trainer_state.get('loss', None)
            
            loss_str = f"{result['loss']:.4f}" if result['loss'] is not None else "N/A"
            print(f"  ✓ Resuming from epoch {result['epoch']}, step {result['step']}, loss {loss_str}")
        else:
            print("  ⚠️  No trainer_state.pt found - optimizer/scheduler not restored")

        print(f"Checkpoint loaded successfully!\n")
        return result

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False}


def train_curriculum_stage(
    model,
    stage: CurriculumStage,
    train_dataset,
    collator,
    config: CurriculumTrainingConfig,
    start_global_step: int = 0,
):
    """Train a single curriculum stage."""
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
            print("Enabling point cloud encoder and projector...")
            model.enable_point_encoder()
            model.enable_point_projector()

    if "image" in stage.modalities:
        if not model.has_image_encoder:
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
    # FIX: Dataset uses 'modality_probs', not 'modality_sample_probs'
    if hasattr(train_dataset, 'modality_probs'):
        train_dataset.modality_probs = stage.modality_sample_probs
        if hasattr(train_dataset, 'available_modalities'):
            train_dataset.available_modalities = stage.modalities
        print(f"Updated dataset sampling: {stage.modality_sample_probs}")

    # Create dataloader for this stage
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
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

    # Setup optimizer
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
    
    # Restore optimizer/scheduler state if resuming from checkpoint
    # Pass checkpoint info via config if available
    if hasattr(config, '_checkpoint_info') and config._checkpoint_info:
        checkpoint_path = Path(config._checkpoint_path)
        trainer_state_path = checkpoint_path / "trainer_state.pt"
        if trainer_state_path.exists():
            print(f"\n{'='*40}")
            print("Restoring optimizer/scheduler state...")
            print(f"{'='*40}")
            try:
                trainer_state = torch.load(trainer_state_path, map_location='cpu')
                
                # Only restore if state exists (backward compatibility)
                if 'optimizer' in trainer_state:
                    try:
                        optimizer.load_state_dict(trainer_state['optimizer'])
                        print("  ✓ Restored optimizer state")
                    except Exception as e:
                        print(f"  ⚠️  Could not restore optimizer state: {e}")
                        print("  → Continuing with fresh optimizer")
                
                if 'scheduler' in trainer_state and trainer_state['scheduler'] is not None:
                    try:
                        scheduler.load_state_dict(trainer_state['scheduler'])
                        print("  ✓ Restored scheduler state")
                    except Exception as e:
                        print(f"  ⚠️  Could not restore scheduler state: {e}")
                        print("  → Continuing with fresh scheduler")
                
                print(f"  ✓ Resuming from step {config._checkpoint_info.get('step', 0)}")
                print(f"{'='*40}\n")
            except Exception as e:
                print(f"  ⚠️  Error loading trainer state: {e}")
                print("  → Continuing with fresh optimizer/scheduler")
                print(f"{'='*40}\n")
        else:
            print(f"\n{'='*40}")
            print("No trainer_state.pt found in checkpoint")
            print("Using fresh optimizer/scheduler")
            print("(This is normal for older checkpoints)")
            print(f"{'='*40}\n")
        
        # Clear the checkpoint info after using it (only restore once per stage)
        delattr(config, '_checkpoint_info')
        delattr(config, '_checkpoint_path')

    print(f"Trainable LLM params: {len(param_groups['llm'])}")
    print(f"Trainable projector params: {len(param_groups['projectors'])}")
    print(f"Trainable encoder params: {len(param_groups['encoders'])}")
    print(f"Total training steps: {num_training_steps}\n")

    # Training loop for this stage
    global_step = start_global_step
    best_loss = float('inf')  # Track best loss for this stage
    interval_checkpoints = []  # Track interval checkpoints for cleanup

    for epoch in range(stage.num_epochs):
        avg_loss, global_step, best_loss, interval_checkpoints = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            epoch,
            stage.name,
            config,
            start_global_step=global_step,
            best_loss=best_loss,
            interval_checkpoints=interval_checkpoints,
        )

        print(f"\n{stage.name} - Epoch {epoch} completed | Average Loss: {avg_loss:.4f}")

        # Log epoch metrics to wandb
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    f"epoch/{stage.name}_loss": avg_loss,
                    "epoch/epoch": epoch,
                    "epoch/stage": stage.name,
                },
                step=global_step,
            )

        # Save checkpoint at end of epoch
        checkpoint_path = os.path.join(config.output_dir, stage.name, f"checkpoint-epoch{epoch}")
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, checkpoint_path)

    # Save stage model
    stage_model_path = os.path.join(config.output_dir, f"{stage.name}_model")
    model.save_pretrained(stage_model_path)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Stage {stage.name} completed!")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Best checkpoint saved at: {os.path.join(config.output_dir, stage.name, 'checkpoint-best')}")
    print(f"{'='*80}\n")
    print(f"\n{stage.name} completed! Model saved to {stage_model_path}")

    return global_step


def main():
    """Main training function."""
    args = parse_args()

    # If running in wandb sweep, override args with wandb.config
    if WANDB_AVAILABLE and wandb.run is not None and hasattr(wandb.config, 'keys'):
        print("\n" + "="*80)
        print("WANDB SWEEP DETECTED - Using wandb.config parameters")
        print("="*80)
        for key in wandb.config.keys():
            if hasattr(args, key):
                old_value = getattr(args, key)
                new_value = wandb.config[key]
                if old_value != new_value:
                    setattr(args, key, new_value)
                    print(f"  {key}: {old_value} → {new_value}")
        print("="*80 + "\n")

    # Set seed
    set_seed(args.seed)

    # Setup data
    use_dummy = args.create_dummy_data

    if use_dummy:
        print("Creating dummy dataset...")
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)

        train_path = data_dir / "train_dummy.json"
        val_path = data_dir / "val_dummy.json"

        create_dummy_dataset(str(train_path), num_samples=args.num_dummy_samples)
        create_dummy_dataset(str(val_path), num_samples=20)

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
    print("\nInitializing CAD-MLLM model...")
    model = CADMLLMModel(model_config)
    
    # Variables to track loaded checkpoint state
    loaded_checkpoint_info = None
    checkpoint_path_for_resume = None
    
    # Load checkpoint if provided (model weights only initially)
    if args.resume_from_ckpt:
        checkpoint_path = Path(args.resume_from_ckpt)
        # If relative path, assume it's in output_dir
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(args.output_dir) / args.resume_from_ckpt

        print(f"\n{'='*80}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*80}")
        # Load model only first, optimizer/scheduler will be loaded after they're created
        loaded_checkpoint_info = load_stage_checkpoint(model, str(checkpoint_path))
        if not loaded_checkpoint_info['success']:
            print("Warning: Failed to load checkpoint. Starting from scratch.")
            loaded_checkpoint_info = None
        else:
            print(f"Successfully loaded model weights from {checkpoint_path}")
            print(f"Will restore optimizer/scheduler after initialization")
            print(f"Will start training from Stage {args.start_from_stage}")
            checkpoint_path_for_resume = checkpoint_path
        print(f"{'='*80}\n")
        
    print_model_info(model)
    verify_lora_training(model)

    # Enable gradient checkpointing if requested
    if args.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Create dataset and collator
    print("\nLoading datasets...")
    if use_dummy:
        # Use simple dummy dataset for text-only initially
        train_dataset = DummyCADDataset(
            data_path=train_config.train_data_path,
            tokenizer=model.tokenizer,
            max_seq_length=model_config.max_seq_length,
        )
        collator = CADCollator(model.tokenizer)
    elif args.use_autocomplete_dataset:
        # Use multimodal autocompletion dataset
        if not args.truncated_json_root:
            raise ValueError("--truncated_json_root is required when using --use_autocomplete_dataset")

        print(f"Using autocompletion dataset with truncated JSONs from {args.truncated_json_root}")
        train_dataset = MultimodalAutocompleteDataset(
            data_path=train_config.train_data_path,
            truncated_json_root=args.truncated_json_root,
            full_json_root=args.omnicad_json_root,
            image_root=args.omnicad_img_root,
            pc_root=args.omnicad_pc_root,
            modality_probs={"text": 0.1, "text+pc": 0.3, "text+img": 0.3, "text+pc+img": 0.3},
            max_samples=args.max_train_samples,
        )
        # Create image processor for DINOv2 normalization
        image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        collator = MultimodalAutocompleteCollator(
            tokenizer=model.tokenizer,
            max_seq_length=model_config.max_seq_length,
            image_processor=image_processor,
        )
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
        collator = MultimodalCADCollator(model.tokenizer)

    print(f"Training samples: {len(train_dataset)}")

    # Initialize wandb
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Skipping wandb logging.")
            train_config.use_wandb = False
        else:
            # Auto-generate run name if not provided
            if args.wandb_run_name is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = args.llm_model_name.split("/")[-1] if "/" in args.llm_model_name else args.llm_model_name
                run_name = f"{model_name}-curriculum-{args.stage1_epochs}+{args.stage2_epochs}+{args.stage3_epochs}ep-{timestamp}"
            else:
                run_name = args.wandb_run_name

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
    print("\n" + "="*80)
    print("STARTING CURRICULUM TRAINING")
    if args.start_from_stage > 1:
        print(f"Resuming from Stage {args.start_from_stage} (skipping Stages 1-{args.start_from_stage-1})")
    print("="*80)

    # Pass checkpoint info to config for optimizer/scheduler restoration
    if loaded_checkpoint_info and checkpoint_path_for_resume:
        train_config._checkpoint_info = loaded_checkpoint_info
        train_config._checkpoint_path = checkpoint_path_for_resume

    global_step = 0
    for stage_idx, stage in enumerate(train_config.curriculum_stages):
        stage_number = stage_idx + 1  # Convert to 1-indexed for display

        # Skip stages if resuming from a checkpoint (convert user input to 0-indexed)
        if stage_number < args.start_from_stage:
            print(f"\n{'#'*80}")
            print(f"# SKIPPING STAGE {stage_number}/{len(train_config.curriculum_stages)}: {stage.name}")
            print(f"# (Already completed in previous run)")
            print(f"{'#'*80}\n")
            continue

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
        )

    # Save final model
    final_path = os.path.join(train_config.output_dir, "final_model")
    model.save_pretrained(final_path)
    print(f"\n\n{'='*80}")
    print(f"CURRICULUM TRAINING COMPLETED!")
    print(f"Final model saved to {final_path}")
    print(f"{'='*80}\n")

    # Finish wandb run
    if train_config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
